#!/usr/bin/env python3
"""Optional, sport-agnostic detection via Ultralytics YOLO + Roboflow supervision.

The stdlib ``detect.py`` recovers the tracked subject as the centroid of the
brightest blob. That is enough to prove the pipeline end-to-end on the synthetic
reference clip, but it does not survive real footage -- jerseys, lights, the
scoreboard and floor reflections are all "brighter blobs" than the thing you
care about. This module is the upgrade path: a real object detector (Ultralytics
YOLO) whose per-frame boxes are normalised by ``supervision``
(https://github.com/roboflow/supervision) into ``sv.Detections`` and given a
stable identity by supervision's ByteTrack, then emitted in the *exact* tracking
schema the rest of the pipeline already consumes.

Sport-agnostic by design
-------------------------
Nothing here is volleyball-specific. The detector tracks a single configurable
*target class*, so the same code extends to other sports by changing weights +
class name:

  * volleyball / basketball / golf -> a ball  (target "sports ball", or tuned
    weights with a "volleyball"/"basketball"/"golf ball" class)
  * karate / gymnastics            -> the athlete (target "person")

The pipeline's tracking schema still calls the per-frame position ``ball`` for
backward compatibility; read it as "the tracked subject's position".

Design constraints (mirrors cosmos_tagger.py)
--------------------------------------------
* Strictly optional. Importing this module never requires the heavy stack. The
  detector, supervision and OpenCV are imported lazily and only when a real run
  is requested; otherwise the pipeline falls back to detect.run_detection.
* The stdlib path stays the default and the self-test still runs on it, so CI
  remains green and dependency-free. This module is selected explicitly, for
  real video, via VOLLEYBALL_DETECTOR=supervision or pipeline --detector.
* Best-effort: a missing dependency raises a clear RuntimeError at setup time,
  but the core decision logic (target selection, class resolution) is pure
  Python so it is unit-tested without torch/opencv present.

Environment variables
---------------------
VOLLEYBALL_DETECTOR       Set to "supervision" to select this backend.
VOLLEYBALL_YOLO_MODEL     YOLO weights to load (default: yolov8n.pt). Point this
                          at sport-tuned weights for best results.
VOLLEYBALL_TARGET_CLASS   Class name to track (default: "sports ball"). Use
                          "person" for athlete-tracked sports like karate.
                          VOLLEYBALL_BALL_CLASS is accepted as a legacy alias.
VOLLEYBALL_YOLO_CONF      Minimum detection confidence to keep (default: 0.25).
"""
import os

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_TARGET_CLASS = "sports ball"   # COCO label; override per sport / weights
DEFAULT_CONF = 0.25
# COCO id for "sports ball"; used as a last resort when a model exposes no names.
COCO_SPORTS_BALL_ID = 32


def is_selected() -> bool:
    """True when this backend has been explicitly requested via env."""
    return os.environ.get("VOLLEYBALL_DETECTOR", "").strip().lower() == "supervision"


def is_available() -> bool:
    """True when the heavy stack (supervision + ultralytics) can be imported."""
    try:
        import supervision  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def target_class_name(override=None):
    """Resolve the configured target class name (sport-agnostic).

    Precedence: explicit override > VOLLEYBALL_TARGET_CLASS > the legacy
    VOLLEYBALL_BALL_CLASS alias > the default. Kept tiny and pure so callers and
    tests can resolve config without importing the heavy stack.
    """
    return (
        override
        or os.environ.get("VOLLEYBALL_TARGET_CLASS")
        or os.environ.get("VOLLEYBALL_BALL_CLASS")
        or DEFAULT_TARGET_CLASS
    )


def resolve_class_id(names, wanted):
    """Resolve a target class id from a model's ``names`` mapping.

    ``names`` is the ``{id: label}`` mapping Ultralytics exposes on a model.
    Matching is case-insensitive on the label. Returns the id, or the COCO
    sports-ball id as a last resort when nothing matches a ball-typed model, or
    None when there is no usable mapping at all.
    """
    wanted_norm = (wanted or "").strip().lower()
    if isinstance(names, dict) and names:
        for class_id, label in names.items():
            if str(label).strip().lower() == wanted_norm:
                return int(class_id)
        return COCO_SPORTS_BALL_ID if COCO_SPORTS_BALL_ID in names else None
    return COCO_SPORTS_BALL_ID


def select_target(boxes, confidences):
    """Pick the tracked subject's centroid from candidate boxes for one frame.

    ``boxes`` is a list of ``(x1, y1, x2, y2)`` already filtered to the target
    class; ``confidences`` is the parallel list of scores. The highest-confidence
    box wins and its centre is returned as ``[x, y]`` (rounded). Returns None when
    there are no candidates -- exactly the "subject out of play" signal that
    highlights.segment_rallies uses to split rallies.
    """
    best_idx = None
    best_conf = None
    for i, conf in enumerate(confidences):
        if best_conf is None or conf > best_conf:
            best_conf = conf
            best_idx = i
    if best_idx is None:
        return None
    x1, y1, x2, y2 = boxes[best_idx]
    return [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)]


def _require_stack():
    """Import and return (supervision, YOLO) or raise a clear RuntimeError."""
    try:
        import supervision as sv
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "the supervision backend needs `supervision` and `ultralytics` "
            "installed (pip install supervision ultralytics)"
        ) from exc
    return sv, YOLO


def run_detection(
    clip_path,
    fps=None,
    events=None,
    source=None,
    model_path=None,
    target_class=None,
    conf=None,
):
    """Detect the target's track in a real video clip and return a tracking dict.

    The returned dict matches detect.run_detection's schema exactly (``fps``,
    ``source``, ``width``, ``height``, ``frame_count``, ``detected_frames``,
    ``frames``, ``events``) so it is a drop-in replacement for the stdlib
    detector on real footage. Event detection stays out of scope -- coaching
    events are passed through from the sidecar, just like the geometric detector.
    """
    sv, YOLO = _require_stack()

    model_path = model_path or os.environ.get("VOLLEYBALL_YOLO_MODEL", DEFAULT_MODEL)
    target_class = target_class_name(target_class)
    conf = conf if conf is not None else float(os.environ.get("VOLLEYBALL_YOLO_CONF", DEFAULT_CONF))

    info = sv.VideoInfo.from_video_path(clip_path)
    video_fps = float(fps) if fps else float(info.fps)

    model = YOLO(model_path)
    class_id = resolve_class_id(getattr(model, "names", None), target_class)
    tracker = sv.ByteTrack(frame_rate=int(round(video_fps)))

    records = []
    detected = 0
    for i, frame in enumerate(sv.get_video_frames_generator(clip_path)):
        result = model(frame, conf=conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        if class_id is not None and detections.class_id is not None:
            detections = detections[detections.class_id == class_id]
        detections = tracker.update_with_detections(detections)

        boxes = [tuple(xyxy) for xyxy in detections.xyxy]
        confs = list(detections.confidence) if detections.confidence is not None else [1.0] * len(boxes)
        ball = select_target(boxes, confs)
        if ball is not None:
            detected += 1
        # Schema key stays "ball" for pipeline compatibility; it is the position
        # of whatever target class this sport tracks.
        records.append({"frame": i, "t": round(i / video_fps, 4), "ball": ball})

    return {
        "fps": video_fps,
        "source": source if source is not None else clip_path,
        "width": info.width,
        "height": info.height,
        "frame_count": len(records),
        "detected_frames": detected,
        "frames": records,
        "events": list(events or []),
    }
