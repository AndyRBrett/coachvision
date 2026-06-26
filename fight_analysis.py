#!/usr/bin/env python3
"""Pose-driven fight analysis: lock onto the fighters, count strike attempts.

This replaces the crude motion-energy detector (which averaged *all* moving
pixels, so background people and camera shake dragged the tracked point around)
with a real detection+pose model. For each frame we:

  1. detect every person (YOLOv8-pose) and **pick the fighters** -- the 1-2
     largest, most central people -- ignoring ringside/background people; then
  2. read each fighter's skeleton and flag **strike attempts** from rapid
     wrist/ankle extensions (a hand-strike when a wrist snaps out fast, a
     leg-strike when an ankle does), debounced so one strike counts once.

The heavy bits (ultralytics + opencv) are imported lazily inside ``analyze`` /
``render_overlay``; the decision logic (who's a fighter, what's a strike) is in
pure functions below so it is unit-tested without a GPU or any model. Everything
runs on CPU.

Strike detection is deliberately honest: it counts *attempts* and splits
hand vs leg. It does NOT name techniques (jab vs cross) -- that needs an
action-recognition model and labelled data, which isn't free/CPU-friendly.
"""
import math
import os
import subprocess
import tempfile

DEFAULT_MODEL = "yolov8n-pose.pt"
DEFAULT_FPS = 10.0
DEFAULT_WIDTH = 640          # normalize to this width before inference
DEFAULT_CONF = 0.25
MIN_KP_CONF = 0.3            # ignore keypoints the model isn't sure about
MAX_FIGHTERS = 2

# COCO-17 keypoint indices.
L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16
HAND_KPS = (L_WRIST, R_WRIST)
LEG_KPS = (L_ANKLE, R_ANKLE)

# Strike heuristic: limb speed is measured in fighter-heights per second so it's
# scale-invariant. A snap above this with a refractory gap counts as one strike.
STRIKE_SPEED_THRESH = 2.5
STRIKE_REFRACTORY_S = 0.25

# Skeleton edges for drawing (COCO-17).
_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]


# --------------------------------------------------------------------------
# Pure decision logic (unit-tested without any model)
# --------------------------------------------------------------------------
def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_fighters(persons, frame_w, frame_h, max_fighters=MAX_FIGHTERS,
                    min_area_frac=0.20, min_height_frac=0.30):
    """Pick the fighters from all detected people: the engaged pair.

    Fighters share two traits a referee and the crowd don't: they're *large* in
    frame and *engaged with each other* -- close together, clashing. We use both:

      1. Keep only people big enough to be a fighter -- box height at least
         ``min_height_frac`` of the frame AND area at least ``min_area_frac`` of
         the biggest detection. This drops ringside/background people outright
         (they're small) and most of the time the referee too (often turned/
         crouched, so shorter than the upright fighters).
      2. If more candidates than slots remain (e.g. two fighters + a referee who
         is also large), keep the most *engaged pair*: the two whose boxes are
         closest together, rewarding combined size. A referee standing off the
         clash is farther from either fighter than they are from each other, so
         the pair selection drops him.

    Each person is a dict with at least ``box`` ([x1,y1,x2,y2]). Returns up to
    ``max_fighters`` persons, biggest first.
    """
    if not persons:
        return []
    biggest = max(_box_area(p["box"]) for p in persons) or 1.0
    diag = math.hypot(frame_w, frame_h) or 1.0

    candidates = []
    for p in persons:
        box = p["box"]
        if (box[3] - box[1]) < min_height_frac * frame_h:
            continue  # too short -> background / ringside / crouched non-fighter
        if _box_area(box) < min_area_frac * biggest:
            continue  # much smaller than the principals -> background
        candidates.append(p)

    candidates.sort(key=lambda p: _box_area(p["box"]), reverse=True)
    if len(candidates) <= max_fighters:
        return candidates

    # Too many large people (fighters + a referee/cornerman). For the usual two
    # fighters, take the most engaged pair -- big and close together. Otherwise
    # fall back to biggest-first (already sorted).
    if max_fighters != 2:
        return candidates[:max_fighters]

    best_pair, best_score = None, -1.0
    for i in range(len(candidates)):
        ca = _box_center(candidates[i]["box"])
        for j in range(i + 1, len(candidates)):
            cb = _box_center(candidates[j]["box"])
            dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / diag
            combined = _box_area(candidates[i]["box"]) + _box_area(candidates[j]["box"])
            score = combined / (1.0 + 3.0 * dist)   # big + close ranks highest
            if score > best_score:
                best_pair, best_score = (candidates[i], candidates[j]), score
    pair = sorted(best_pair, key=lambda p: _box_area(p["box"]), reverse=True)
    return pair


def _kp_xy(kpts, idx):
    """Return (x, y) for keypoint ``idx`` if confident enough, else None."""
    if idx >= len(kpts):
        return None
    x, y, c = kpts[idx]
    return (x, y) if c >= MIN_KP_CONF else None


def detect_strikes(frame_records, speed_thresh=STRIKE_SPEED_THRESH,
                   refractory_s=STRIKE_REFRACTORY_S):
    """Flag strike attempts from per-frame fighter keypoints.

    ``frame_records`` is a list of {"t": seconds, "fighters": [fighter, ...]}
    where each fighter has ``id``, ``box`` and ``kpts`` (17x[x,y,conf]). For each
    fighter+limb we track speed in fighter-heights/sec between confident samples;
    a sample whose speed crosses ``speed_thresh`` (with a per-limb refractory gap)
    is one strike. Returns events: {"t","type","pos","fighter"} with type
    ``hand_strike`` or ``leg_strike``.
    """
    last = {}    # (fighter_id, group) -> last fire time
    prev = {}    # (fighter_id, kp_idx) -> (t, x, y)
    events = []

    for rec in frame_records:
        t = rec["t"]
        for f in rec.get("fighters", []):
            fid = f.get("id", 0)
            box = f["box"]
            height = max(1.0, box[3] - box[1])  # fighter height for scale
            for group, idxs in (("hand_strike", HAND_KPS), ("leg_strike", LEG_KPS)):
                for idx in idxs:
                    xy = _kp_xy(f["kpts"], idx)
                    key = (fid, idx)
                    if xy is None:
                        prev.pop(key, None)
                        continue
                    if key in prev:
                        pt, px, py = prev[key]
                        dt = t - pt
                        if dt > 0:
                            speed = math.hypot(xy[0] - px, xy[1] - py) / height / dt
                            gkey = (fid, group)
                            if speed >= speed_thresh and (t - last.get(gkey, -1e9)) >= refractory_s:
                                events.append({
                                    "t": round(t, 3),
                                    "type": group,
                                    "pos": [round(xy[0], 1), round(xy[1], 1)],
                                    "fighter": fid,
                                })
                                last[gkey] = t
                    prev[key] = (t, xy[0], xy[1])
    return events


def build_tracking(frame_records, width, height, fps, source=None, domain="martial_arts"):
    """Turn per-frame fighter records into the pipeline's tracking schema.

    The per-frame ``subject`` point is the centroid of the fighters' box centres
    (or null when no fighter is in frame -- a real lull), so the existing
    segmentation/speed code keys on the *fighters* rather than all motion.
    Strike events come from :func:`detect_strikes`.
    """
    frames = []
    for i, rec in enumerate(frame_records):
        fighters = rec.get("fighters", [])
        if fighters:
            centers = [_box_center(f["box"]) for f in fighters]
            subject = [round(sum(c[0] for c in centers) / len(centers), 2),
                       round(sum(c[1] for c in centers) / len(centers), 2)]
        else:
            subject = None
        frames.append({"frame": i, "t": round(rec["t"], 4), "subject": subject})

    detected = sum(1 for f in frames if f["subject"] is not None)
    return {
        "fps": fps,
        "source": source,
        "domain": domain,
        "width": width,
        "height": height,
        "frame_count": len(frames),
        "detected_frames": detected,
        "frames": frames,
        "events": detect_strikes(frame_records),
    }


# --------------------------------------------------------------------------
# Model + video I/O (lazy deps: ultralytics, opencv, ffmpeg)
# --------------------------------------------------------------------------
def _normalize_video(src, fps, width):
    """ffmpeg the source to a fixed fps/width mp4 so inference cost is bounded."""
    tmp = tempfile.mkstemp(prefix="coachvision_norm_", suffix=".mp4")[1]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", f"fps={fps},scale={int(width)}:-2", "-an", tmp],
        check=True,
    )
    return tmp


def _persons_from_result(result):
    """Extract [{id, box, kpts}] for every detected person in a YOLO result."""
    persons = []
    boxes = getattr(result, "boxes", None)
    kpts = getattr(result, "keypoints", None)
    if boxes is None or kpts is None or boxes.xyxy is None:
        return persons
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)
    kdata = kpts.data.cpu().numpy()  # (n, 17, 3)
    for i in range(len(xyxy)):
        persons.append({
            "id": int(ids[i]) if ids[i] is not None else i,
            "box": [float(v) for v in xyxy[i]],
            "kpts": [[float(x), float(y), float(c)] for x, y, c in kdata[i]],
        })
    return persons


def analyze(video_path, fps=DEFAULT_FPS, width=DEFAULT_WIDTH, conf=DEFAULT_CONF,
            model=DEFAULT_MODEL, source_label=None):
    """Run pose tracking over ``video_path`` and return (tracking, records, norm).

    ``tracking`` is the pipeline schema (fighter-centroid frames + strike events);
    ``records`` are the per-frame fighter detections (reused for the overlay);
    ``norm`` is the normalized video the overlay should be drawn on. Caller is
    responsible for removing ``norm`` when done.
    """
    from ultralytics import YOLO

    norm = _normalize_video(video_path, fps, width)
    yolo = YOLO(model)
    records = []
    fw = fh = 0
    for i, result in enumerate(yolo.track(source=norm, stream=True, persist=True,
                                          conf=conf, verbose=False)):
        if result.orig_shape is not None:
            fh, fw = int(result.orig_shape[0]), int(result.orig_shape[1])
        persons = _persons_from_result(result)
        fighters = select_fighters(persons, fw or width, fh or width)
        records.append({"t": i / fps, "fighters": fighters})

    tracking = build_tracking(records, fw or int(width), fh or int(width), fps,
                              source=source_label)
    return tracking, records, norm


def render_overlay(norm_video, out_path, records, events, fps=DEFAULT_FPS):
    """Draw fighters-only boxes + skeletons + strike flashes onto ``norm_video``.

    Background people are never drawn (only the selected fighters are in
    ``records``). A strike event flashes a marker + HAND/LEG label near where it
    happened for a few frames. Re-encodes browser-friendly with ffmpeg.
    """
    import cv2

    events_by_frame = {}
    for ev in events:
        fi = int(round(ev["t"] * fps))
        for k in range(fi, fi + 3):  # hold the flash ~3 frames
            events_by_frame.setdefault(k, []).append(ev)

    cap = cv2.VideoCapture(norm_video)
    tmp = out_path + ".annot.avi"
    writer = None
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            rec = records[i] if i < len(records) else {"fighters": []}
            _draw_fighters(cv2, frame, rec.get("fighters", []))
            for ev in events_by_frame.get(i, []):
                _draw_strike(cv2, frame, ev)
            writer.write(frame)
            i += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if writer is None:
        return None
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        check=True,
    )
    try:
        os.remove(tmp)
    except OSError:
        pass
    return out_path


def _draw_fighters(cv2, frame, fighters):
    colors = [(0, 220, 255), (255, 180, 0)]  # one per fighter id slot
    for n, f in enumerate(fighters):
        color = colors[n % len(colors)]
        x1, y1, x2, y2 = (int(v) for v in f["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"fighter {f.get('id', n)}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        kpts = f["kpts"]
        for a, b in _SKELETON:
            pa, pb = _kp_xy(kpts, a), _kp_xy(kpts, b)
            if pa and pb:
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
        for idx in range(len(kpts)):
            p = _kp_xy(kpts, idx)
            if p:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, color, -1)


def _draw_strike(cv2, frame, ev):
    x, y = int(ev["pos"][0]), int(ev["pos"][1])
    label = "HAND" if ev["type"] == "hand_strike" else "LEG"
    cv2.circle(frame, (x, y), 16, (0, 0, 255), 3)
    cv2.putText(frame, label, (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Pose-analyze a clip (fighters + strikes).")
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = parser.parse_args()
    tracking, records, norm = analyze(args.video, fps=args.fps)
    os.remove(norm)
    print(json.dumps({"detected_frames": tracking["detected_frames"],
                      "frame_count": tracking["frame_count"],
                      "strikes": len(tracking["events"])}, indent=2))
