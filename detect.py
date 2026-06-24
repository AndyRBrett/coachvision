#!/usr/bin/env python3
"""Subject detection: turn raw clip frames into tracking data (the CV front-end).

This is the stage that was missing -- the reason the pipeline had "processed 0
frames ever". Everything downstream (highlights.py, coaching.py) consumes
*tracking* data; nothing produced it from actual pixels. This module closes that
gap with deliberately small, dependency-free detectors so the pipeline can be
proven end-to-end in CI without ffmpeg, opencv, torch, or a GPU.

It reads a clip stored as a gzipped sequence of Netpbm P5/PGM frames (see
fixtures/make_reference_clip.py) and recovers the per-frame *subject point* the
rest of the pipeline tracks. Two detectors, picked by the active domain:

  * ``brightest_blob`` (volleyball) -- the subject is the ball, found as the
    centroid of the brightest blob in each frame.
  * ``motion_energy`` (martial arts) -- there is no ball; the subject is the
    fighter, found as the centroid of pixels that *changed* since the previous
    frame. Quiet moments produce no motion, so they read as gaps between
    exchanges -- the direct analogue of a ball going out of play.

Either way, frames with no subject yield a null position -- exactly the signal
highlights.segment_plays uses to split continuous play into segments. The
per-frame position is stored under the ``subject`` key (the ball in volleyball,
the fighter in martial arts).

The output schema matches the tracking JSON the rest of the pipeline already
expects (see highlights.py), with ``width``/``height`` added so the coaching
heatmap knows the surface dimensions, and ``domain`` recording which sport ran.
"""
import gzip
import json
import os

import domains

DEFAULT_THRESHOLD = 200   # pixel value at/above which a pixel is "ball-bright"
DEFAULT_MIN_PIXELS = 3     # fewer bright pixels than this -> treat as no ball


def _open_maybe_gzip(path):
    """Open ``path`` for binary reading, transparently decompressing .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def load_pgm_frames(path):
    """Load a (possibly gzipped) P5/PGM frame sequence.

    Returns ``(width, height, frames)`` where ``frames`` is a list of ``bytes``
    buffers, one per frame, each ``width * height`` bytes (row-major, 1 byte per
    pixel). All frames in a clip are required to share dimensions.
    """
    with _open_maybe_gzip(path) as fh:
        data = fh.read()

    frames = []
    width = height = None
    pos = 0
    n = len(data)
    while pos < n:
        magic, pos = _read_token(data, pos)
        if magic != b"P5":
            raise ValueError(f"unsupported frame magic {magic!r} (expected P5)")
        w_tok, pos = _read_token(data, pos)
        h_tok, pos = _read_token(data, pos)
        maxval_tok, pos = _read_token(data, pos)
        w, h = int(w_tok), int(h_tok)
        # Exactly one whitespace byte separates the header from the raster.
        pos += 1
        size = w * h
        raster = data[pos:pos + size]
        if len(raster) != size:
            raise ValueError("truncated PGM raster")
        pos += size
        if width is None:
            width, height = w, h
        elif (w, h) != (width, height):
            raise ValueError("inconsistent frame dimensions in clip")
        frames.append(raster)
    if width is None:
        raise ValueError("no frames found in clip")
    return width, height, frames


def _read_token(data, pos):
    """Read one whitespace-delimited token from ``data`` starting at ``pos``."""
    n = len(data)
    while pos < n and data[pos] in b" \t\r\n":
        pos += 1
    start = pos
    while pos < n and data[pos] not in b" \t\r\n":
        pos += 1
    return data[start:pos], pos


def detect_blob(raster, width, height, threshold=DEFAULT_THRESHOLD, min_pixels=DEFAULT_MIN_PIXELS):
    """Return the ``[x, y]`` centroid of the brightest blob, or None.

    Pixels at/above ``threshold`` are treated as ball-bright; their centroid is
    the detected ball position. When fewer than ``min_pixels`` qualify the ball
    is considered out of frame and None is returned.
    """
    count = 0
    sum_x = 0
    sum_y = 0
    for idx, value in enumerate(raster):
        if value >= threshold:
            count += 1
            sum_x += idx % width
            sum_y += idx // width
    if count < min_pixels:
        return None
    return [round(sum_x / count, 2), round(sum_y / count, 2)]


def detect_motion(prev_raster, raster, width, height, threshold=DEFAULT_THRESHOLD, min_pixels=DEFAULT_MIN_PIXELS):
    """Return the ``[x, y]`` centroid of pixels that changed since ``prev``, or None.

    Motion-energy detection for sports with no ball: a pixel is "moving" when its
    absolute difference from the previous frame is at/above ``threshold``. The
    centroid of the moving pixels follows the action (the fighter). When fewer
    than ``min_pixels`` moved -- a still moment between exchanges -- None is
    returned, mirroring the ball going out of play.
    """
    count = 0
    sum_x = 0
    sum_y = 0
    for idx, value in enumerate(raster):
        if abs(value - prev_raster[idx]) >= threshold:
            count += 1
            sum_x += idx % width
            sum_y += idx // width
    if count < min_pixels:
        return None
    return [round(sum_x / count, 2), round(sum_y / count, 2)]


def detect_frames(width, height, frames, domain=None, fps=10.0):
    """Run the domain's detector over every frame, returning tracking records.

    For ``brightest_blob`` each frame is detected independently. For
    ``motion_energy`` each frame is compared against its predecessor, so the
    very first frame has no motion reference and yields a null subject.
    """
    domain = domains.get_domain(domain)
    records = []
    prev = None
    for i, raster in enumerate(frames):
        if domain.detector == "motion_energy":
            pos = (
                None if prev is None
                else detect_motion(prev, raster, width, height,
                                   threshold=domain.detect_threshold,
                                   min_pixels=domain.detect_min_pixels)
            )
        else:
            pos = detect_blob(raster, width, height,
                              threshold=domain.detect_threshold,
                              min_pixels=domain.detect_min_pixels)
        records.append({"frame": i, "t": round(i / fps, 4), "subject": pos})
        prev = raster
    return records


def run_detection(clip_path, fps=10.0, events=None, source=None, domain=None):
    """Detect the subject track in ``clip_path`` and return a tracking dict.

    ``domain`` (a Domain, its name, or None for the configured default) selects
    the detector and tuning. The returned dict matches the tracking schema
    highlights.py consumes: ``fps``, ``source``, ``width``, ``height``,
    ``domain``, ``frames`` (per-frame subject position or null) and ``events``
    (passed through, since event detection is out of scope for these geometric
    detectors -- the reference clips bundle ground-truth events alongside the
    pixels).
    """
    domain = domains.get_domain(domain)
    width, height, frames = load_pgm_frames(clip_path)
    records = detect_frames(width, height, frames, domain=domain, fps=fps)
    detected = sum(1 for r in records if r["subject"] is not None)
    return {
        "fps": fps,
        "source": source if source is not None else clip_path,
        "domain": domain.key,
        "width": width,
        "height": height,
        "frame_count": len(records),
        "detected_frames": detected,
        "frames": records,
        "events": list(events or []),
    }


def _load_events_sidecar(path):
    """Load a bundled events sidecar, returning (events, fps_or_None, source)."""
    with open(path) as fh:
        doc = json.load(fh)
    return doc.get("events", []), doc.get("fps"), doc.get("source")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Detect the subject track from a clip and emit tracking JSON.")
    parser.add_argument("clip", help="Path to the clip (gzipped P5/PGM frame sequence)")
    parser.add_argument("--events", help="Optional bundled events sidecar JSON")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--domain", default=None,
                        help="Sport domain (volleyball|martial_arts); default from COACHVISION_DOMAIN")
    parser.add_argument("--output", help="Write tracking JSON here (default: stdout)")
    args = parser.parse_args()

    events = []
    fps = args.fps
    source = None
    if args.events:
        events, sidecar_fps, source = _load_events_sidecar(args.events)
        if sidecar_fps:
            fps = float(sidecar_fps)

    tracking = run_detection(args.clip, fps=fps, events=events, source=source, domain=args.domain)
    subject = domains.get_domain(tracking["domain"]).subject_noun
    text = json.dumps(tracking, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text + "\n")
        print(f"Wrote {args.output}: {tracking['detected_frames']}/{tracking['frame_count']} frames with {subject}")
    else:
        print(text)


if __name__ == "__main__":
    main()
