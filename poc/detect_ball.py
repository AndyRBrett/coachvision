#!/usr/bin/env python3
"""Ball + player detection proof-of-concept for volleyball footage.

This is step 1 of the pipeline: prove we can locate the ball and track players
on a video clip, frame by frame, and emit structured output. It deliberately
uses the stock YOLOv8 COCO model (pip-installable, no broken weight links) so it
runs anywhere on CPU. The COCO model gives us two relevant classes out of the
box:

    class 0  -> "person"       (players)
    class 32 -> "sports ball"  (the volleyball, approximately)

"Approximately" is the whole point of this PoC: COCO's sports-ball class was not
trained on volleyball footage, so on YOUR camera angle / lighting it will miss
fast-moving or partially-occluded balls. That is the expected, known limitation
(see README). When we want better ball accuracy, we swap `--model` for
volleyball-trained weights -- the rest of this script does not change.

Outputs:
  * an annotated .mp4 (player boxes + IDs, ball marker + recent trail)
  * a .json of per-frame detections (the structured data the metrics layer and
    Claude coaching call will eventually consume)

Designed to be slow-but-fine on an old Intel MacBook: small default model,
optional frame striding, optional frame cap for quick smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

# COCO class indices we care about.
PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect the ball and track players in a volleyball clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="Path to the input video clip (mp4/mov/etc).")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Annotated video path. Defaults to <input>_annotated.mp4")
    p.add_argument("--json", type=Path, default=None,
                   help="Per-frame detections JSON. Defaults to <input>_events.json")
    p.add_argument("--model", "-m", default="yolov8n.pt",
                   help="YOLO weights. yolov8n.pt = smallest/fastest (CPU-friendly). "
                        "Swap for volleyball-trained weights to improve ball accuracy.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detection confidence threshold.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size. Lower (e.g. 416) = faster, less accurate.")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame. >1 speeds things up on slow machines.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many processed frames (0 = whole clip). "
                        "Use a small value for a quick smoke test.")
    p.add_argument("--trail", type=int, default=20,
                   help="How many recent ball positions to draw as a trail.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip writing the annotated video (JSON only, faster).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Imported here so --help works even before deps are installed.
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    out_video = args.output or args.input.with_name(f"{args.input.stem}_annotated.mp4")
    out_json = args.json or args.input.with_name(f"{args.input.stem}_events.json")

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)  # auto-downloads stock weights on first run

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Could not open video: {args.input}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Input: {width}x{height} @ {fps:.1f}fps, ~{total} frames")

    writer = None
    if not args.no_video:
        # mp4v is broadly available; stride changes effective playback fps.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps / args.stride,
                                 (width, height))

    ball_trail: deque = deque(maxlen=args.trail)
    events: list[dict] = []

    frame_idx = -1
    processed = 0
    ball_seen = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.stride > 1 and frame_idx % args.stride != 0:
            continue

        # persist=True keeps ByteTrack IDs stable across frames.
        results = model.track(
            frame, persist=True, conf=args.conf, imgsz=args.imgsz,
            classes=[PERSON_CLASS, SPORTS_BALL_CLASS], verbose=False,
        )
        r = results[0]

        players: list[dict] = []
        ball: dict | None = None

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else [None] * len(clss))

            for box, conf, cls, tid in zip(xyxy, confs, clss, ids):
                x1, y1, x2, y2 = (float(v) for v in box)
                if cls == PERSON_CLASS:
                    players.append({
                        "track_id": int(tid) if tid is not None else None,
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf),
                    })
                elif cls == SPORTS_BALL_CLASS:
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    # Keep only the most confident ball per frame.
                    if ball is None or conf > ball["conf"]:
                        ball = {"center": [cx, cy], "bbox": [x1, y1, x2, y2],
                                "conf": float(conf)}

        if ball is not None:
            ball_seen += 1
            ball_trail.append((int(ball["center"][0]), int(ball["center"][1])))

        events.append({
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 3),
            "ball": ball,
            "players": players,
        })

        if writer is not None:
            _draw(cv2, np, frame, players, ball, ball_trail)
            writer.write(frame)

        processed += 1
        if processed % 50 == 0:
            rate = processed / (time.time() - t0)
            print(f"  {processed} frames | ball in {ball_seen} | {rate:.1f} fps")
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t0
    out_json.write_text(json.dumps({
        "source": str(args.input),
        "model": args.model,
        "fps": fps,
        "frame_size": [width, height],
        "stride": args.stride,
        "frames_processed": processed,
        "frames_with_ball": ball_seen,
        "events": events,
    }, indent=2))

    print(f"\nDone in {elapsed:.1f}s ({processed / elapsed:.1f} fps).")
    pct = (100.0 * ball_seen / processed) if processed else 0.0
    print(f"Ball detected in {ball_seen}/{processed} frames ({pct:.0f}%).")
    print(f"  JSON:  {out_json}")
    if writer is not None:
        print(f"  Video: {out_video}")
    if pct < 40:
        print("\nNOTE: low ball-detection rate is expected with the stock COCO model "
              "on volleyball footage.\nThis is the known limitation; swap --model for "
              "volleyball-trained weights to improve it.")
    return 0


def _draw(cv2, np, frame, players, ball, ball_trail) -> None:
    """Annotate a frame in place: player boxes + IDs, ball marker + trail."""
    for pl in players:
        x1, y1, x2, y2 = (int(v) for v in pl["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"P{pl['track_id']}" if pl["track_id"] is not None else "player"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    # Fading ball trail.
    pts = list(ball_trail)
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 165, 255), 2)

    if ball is not None:
        cx, cy = (int(ball["center"][0]), int(ball["center"][1]))
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"ball {ball['conf']:.2f}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


if __name__ == "__main__":
    raise SystemExit(main())
