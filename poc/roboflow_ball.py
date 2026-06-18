#!/usr/bin/env python3
"""Validate a Roboflow-hosted volleyball model on a clip (ball detection).

This is the "simplest route" probe: instead of installing a heavy local
inference stack or training a model, we send frames to a Roboflow-hosted model
over plain HTTP (only `requests` needed) and see whether it actually detects the
ball on YOUR footage. If the detection rate is good, we graduate this into the
main pipeline (ball from Roboflow, players from the local YOLOv8 we already
have).

You need:
  * a free Roboflow account -> Settings -> API Keys -> copy the Private API Key
  * a model id of the form "project-slug/version", e.g. "volleyball_detection/3"
    (shown on the project's page / "Deploy" tab on Roboflow Universe)

Example:
  python3 poc/roboflow_ball.py \
      --api-key YOUR_KEY \
      --model "volleyball_detection/3" \
      -i clip.mp4 --max-frames 40 --stride 2

It prints the ball-detection rate, the set of class names the model returned
(so we learn what it calls the ball), and writes an annotated video.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from collections import deque
from pathlib import Path

DETECT_URL = "https://detect.roboflow.com/{model}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Probe a Roboflow-hosted volleyball model for ball detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api-key", required=True, help="Roboflow Private API Key.")
    p.add_argument("--model", required=True,
                   help='Model id as "project-slug/version", e.g. volleyball_detection/3')
    p.add_argument("--input", "-i", required=True, type=Path, help="Input video clip.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Annotated video. Defaults to <input>_roboflow.mp4")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1).")
    p.add_argument("--stride", type=int, default=2, help="Process every Nth frame.")
    p.add_argument("--max-frames", type=int, default=40,
                   help="Stop after this many processed frames (0 = whole clip). "
                        "Keep small at first: every frame is a network call.")
    p.add_argument("--trail", type=int, default=15, help="Recent ball positions to draw.")
    p.add_argument("--no-video", action="store_true", help="Skip the annotated video.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import cv2
        import requests
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Run: pip install requests "
              "(opencv should already be installed).", file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    out_video = args.output or args.input.with_name(f"{args.input.stem}_roboflow.mp4")
    url = DETECT_URL.format(model=args.model)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Could not open video: {args.input}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps / args.stride, (w, h))

    classes_seen: dict[str, int] = {}
    trail: deque = deque(maxlen=args.trail)
    frame_idx, processed, ball_seen = -1, 0, 0
    t0 = time.time()
    session = requests.Session()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.stride > 1 and frame_idx % args.stride != 0:
            continue

        ok_enc, buf = cv2.imencode(".jpg", frame)
        if not ok_enc:
            continue
        b64 = base64.b64encode(buf).decode("ascii")

        try:
            resp = session.post(
                url, params={"api_key": args.api_key, "confidence": int(args.conf * 100)},
                data=b64, headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"Request failed on frame {frame_idx}: {e}", file=sys.stderr)
            break

        if resp.status_code != 200:
            print(f"\nRoboflow returned HTTP {resp.status_code}:\n{resp.text[:400]}",
                  file=sys.stderr)
            print("Check your --api-key and --model (project-slug/version).",
                  file=sys.stderr)
            return 2

        preds = resp.json().get("predictions", [])
        ball = None
        for d in preds:
            name = str(d.get("class", "")).lower()
            classes_seen[name] = classes_seen.get(name, 0) + 1
            if "ball" in name and (ball is None or d["confidence"] > ball["confidence"]):
                ball = d
        if ball is not None:
            ball_seen += 1
            trail.append((int(ball["x"]), int(ball["y"])))

        if writer is not None:
            _draw(cv2, frame, preds, ball, trail)
            writer.write(frame)

        processed += 1
        rate = processed / (time.time() - t0)
        print(f"  frame {frame_idx}: {len(preds)} dets, ball={'Y' if ball else 'n'} "
              f"| {ball_seen}/{processed} have ball | {rate:.1f} fps", flush=True)
        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    pct = (100.0 * ball_seen / processed) if processed else 0.0
    print(f"\nBall detected in {ball_seen}/{processed} frames ({pct:.0f}%).")
    print(f"Class names the model returned: {classes_seen or '(none)'}")
    if writer is not None:
        print(f"Annotated video: {out_video}")
    print("\nPaste this whole summary back. If the ball % is decent, I wire this "
          "into the main pipeline (ball from Roboflow, players from local YOLOv8).")
    return 0


def _draw(cv2, frame, preds, ball, trail) -> None:
    for d in preds:
        x, y, bw, bh = d["x"], d["y"], d["width"], d["height"]
        x1, y1 = int(x - bw / 2), int(y - bh / 2)
        x2, y2 = int(x + bw / 2), int(y + bh / 2)
        is_ball = "ball" in str(d.get("class", "")).lower()
        color = (0, 0, 255) if is_ball else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d.get('class','?')} {d['confidence']:.2f}",
                    (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    pts = list(trail)
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 165, 255), 2)


if __name__ == "__main__":
    raise SystemExit(main())
