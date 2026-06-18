#!/usr/bin/env python3
"""Metrics layer: turn pipeline events.json into derived stats.

Pure local computation on the per-frame events the pipeline produces — no models,
no API calls. Reads an events.json and writes a metrics.json with:

  * ball.path        - the ball trajectory with short detection gaps filled in
                       (linear interpolation), so the path is continuous through
                       brief misses but NOT across long gaps (ball off-screen).
  * ball speed       - per-step speed series + avg/max (pixels/second).
  * rallies          - heuristic segmentation into rallies vs. breaks, based on
                       how long the ball is continuously present.
  * players          - per-track distance travelled + an occupancy heatmap grid
                       built from player foot positions.

UNITS ARE PIXELS. Court calibration (a later step) converts these to metres /
real speed. Rally segmentation is a heuristic proxy, not ground truth.

CLI:
    python3 poc/metrics.py -i clip_pipeline.json [-o clip_metrics.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Tunables (in seconds where noted).
MAX_INTERP_GAP_S = 0.4   # fill ball gaps no longer than this; longer = off-screen
RALLY_BREAK_S = 1.5      # ball absent longer than this ends a rally
MIN_RALLY_S = 1.0        # ignore "rallies" shorter than this
SPEED_OUTLIER_PX_S = 6000  # ignore implausible single-step speeds (detection jitter)
HEATMAP_COLS = 32
HEATMAP_ROWS = 18        # 32x18 ~ 16:9


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute metrics from pipeline events JSON.")
    p.add_argument("--input", "-i", required=True, type=Path, help="events.json from the pipeline.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Metrics JSON. Defaults to <input stem>_metrics.json")
    return p.parse_args(argv)


def compute_metrics(data: dict) -> dict:
    fps = data.get("fps") or 30.0
    stride = data.get("stride") or 1
    fw, fh = (data.get("frame_size") or [0, 0])[:2]
    events = data.get("events") or []

    # Real time between two *processed* frames.
    step_dt = stride / fps if fps else 0.0

    ball_path = _interpolate_ball(events, fps)
    speed = _ball_speed(ball_path)
    rallies = _segment_rallies(events, fps)
    players = _player_metrics(events, fw, fh)

    processed = data.get("frames_processed", len(events))
    raw_with_ball = sum(1 for e in events if e.get("ball"))
    return {
        "source": data.get("source"),
        "fps": fps, "stride": stride, "frame_size": [fw, fh],
        "frames_processed": processed,
        "units": "pixels (calibrate for metres / real speed)",
        "ball": {
            "raw_detection_rate": round(raw_with_ball / processed, 3) if processed else 0,
            "path_points": len(ball_path),
            "interpolated_points": sum(1 for p in ball_path if p["interpolated"]),
            "avg_speed_px_s": speed["avg"],
            "max_speed_px_s": speed["max"],
            "speed_series": speed["series"],
            "path": ball_path,
        },
        "rallies": rallies,
        "rally_count": len(rallies),
        "players": players,
        "_notes": {
            "max_interp_gap_s": MAX_INTERP_GAP_S,
            "rally_break_s": RALLY_BREAK_S,
            "step_dt_s": round(step_dt, 4),
        },
    }


def _interpolate_ball(events, fps) -> list[dict]:
    """Fill short gaps between ball detections with linear interpolation."""
    detected = [(e["frame"], e["time_s"], e["ball"]["center"])
                for e in events if e.get("ball")]
    if not detected:
        return []
    max_gap_frames = MAX_INTERP_GAP_S * fps
    path: list[dict] = []
    for i, (frame, t, c) in enumerate(detected):
        path.append({"frame": frame, "time_s": t, "x": c[0], "y": c[1],
                     "interpolated": False})
        if i + 1 < len(detected):
            nframe, nt, nc = detected[i + 1]
            gap = nframe - frame
            if 1 < gap <= max_gap_frames:
                # Interpolate one point per missing processed-frame slot.
                n_missing = max(1, int(gap) - 1)
                for k in range(1, n_missing + 1):
                    a = k / (n_missing + 1)
                    path.append({
                        "frame": frame + (nframe - frame) * a,
                        "time_s": t + (nt - t) * a,
                        "x": c[0] + (nc[0] - c[0]) * a,
                        "y": c[1] + (nc[1] - c[1]) * a,
                        "interpolated": True,
                    })
    path.sort(key=lambda p: p["time_s"])
    return path


def _ball_speed(path) -> dict:
    series, speeds = [], []
    for a, b in zip(path, path[1:]):
        dt = b["time_s"] - a["time_s"]
        if dt <= 0:
            continue
        d = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        v = d / dt
        if v > SPEED_OUTLIER_PX_S:
            continue
        series.append({"time_s": round(b["time_s"], 3), "speed_px_s": round(v, 1)})
        speeds.append(v)
    return {
        "series": series,
        "avg": round(sum(speeds) / len(speeds), 1) if speeds else 0.0,
        "max": round(max(speeds), 1) if speeds else 0.0,
    }


def _segment_rallies(events, fps) -> list[dict]:
    """A rally = a run of frames where the ball is present without a long gap."""
    times = [e["time_s"] for e in events if e.get("ball")]
    if not times:
        return []
    rallies = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev > RALLY_BREAK_S:
            _maybe_add_rally(rallies, start, prev)
            start = t
        prev = t
    _maybe_add_rally(rallies, start, prev)
    for i, r in enumerate(rallies):
        r["index"] = i + 1
    return rallies


def _maybe_add_rally(rallies, start, end):
    dur = end - start
    if dur >= MIN_RALLY_S:
        rallies.append({"start_s": round(start, 2), "end_s": round(end, 2),
                        "duration_s": round(dur, 2)})


def _player_metrics(events, fw, fh) -> dict:
    """Per-track distance + an occupancy heatmap from player foot positions."""
    grid = [[0] * HEATMAP_COLS for _ in range(HEATMAP_ROWS)]
    last_pos: dict[int, tuple[float, float]] = {}
    dist: dict[int, float] = {}
    seen: dict[int, int] = {}

    for e in events:
        for pl in e.get("players", []):
            tid = pl.get("track_id")
            x1, y1, x2, y2 = pl["bbox"]
            foot = ((x1 + x2) / 2.0, y2)  # bottom-center ~ where the player stands
            if fw and fh:
                col = min(HEATMAP_COLS - 1, max(0, int(foot[0] / fw * HEATMAP_COLS)))
                row = min(HEATMAP_ROWS - 1, max(0, int(foot[1] / fh * HEATMAP_ROWS)))
                grid[row][col] += 1
            if tid is None:
                continue
            seen[tid] = seen.get(tid, 0) + 1
            if tid in last_pos:
                dist[tid] = dist.get(tid, 0.0) + math.hypot(
                    foot[0] - last_pos[tid][0], foot[1] - last_pos[tid][1])
            last_pos[tid] = foot

    per_track = [{"track_id": tid, "frames_seen": seen[tid],
                  "distance_px": round(dist.get(tid, 0.0), 1)}
                 for tid in sorted(seen)]
    return {
        "track_count": len(seen),
        "per_track": per_track,
        "heatmap": {"cols": HEATMAP_COLS, "rows": HEATMAP_ROWS, "grid": grid},
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return 2
    data = json.loads(args.input.read_text())
    metrics = compute_metrics(data)
    out = args.output or args.input.with_name(args.input.stem.replace("_pipeline", "")
                                              + "_metrics.json")
    out.write_text(json.dumps(metrics, indent=2))

    b = metrics["ball"]
    print(f"Rallies: {metrics['rally_count']} | tracks: {metrics['players']['track_count']}")
    print(f"Ball path points: {b['path_points']} "
          f"({b['interpolated_points']} interpolated) | "
          f"avg speed {b['avg_speed_px_s']} px/s, max {b['max_speed_px_s']} px/s")
    print(f"Metrics: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
