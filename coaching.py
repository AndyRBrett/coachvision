#!/usr/bin/env python3
"""Per-clip coaching reports: segment length, subject speed, action-zone heatmap.

This is the project's core user-facing value: for each ingested clip, turn the
tracking data into a coaching-feedback summary a player or coach can actually
read. It builds directly on the tracking schema produced by detect.py and the
play segments produced by highlights.segment_plays, so it is verifiable
end-to-end against the bundled reference clips (see pipeline.py --self-test).

Wording follows the active domain (rally/ball/contact for volleyball,
exchange/fighter/strike for martial arts). Three signals per clip (and per
segment):

  * segment length  -- duration in seconds and the number of tracked actions.
  * subject speed   -- average and peak speed from frame-to-frame subject
                       motion, in pixels/second, plus meters/second when a
                       surface calibration (meters_per_pixel) is supplied.
  * action heatmap  -- a coarse grid over the surface counting where coaching
                       actions (the domain's events) happened, so hot zones are
                       visible at a glance.

Everything here is pure and stdlib-only: no rendering, no model calls.
"""
import json
import math
import os
from datetime import datetime, timezone

import domains
import highlights

DEFAULT_HEATMAP_COLS = 6
DEFAULT_HEATMAP_ROWS = 4
# ASCII ramp from empty to hot, used to render the heatmap in the text summary.
_HEAT_RAMP = " .:-=+*#%@"


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frames_in_window(frames, start, end, fps):
    """Tracked (subject-present) frames whose timestamp falls within [start, end]."""
    out = []
    for fr in frames:
        if fr.get("subject") is None:
            continue
        t = highlights._frame_time(fr, fps)
        if start <= t <= end:
            out.append((t, fr["subject"]))
    return out


def subject_speed(frames, start, end, fps, meters_per_pixel=None):
    """Average/peak subject speed across a window from frame-to-frame motion.

    Speed is measured between consecutive tracked positions as Euclidean pixel
    distance over elapsed time. When ``meters_per_pixel`` is given, metric
    speeds are added (clearly calibration-dependent, never fabricated when it
    isn't). Returns None when there isn't enough motion to measure.
    """
    samples = _frames_in_window(frames, start, end, fps)
    speeds = []
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        speeds.append(dist / dt)
    if not speeds:
        return None
    avg = sum(speeds) / len(speeds)
    peak = max(speeds)
    result = {
        "avg_px_per_s": round(avg, 2),
        "peak_px_per_s": round(peak, 2),
        "samples": len(speeds),
    }
    if meters_per_pixel:
        result["avg_m_per_s"] = round(avg * meters_per_pixel, 2)
        result["peak_m_per_s"] = round(peak * meters_per_pixel, 2)
    return result


def _subject_at(frames, t, fps):
    """Subject position of the tracked frame nearest in time to ``t`` (or None)."""
    best = None
    best_dt = None
    for fr in frames:
        if fr.get("subject") is None:
            continue
        dt = abs(highlights._frame_time(fr, fps) - t)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = fr["subject"]
    return best


def _cell_for(point, width, height, cols, rows):
    """Map a pixel point to a (col, row) grid cell, clamped to the grid."""
    x, y = point
    col = min(cols - 1, max(0, int(x / width * cols))) if width else 0
    row = min(rows - 1, max(0, int(y / height * rows))) if height else 0
    return col, row


def action_heatmap(frames, events, width, height, fps, cols=DEFAULT_HEATMAP_COLS, rows=DEFAULT_HEATMAP_ROWS):
    """Bin coaching actions into a ``rows x cols`` grid of counts.

    Each event is placed at the subject position nearest its timestamp (or its
    own ``pos`` if the event carries one), then counted into a surface cell.
    Returns the grid (row-major list of lists), its dimensions, and the total
    binned.
    """
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    binned = 0
    for ev in events or []:
        t = ev.get("t")
        if t is None:
            continue
        point = ev.get("pos") or _subject_at(frames, t, fps)
        if point is None:
            continue
        col, row = _cell_for(point, width, height, cols, rows)
        grid[row][col] += 1
        binned += 1
    return {"cols": cols, "rows": rows, "grid": grid, "actions_binned": binned}


def render_heatmap(heatmap):
    """Render a heatmap grid as ASCII rows (hottest cell = densest glyph)."""
    grid = heatmap["grid"]
    peak = max((max(row) for row in grid), default=0)
    lines = []
    for row in grid:
        chars = []
        for count in row:
            if peak == 0:
                chars.append(_HEAT_RAMP[0])
            else:
                level = int(round(count / peak * (len(_HEAT_RAMP) - 1)))
                chars.append(_HEAT_RAMP[level])
        lines.append("".join(chars))
    return lines


def segment_report(segment, index, tracking, meters_per_pixel=None, domain=None):
    """Build the coaching report for a single segment window."""
    domain = domains.get_domain(domain if domain is not None else tracking.get("domain"))
    frames = tracking.get("frames", [])
    events = tracking.get("events", [])
    fps = float(tracking.get("fps") or 30.0)
    start, end = segment["start"], segment["end"]
    actions = [ev for ev in events if ev.get("t") is not None and start <= ev["t"] <= end]
    return {
        "id": f"{domain.segment_noun}_{index:03d}",
        "start": round(start, 3),
        "end": round(end, 3),
        "length_s": round(end - start, 3),
        "actions": len(actions),
        "tags": highlights.tag_segment(segment, events, known_tags=domain.tags),
        "subject_speed": subject_speed(frames, start, end, fps, meters_per_pixel=meters_per_pixel),
    }


def build_report(
    tracking,
    segments=None,
    meters_per_pixel=None,
    heatmap_cols=DEFAULT_HEATMAP_COLS,
    heatmap_rows=DEFAULT_HEATMAP_ROWS,
    max_gap_s=None,
    min_segment_s=None,
    domain=None,
):
    """Build a full coaching report for a clip's tracking data.

    The active ``domain`` (argument, the tracking record's ``domain``, or the
    configured default) supplies the tag vocabulary, segment id prefix, report
    wording, and segmentation defaults. Segments play (unless ``segments`` are
    supplied), reports per-segment length / speed / tags, and a clip-wide
    action-zone heatmap plus roll-up totals. Internal JSON keys (``segment_count``,
    ``subject_speed``, ``action_heatmap``, ``surface``) stay stable across
    domains; ``domain`` is recorded so render_summary can pick the right words.
    """
    domain = domains.get_domain(domain if domain is not None else tracking.get("domain"))
    max_gap_s = domain.max_gap_s if max_gap_s is None else max_gap_s
    min_segment_s = domain.min_segment_s if min_segment_s is None else min_segment_s

    fps = float(tracking.get("fps") or 30.0)
    frames = tracking.get("frames", [])
    events = tracking.get("events", [])
    width = tracking.get("width") or 0
    height = tracking.get("height") or 0

    if segments is None:
        segments = highlights.segment_plays(frames, fps, max_gap_s=max_gap_s, min_segment_s=min_segment_s)

    segment_reports = [segment_report(s, i, tracking, meters_per_pixel, domain=domain)
                       for i, s in enumerate(segments, start=1)]
    heatmap = action_heatmap(frames, events, width, height, fps, cols=heatmap_cols, rows=heatmap_rows)

    total_play = round(sum(r["length_s"] for r in segment_reports), 3)
    speeds = [r["subject_speed"]["peak_px_per_s"] for r in segment_reports if r["subject_speed"]]
    overall_peak = max(speeds) if speeds else None

    return {
        "generated_at": _utc_now_iso(),
        "source": tracking.get("source"),
        "domain": domain.key,
        "fps": fps,
        "surface": {"width": width, "height": height, "meters_per_pixel": meters_per_pixel},
        "segment_count": len(segment_reports),
        "total_play_s": total_play,
        "longest_segment_s": max((r["length_s"] for r in segment_reports), default=0.0),
        "peak_subject_speed_px_per_s": overall_peak,
        "segments": segment_reports,
        "action_heatmap": heatmap,
    }


def render_summary(report):
    """Render a human-readable coaching summary (the coach-facing artifact).

    Wording follows the report's ``domain`` (segment/subject/surface/action
    nouns) so a volleyball clip reads in rallies and ball speed while a martial
    -arts clip reads in exchanges and fighter speed.
    """
    domain = domains.get_domain(report.get("domain"))
    lines = []
    lines.append(f"Coaching report ({domain.label}) -- {report.get('source')}")
    lines.append(f"  {domain.segment_plural}: {report['segment_count']}"
                 f"   total play: {report['total_play_s']}s"
                 f"   longest: {report['longest_segment_s']}s")
    peak = report.get("peak_subject_speed_px_per_s")
    if peak is not None:
        lines.append(f"  peak {domain.subject_noun} speed: {peak} px/s")
    lines.append("")
    for r in report["segments"]:
        spd = r["subject_speed"]
        if spd:
            extra = f", peak {spd['peak_px_per_s']} px/s"
            if "peak_m_per_s" in spd:
                extra += f" ({spd['peak_m_per_s']} m/s)"
        else:
            extra = ", speed n/a"
        tags = ", ".join(r["tags"]) or "untagged"
        lines.append(f"  {r['id']}: {r['length_s']}s, {r['actions']} {domain.action_plural} [{tags}]{extra}")
    lines.append("")
    lines.append(f"  {domain.action_noun}-zone heatmap ({domain.surface_noun} top-left origin):")
    for row in render_heatmap(report["action_heatmap"]):
        lines.append(f"    |{row}|")
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a coaching report from tracking JSON.")
    parser.add_argument("tracking", help="Path to tracking JSON (see detect.py / highlights.py)")
    parser.add_argument("--output-dir", default="coaching")
    parser.add_argument("--meters-per-pixel", type=float, default=None,
                        help="Surface calibration to add metric subject speeds")
    parser.add_argument("--domain", default=None,
                        help="Sport domain (volleyball|martial_arts); default from COACHVISION_DOMAIN")
    args = parser.parse_args()

    with open(args.tracking) as fh:
        tracking = json.load(fh)

    report = build_report(tracking, meters_per_pixel=args.meters_per_pixel, domain=args.domain)
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "report.json")
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    summary = render_summary(report)
    with open(summary_path, "w") as fh:
        fh.write(summary + "\n")
    print(summary)
    print(f"\nWrote {report_path} and {summary_path}")


if __name__ == "__main__":
    main()
