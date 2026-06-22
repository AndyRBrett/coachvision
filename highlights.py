#!/usr/bin/env python3
"""Auto-generate per-rally highlight clips with coaching tags (Overseer #9).

Scope (deliberately bounded before building)
-------------------------------------------
This turns existing ball+player tracking data into rewatchable, tagged clips --
the project's stated purpose (coaching feedback). It is NOT a detector; it
consumes tracking output the pipeline already produces. Three stages:

  1. segment_rallies   -- split continuous play into rallies from ball-motion
                          gaps (ball missing/still for longer than a threshold).
  2. tag_rally         -- attach coaching tags (serve/attack/block/dig/...) whose
                          event timestamps fall inside each rally window.
  3. build_manifest    -- emit a dashboard manifest, one entry per rally, with an
                          ffmpeg trim+overlay command per clip.

Rendering (ffmpeg) is OPTIONAL and guarded: the command is always recorded in
the manifest, but only executed when ffmpeg is installed and render=True. This
keeps the core logic pure and testable in environments without ffmpeg.

Optional tag enrichment via NVIDIA Cosmos Reason (a reasoning vision-language
model) is delegated to cosmos_tagger.py and used only when configured; the
heuristic event-window tags are always the baseline.

Tracking input schema (JSON)
----------------------------
{
  "fps": 30,
  "source": "drop/match1.mp4",          # optional source video for rendering
  "frames": [                            # per-frame ball position (or null)
    {"frame": 0, "t": 0.0, "ball": [x, y]},
    {"frame": 1, "t": 0.033, "ball": null},
    ...
  ],
  "events": [                            # detected coaching events
    {"t": 1.2, "type": "serve"},
    {"t": 3.4, "type": "attack", "player": 7},
    ...
  ]
}
"""
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

DEFAULT_OUTPUT_DIR = "highlights"
DEFAULT_MANIFEST = os.path.join(DEFAULT_OUTPUT_DIR, "manifest.json")

# Coaching events we recognise as tags, in a stable display order.
KNOWN_TAGS = ("serve", "reception", "set", "attack", "block", "dig", "ace", "error")

# Defaults for rally segmentation.
DEFAULT_MAX_GAP_S = 2.0   # ball missing/still longer than this ends a rally
DEFAULT_MIN_RALLY_S = 1.0  # discard blips shorter than this
DEFAULT_PAD_S = 1.0        # padding added before/after each rally for context


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frame_time(frame: dict, fps: float) -> float:
    """Timestamp of a frame, preferring an explicit ``t`` then frame/fps."""
    if "t" in frame and frame["t"] is not None:
        return float(frame["t"])
    return float(frame.get("frame", 0)) / fps if fps else 0.0


def segment_rallies(
    frames,
    fps,
    max_gap_s=DEFAULT_MAX_GAP_S,
    min_rally_s=DEFAULT_MIN_RALLY_S,
):
    """Split frames into rally [start, end] windows from ball-motion gaps.

    A rally is a run of frames where the ball is tracked. When the ball goes
    missing (``ball`` is null/absent) for longer than ``max_gap_s``, the current
    rally ends. Rallies shorter than ``min_rally_s`` are dropped as noise.

    Returns a list of dicts: {"start": float, "end": float}.
    """
    rallies = []
    start = None
    last_seen = None
    for fr in frames:
        t = _frame_time(fr, fps)
        has_ball = fr.get("ball") is not None
        if has_ball:
            if start is None:
                start = t
            last_seen = t
        else:
            if start is not None and last_seen is not None and (t - last_seen) > max_gap_s:
                rallies.append({"start": start, "end": last_seen})
                start = None
                last_seen = None
    if start is not None and last_seen is not None:
        rallies.append({"start": start, "end": last_seen})

    return [r for r in rallies if (r["end"] - r["start"]) >= min_rally_s]


def tag_rally(rally, events):
    """Return the sorted set of coaching tags whose events fall in the rally.

    Padding is intentionally not applied here -- tags reflect events inside the
    actual rally window. Unknown event types are passed through so custom tags
    aren't silently lost, but known tags sort first in canonical order.
    """
    tags = set()
    for ev in events:
        t = ev.get("t")
        if t is None:
            continue
        if rally["start"] <= t <= rally["end"]:
            etype = ev.get("type")
            if etype:
                tags.add(etype)

    def sort_key(tag):
        return (KNOWN_TAGS.index(tag) if tag in KNOWN_TAGS else len(KNOWN_TAGS), tag)

    return sorted(tags, key=sort_key)


def ffmpeg_trim_cmd(source, start, end, out_path, tags=None, pad_s=DEFAULT_PAD_S):
    """Build an ffmpeg command that trims [start-pad, end+pad] and overlays tags.

    The command is returned as an argv list (never shell-joined) so it is safe
    to pass to subprocess. Returns None when no source video is available, in
    which case the manifest entry records that the clip can't be rendered yet.
    """
    if not source:
        return None
    clip_start = max(0.0, start - pad_s)
    duration = (end + pad_s) - clip_start
    cmd = ["ffmpeg", "-y", "-ss", f"{clip_start:.3f}", "-i", source, "-t", f"{duration:.3f}"]
    label = ", ".join(tags or [])
    if label:
        # Escape characters special to ffmpeg's drawtext filter.
        safe = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        drawtext = (
            f"drawtext=text='{safe}':x=20:y=20:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.5:boxborderw=8"
        )
        cmd += ["-vf", drawtext]
    cmd += [out_path]
    return cmd


def build_manifest(
    tracking,
    output_dir=DEFAULT_OUTPUT_DIR,
    max_gap_s=DEFAULT_MAX_GAP_S,
    min_rally_s=DEFAULT_MIN_RALLY_S,
    pad_s=DEFAULT_PAD_S,
    tag_enricher=None,
):
    """Build a highlight-clip manifest from tracking data.

    ``tag_enricher`` is an optional callable (source, start, end, base_tags) ->
    tags, used to fold in VLM-derived tags (e.g. NVIDIA Cosmos Reason). It is
    only consulted when provided; the heuristic event-window tags are always the
    baseline so the feature degrades gracefully without a model.
    """
    fps = float(tracking.get("fps") or 30.0)
    source = tracking.get("source")
    frames = tracking.get("frames", [])
    events = tracking.get("events", [])

    rallies = segment_rallies(frames, fps, max_gap_s=max_gap_s, min_rally_s=min_rally_s)
    clips = []
    for i, rally in enumerate(rallies, start=1):
        tags = tag_rally(rally, events)
        if tag_enricher is not None:
            try:
                tags = tag_enricher(source, rally["start"], rally["end"], tags)
            except Exception as exc:  # enrichment is best-effort, never fatal
                clips_warning = f"tag_enricher failed: {exc}"
            else:
                clips_warning = None
        else:
            clips_warning = None
        clip_id = f"rally_{i:03d}"
        out_path = os.path.join(output_dir, f"{clip_id}.mp4")
        cmd = ffmpeg_trim_cmd(source, rally["start"], rally["end"], out_path, tags, pad_s)
        entry = {
            "id": clip_id,
            "start": round(rally["start"], 3),
            "end": round(rally["end"], 3),
            "duration": round(rally["end"] - rally["start"], 3),
            "tags": tags,
            "output": out_path,
            "renderable": cmd is not None,
            "ffmpeg_cmd": cmd,
        }
        if clips_warning:
            entry["warning"] = clips_warning
        clips.append(entry)

    return {
        "generated_at": _utc_now_iso(),
        "source": source,
        "fps": fps,
        "rally_count": len(clips),
        "clips": clips,
    }


def render_clips(manifest, dry_run=False):
    """Execute the ffmpeg commands recorded in a manifest.

    Returns a list of {"id", "status", ...}. When ffmpeg is missing or a clip is
    not renderable, the clip is skipped (status "skipped") rather than failing
    the whole run. ``dry_run`` reports what would run without invoking ffmpeg.
    """
    have_ffmpeg = shutil.which("ffmpeg") is not None
    results = []
    for clip in manifest.get("clips", []):
        cmd = clip.get("ffmpeg_cmd")
        if not cmd:
            results.append({"id": clip["id"], "status": "skipped", "reason": "no source"})
            continue
        if dry_run:
            results.append({"id": clip["id"], "status": "dry-run", "cmd": cmd})
            continue
        if not have_ffmpeg:
            results.append({"id": clip["id"], "status": "skipped", "reason": "ffmpeg not installed"})
            continue
        os.makedirs(os.path.dirname(cmd[-1]) or ".", exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            results.append({"id": clip["id"], "status": "rendered", "output": cmd[-1]})
        else:
            results.append({"id": clip["id"], "status": "failed", "stderr": proc.stderr[-500:]})
    return results


def write_manifest(manifest, manifest_path=DEFAULT_MANIFEST):
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate tagged highlight clips from tracking data.")
    parser.add_argument("tracking", help="Path to tracking JSON")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=None, help="Manifest path (default: <output-dir>/manifest.json)")
    parser.add_argument("--render", action="store_true", help="Run ffmpeg to produce the clips")
    parser.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands without running them")
    parser.add_argument("--cosmos", action="store_true", help="Enrich tags via NVIDIA Cosmos Reason if configured")
    args = parser.parse_args()

    with open(args.tracking) as fh:
        tracking = json.load(fh)

    enricher = None
    if args.cosmos:
        try:
            from cosmos_tagger import make_enricher

            enricher = make_enricher()
        except Exception as exc:  # noqa: BLE001
            print(f"cosmos tagging unavailable, using heuristic tags only: {exc}")

    manifest = build_manifest(tracking, output_dir=args.output_dir, tag_enricher=enricher)
    manifest_path = args.manifest or os.path.join(args.output_dir, "manifest.json")
    write_manifest(manifest, manifest_path)
    print(f"Wrote {manifest_path}: {manifest['rally_count']} rallies")

    if args.render or args.dry_run:
        for res in render_clips(manifest, dry_run=args.dry_run):
            print(json.dumps(res))


if __name__ == "__main__":
    main()
