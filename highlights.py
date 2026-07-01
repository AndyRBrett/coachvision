#!/usr/bin/env python3
"""Auto-generate per-segment highlight clips with coaching tags (Overseer #9).

Scope (deliberately bounded before building)
-------------------------------------------
This turns existing subject tracking data into rewatchable, tagged clips --
the project's stated purpose (coaching feedback). It is NOT a detector; it
consumes tracking output the pipeline already produces. Three stages:

  1. segment_plays   -- split continuous play into segments from subject-motion
                          gaps (subject missing/still for longer than a threshold).
  2. tag_segment       -- attach coaching tags (the active domain's vocabulary,
                          e.g. serve/attack/block or jab/cross/kick) whose event
                          timestamps fall inside each segment window.
  3. build_manifest    -- emit a dashboard manifest, one entry per segment, with
                          an ffmpeg trim+overlay command per clip.

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
  "frames": [                            # per-frame subject position (or null)
    {"frame": 0, "t": 0.0, "subject": [x, y]},
    {"frame": 1, "t": 0.033, "subject": null},
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
import re
import shutil
import subprocess
from datetime import datetime, timezone

import domains

DEFAULT_OUTPUT_DIR = "highlights"
DEFAULT_MANIFEST = os.path.join(DEFAULT_OUTPUT_DIR, "manifest.json")

# Coaching events we recognise as tags, in a stable display order. Kept as the
# volleyball default; the active domain supplies the real vocabulary.
KNOWN_TAGS = domains.VOLLEYBALL.tags

# Defaults for segmentation (volleyball values; per-domain overrides live in
# domains.py and are applied by build_manifest).
DEFAULT_MAX_GAP_S = domains.VOLLEYBALL.max_gap_s   # subject missing/still longer ends a segment
DEFAULT_MIN_SEGMENT_S = domains.VOLLEYBALL.min_segment_s  # discard blips shorter than this
DEFAULT_PAD_S = domains.VOLLEYBALL.pad_s        # padding added before/after each segment


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _frame_time(frame: dict, fps: float) -> float:
    """Timestamp of a frame, preferring an explicit ``t`` then frame/fps."""
    if "t" in frame and frame["t"] is not None:
        return float(frame["t"])
    return float(frame.get("frame", 0)) / fps if fps else 0.0


def segment_plays(
    frames,
    fps,
    max_gap_s=DEFAULT_MAX_GAP_S,
    min_segment_s=DEFAULT_MIN_SEGMENT_S,
):
    """Split frames into play [start, end] windows from subject-motion gaps.

    A play (a rally in volleyball, an exchange in martial arts) is a run of
    frames where the subject is tracked. When the subject goes missing
    (``subject`` is null/absent) for longer than ``max_gap_s``, the current play
    ends. Plays shorter than ``min_segment_s`` are dropped as noise.

    Returns a list of dicts: {"start": float, "end": float}.
    """
    segments = []
    start = None
    last_seen = None
    for fr in frames:
        t = _frame_time(fr, fps)
        has_subject = fr.get("subject") is not None
        if has_subject:
            if start is None:
                start = t
            last_seen = t
        else:
            if start is not None and last_seen is not None and (t - last_seen) > max_gap_s:
                segments.append({"start": start, "end": last_seen})
                start = None
                last_seen = None
    if start is not None and last_seen is not None:
        segments.append({"start": start, "end": last_seen})

    return [r for r in segments if (r["end"] - r["start"]) >= min_segment_s]


def tag_segment(segment, events, known_tags=KNOWN_TAGS):
    """Return the sorted set of coaching tags whose events fall in the segment.

    Padding is intentionally not applied here -- tags reflect events inside the
    actual segment window. Unknown event types are passed through so custom tags
    aren't silently lost, but ``known_tags`` (the active domain's vocabulary)
    sort first in canonical order.
    """
    tags = set()
    for ev in events:
        t = ev.get("t")
        if t is None:
            continue
        if segment["start"] <= t <= segment["end"]:
            etype = ev.get("type")
            if etype:
                tags.add(etype)

    def sort_key(tag):
        return (known_tags.index(tag) if tag in known_tags else len(known_tags), tag)

    return sorted(tags, key=sort_key)


# Characters kept in a drawtext label; everything else is dropped. Rather than
# trying to correctly escape every character ffmpeg's filtergraph parser
# treats specially (backslash, single quote, colon, comma, semicolon,
# brackets, percent -- whose interactions get subtle once the value is also
# single-quoted), whitelist a safe set. Tag vocabulary is always plain words
# (letters/digits/underscores/hyphens), so this never affects legitimate tags
# and closes off the whole escaping-correctness question for anything else
# (e.g. an unknown event type from a hand-fed tracking JSON) that ends up here.
_DRAWTEXT_UNSAFE = re.compile(r"[^A-Za-z0-9 ,_-]")


def _sanitize_drawtext_label(label):
    return _DRAWTEXT_UNSAFE.sub("", label)


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
    label = _sanitize_drawtext_label(", ".join(tags or []))
    if label:
        drawtext = (
            f"drawtext=text='{label}':x=20:y=20:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.5:boxborderw=8"
        )
        cmd += ["-vf", drawtext]
    # Re-encode browser-friendly: yuv420p H.264 + faststart so the rendered clip
    # plays inline in the PWA's <video> tag (Safari/iOS are picky about this).
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    return cmd


def build_manifest(
    tracking,
    output_dir=DEFAULT_OUTPUT_DIR,
    max_gap_s=None,
    min_segment_s=None,
    pad_s=None,
    tag_enricher=None,
    domain=None,
    video_path=None,
):
    """Build a highlight-clip manifest from tracking data.

    The active ``domain`` (resolved from the argument, the tracking record's
    ``domain``, or the configured default) supplies the tag vocabulary, the
    segment id prefix, and segmentation defaults; explicit ``max_gap_s`` /
    ``min_segment_s`` / ``pad_s`` override those defaults when given.

    ``video_path`` is the actual local file the ffmpeg trim commands read from
    (``-i``). It defaults to the tracking record's ``source`` -- but that field
    is often a display label, so callers that intend to *render* clips pass the
    real video path here while ``source`` stays human-readable in the manifest.

    ``tag_enricher`` is an optional callable (source, start, end, base_tags) ->
    tags, used to fold in VLM-derived tags (e.g. NVIDIA Cosmos Reason). It is
    only consulted when provided; the heuristic event-window tags are always the
    baseline so the feature degrades gracefully without a model.
    """
    domain = domains.get_domain(domain if domain is not None else tracking.get("domain"))
    max_gap_s = domain.max_gap_s if max_gap_s is None else max_gap_s
    min_segment_s = domain.min_segment_s if min_segment_s is None else min_segment_s
    pad_s = domain.pad_s if pad_s is None else pad_s

    fps = float(tracking.get("fps") or 30.0)
    source = tracking.get("source")
    ffmpeg_input = video_path if video_path is not None else source
    frames = tracking.get("frames", [])
    events = tracking.get("events", [])

    segments = segment_plays(frames, fps, max_gap_s=max_gap_s, min_segment_s=min_segment_s)
    clips = []
    for i, segment in enumerate(segments, start=1):
        tags = tag_segment(segment, events, known_tags=domain.tags)
        if tag_enricher is not None:
            try:
                tags = tag_enricher(source, segment["start"], segment["end"], tags)
            except Exception as exc:  # enrichment is best-effort, never fatal
                clips_warning = f"tag_enricher failed: {exc}"
            else:
                clips_warning = None
        else:
            clips_warning = None
        clip_id = f"{domain.segment_noun}_{i:03d}"
        out_path = os.path.join(output_dir, f"{clip_id}.mp4")
        cmd = ffmpeg_trim_cmd(ffmpeg_input, segment["start"], segment["end"], out_path, tags, pad_s)
        entry = {
            "id": clip_id,
            "start": round(segment["start"], 3),
            "end": round(segment["end"], 3),
            "duration": round(segment["end"] - segment["start"], 3),
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
        "domain": domain.key,
        "fps": fps,
        # ``segment_count`` is kept as the stable, machine-readable segment count
        # the overseer status contract depends on, across all domains.
        "segment_count": len(clips),
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
    parser.add_argument("--domain", default=None,
                        help="Sport domain (volleyball|martial_arts); default from COACHVISION_DOMAIN")
    args = parser.parse_args()

    with open(args.tracking) as fh:
        tracking = json.load(fh)

    domain = domains.get_domain(args.domain if args.domain is not None else tracking.get("domain"))

    enricher = None
    if args.cosmos:
        try:
            from cosmos_tagger import make_enricher

            enricher = make_enricher(domain=domain)
        except Exception as exc:  # noqa: BLE001
            print(f"cosmos tagging unavailable, using heuristic tags only: {exc}")

    manifest = build_manifest(tracking, output_dir=args.output_dir, tag_enricher=enricher, domain=domain)
    manifest_path = args.manifest or os.path.join(args.output_dir, "manifest.json")
    write_manifest(manifest, manifest_path)
    print(f"Wrote {manifest_path}: {manifest['segment_count']} {domain.segment_plural}")

    if args.render or args.dry_run:
        for res in render_clips(manifest, dry_run=args.dry_run):
            print(json.dumps(res))


if __name__ == "__main__":
    main()
