#!/usr/bin/env python3
"""Process one real recording end-to-end on GitHub (no Mac, no GPU).

This is the glue the weekly/dispatch workflow calls: take a real video (a path
in the repo or a URL to download), decode it to frames with ffmpeg, run the full
CV pipeline for the chosen sport domain, and publish browsable coaching outputs
under ``reports/<clip>/`` plus a single ``reports/index.json`` catalog.

The catalog is deliberately a flat JSON list of processed clips with relative
paths to each artifact -- exactly what a static phone PWA can fetch (from GitHub
Pages or the raw repo) to render a gallery of sessions, with no server to run.

Everything here is CPU-only: ffmpeg's software decoder + the pure-Python
pipeline. The only moving part that needs ffmpeg is the decode step, isolated in
decode_video.py.
"""
import json
import os
import re
from datetime import datetime, timezone

import decode_video
import pipeline

DEFAULT_REPORTS_DIR = "reports"
INDEX_NAME = "index.json"


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name):
    """Filesystem/URL-safe id from a clip filename (stem only, lower-kebab)."""
    stem = os.path.splitext(os.path.basename(name))[0]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-._").lower()
    return slug or "clip"


def update_index(reports_dir, entry):
    """Upsert ``entry`` (keyed by ``id``) into reports/index.json, newest first."""
    index_path = os.path.join(reports_dir, INDEX_NAME)
    try:
        with open(index_path) as fh:
            index = json.load(fh)
    except (FileNotFoundError, ValueError):
        index = {}
    clips = [c for c in index.get("clips", []) if c.get("id") != entry["id"]]
    clips.append(entry)
    clips.sort(key=lambda c: c.get("processed_at", ""), reverse=True)
    index = {"updated_at": _utc_now_iso(), "clips": clips}
    os.makedirs(reports_dir, exist_ok=True)
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return index_path


def process(
    src_video,
    domain,
    reports_dir=DEFAULT_REPORTS_DIR,
    fps=decode_video.DEFAULT_FPS,
    width=decode_video.DEFAULT_WIDTH,
    meters_per_pixel=None,
    source_label=None,
    work_dir=None,
):
    """Decode + run the pipeline for one video, publishing under reports/<id>/.

    Returns a summary dict (the catalog entry). ``src_video`` is a local path
    (already resolved/downloaded by the caller or decode_video.resolve_source).
    Artifacts: ``reports/<id>/coaching/{report.json,summary.txt}``,
    ``reports/<id>/highlights/manifest.json``, ``reports/<id>/results/metrics.json``.
    """
    clip_id = slugify(src_video if source_label is None else source_label)
    clip_dir = os.path.join(reports_dir, clip_id)
    work_dir = work_dir or clip_dir
    os.makedirs(work_dir, exist_ok=True)

    pgm_path = os.path.join(work_dir, f"{clip_id}.pgm.gz")
    decoded = decode_video.decode_to_pgm_gz(src_video, pgm_path, fps=fps, width=width)

    output_dir = os.path.join(clip_dir, "highlights")
    coaching_dir = os.path.join(clip_dir, "coaching")
    results_dir = os.path.join(clip_dir, "results")
    result = pipeline.run_pipeline(
        pgm_path,
        fps=fps,
        source=source_label or os.path.basename(src_video),
        meters_per_pixel=meters_per_pixel,
        output_dir=output_dir,
        coaching_dir=coaching_dir,
        domain=domain,
    )
    pipeline._write_artifacts(result, output_dir, coaching_dir, results_dir)

    # Decoded frames are an intermediate, not an artifact worth committing.
    try:
        os.remove(pgm_path)
    except OSError:
        pass

    metrics = result["metrics"]
    entry = {
        "id": clip_id,
        "domain": metrics["domain"],
        "source": result["tracking"].get("source"),
        "processed_at": _utc_now_iso(),
        "frames_processed": metrics["frames_processed"],
        "detected_frames": metrics["detected_frames"],
        "segment_count": metrics["segment_count"],
        "fps": fps,
        "frame_size": [decoded["width"], decoded["height"]],
        "report": os.path.join(clip_dir, "coaching", "report.json"),
        "summary": os.path.join(clip_dir, "coaching", "summary.txt"),
        "manifest": os.path.join(output_dir, "manifest.json"),
    }
    update_index(reports_dir, entry)
    return entry


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decode + analyze one recording, publish reports.")
    parser.add_argument("src", nargs="?", help="Path to a local video file")
    parser.add_argument("--url", help="Download the video from this URL instead")
    parser.add_argument("--domain", default=None,
                        help="Sport domain (volleyball|martial_arts); default from COACHVISION_DOMAIN")
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--fps", type=float, default=decode_video.DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=decode_video.DEFAULT_WIDTH)
    parser.add_argument("--meters-per-pixel", type=float, default=None)
    args = parser.parse_args()

    src = decode_video.resolve_source(clip_path=args.src, clip_url=args.url)
    # Prefer the original filename (from a URL or path) as the human label/id.
    label = os.path.basename(args.src) if args.src else (args.url or src).rstrip("/").split("/")[-1]
    entry = process(
        src,
        domain=args.domain,
        reports_dir=args.reports_dir,
        fps=args.fps,
        width=args.width,
        meters_per_pixel=args.meters_per_pixel,
        source_label=label,
    )
    print(f"Processed {entry['source']} [{entry['domain']}]: "
          f"{entry['frames_processed']} frames -> {entry['segment_count']} segments. "
          f"Reports under {os.path.join(args.reports_dir, entry['id'])}/")


if __name__ == "__main__":
    main()
