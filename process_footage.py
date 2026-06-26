#!/usr/bin/env python3
"""Process one real recording end-to-end on GitHub (no Mac, no GPU).

This is the glue the weekly/dispatch workflow calls: take a real video (a path
in the repo or a URL to download), decode it to frames with ffmpeg, run the full
CV pipeline for the chosen sport domain, and publish browsable coaching outputs
under ``reports/<clip>/`` plus a single ``reports/index.json`` catalog.

The catalog is deliberately a flat JSON list of processed clips with relative
paths to each artifact -- exactly what a static phone PWA can fetch (from GitHub
Pages or the raw repo) to render a gallery of sessions, with no server to run.

For martial arts with overlay on (and the pose deps installed), the analysis is
pose-driven (fight_analysis): it locks onto the fighters and detects strike
attempts. Otherwise it uses the dependency-free motion-energy detector.
"""
import importlib.util
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import coach_feedback
import coaching
import decode_video
import domains
import fight_analysis
import highlights
import pipeline
import pose_overlay

DEFAULT_REPORTS_DIR = "reports"
INDEX_NAME = "index.json"
# Slugs too generic to make a useful session id (e.g. a Drive ".../view" tail);
# fall back to a timestamp so sessions stay distinct and readable.
_GENERIC_IDS = {"clip", "view", "uc", "download", "input-video", "input_video", "video"}


def _pose_available():
    """True when the optional pose deps (ultralytics + opencv) are importable."""
    return all(importlib.util.find_spec(m) for m in ("ultralytics", "cv2"))


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


def _coaching_stills(manifest, work_dir, n=coach_feedback.MAX_STILLS):
    """Grab one annotated frame from up to ``n`` rendered clips, for the LLM.

    Best-effort: returns the still paths that ffmpeg actually wrote (missing
    ffmpeg or a bad clip is skipped). The clips here are the fighters-only
    annotated highlights, so the frames already show the tracking + strike marks.
    """
    stills = []
    for clip in manifest.get("clips", []):
        if len(stills) >= n:
            break
        vid = clip.get("output")
        if not (clip.get("rendered") and vid and os.path.exists(vid)):
            continue
        still = os.path.join(work_dir, f"still_{clip['id']}.jpg")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", vid,
                 "-frames:v", "1", "-q:v", "3", still],
                check=True,
            )
            if os.path.exists(still):
                stills.append(still)
        except Exception as exc:  # noqa: BLE001 -- stills are best-effort
            print(f"still extraction skipped for {clip['id']}: {exc}")
    return stills


def _write_feedback(result, manifest, coaching_dir, work_dir, domain):
    """Generate Claude coaching feedback (if a key is set) and save it.

    Returns the feedback file path if written, else None. Cost-gated and
    best-effort: no ANTHROPIC_API_KEY -> skipped silently; any error is logged
    and never fatal to the run.
    """
    dom = domains.get_domain(domain)
    stills = _coaching_stills(manifest, work_dir)
    try:
        feedback = coach_feedback.generate_feedback(
            result["tracking"], stills,
            role=f"{dom.label.lower()} coach",
            segment_count=result["metrics"].get("segment_count"),
        )
    except Exception as exc:  # noqa: BLE001 -- feedback is best-effort
        print(f"coaching feedback skipped: {exc}")
        feedback = None
    finally:
        for s in stills:
            try:
                os.remove(s)
            except OSError:
                pass
    if not feedback:
        return None
    os.makedirs(coaching_dir, exist_ok=True)
    path = os.path.join(coaching_dir, "feedback.md")
    with open(path, "w") as fh:
        fh.write(feedback.rstrip() + "\n")
    return path


def _run_pose(src_video, title, fps, meters_per_pixel, output_dir, domain, work_dir, cleanup):
    """Pose-driven analysis path: fighters-only tracking + strike events + overlay.

    Returns ``(result, decoded, trim_is_annotated)`` where ``result`` matches
    pipeline.run_pipeline's shape and ``trim_is_annotated`` says the trim source
    is the fighters-only annotated video (so highlight clips come out annotated).
    """
    dom = domains.get_domain(domain)
    tracking, records, norm = fight_analysis.analyze(src_video, fps=fps, source_label=title)
    cleanup.append(norm)

    trim_video = src_video
    trim_is_annotated = False
    annotated = os.path.join(work_dir, "annotated.mp4")
    try:
        if fight_analysis.render_overlay(norm, annotated, records, tracking["events"], fps=fps):
            trim_video, trim_is_annotated = annotated, True
            cleanup.append(annotated)
    except Exception as exc:  # noqa: BLE001 -- overlay is best-effort
        print(f"fighter overlay failed, using raw clips: {exc}")

    manifest = highlights.build_manifest(tracking, output_dir=output_dir, domain=dom, video_path=trim_video)
    report = coaching.build_report(tracking, meters_per_pixel=meters_per_pixel, domain=dom)
    metrics = {
        "generated_at": _utc_now_iso(),
        "source": tracking.get("source"),
        "domain": dom.key,
        "footage_processed": 1,
        "expected_frames": tracking["frame_count"],
        "actual_frames": tracking["frame_count"],
        "frames_processed": tracking["frame_count"],
        "detected_frames": tracking["detected_frames"],
        "failed_frames": 0,
        "segment_count": manifest["segment_count"],
        "errors": [],
    }
    result = {"tracking": tracking, "manifest": manifest, "report": report, "metrics": metrics}
    decoded = {"width": tracking["width"], "height": tracking["height"]}
    return result, decoded, trim_is_annotated


def process(
    src_video,
    domain,
    reports_dir=DEFAULT_REPORTS_DIR,
    fps=decode_video.DEFAULT_FPS,
    width=decode_video.DEFAULT_WIDTH,
    meters_per_pixel=None,
    source_label=None,
    name=None,
    annotate=False,
    work_dir=None,
):
    """Decode + run the pipeline for one video, publishing under reports/<id>/.

    Returns a summary dict (the catalog entry). ``src_video`` is a local path
    (already resolved/downloaded by the caller or decode_video.resolve_source).
    The tagged highlight clips are rendered with ffmpeg so the PWA can play them.
    Artifacts: ``reports/<id>/coaching/{report.json,summary.txt}``,
    ``reports/<id>/highlights/{manifest.json,<segment>.mp4}``,
    ``reports/<id>/results/metrics.json``.
    """
    clip_id = slugify(name or source_label or src_video)
    if clip_id in _GENERIC_IDS:
        clip_id = "clip-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    title = name or source_label or clip_id
    clip_dir = os.path.join(reports_dir, clip_id)
    work_dir = work_dir or clip_dir
    os.makedirs(work_dir, exist_ok=True)

    output_dir = os.path.join(clip_dir, "highlights")
    coaching_dir = os.path.join(clip_dir, "coaching")
    results_dir = os.path.join(clip_dir, "results")
    cleanup = []  # temp videos/frames to remove once the report is written

    # Pose path (martial arts + overlay + deps present): lock onto the fighters,
    # detect strike attempts, and trim clips from a fighters-only annotated video.
    # Otherwise fall back to the dependency-free motion-energy detector.
    use_pose = bool(annotate) and domains.get_domain(domain).key == "martial_arts" and _pose_available()
    if use_pose:
        result, decoded, trim_is_annotated = _run_pose(
            src_video, title, fps, meters_per_pixel, output_dir, domain, work_dir, cleanup)
    else:
        pgm_path = os.path.join(work_dir, f"{clip_id}.pgm.gz")
        decoded = decode_video.decode_to_pgm_gz(src_video, pgm_path, fps=fps, width=width)
        cleanup.append(pgm_path)
        # video_path points the highlight ffmpeg commands at the real file
        # (tracking["source"] is just a display label).
        result = pipeline.run_pipeline(
            pgm_path, fps=fps, source=title, meters_per_pixel=meters_per_pixel,
            output_dir=output_dir, coaching_dir=coaching_dir, domain=domain,
            video_path=src_video,
        )
        trim_is_annotated = False

    pipeline._write_artifacts(result, output_dir, coaching_dir, results_dir)

    # Render the tagged highlight clips (ffmpeg trims). Best-effort: missing ffmpeg
    # or a single bad clip is skipped, never fatal. Record which actually rendered.
    manifest = result["manifest"]
    render_status = {r["id"]: r.get("status") for r in highlights.render_clips(manifest)}
    for clip in manifest["clips"]:
        clip["rendered"] = render_status.get(clip["id"]) == "rendered"
        clip.pop("ffmpeg_cmd", None)  # internal; not needed in the published manifest

    if trim_is_annotated:
        # Clips are trims of the fighters-only annotated video -> already drawn.
        for clip in manifest["clips"]:
            clip["annotated"] = bool(clip.get("rendered"))
    elif annotate:
        # Fallback overlay (volleyball, or no pose deps): annotate each clip,
        # best-effort -- a missing model/dep or a bad clip is logged, never fatal.
        for clip in manifest["clips"]:
            if not clip.get("rendered"):
                continue
            try:
                pose_overlay.annotate(clip["output"])
                clip["annotated"] = True
            except Exception as exc:  # noqa: BLE001 -- overlay is best-effort
                clip["annotated"] = False
                print(f"pose overlay skipped for {clip['id']}: {exc}")

    highlights.write_manifest(manifest, os.path.join(output_dir, "manifest.json"))

    # Coaching feedback from Claude (vision over the annotated stills + stats).
    # Pose path only -- the stills need to show the fighter tracking. Cost-gated:
    # no ANTHROPIC_API_KEY -> skipped. Best-effort: never fatal to the run.
    feedback_path = _write_feedback(result, manifest, coaching_dir, work_dir, domain) if use_pose else None

    # Intermediates (decoded frames, normalized/annotated working videos) are not
    # artifacts worth committing.
    for path in cleanup:
        try:
            os.remove(path)
        except OSError:
            pass

    metrics = result["metrics"]
    clips = [
        {
            "id": c["id"],
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "tags": c["tags"],
            "video": c["output"] if c.get("rendered") else None,
            "annotated": bool(c.get("annotated")),
        }
        for c in manifest["clips"]
    ]
    entry = {
        "id": clip_id,
        "title": title,
        "domain": metrics["domain"],
        "source": result["tracking"].get("source"),
        "processed_at": _utc_now_iso(),
        "frames_processed": metrics["frames_processed"],
        "detected_frames": metrics["detected_frames"],
        "segment_count": metrics["segment_count"],
        "rendered_count": sum(1 for c in clips if c["video"]),
        "fps": fps,
        "frame_size": [decoded["width"], decoded["height"]],
        "report": os.path.join(clip_dir, "coaching", "report.json"),
        "summary": os.path.join(clip_dir, "coaching", "summary.txt"),
        "feedback": feedback_path,  # Claude coaching feedback, or None if skipped
        "manifest": os.path.join(output_dir, "manifest.json"),
        "clips": clips,
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
    parser.add_argument("--name", default=None, help="Session name (id/title); else derived from the file")
    parser.add_argument("--annotate", action="store_true",
                        help="Draw boxes + ids + skeletons on clips (needs ultralytics+opencv)")
    args = parser.parse_args()

    src = decode_video.resolve_source(clip_path=args.src, clip_url=args.url)
    # Prefer the original filename (from a URL or path) as the human label/id,
    # stripping any query string so a Drive '...mp4?usp=...' tail stays clean.
    raw = args.src if args.src else (args.url or src)
    label = os.path.basename(raw.split("?", 1)[0].rstrip("/"))
    entry = process(
        src,
        domain=args.domain,
        reports_dir=args.reports_dir,
        fps=args.fps,
        width=args.width,
        meters_per_pixel=args.meters_per_pixel,
        source_label=label,
        name=args.name,
        annotate=args.annotate,
    )
    print(f"Processed {entry['source']} [{entry['domain']}]: "
          f"{entry['frames_processed']} frames -> {entry['segment_count']} segments. "
          f"Reports under {os.path.join(args.reports_dir, entry['id'])}/")


if __name__ == "__main__":
    main()
