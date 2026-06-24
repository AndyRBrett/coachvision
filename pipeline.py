#!/usr/bin/env python3
"""End-to-end CV pipeline: clip -> detection -> highlights -> coaching report.

This ties the stages together and, crucially, provides a SELF-TEST that runs the
whole pipeline on the bundled reference clip (fixtures/reference_clip.pgm.gz).
The self-test is wired into CI, so a broken pipeline now *fails the build*
instead of silently sitting at "0 frames processed" for weeks.

Stages
------
  1. detect.run_detection   -- pixels -> per-frame ball track (tracking JSON).
  2. highlights.build_manifest -- tracking -> tagged per-rally clip manifest.
  3. coaching.build_report  -- tracking -> rally length / ball speed / heatmap.

Running it on a real clip writes the manifest + coaching report + a results
metrics file (results/metrics.json) that write_status.py reads, so processed
frames show up in the overseer status. Running --self-test additionally writes
results/selftest.json proving the pipeline is alive.
"""
import json
import os
from datetime import datetime, timezone

import coaching
import detect
import detect_supervision
import highlights

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_CLIP = os.path.join(HERE, "fixtures", "reference_clip.pgm.gz")
REFERENCE_EVENTS = os.path.join(HERE, "fixtures", "reference_clip.events.json")

DEFAULT_RESULTS_DIR = "results"
# Nominal calibration for the reference clip: a 9 m court width spans the 80 px
# frame -> 0.1125 m/px. Used only to show illustrative metric speeds.
REFERENCE_M_PER_PX = 9.0 / 80.0


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_detector(detector):
    """Pick the detection backend.

    Default is the stdlib geometric detector (detect.py), which keeps the
    pipeline dependency-free and is what the self-test runs on. The optional
    supervision backend (Ultralytics YOLO + ByteTrack) is used for real footage
    when requested explicitly via the ``detector`` argument or
    VOLLEYBALL_DETECTOR=supervision, and only when its heavy stack is installed.
    """
    use_supervision = detector == "supervision" or (
        detector in (None, "auto") and detect_supervision.is_selected()
    )
    if use_supervision:
        if not detect_supervision.is_available():
            raise SystemExit(
                "supervision backend requested but not installed "
                "(pip install supervision ultralytics)"
            )
        return detect_supervision.run_detection
    return detect.run_detection


def run_pipeline(
    clip_path,
    events_path=None,
    fps=10.0,
    source=None,
    meters_per_pixel=None,
    output_dir="highlights",
    coaching_dir="coaching",
    detector=None,
):
    """Run detection -> highlights -> coaching for one clip and return artifacts.

    Returns a dict with the ``tracking`` data, the highlight ``manifest``, the
    coaching ``report``, and a ``metrics`` roll-up (frame counts, rally count).
    Nothing is written to disk here; callers choose what to persist.

    ``detector`` selects the detection backend: None/"auto" honours the
    VOLLEYBALL_DETECTOR env var (default: stdlib geometric), "stdlib" forces the
    dependency-free detector, "supervision" forces the YOLO+ByteTrack backend.
    """
    events = []
    if events_path:
        events, sidecar_fps, sidecar_source = detect._load_events_sidecar(events_path)
        if sidecar_fps:
            fps = float(sidecar_fps)
        if source is None:
            source = sidecar_source

    run_detection = _resolve_detector(detector)
    tracking = run_detection(clip_path, fps=fps, events=events, source=source)
    manifest = highlights.build_manifest(tracking, output_dir=output_dir)
    report = coaching.build_report(tracking, meters_per_pixel=meters_per_pixel)

    metrics = {
        "generated_at": _utc_now_iso(),
        "source": tracking.get("source"),
        "footage_processed": 1,
        "expected_frames": tracking["frame_count"],
        "actual_frames": tracking["frame_count"],
        "frames_processed": tracking["frame_count"],
        "detected_frames": tracking["detected_frames"],
        "failed_frames": 0,
        "rally_count": manifest["rally_count"],
        "errors": [],
    }
    return {"tracking": tracking, "manifest": manifest, "report": report, "metrics": metrics}


def self_test(results_dir=DEFAULT_RESULTS_DIR, verbose=True):
    """Run the full pipeline on the bundled reference clip and validate it.

    Returns the result dict on success; raises AssertionError if any stage
    produces an obviously-broken result (no frames, no rallies, empty report).
    Also writes ``results/selftest.json`` so the overseer can see the pipeline
    was verified end-to-end and on what date.
    """
    # The self-test always uses the stdlib detector: the reference clip is a
    # gzipped PGM sequence (not a video), and the whole point is a dependency-free
    # end-to-end proof that runs anywhere CI does.
    result = run_pipeline(
        REFERENCE_CLIP,
        events_path=REFERENCE_EVENTS,
        meters_per_pixel=REFERENCE_M_PER_PX,
        detector="stdlib",
    )
    tracking = result["tracking"]
    report = result["report"]

    # Guardrails: these are exactly the silent regressions the self-test exists
    # to catch -- a detector that finds nothing, or a pipeline that produces no
    # rallies/coaching output, would otherwise pass unnoticed as "0 frames".
    assert tracking["frame_count"] > 0, "detector read zero frames"
    assert tracking["detected_frames"] > 0, "detector found the ball in zero frames"
    assert report["rally_count"] >= 1, "no rallies segmented from the reference clip"
    assert report["total_play_s"] > 0, "rallies have zero total play time"
    assert any(r["tags"] for r in report["rallies"]), "no coaching tags attached to any rally"
    assert report["contact_heatmap"]["contacts_binned"] > 0, "no contacts binned into the heatmap"
    assert any(r["ball_speed"] for r in report["rallies"]), "ball speed could not be measured"

    selftest = {
        "ok": True,
        "verified_at": _utc_now_iso(),
        "clip": "fixtures/reference_clip.pgm.gz",
        "frames_processed": tracking["frame_count"],
        "detected_frames": tracking["detected_frames"],
        "rally_count": report["rally_count"],
    }
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "selftest.json"), "w") as fh:
            json.dump(selftest, fh, indent=2, sort_keys=True)
            fh.write("\n")

    if verbose:
        print(coaching.render_summary(report))
        print()
        print(f"SELF-TEST OK: processed {selftest['frames_processed']} frames, "
              f"{selftest['rally_count']} rallies, "
              f"{report['contact_heatmap']['contacts_binned']} contacts binned.")
    return result


def _write_artifacts(result, output_dir, coaching_dir, results_dir):
    """Persist manifest, coaching report/summary, and results metrics to disk."""
    highlights.write_manifest(result["manifest"], os.path.join(output_dir, "manifest.json"))

    os.makedirs(coaching_dir, exist_ok=True)
    with open(os.path.join(coaching_dir, "report.json"), "w") as fh:
        json.dump(result["report"], fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(coaching_dir, "summary.txt"), "w") as fh:
        fh.write(coaching.render_summary(result["report"]) + "\n")

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "metrics.json"), "w") as fh:
            json.dump(result["metrics"], fh, indent=2, sort_keys=True)
            fh.write("\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the volleyball CV pipeline end-to-end.")
    parser.add_argument("clip", nargs="?", help="Clip to process (gzipped P5/PGM frame sequence)")
    parser.add_argument("--events", help="Bundled events sidecar JSON")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--meters-per-pixel", type=float, default=None)
    parser.add_argument("--output-dir", default="highlights", help="Highlight manifest dir")
    parser.add_argument("--coaching-dir", default="coaching", help="Coaching report dir")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Metrics output dir")
    parser.add_argument("--detector", choices=("auto", "stdlib", "supervision"), default="auto",
                        help="Detection backend: stdlib geometric (default), or "
                             "supervision (Ultralytics YOLO + ByteTrack) for real video. "
                             "'auto' honours VOLLEYBALL_DETECTOR.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the full pipeline on the bundled reference clip and validate it")
    args = parser.parse_args()

    if args.self_test:
        try:
            self_test(results_dir=args.results_dir)
        except AssertionError as exc:
            raise SystemExit(f"SELF-TEST FAILED: {exc}")
        return

    if not args.clip:
        raise SystemExit("a clip path is required (or pass --self-test)")

    result = run_pipeline(
        args.clip,
        events_path=args.events,
        fps=args.fps,
        meters_per_pixel=args.meters_per_pixel,
        output_dir=args.output_dir,
        coaching_dir=args.coaching_dir,
        detector=args.detector,
    )
    _write_artifacts(result, args.output_dir, args.coaching_dir, args.results_dir)
    m = result["metrics"]
    print(f"Processed {m['frames_processed']} frames -> {m['rally_count']} rallies. "
          f"Wrote manifest, coaching report, and {args.results_dir}/metrics.json")


if __name__ == "__main__":
    main()
