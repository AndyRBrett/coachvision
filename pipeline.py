#!/usr/bin/env python3
"""End-to-end CV pipeline: clip -> detection -> highlights -> coaching report.

This ties the stages together and, crucially, provides a SELF-TEST that runs the
whole pipeline on each domain's bundled reference clip. The self-test is wired
into CI, so a broken pipeline now *fails the build* instead of silently sitting
at "0 frames processed" for weeks.

Stages
------
  1. detect.run_detection   -- pixels -> per-frame subject track (tracking JSON).
  2. highlights.build_manifest -- tracking -> tagged per-segment clip manifest.
  3. coaching.build_report  -- tracking -> segment length / subject speed / heatmap.

Running it on a real clip writes the manifest + coaching report + a results
metrics file (results/metrics.json) that write_status.py reads, so processed
frames show up in the overseer status. Running --self-test additionally writes
results/selftest.json proving the pipeline is alive, and appends the run to
results/selftest_history.json so a scheduled recurring self-test (independent
of whether any footage was ingested) can be compared against its established
baseline and flag drift.
"""
import json
import os
from datetime import datetime, timezone

import coaching
import detect
import domains
import highlights

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_CLIP = os.path.join(HERE, "fixtures", "reference_clip.pgm.gz")
REFERENCE_EVENTS = os.path.join(HERE, "fixtures", "reference_clip.events.json")

DEFAULT_RESULTS_DIR = "results"
HISTORY_FILENAME = "selftest_history.json"
# ~a year of weekly overseer runs; oldest entries roll off so the history file
# doesn't grow unbounded.
MAX_HISTORY_ENTRIES = 52
# Nominal calibration for the volleyball reference clip: a 9 m court width spans
# the 80 px frame -> 0.1125 m/px. Used only to show illustrative metric speeds.
REFERENCE_M_PER_PX = 9.0 / 80.0

# Per-domain self-test fixtures: each proves its own detector end-to-end.
# (clip, events sidecar, illustrative meters-per-pixel calibration).
DOMAIN_FIXTURES = {
    "volleyball": (REFERENCE_CLIP, REFERENCE_EVENTS, REFERENCE_M_PER_PX),
    "martial_arts": (
        os.path.join(HERE, "fixtures", "martialarts_clip.pgm.gz"),
        os.path.join(HERE, "fixtures", "martialarts_clip.events.json"),
        8.0 / 80.0,  # ~8 m mat spanning the 80 px frame
    ),
}


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_pipeline(
    clip_path,
    events_path=None,
    fps=10.0,
    source=None,
    meters_per_pixel=None,
    output_dir="highlights",
    coaching_dir="coaching",
    domain=None,
    video_path=None,
):
    """Run detection -> highlights -> coaching for one clip and return artifacts.

    ``domain`` (a Domain, its name, or None for the configured default) selects
    the detector and coaching vocabulary used across all three stages.
    ``video_path`` is the real video file the highlight ffmpeg commands read from
    (so the clips can actually be rendered); it defaults to ``source``. Returns a
    dict with the ``tracking`` data, the highlight ``manifest``, the coaching
    ``report``, and a ``metrics`` roll-up (frame counts, segment count). Nothing
    is written to disk here; callers choose what to persist.
    """
    domain = domains.get_domain(domain)
    events = []
    if events_path:
        events, sidecar_fps, sidecar_source = detect._load_events_sidecar(events_path)
        if sidecar_fps:
            fps = float(sidecar_fps)
        if source is None:
            source = sidecar_source

    tracking = detect.run_detection(clip_path, fps=fps, events=events, source=source, domain=domain)
    manifest = highlights.build_manifest(tracking, output_dir=output_dir, domain=domain, video_path=video_path)
    report = coaching.build_report(tracking, meters_per_pixel=meters_per_pixel, domain=domain)

    metrics = {
        "generated_at": _utc_now_iso(),
        "source": tracking.get("source"),
        "domain": domain.key,
        "footage_processed": 1,
        "expected_frames": tracking["frame_count"],
        "actual_frames": tracking["frame_count"],
        "frames_processed": tracking["frame_count"],
        "detected_frames": tracking["detected_frames"],
        "failed_frames": 0,
        "segment_count": manifest["segment_count"],
        "errors": [],
    }
    return {"tracking": tracking, "manifest": manifest, "report": report, "metrics": metrics}


def _record_history(results_dir, domain_key, entry):
    """Append one self-test run to results/selftest_history.json and compare it
    against the established baseline (the earliest recorded run for this domain).

    The baseline is set once, by the first run ever recorded for a domain, and
    never moves -- so a dependency upgrade that silently changes detection or
    segmentation shows up as drift even on a run that still technically passes.
    Returns a list of human-readable drift descriptions (empty if none).
    """
    path = os.path.join(results_dir, HISTORY_FILENAME)
    try:
        with open(path) as fh:
            history = json.load(fh)
    except (FileNotFoundError, ValueError):
        history = {}

    domain_history = history.setdefault(domain_key, [])
    baseline = domain_history[0] if domain_history else None

    drift = []
    if baseline is not None:
        if bool(entry.get("ok")) != bool(baseline.get("ok")):
            drift.append(f"ok: {baseline.get('ok')} -> {entry.get('ok')}")
        for key in ("frames_processed", "segment_count"):
            if baseline.get(key) != entry.get(key):
                drift.append(f"{key}: {baseline.get(key)} -> {entry.get(key)}")

    domain_history.append(entry)
    del domain_history[:-MAX_HISTORY_ENTRIES]
    os.makedirs(results_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(history, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return drift


def self_test(results_dir=DEFAULT_RESULTS_DIR, verbose=True, domain=None):
    """Run the full pipeline on a domain's bundled reference clip and validate it.

    ``domain`` selects which sport's fixture/detector to prove (default from
    COACHVISION_DOMAIN). Returns the result dict on success; raises AssertionError
    if any stage produces an obviously-broken result (no frames, no segments,
    empty report). Also writes ``results/selftest.json`` so the overseer can see
    the pipeline was verified end-to-end, for which domain, and on what date, and
    appends the run to ``results/selftest_history.json`` so a scheduled recurring
    run (e.g. weekly, independent of whether any real footage was ingested) can
    be compared against its established baseline -- catching a silent drift in
    frames_processed/segment_count/ok (from a dependency upgrade, say) even on a
    run that still technically passes.
    """
    domain = domains.get_domain(domain)
    if domain.key not in DOMAIN_FIXTURES:
        raise AssertionError(f"no self-test fixture bundled for domain {domain.key!r}")
    clip, events, m_per_px = DOMAIN_FIXTURES[domain.key]

    result = run_pipeline(clip, events_path=events, meters_per_pixel=m_per_px, domain=domain)
    tracking = result["tracking"]
    report = result["report"]

    seg = domain.segment_plural
    selftest = {
        "verified_at": _utc_now_iso(),
        "domain": domain.key,
        "clip": os.path.relpath(clip, HERE),
        "frames_processed": tracking["frame_count"],
        "detected_frames": tracking["detected_frames"],
        "segment_count": report["segment_count"],
    }

    # Guardrails: these are exactly the silent regressions the self-test exists
    # to catch -- a detector that finds nothing, or a pipeline that produces no
    # segments/coaching output, would otherwise pass unnoticed as "0 frames".
    # A failure is still recorded to history (ok: False) before re-raising, so
    # the drift trend captures the outage instead of just losing that run.
    try:
        assert tracking["frame_count"] > 0, "detector read zero frames"
        assert tracking["detected_frames"] > 0, f"detector found the {domain.subject_noun} in zero frames"
        assert report["segment_count"] >= 1, f"no {seg} segmented from the reference clip"
        assert report["total_play_s"] > 0, f"{seg} have zero total play time"
        assert any(r["tags"] for r in report["segments"]), f"no coaching tags attached to any {domain.segment_noun}"
        assert report["action_heatmap"]["actions_binned"] > 0, f"no {domain.action_plural} binned into the heatmap"
        assert any(r["subject_speed"] for r in report["segments"]), f"{domain.subject_noun} speed could not be measured"
    except AssertionError:
        selftest["ok"] = False
        if results_dir:
            _record_history(results_dir, domain.key, selftest)
        raise
    selftest["ok"] = True

    drift = _record_history(results_dir, domain.key, selftest) if results_dir else []
    if drift:
        selftest["drift"] = drift
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "selftest.json"), "w") as fh:
            json.dump(selftest, fh, indent=2, sort_keys=True)
            fh.write("\n")

    if verbose:
        print(coaching.render_summary(report))
        print()
        print(f"SELF-TEST OK ({domain.label}): processed {selftest['frames_processed']} frames, "
              f"{selftest['segment_count']} {seg}, "
              f"{report['action_heatmap']['actions_binned']} {domain.action_plural} binned.")
        if drift:
            print(f"SELF-TEST DRIFT ({domain.label}) vs baseline: {'; '.join(drift)}")
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

    parser = argparse.ArgumentParser(description="Run the CV coaching pipeline end-to-end.")
    parser.add_argument("clip", nargs="?", help="Clip to process (gzipped P5/PGM frame sequence)")
    parser.add_argument("--events", help="Bundled events sidecar JSON")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--meters-per-pixel", type=float, default=None)
    parser.add_argument("--domain", default=None,
                        help="Sport domain (volleyball|martial_arts); default from COACHVISION_DOMAIN")
    parser.add_argument("--output-dir", default="highlights", help="Highlight manifest dir")
    parser.add_argument("--coaching-dir", default="coaching", help="Coaching report dir")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Metrics output dir")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the full pipeline on the bundled reference clip and validate it")
    args = parser.parse_args()

    if args.self_test:
        try:
            self_test(results_dir=args.results_dir, domain=args.domain)
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
        domain=args.domain,
    )
    _write_artifacts(result, args.output_dir, args.coaching_dir, args.results_dir)
    m = result["metrics"]
    seg = domains.get_domain(m.get("domain")).segment_plural
    print(f"Processed {m['frames_processed']} frames -> {m['segment_count']} {seg}. "
          f"Wrote manifest, coaching report, and {args.results_dir}/metrics.json")


if __name__ == "__main__":
    main()
