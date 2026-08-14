#!/usr/bin/env python3
"""
Publish the bundled fixture clip as a demo coaching session (issue #21).

coachvision has never ingested real footage (``needs_footage: true``), so every
capability it has is invisible: the dashboard opens on an empty gallery and
nothing about it suggests what a processed session looks like. Meanwhile the
self-test already runs the *entire* pipeline on a bundled clip on every CI run —
detection, segmentation, coaching tags, heatmap — and then throws the report
away, keeping only a pass/fail summary.

This publishes that work instead of discarding it. The demo lands in the same
``reports/<id>/`` layout real footage produces and is upserted into the same
``reports/index.json`` catalog, so the dashboard renders it through the existing
session path with no parallel rendering code to drift out of sync. The entry is
flagged ``demo: true`` so the UI can badge it and the overseer can tell a
demonstration apart from real ingested footage — the distinction matters,
because a demo must never make ``footage_processed`` look non-zero.

Two things fall out of that:

  * a first-run "here is what you'll get" session, instead of an empty gallery
  * continuous end-to-end coverage of the *report-writing* path, not just the
    pipeline compute path the self-test asserts on. A report renderer that
    breaks is now a red CI run rather than a surprise on the day real footage
    finally arrives.

Usage:
    python demo.py                 # publish to reports/demo/
    python demo.py --check         # verify the published demo is current, don't write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import coaching
import domains
import pipeline
from process_footage import update_index

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_ID = "demo"
DEMO_TITLE = "Demo — bundled sample clip"


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_demo(reports_dir="reports", domain=None, now=None):
    """Run the bundled fixture through the pipeline and publish it as a session.

    Returns the catalog entry. Reuses ``pipeline.run_pipeline`` and the fixture
    table the self-test uses, so the demo can never drift from what the pipeline
    actually does — if the two disagree, one of them is broken and CI says so.
    """
    domain_obj = domains.get_domain(domain)
    if domain_obj.key not in pipeline.DOMAIN_FIXTURES:
        raise SystemExit(f"no bundled fixture for domain {domain_obj.key!r}")
    clip, events, m_per_px = pipeline.DOMAIN_FIXTURES[domain_obj.key]

    result = pipeline.run_pipeline(clip, events_path=events,
                                   meters_per_pixel=m_per_px, domain=domain_obj)
    report, tracking, metrics = result["report"], result["tracking"], result["metrics"]

    clip_dir = os.path.join(reports_dir, DEMO_ID)
    coaching_dir = os.path.join(clip_dir, "coaching")
    results_dir = os.path.join(clip_dir, "results")
    os.makedirs(coaching_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(coaching_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(coaching_dir, "summary.txt"), "w") as fh:
        fh.write(coaching.render_summary(report) + "\n")
    with open(os.path.join(results_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
        fh.write("\n")

    entry = {
        "id": DEMO_ID,
        "title": DEMO_TITLE,
        # The flag the rest of the system keys off. Without it a demo would be
        # indistinguishable from ingested footage and would quietly answer
        # "has this ever processed anything real?" with a false yes.
        "demo": True,
        "domain": metrics["domain"],
        "source": os.path.relpath(clip, HERE),
        "processed_at": now or _utc_now_iso(),
        "frames_processed": metrics["frames_processed"],
        "detected_frames": metrics["detected_frames"],
        "segment_count": metrics["segment_count"],
        # No rendered highlight videos: the fixture is a frame sequence, not a
        # decodable video, so there is nothing for ffmpeg to trim. The dashboard
        # already handles a session whose segments have no playable clip.
        "rendered_count": 0,
        "report": os.path.join(clip_dir, "coaching", "report.json"),
        "summary": os.path.join(clip_dir, "coaching", "summary.txt"),
        "feedback": None,
        "clips": [],
    }
    update_index(reports_dir, entry)
    return entry


def check_demo(reports_dir="reports"):
    """Verify a published demo exists and matches what the pipeline produces now.

    Returns a list of problems (empty when healthy). This is the continuous
    coverage the issue asked for: it fails when the report path breaks, when the
    published demo goes stale against a changed pipeline, or when the demo entry
    loses its flag and starts masquerading as real footage.
    """
    problems = []
    index_path = os.path.join(reports_dir, "index.json")
    try:
        with open(index_path) as fh:
            index = json.load(fh)
    except (FileNotFoundError, ValueError) as exc:
        return [f"{index_path} unreadable: {exc}"]

    entry = next((c for c in index.get("clips", []) if c.get("id") == DEMO_ID), None)
    if entry is None:
        return [f"no '{DEMO_ID}' entry in {index_path} — run `python demo.py`"]
    if not entry.get("demo"):
        problems.append("demo entry is missing demo:true — it would count as real footage")

    report_path = os.path.join(reports_dir, DEMO_ID, "coaching", "report.json")
    try:
        with open(report_path) as fh:
            published = json.load(fh)
    except (FileNotFoundError, ValueError) as exc:
        return problems + [f"{report_path} unreadable: {exc}"]

    domain_obj = domains.get_domain(entry.get("domain"))
    clip, events, m_per_px = pipeline.DOMAIN_FIXTURES[domain_obj.key]
    fresh = pipeline.run_pipeline(clip, events_path=events,
                                  meters_per_pixel=m_per_px, domain=domain_obj)["report"]
    for key in ("segment_count", "total_play_s"):
        if published.get(key) != fresh.get(key):
            problems.append(
                f"published demo is stale: {key} is {published.get(key)!r} on disk but "
                f"{fresh.get(key)!r} from the current pipeline — re-run `python demo.py`")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--check", action="store_true",
                        help="verify the published demo is current instead of rewriting it")
    args = parser.parse_args()

    if args.check:
        problems = check_demo(args.reports_dir)
        for p in problems:
            print(f"[demo] FAIL: {p}", file=sys.stderr)
        if problems:
            return 1
        print("[demo] OK — published demo matches the current pipeline.")
        return 0

    entry = build_demo(args.reports_dir, domain=args.domain)
    print(f"[demo] published {entry['id']}: {entry['frames_processed']} frames, "
          f"{entry['segment_count']} segments → {entry['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
