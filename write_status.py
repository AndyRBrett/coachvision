#!/usr/bin/env python3
"""Publish overseer-status.json at the repo root on every run.

The Project Overseer reads this file weekly via the GitHub API to tell whether
the pipeline is healthy-but-idle or broken. To keep that distinction meaningful,
the file is written on EVERY run -- including weeks where no footage was
processed (footage_processed: 0).

Metrics are sourced from the pipeline's results JSON, whose path is given by
the VOLLEYBALL_RESULTS_PATH environment variable (default: results/metrics.json).
Any metric the pipeline did not produce is omitted rather than invented.
"""
import json
import os
from datetime import datetime, timezone

STATUS_PATH = "overseer-status.json"
DEFAULT_RESULTS_PATH = "results/metrics.json"


def _utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a trailing Z (used for staleness)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_results() -> dict:
    """Read the pipeline's results JSON, or {} if none exists (idle week)."""
    path = os.environ.get("VOLLEYBALL_RESULTS_PATH", DEFAULT_RESULTS_PATH)
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def build_status(results: dict) -> dict:
    """Map pipeline results onto the overseer status schema.

    Always-emitted fields default to a healthy-idle record. Optional fields
    (detection_rate, model_version) are omitted when the pipeline didn't
    produce them rather than reported as fabricated values.
    """
    status = {
        "generated_at": _utc_now_iso(),
        "footage_processed": results.get("footage_processed", 0),
        # Distinguish "no new footage" (idle, fine) from "footage uploaded but
        # 0 frames came out" (broken ingest). last_footage_at is null when no
        # footage has ever been ingested; expected_frames > actual_frames flags
        # a stuck/failed run that would otherwise look identical to an idle week.
        "last_footage_at": results.get("last_footage_at"),
        "expected_frames": results.get("expected_frames", 0),
        "actual_frames": results.get("actual_frames", results.get("frames_processed", 0)),
        "failed_frames": results.get("failed_frames", 0),
        "frames_processed": results.get("frames_processed", 0),
        "errors": results.get("errors", []),
    }

    detection_rate = results.get("detection_rate")
    if detection_rate is not None:
        status["detection_rate"] = detection_rate

    model_version = results.get("model_version")
    if model_version is not None:
        status["model_version"] = model_version

    return status


def main() -> None:
    status = build_status(load_results())
    with open(STATUS_PATH, "w") as fh:
        json.dump(status, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote {STATUS_PATH}: {json.dumps(status)}")


if __name__ == "__main__":
    main()
