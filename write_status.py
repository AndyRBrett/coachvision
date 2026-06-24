#!/usr/bin/env python3
"""Publish overseer-status.json at the repo root on every run.

The Project Overseer reads this file weekly via the GitHub API to tell whether
the pipeline is healthy-but-idle or broken. To keep that distinction meaningful,
the file is written on EVERY run -- including weeks where no footage was
processed (footage_processed: 0).

Metrics are sourced from the pipeline's results JSON, whose path is given by
the COACHVISION_RESULTS_PATH environment variable (default: results/metrics.json).
Any metric the pipeline did not produce is omitted rather than invented.
"""
import json
import os
from datetime import datetime, timezone

STATUS_PATH = "overseer-status.json"
DEFAULT_RESULTS_PATH = "results/metrics.json"
DEFAULT_SELFTEST_PATH = "results/selftest.json"
DEFAULT_QUEUE_PATH = "ingest_queue.json"
# Days of idleness after which the overseer should be nudged to feed footage.
DEFAULT_IDLE_THRESHOLD_DAYS = 14


def _utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a trailing Z (used for staleness)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _days_since(last_footage_at, now: datetime):
    """Whole days between last_footage_at (ISO-8601) and now, or None if never.

    Returns None when no footage has ever been ingested (last_footage_at is
    null) or when the timestamp can't be parsed, so the overseer never sees a
    fabricated age.
    """
    if not last_footage_at:
        return None
    try:
        then = datetime.fromisoformat(last_footage_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).days


def load_results() -> dict:
    """Read the pipeline's results JSON, or {} if none exists (idle week)."""
    path = os.environ.get("COACHVISION_RESULTS_PATH", DEFAULT_RESULTS_PATH)
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def load_pending_footage() -> int:
    """Count clips waiting in the ingest queue (0 when none/unconfigured).

    The queue is produced by ingest_watch.run_scan; surfacing its depth lets the
    overseer distinguish "idle because nothing was dropped" from "footage was
    dropped but the pipeline hasn't consumed it yet" (a real stall).
    """
    path = os.environ.get("COACHVISION_INGEST_QUEUE", DEFAULT_QUEUE_PATH)
    try:
        with open(path) as fh:
            return len(json.load(fh).get("pending", []))
    except (FileNotFoundError, ValueError):
        return 0


def load_selftest() -> dict:
    """Read the pipeline self-test result, or {} if it has never run.

    pipeline.py --self-test writes results/selftest.json when the full CV
    pipeline runs cleanly on the bundled reference clip. Surfacing it lets the
    overseer tell "healthy but idle" (pipeline verified, just no new footage)
    from "broken" (self-test failing/absent) -- the exact distinction that was
    impossible while the pipeline sat at 0 frames with no proof it worked.
    """
    path = os.environ.get("COACHVISION_SELFTEST_PATH", DEFAULT_SELFTEST_PATH)
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        return {}


def _idle_threshold_days() -> int:
    try:
        return int(os.environ.get("COACHVISION_IDLE_THRESHOLD_DAYS", DEFAULT_IDLE_THRESHOLD_DAYS))
    except ValueError:
        return DEFAULT_IDLE_THRESHOLD_DAYS


def build_nudge(days_since, threshold, pending):
    """Return a human-readable idle nudge, or None when footage is flowing.

    Fires when footage has never been ingested, or when the last ingest is older
    than ``threshold`` days. If clips are already queued, the nudge points at the
    stalled queue instead of asking for more footage.
    """
    if pending > 0 and (days_since is None or days_since > threshold):
        return f"{pending} clip(s) queued but unprocessed -- pipeline may be stalled."
    if days_since is None:
        return "No footage has ever been ingested -- drop clips in the watched folder to start."
    if days_since > threshold:
        return f"No new footage in {days_since} days (threshold {threshold}) -- pipeline is idle."
    return None


def build_selftest_summary(selftest: dict) -> dict:
    """Map a raw self-test record onto the status schema's pipeline block.

    ``ok`` is False (not absent) when no self-test record exists, so a pipeline
    that has never proven itself reads as unverified rather than silently fine.
    """
    if not selftest:
        return {"ok": False, "verified_at": None}
    summary = {"ok": bool(selftest.get("ok"))}
    for key in ("verified_at", "frames_processed", "segment_count", "clip", "domain"):
        if key in selftest:
            summary[key] = selftest[key]
    return summary


def build_status(results: dict, pending_footage: int = 0, selftest: dict = None) -> dict:
    """Map pipeline results onto the overseer status schema.

    Always-emitted fields default to a healthy-idle record. Optional fields
    (detection_rate, model_version) are omitted when the pipeline didn't
    produce them rather than reported as fabricated values.
    """
    now = datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    last_footage_at = results.get("last_footage_at")
    status = {
        "generated_at": now_iso,
        # Heartbeat: written on EVERY run regardless of footage. Lets the
        # overseer distinguish "pipeline ran, no new footage" (last_run_at
        # recent, footage_processed 0) from "pipeline didn't run at all"
        # (last_run_at stale), which otherwise look identical.
        "last_run_at": now_iso,
        "footage_processed": results.get("footage_processed", 0),
        # Distinguish "no new footage" (idle, fine) from "footage uploaded but
        # 0 frames came out" (broken ingest). last_footage_at is null when no
        # footage has ever been ingested; expected_frames > actual_frames flags
        # a stuck/failed run that would otherwise look identical to an idle week.
        "last_footage_at": last_footage_at,
        # Whole days since the most recent footage was ingested, or null if
        # none ever has been. Lets the overseer flag prolonged idleness.
        "days_since_last_footage": _days_since(last_footage_at, now),
        "expected_frames": results.get("expected_frames", 0),
        "actual_frames": results.get("actual_frames", results.get("frames_processed", 0)),
        "failed_frames": results.get("failed_frames", 0),
        "frames_processed": results.get("frames_processed", 0),
        "errors": results.get("errors", []),
    }

    # Which sport the pipeline is currently configured for, when the metrics
    # record it. Omitted (not invented) for legacy results that predate domains.
    if results.get("domain") is not None:
        status["domain"] = results["domain"]

    # Idle-footage nudge (Overseer #8): surface prolonged idleness and any
    # footage that was dropped but never consumed, so "works but unused" is
    # visible rather than silently passing as healthy.
    threshold = _idle_threshold_days()
    days_since = status["days_since_last_footage"]
    nudge = build_nudge(days_since, threshold, pending_footage)
    status["idle_threshold_days"] = threshold
    status["pending_footage"] = pending_footage
    status["needs_footage"] = nudge is not None
    status["nudge"] = nudge

    # Pipeline self-test verification: proof the CV pipeline actually processes
    # frames end-to-end (resolves the "0 frames ever -- idle or broken?"
    # ambiguity that got the project flagged). Written by pipeline.py --self-test.
    status["pipeline_selftest"] = build_selftest_summary(selftest or {})

    detection_rate = results.get("detection_rate")
    if detection_rate is not None:
        status["detection_rate"] = detection_rate

    model_version = results.get("model_version")
    if model_version is not None:
        status["model_version"] = model_version

    return status


def main() -> None:
    status = build_status(
        load_results(),
        pending_footage=load_pending_footage(),
        selftest=load_selftest(),
    )
    with open(STATUS_PATH, "w") as fh:
        json.dump(status, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote {STATUS_PATH}: {json.dumps(status)}")


if __name__ == "__main__":
    main()
