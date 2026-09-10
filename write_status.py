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
DEFAULT_QUALITY_GATE_PATH = "results/quality_gate.json"
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


def load_quality_gate() -> dict:
    """Read the clip quality gate's rejection record, or {} if none yet exists.

    Written by quality_gate.record_rejection when process_footage.py rejects a
    clip (too_short/too_dark/resolution_too_low/corrupt_file) before it reaches
    the CV pipeline. Surfacing it here is what keeps a rejected clip visible
    instead of just quietly not appearing in reports/index.json.
    """
    path = os.environ.get("COACHVISION_QUALITY_GATE_PATH", DEFAULT_QUALITY_GATE_PATH)
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


def quality_gate_warning(days_since, threshold, pending, quality_gate=None, now=None):
    """The "arriving but refused" message, or None when it does not apply.

    Separate from build_nudge because the two mean OPPOSITE things to a machine.
    `needs_footage` is derived from the nudge being non-null, so folding this in
    would publish prose saying "check the source rather than sending more"
    beside a boolean still saying send more -- and any consumer reading the
    field instead of the sentence would take exactly the action this exists to
    prevent (Codex, #46). The caller distinguishes them; see build_status.
    """
    # Only where an idle nudge would otherwise fire. While footage is flowing
    # there is nothing to nudge about, and a rejection alongside working ingest
    # is already visible in the quality_gate block without a headline.
    if not (days_since is None or days_since > threshold):
        return None
    # A stalled queue speaks first -- clips sitting unprocessed is a different
    # failure from clips being refused, and the more urgent one.
    if pending > 0:
        return None

    rejection = (quality_gate or {}).get("last_rejection") or {}
    age = _days_since(rejection.get("rejected_at"), now or datetime.now(timezone.utc))
    if age is None or age > threshold:
        return None

    total = (quality_gate or {}).get("rejected_total") or 0
    reason = rejection.get("reason") or "unknown"
    return (f"Footage is arriving but not usable -- {total} clip(s) rejected "
            f"before processing, most recently {reason}. "
            f"Check the source rather than sending more.")


def build_nudge(days_since, threshold, pending, quality_gate=None, now=None):
    """Return a human-readable idle nudge, or None when footage is flowing.

    Fires when footage has never been ingested, or when the last ingest is older
    than ``threshold`` days. If clips are already queued, the nudge points at the
    stalled queue instead of asking for more footage.

    REJECTED FOOTAGE IS NOT ABSENT FOOTAGE (Overseer #45). A clip the quality
    gate turns away never reaches the pipeline, so it never updates
    ``last_footage_at`` and ``days_since`` keeps climbing exactly as though
    nothing had been sent. Upload daily and have every clip refused and the
    nudge would read "no new footage -- pipeline is idle", asking for the one
    action already being taken, while the real problem sits unmentioned in the
    same status file. The two states are opposites and they rendered
    identically; worse than silence, because it sends someone to do the wrong
    thing.

    A rejection only speaks for the window the idle nudge itself uses. Reusing
    ``threshold`` rather than inventing a second constant is deliberate: this
    file would then hold two definitions of "lately", and a second definition of
    recency is how the dashboard ended up with two clocks. Past that window
    nothing has arrived for a threshold period either way, so the plain idle
    reading is the true one again.
    """
    if pending > 0 and (days_since is None or days_since > threshold):
        return f"{pending} clip(s) queued but unprocessed -- pipeline may be stalled."

    warning = quality_gate_warning(days_since, threshold, pending, quality_gate, now)
    if warning:
        return warning

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
    for key in ("verified_at", "frames_processed", "segment_count", "clip", "domain", "drift"):
        if key in selftest:
            summary[key] = selftest[key]
    return summary


def build_status(results: dict, pending_footage: int = 0, selftest: dict = None,
                  quality_gate: dict = None) -> dict:
    """Map pipeline results onto the overseer status schema.

    Always-emitted fields default to a healthy-idle record. Optional fields
    (detection_rate, model_version) are omitted when the pipeline didn't
    produce them rather than reported as fabricated values.
    """
    now = datetime.now(timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    last_footage_at = results.get("last_footage_at")
    status = {
        # Product identity: the heartbeat leads with the app name so the status
        # reads as "coachvision" rather than any one sport. The sport actually
        # exercised is recorded separately (`domain` / `pipeline_selftest.domain`).
        "app": "coachvision",
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
    warning = quality_gate_warning(days_since, threshold, pending_footage,
                                   quality_gate, now)
    nudge = build_nudge(days_since, threshold, pending_footage,
                        quality_gate=quality_gate, now=now)
    status["idle_threshold_days"] = threshold
    status["pending_footage"] = pending_footage
    # NOT simply `nudge is not None` (Codex, #46). A quality-gate warning says
    # the opposite of "send footage" -- footage IS arriving, it is being
    # refused -- so flagging it here would publish a machine-readable field
    # telling any consumer to do the one thing the sentence beside it warns
    # against. The prose was right and the boolean was still wrong, which is
    # this bug one layer down.
    status["needs_footage"] = nudge is not None and warning is None
    status["nudge"] = nudge
    # Which KIND of attention, so a consumer never has to parse the sentence to
    # find out. A bool that only distinguishes "send footage" from "nothing"
    # cannot express a third state, and there are now four.
    if warning is not None:
        status["nudge_kind"] = "quality_gate"
    elif nudge is None:
        status["nudge_kind"] = None
    elif pending_footage > 0:
        status["nudge_kind"] = "queue_stalled"
    else:
        status["nudge_kind"] = "idle"

    # Pipeline self-test verification: proof the CV pipeline actually processes
    # frames end-to-end (resolves the "0 frames ever -- idle or broken?"
    # ambiguity that got the project flagged). Written by pipeline.py --self-test.
    status["pipeline_selftest"] = build_selftest_summary(selftest or {})

    # Clip quality gate (Overseer #43): a clip rejected before reaching the CV
    # pipeline (too_short/too_dark/resolution_too_low/corrupt_file) shows up
    # here with its reason, instead of silently producing degraded/empty
    # technique-tracking output or just not appearing anywhere.
    quality_gate = quality_gate or {}
    status["quality_gate"] = {
        "rejected_total": quality_gate.get("rejected_total", 0),
        "last_rejection": quality_gate.get("last_rejection"),
    }

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
        quality_gate=load_quality_gate(),
    )
    with open(STATUS_PATH, "w") as fh:
        json.dump(status, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Wrote {STATUS_PATH}: {json.dumps(status)}")


if __name__ == "__main__":
    main()
