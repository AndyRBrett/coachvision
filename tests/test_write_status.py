import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import write_status  # noqa: E402


class TestRejectedFootageIsNotAbsentFootage(unittest.TestCase):
    """Overseer #45.

    A clip the quality gate turns away never reaches the pipeline, so it never
    updates last_footage_at and days_since climbs exactly as if nothing had been
    sent. Upload daily, have every clip refused, and the nudge asked for the one
    action already being taken while the real problem sat unmentioned in the
    same file. Opposite states, identical rendering.
    """

    NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)

    def _gate(self, days_ago, reason="too_dark", total=3):
        at = (self.NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"rejected_total": total,
                "last_rejection": {"rejected_at": at, "reason": reason}}

    def test_recent_rejections_replace_the_idle_nudge(self):
        msg = write_status.build_nudge(20, 14, 0, quality_gate=self._gate(1),
                                       now=self.NOW)
        self.assertIn("arriving but not usable", msg)
        self.assertIn("too_dark", msg)
        self.assertNotIn("pipeline is idle", msg)

    def test_they_also_replace_the_never_ingested_nudge(self):
        # The likeliest shape of the bug: every clip ever sent was rejected, so
        # last_footage_at was never set at all and the panel said nobody had
        # ever sent anything.
        msg = write_status.build_nudge(None, 14, 0, quality_gate=self._gate(2),
                                       now=self.NOW)
        self.assertIn("arriving but not usable", msg)
        self.assertNotIn("never been ingested", msg)

    def test_a_stale_rejection_does_not_mask_real_idleness(self):
        # Past the window nothing has arrived for a threshold period either way,
        # so the plain idle reading is the true one again.
        msg = write_status.build_nudge(40, 14, 0, quality_gate=self._gate(30),
                                       now=self.NOW)
        self.assertIn("pipeline is idle", msg)

    def test_the_rejection_window_is_the_idle_threshold(self):
        # Not a second constant. Two definitions of "lately" in one file is how
        # the dashboard ended up with two clocks.
        inside = write_status.build_nudge(20, 14, 0, quality_gate=self._gate(14),
                                          now=self.NOW)
        outside = write_status.build_nudge(20, 14, 0, quality_gate=self._gate(15),
                                           now=self.NOW)
        self.assertIn("arriving but not usable", inside)
        self.assertIn("pipeline is idle", outside)

    def test_flowing_footage_still_nudges_about_nothing(self):
        # A rejection alongside working ingest is already visible in the
        # quality_gate block; it does not need a headline.
        self.assertIsNone(write_status.build_nudge(3, 14, 0,
                                                   quality_gate=self._gate(1),
                                                   now=self.NOW))

    def test_a_stalled_queue_still_wins(self):
        # Clips sitting unprocessed is a different failure from clips being
        # refused, and the more urgent one.
        msg = write_status.build_nudge(None, 14, 2, quality_gate=self._gate(1),
                                       now=self.NOW)
        self.assertIn("queued but unprocessed", msg)

    def test_an_unparseable_rejection_timestamp_is_ignored(self):
        # Never fabricate an age. A rejection whose stamp cannot be read must
        # not silently suppress the idle nudge.
        gate = {"rejected_total": 1,
                "last_rejection": {"rejected_at": "not-a-date", "reason": "too_dark"}}
        msg = write_status.build_nudge(20, 14, 0, quality_gate=gate, now=self.NOW)
        self.assertIn("pipeline is idle", msg)

    def test_a_rejection_warning_does_not_ask_for_footage(self):
        # Codex P2 on #46, and it is this bug one layer down. needs_footage was
        # derived from the nudge being non-null, so the sentence said "check the
        # source rather than sending more" while the machine-readable field
        # still said SEND MORE -- and a consumer reading the field instead of
        # the prose would take exactly the action this change exists to prevent.
        import write_status as ws
        gate = self._gate(1)
        status = ws.build_status({"last_footage_at": None}, pending_footage=0,
                                 quality_gate=gate)
        self.assertIn("arriving but not usable", status["nudge"])
        self.assertFalse(status["needs_footage"])
        self.assertEqual(status["nudge_kind"], "quality_gate")

    def test_a_real_idle_status_still_asks_for_footage(self):
        # The other half: narrowing needs_footage must not silence the case it
        # was there for.
        import write_status as ws
        status = ws.build_status({"last_footage_at": None}, pending_footage=0)
        self.assertTrue(status["needs_footage"])
        self.assertEqual(status["nudge_kind"], "idle")

    def test_nudge_kind_agrees_with_the_sentence(self):
        # The kind is computed beside build_nudge rather than inside it, so this
        # pins the two against each other -- a reordering in one that is not
        # mirrored in the other shows up here rather than in a consumer.
        import write_status as ws
        cases = [
            ({"last_footage_at": None}, 0, self._gate(1), "quality_gate",
             "arriving but not usable"),
            ({"last_footage_at": None}, 0, None, "idle", "has ever been ingested"),
            ({"last_footage_at": None}, 2, self._gate(1), "queue_stalled",
             "queued but unprocessed"),
        ]
        for results, pending, gate, kind, fragment in cases:
            status = ws.build_status(results, pending_footage=pending,
                                     quality_gate=gate)
            self.assertEqual(status["nudge_kind"], kind, fragment)
            self.assertIn(fragment, status["nudge"])

    def test_a_healthy_status_has_no_kind(self):
        import write_status as ws
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        status = ws.build_status({"last_footage_at": recent}, pending_footage=0)
        self.assertIsNone(status["nudge"])
        self.assertIsNone(status["nudge_kind"])
        self.assertFalse(status["needs_footage"])

    def test_no_gate_block_behaves_exactly_as_before(self):
        for gate in (None, {}, {"rejected_total": 0, "last_rejection": None}):
            msg = write_status.build_nudge(20, 14, 0, quality_gate=gate, now=self.NOW)
            self.assertIn("pipeline is idle", msg, gate)


class TestBuildNudge(unittest.TestCase):
    def test_never_ingested(self):
        msg = write_status.build_nudge(None, 14, 0)
        self.assertIn("ever been ingested", msg.lower())

    def test_idle_past_threshold(self):
        msg = write_status.build_nudge(20, 14, 0)
        self.assertIn("20 days", msg)

    def test_within_threshold_no_nudge(self):
        self.assertIsNone(write_status.build_nudge(3, 14, 0))

    def test_stalled_queue_takes_priority(self):
        msg = write_status.build_nudge(None, 14, 2)
        self.assertIn("queued", msg)
        self.assertIn("2", msg)

    def test_queue_with_recent_footage_no_nudge(self):
        # Footage recent (within threshold) -> not idle even if queue nonempty.
        self.assertIsNone(write_status.build_nudge(1, 14, 2))


class TestBuildStatus(unittest.TestCase):
    def test_idle_record_flags_needs_footage(self):
        status = write_status.build_status({}, pending_footage=0)
        self.assertTrue(status["needs_footage"])
        self.assertIsNotNone(status["nudge"])
        self.assertEqual(status["pending_footage"], 0)
        self.assertEqual(status["idle_threshold_days"], write_status.DEFAULT_IDLE_THRESHOLD_DAYS)

    def test_recent_footage_no_nudge(self):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        status = write_status.build_status({"last_footage_at": now, "footage_processed": 3})
        self.assertFalse(status["needs_footage"])
        self.assertIsNone(status["nudge"])

    def test_pending_footage_surfaced(self):
        status = write_status.build_status({}, pending_footage=5)
        self.assertEqual(status["pending_footage"], 5)
        self.assertIn("queued", status["nudge"])

    def test_existing_fields_preserved(self):
        status = write_status.build_status({"frames_processed": 10, "footage_processed": 2})
        self.assertEqual(status["frames_processed"], 10)
        self.assertEqual(status["footage_processed"], 2)
        self.assertIn("generated_at", status)
        self.assertIn("last_run_at", status)


class TestQualityGateSummary(unittest.TestCase):
    def test_defaults_to_no_rejections(self):
        status = write_status.build_status({})
        self.assertEqual(status["quality_gate"]["rejected_total"], 0)
        self.assertIsNone(status["quality_gate"]["last_rejection"])

    def test_surfaces_rejection_record(self):
        record = {
            "rejected_total": 3,
            "last_rejection": {"reason": "too_dark", "detail": "mean luminance 2.0"},
        }
        status = write_status.build_status({}, quality_gate=record)
        self.assertEqual(status["quality_gate"]["rejected_total"], 3)
        self.assertEqual(status["quality_gate"]["last_rejection"]["reason"], "too_dark")


class TestSelfTestSummary(unittest.TestCase):
    def test_missing_selftest_reads_unverified(self):
        summary = write_status.build_selftest_summary({})
        self.assertFalse(summary["ok"])
        self.assertIsNone(summary["verified_at"])

    def test_passing_selftest_surfaced(self):
        record = {
            "ok": True,
            "verified_at": "2026-06-22T06:00:00Z",
            "frames_processed": 104,
            "segment_count": 3,
            "clip": "fixtures/reference_clip.pgm.gz",
        }
        summary = write_status.build_selftest_summary(record)
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["frames_processed"], 104)
        self.assertEqual(summary["segment_count"], 3)

    def test_status_includes_pipeline_selftest(self):
        status = write_status.build_status({}, selftest={"ok": True, "frames_processed": 104})
        self.assertIn("pipeline_selftest", status)
        self.assertTrue(status["pipeline_selftest"]["ok"])

    def test_status_selftest_defaults_unverified(self):
        status = write_status.build_status({})
        self.assertFalse(status["pipeline_selftest"]["ok"])


if __name__ == "__main__":
    unittest.main()
