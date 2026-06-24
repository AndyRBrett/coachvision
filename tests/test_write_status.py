import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import write_status  # noqa: E402


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
