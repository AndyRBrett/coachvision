import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline  # noqa: E402


class TestPipelineSelfTest(unittest.TestCase):
    """The end-to-end guard: run the full CV pipeline on the bundled reference
    clip. If detection, segmentation, tagging, or coaching breaks, this fails
    the build instead of letting the pipeline sit silently at zero frames."""

    def test_self_test_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.self_test(results_dir=tmp, verbose=False)

            tracking = result["tracking"]
            report = result["report"]
            manifest = result["manifest"]

            # Detection actually processed frames (the whole point).
            self.assertGreater(tracking["frame_count"], 0)
            self.assertGreater(tracking["detected_frames"], 0)

            # Highlights + coaching produced real output.
            self.assertGreaterEqual(manifest["segment_count"], 1)
            self.assertEqual(report["segment_count"], manifest["segment_count"])
            self.assertTrue(any(r["tags"] for r in report["segments"]))
            self.assertGreater(report["action_heatmap"]["actions_binned"], 0)
            self.assertTrue(any(r["subject_speed"] for r in report["segments"]))

            # Metrics roll-up is consistent (this is what write_status reads).
            metrics = result["metrics"]
            self.assertEqual(metrics["frames_processed"], tracking["frame_count"])
            self.assertEqual(metrics["footage_processed"], 1)

            # selftest.json was written for the overseer.
            self.assertTrue(os.path.exists(os.path.join(tmp, "selftest.json")))

    def test_reference_clip_has_three_rallies(self):
        # Locks in the structure of the bundled fixture so a fixture or
        # segmentation regression is caught explicitly, not just "segment_count>=1".
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.self_test(results_dir=tmp, verbose=False)
        self.assertEqual(result["report"]["segment_count"], 3)


class TestSelfTestHistory(unittest.TestCase):
    """Recurring self-test runs are tracked over time (Overseer #26) so a
    dependency upgrade that silently shifts detection/segmentation is visible
    as drift against the established baseline, not just a still-passing run."""

    def test_first_run_becomes_baseline_with_no_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.self_test(results_dir=tmp, verbose=False, domain="volleyball")
            with open(os.path.join(tmp, "selftest.json")) as fh:
                selftest = json.load(fh)
            self.assertNotIn("drift", selftest)

            with open(os.path.join(tmp, pipeline.HISTORY_FILENAME)) as fh:
                history = json.load(fh)
            self.assertEqual(len(history["volleyball"]), 1)
            self.assertTrue(history["volleyball"][0]["ok"])

    def test_repeated_run_matches_baseline_with_no_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.self_test(results_dir=tmp, verbose=False, domain="volleyball")
            pipeline.self_test(results_dir=tmp, verbose=False, domain="volleyball")
            with open(os.path.join(tmp, pipeline.HISTORY_FILENAME)) as fh:
                history = json.load(fh)
            self.assertEqual(len(history["volleyball"]), 2)
            with open(os.path.join(tmp, "selftest.json")) as fh:
                selftest = json.load(fh)
            self.assertNotIn("drift", selftest)

    def test_segment_count_drift_is_flagged_against_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            drift = pipeline._record_history(
                tmp, "volleyball", {"ok": True, "frames_processed": 77, "segment_count": 3},
            )
            self.assertEqual(drift, [])
            drift = pipeline._record_history(
                tmp, "volleyball", {"ok": True, "frames_processed": 77, "segment_count": 2},
            )
            self.assertEqual(drift, ["segment_count: 3 -> 2"])

    def test_ok_flip_is_flagged_and_history_keeps_the_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline._record_history(
                tmp, "volleyball", {"ok": True, "frames_processed": 77, "segment_count": 3},
            )
            drift = pipeline._record_history(
                tmp, "volleyball", {"ok": False, "frames_processed": 0, "segment_count": 0},
            )
            self.assertIn("ok: True -> False", drift)
            with open(os.path.join(tmp, pipeline.HISTORY_FILENAME)) as fh:
                history = json.load(fh)
            self.assertEqual(len(history["volleyball"]), 2)
            self.assertFalse(history["volleyball"][-1]["ok"])

    def test_history_is_capped_at_max_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(pipeline.MAX_HISTORY_ENTRIES + 5):
                pipeline._record_history(
                    tmp, "volleyball", {"ok": True, "frames_processed": 77, "segment_count": 3, "run": i},
                )
            with open(os.path.join(tmp, pipeline.HISTORY_FILENAME)) as fh:
                history = json.load(fh)
            self.assertEqual(len(history["volleyball"]), pipeline.MAX_HISTORY_ENTRIES)
            # Oldest entries roll off; the newest run is kept.
            self.assertEqual(history["volleyball"][-1]["run"], pipeline.MAX_HISTORY_ENTRIES + 4)


if __name__ == "__main__":
    unittest.main()
