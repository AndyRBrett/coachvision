"""Tests for the bundled demo session (issue #21).

The demo exists so the dashboard isn't empty before real footage arrives, and so
the report-WRITING path is exercised continuously rather than first attempted on
the day real footage lands. The load-bearing property is the separation: a demo
must never be counted as ingested footage.

unittest, not pytest — this repo's CI runs `python -m unittest discover` and
pytest is not installed on the runner.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demo  # noqa: E402


class DemoTestCase(unittest.TestCase):
    """Publishes a demo into a throwaway reports dir for each test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reports = os.path.join(self.tmp.name, "reports")
        self.entry = demo.build_demo(self.reports, domain="martial_arts")

    def tearDown(self):
        self.tmp.cleanup()

    def _index(self):
        with open(os.path.join(self.reports, "index.json")) as fh:
            return json.load(fh)

    def _write_index(self, index):
        with open(os.path.join(self.reports, "index.json"), "w") as fh:
            json.dump(index, fh)


class TestDemoArtifacts(DemoTestCase):
    def test_demo_publishes_a_full_report(self):
        path = os.path.join(self.reports, "demo", "coaching", "report.json")
        with open(path) as fh:
            report = json.load(fh)
        self.assertGreaterEqual(report["segment_count"], 1)
        self.assertGreater(report["total_play_s"], 0)
        self.assertTrue(any(s["tags"] for s in report["segments"]),
                        "no coaching tags in the demo report")

    def test_demo_writes_every_artifact_a_real_session_does(self):
        for rel in ("coaching/report.json", "coaching/summary.txt", "results/metrics.json"):
            self.assertTrue(os.path.exists(os.path.join(self.reports, "demo", rel)),
                            f"missing {rel}")

    def test_demo_entry_is_flagged_so_it_cannot_pass_as_real_footage(self):
        # THE ONE THAT MATTERS. Without demo:true the entry is indistinguishable
        # from ingested footage, and "has this ever processed anything real?"
        # silently starts answering yes.
        self.assertIs(self.entry["demo"], True)

    def test_demo_lands_in_the_shared_catalog(self):
        ids = [c["id"] for c in self._index()["clips"]]
        self.assertIn(self.entry["id"], ids)

    def test_republishing_does_not_duplicate_the_catalog_entry(self):
        demo.build_demo(self.reports, domain="martial_arts")
        matches = [c for c in self._index()["clips"] if c["id"] == demo.DEMO_ID]
        self.assertEqual(len(matches), 1)


class TestDemoLeavesRealSessionsAlone(unittest.TestCase):
    def test_demo_does_not_disturb_existing_real_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = os.path.join(tmp, "reports")
            os.makedirs(reports)
            real = {"id": "clip-real", "title": "Sparring",
                    "processed_at": "2026-08-01T00:00:00Z"}
            with open(os.path.join(reports, "index.json"), "w") as fh:
                json.dump({"clips": [real]}, fh)

            demo.build_demo(reports, domain="martial_arts")

            with open(os.path.join(reports, "index.json")) as fh:
                index = json.load(fh)
            kept = next(c for c in index["clips"] if c["id"] == "clip-real")
            self.assertEqual(kept["title"], "Sparring")
            self.assertFalse(kept.get("demo"))


class TestDemoCheck(DemoTestCase):
    def test_check_passes_on_a_freshly_published_demo(self):
        self.assertEqual(demo.check_demo(self.reports), [])

    def test_check_detects_a_stale_published_report(self):
        # The continuous-coverage case: the pipeline changed, the committed demo
        # didn't. Silence here would let the sample drift away from reality.
        path = os.path.join(self.reports, "demo", "coaching", "report.json")
        with open(path) as fh:
            report = json.load(fh)
        report["segment_count"] += 99
        with open(path, "w") as fh:
            json.dump(report, fh)
        self.assertTrue(any("stale" in p for p in demo.check_demo(self.reports)))

    def test_check_flags_a_demo_entry_that_lost_its_flag(self):
        index = self._index()
        for c in index["clips"]:
            if c["id"] == demo.DEMO_ID:
                c["demo"] = False
        self._write_index(index)
        self.assertTrue(any("real footage" in p for p in demo.check_demo(self.reports)))


class TestDemoCheckEdges(unittest.TestCase):
    def test_check_reports_a_missing_demo_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = os.path.join(tmp, "reports")
            os.makedirs(reports)
            with open(os.path.join(reports, "index.json"), "w") as fh:
                json.dump({"clips": []}, fh)
            problems = demo.check_demo(reports)
            self.assertTrue(problems)
            self.assertIn("no 'demo' entry", problems[0])

    def test_committed_demo_matches_the_current_pipeline(self):
        # Guards the artifact actually in the repo, the way the workflow runs it.
        os.environ.setdefault("COACHVISION_DOMAIN", "martial_arts")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual(demo.check_demo(os.path.join(repo_root, "reports")), [])


if __name__ == "__main__":
    unittest.main()
