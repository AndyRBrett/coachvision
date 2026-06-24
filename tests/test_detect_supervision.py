import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detect_supervision  # noqa: E402


class TestSelectTarget(unittest.TestCase):
    def test_highest_confidence_box_wins(self):
        boxes = [(0, 0, 10, 10), (100, 100, 110, 130)]
        confs = [0.4, 0.9]
        # Centre of the second (higher-confidence) box: ((100+110)/2, (100+130)/2)
        self.assertEqual(detect_supervision.select_target(boxes, confs), [105.0, 115.0])

    def test_no_candidates_returns_none(self):
        self.assertIsNone(detect_supervision.select_target([], []))

    def test_single_box_centroid(self):
        self.assertEqual(detect_supervision.select_target([(0, 0, 4, 6)], [0.5]), [2.0, 3.0])


class TestResolveClassId(unittest.TestCase):
    def test_matches_by_label_case_insensitive(self):
        names = {0: "person", 1: "Volleyball", 2: "net"}
        self.assertEqual(detect_supervision.resolve_class_id(names, "volleyball"), 1)

    def test_person_target_for_athlete_sports(self):
        names = {0: "person", 1: "racket"}
        self.assertEqual(detect_supervision.resolve_class_id(names, "person"), 0)

    def test_falls_back_to_coco_sports_ball(self):
        names = {32: "sports ball", 0: "person"}
        self.assertEqual(
            detect_supervision.resolve_class_id(names, "no-such-class"),
            detect_supervision.COCO_SPORTS_BALL_ID,
        )

    def test_no_names_returns_coco_default(self):
        self.assertEqual(
            detect_supervision.resolve_class_id(None, "anything"),
            detect_supervision.COCO_SPORTS_BALL_ID,
        )


class TestTargetClassName(unittest.TestCase):
    _VARS = ("VOLLEYBALL_TARGET_CLASS", "VOLLEYBALL_BALL_CLASS")

    def setUp(self):
        saved = {v: os.environ.pop(v, None) for v in self._VARS}

        def restore():
            for v, val in saved.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val

        self.addCleanup(restore)

    def test_default(self):
        self.assertEqual(detect_supervision.target_class_name(), "sports ball")

    def test_explicit_override_wins(self):
        os.environ["VOLLEYBALL_TARGET_CLASS"] = "person"
        self.assertEqual(detect_supervision.target_class_name("golf ball"), "golf ball")

    def test_target_env_preferred_over_legacy_alias(self):
        os.environ["VOLLEYBALL_TARGET_CLASS"] = "basketball"
        os.environ["VOLLEYBALL_BALL_CLASS"] = "volleyball"
        self.assertEqual(detect_supervision.target_class_name(), "basketball")

    def test_legacy_ball_class_alias(self):
        os.environ["VOLLEYBALL_BALL_CLASS"] = "volleyball"
        self.assertEqual(detect_supervision.target_class_name(), "volleyball")


class TestSelection(unittest.TestCase):
    def test_is_selected_reads_env(self):
        saved = os.environ.pop("VOLLEYBALL_DETECTOR", None)
        try:
            self.assertFalse(detect_supervision.is_selected())
            os.environ["VOLLEYBALL_DETECTOR"] = "supervision"
            self.assertTrue(detect_supervision.is_selected())
            os.environ["VOLLEYBALL_DETECTOR"] = "stdlib"
            self.assertFalse(detect_supervision.is_selected())
        finally:
            if saved is None:
                os.environ.pop("VOLLEYBALL_DETECTOR", None)
            else:
                os.environ["VOLLEYBALL_DETECTOR"] = saved


if __name__ == "__main__":
    unittest.main()
