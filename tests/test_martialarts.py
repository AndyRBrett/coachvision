import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import coaching  # noqa: E402
import detect  # noqa: E402
import domains  # noqa: E402
import pipeline  # noqa: E402


def _frame(width, height, bright_pixels):
    """A raster (bytes) with the given (x, y) pixels set to 255."""
    buf = bytearray(width * height)
    for x, y in bright_pixels:
        buf[y * width + x] = 255
    return bytes(buf)


class TestMotionDetector(unittest.TestCase):
    def test_motion_centroid_when_subject_moves(self):
        w, h = 10, 8
        prev = _frame(w, h, [(2, 2), (3, 2), (2, 3), (3, 3)])
        cur = _frame(w, h, [(6, 2), (7, 2), (6, 3), (7, 3)])
        pos = detect.detect_motion(prev, cur, w, h, threshold=60, min_pixels=3)
        self.assertIsNotNone(pos)
        # Centroid spans the vacated and newly-occupied blocks (x 2..3 and 6..7).
        self.assertAlmostEqual(pos[0], 4.5, places=1)

    def test_still_subject_is_no_motion(self):
        w, h = 10, 8
        frame = _frame(w, h, [(4, 4), (5, 4), (4, 5), (5, 5)])
        # Identical consecutive frames -> nothing changed -> no subject.
        self.assertIsNone(detect.detect_motion(frame, frame, w, h, threshold=60, min_pixels=3))


class TestMartialArtsDetection(unittest.TestCase):
    def setUp(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.clip = os.path.join(here, "fixtures", "martialarts_clip.pgm.gz")
        self.events = os.path.join(here, "fixtures", "martialarts_clip.events.json")

    def test_motion_energy_tracks_and_segments(self):
        ev, fps, _ = detect._load_events_sidecar(self.events)
        tracking = detect.run_detection(self.clip, fps=fps, events=ev, domain="martial_arts")
        self.assertEqual(tracking["domain"], "martial_arts")
        self.assertGreater(tracking["detected_frames"], 0)
        # The fighter freezes between exchanges, so some frames carry no subject.
        self.assertLess(tracking["detected_frames"], tracking["frame_count"])

        report = coaching.build_report(tracking, domain="martial_arts")
        self.assertEqual(report["domain"], "martial_arts")
        self.assertGreaterEqual(report["segment_count"], 2)
        # Segment ids and wording follow the martial-arts vocabulary.
        self.assertTrue(report["segments"][0]["id"].startswith("exchange_"))
        summary = coaching.render_summary(report)
        self.assertIn("exchanges", summary)
        self.assertIn("strike-zone heatmap", summary)
        self.assertIn("fighter speed", summary)


class TestMartialArtsSelfTest(unittest.TestCase):
    def test_self_test_passes(self):
        result = pipeline.self_test(results_dir=None, verbose=False, domain="martial_arts")
        self.assertEqual(result["metrics"]["domain"], "martial_arts")
        self.assertGreaterEqual(result["metrics"]["segment_count"], 1)
        # Strikes were tagged from the bundled events.
        self.assertTrue(any(r["tags"] for r in result["report"]["segments"]))


if __name__ == "__main__":
    unittest.main()
