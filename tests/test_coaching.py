import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import coaching  # noqa: E402


def _frames(spec, fps=10):
    """Frames from a list of ball positions (None for missing) at given fps."""
    out = []
    for i, ball in enumerate(spec):
        out.append({"frame": i, "t": i / fps, "subject": ball})
    return out


class TestBallSpeed(unittest.TestCase):
    def test_constant_speed(self):
        # Ball moves 10 px every 0.1s -> 100 px/s, constant.
        spec = [[x, 0] for x in range(0, 50, 10)]  # x = 0,10,20,30,40
        frames = _frames(spec, fps=10)
        speed = coaching.subject_speed(frames, 0.0, 0.4, fps=10)
        self.assertAlmostEqual(speed["avg_px_per_s"], 100.0)
        self.assertAlmostEqual(speed["peak_px_per_s"], 100.0)
        self.assertEqual(speed["samples"], 4)

    def test_metric_speed_added_with_calibration(self):
        spec = [[0, 0], [10, 0]]
        frames = _frames(spec, fps=10)
        speed = coaching.subject_speed(frames, 0.0, 0.1, fps=10, meters_per_pixel=0.1)
        self.assertAlmostEqual(speed["avg_m_per_s"], 10.0)  # 100 px/s * 0.1 m/px

    def test_no_motion_returns_none(self):
        frames = _frames([[5, 5]], fps=10)  # single point, no pair
        self.assertIsNone(coaching.subject_speed(frames, 0.0, 0.1, fps=10))


class TestContactHeatmap(unittest.TestCase):
    def test_event_binned_to_cell(self):
        # Ball at (75, 40) in an 80x45 court, 4x2 grid -> rightmost, bottom.
        frames = _frames([[75, 40]], fps=10)
        events = [{"t": 0.0, "type": "attack"}]
        hm = coaching.action_heatmap(frames, events, 80, 45, fps=10, cols=4, rows=2)
        self.assertEqual(hm["actions_binned"], 1)
        self.assertEqual(hm["grid"][1][3], 1)

    def test_event_pos_used_when_present(self):
        events = [{"t": 5.0, "type": "serve", "pos": [0, 0]}]
        hm = coaching.action_heatmap([], events, 80, 45, fps=10, cols=4, rows=2)
        self.assertEqual(hm["grid"][0][0], 1)

    def test_render_ascii_rows(self):
        hm = {"cols": 2, "rows": 2, "grid": [[0, 2], [0, 0]], "actions_binned": 2}
        rows = coaching.render_heatmap(hm)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(rows[0][0], " ")  # empty cell
        self.assertEqual(rows[0][1], "@")  # peak cell


class TestBuildReport(unittest.TestCase):
    def _tracking(self):
        spec = [[x, 20] for x in range(0, 45, 3)]  # 15 frames of play (1.4s)
        spec += [None] * 30                         # > 2s gap
        spec += [[x, 25] for x in range(45, 0, -3)]  # 15 more frames of play
        return {
            "fps": 10,
            "source": "drop/m.mp4",
            "width": 80,
            "height": 45,
            "frames": _frames(spec, fps=10),
            "events": [
                {"t": 0.2, "type": "serve"},
                {"t": 0.7, "type": "attack"},
                {"t": 5.2, "type": "dig"},
            ],
        }

    def test_report_shape(self):
        report = coaching.build_report(self._tracking(), meters_per_pixel=0.1)
        self.assertEqual(report["segment_count"], 2)
        self.assertEqual(len(report["segments"]), 2)
        r0 = report["segments"][0]
        self.assertIn("length_s", r0)
        self.assertIn("subject_speed", r0)
        self.assertEqual(r0["tags"], ["serve", "attack"])
        self.assertGreater(report["total_play_s"], 0)
        self.assertEqual(report["action_heatmap"]["actions_binned"], 3)

    def test_summary_renders(self):
        report = coaching.build_report(self._tracking())
        summary = coaching.render_summary(report)
        self.assertIn("Coaching report", summary)
        self.assertIn("rally_001", summary)
        self.assertIn("heatmap", summary)


if __name__ == "__main__":
    unittest.main()
