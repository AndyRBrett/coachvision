import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import highlights  # noqa: E402


def _frames(spec, fps=10):
    """Build frames from a (has_ball) bool list at the given fps."""
    out = []
    for i, has in enumerate(spec):
        out.append({"frame": i, "t": i / fps, "subject": [1, 2] if has else None})
    return out


class TestSegmentRallies(unittest.TestCase):
    def test_single_continuous_rally(self):
        frames = _frames([True] * 30, fps=10)  # 3s of play
        rallies = highlights.segment_plays(frames, fps=10, min_segment_s=1.0)
        self.assertEqual(len(rallies), 1)
        self.assertAlmostEqual(rallies[0]["start"], 0.0)
        self.assertAlmostEqual(rallies[0]["end"], 2.9)

    def test_gap_splits_into_two_rallies(self):
        # 2s play, 3s ball missing (> 2s gap), 2s play
        spec = [True] * 20 + [False] * 30 + [True] * 20
        frames = _frames(spec, fps=10)
        rallies = highlights.segment_plays(frames, fps=10, max_gap_s=2.0, min_segment_s=1.0)
        self.assertEqual(len(rallies), 2)

    def test_short_gap_does_not_split(self):
        # 1s missing < 2s threshold -> still one rally
        spec = [True] * 20 + [False] * 10 + [True] * 20
        frames = _frames(spec, fps=10)
        rallies = highlights.segment_plays(frames, fps=10, max_gap_s=2.0, min_segment_s=1.0)
        self.assertEqual(len(rallies), 1)

    def test_short_rally_discarded(self):
        spec = [True] * 5  # 0.5s < 1.0s min
        frames = _frames(spec, fps=10)
        rallies = highlights.segment_plays(frames, fps=10, min_segment_s=1.0)
        self.assertEqual(rallies, [])


class TestTagRally(unittest.TestCase):
    def test_tags_within_window_only(self):
        rally = {"start": 2.0, "end": 6.0}
        events = [
            {"t": 1.0, "type": "serve"},   # before
            {"t": 2.5, "type": "attack"},  # inside
            {"t": 5.0, "type": "dig"},     # inside
            {"t": 9.0, "type": "block"},   # after
        ]
        self.assertEqual(highlights.tag_segment(rally, events), ["attack", "dig"])

    def test_canonical_tag_order(self):
        rally = {"start": 0.0, "end": 10.0}
        events = [{"t": 1, "type": "dig"}, {"t": 2, "type": "serve"}, {"t": 3, "type": "block"}]
        self.assertEqual(highlights.tag_segment(rally, events), ["serve", "block", "dig"])


class TestFfmpegCmd(unittest.TestCase):
    def test_none_without_source(self):
        self.assertIsNone(highlights.ffmpeg_trim_cmd(None, 1.0, 2.0, "out.mp4"))

    def test_cmd_has_trim_and_overlay(self):
        cmd = highlights.ffmpeg_trim_cmd("in.mp4", 5.0, 8.0, "out.mp4", tags=["attack"], pad_s=1.0)
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-ss", cmd)
        self.assertIn("4.000", cmd)  # 5.0 - 1.0 pad
        self.assertIn("-t", cmd)
        self.assertIn("-vf", cmd)
        self.assertTrue(any("attack" in part for part in cmd))
        self.assertEqual(cmd[-1], "out.mp4")

    def test_start_pad_clamped_at_zero(self):
        cmd = highlights.ffmpeg_trim_cmd("in.mp4", 0.5, 2.0, "out.mp4", pad_s=1.0)
        self.assertIn("0.000", cmd)


class TestBuildManifest(unittest.TestCase):
    def test_manifest_shape(self):
        tracking = {
            "fps": 10,
            "source": "drop/m.mp4",
            "frames": _frames([True] * 20 + [False] * 30 + [True] * 20, fps=10),
            "events": [{"t": 0.5, "type": "serve"}, {"t": 5.5, "type": "attack"}],
        }
        manifest = highlights.build_manifest(tracking)
        self.assertEqual(manifest["segment_count"], 2)
        self.assertEqual(manifest["clips"][0]["tags"], ["serve"])
        self.assertEqual(manifest["clips"][1]["tags"], ["attack"])
        self.assertTrue(manifest["clips"][0]["renderable"])
        self.assertEqual(manifest["clips"][0]["id"], "rally_001")

    def test_enricher_invoked(self):
        tracking = {
            "fps": 10,
            "source": "drop/m.mp4",
            "frames": _frames([True] * 30, fps=10),
            "events": [{"t": 1.0, "type": "serve"}],
        }

        def enricher(source, start, end, base_tags):
            return base_tags + ["block"]

        manifest = highlights.build_manifest(tracking, tag_enricher=enricher)
        self.assertIn("block", manifest["clips"][0]["tags"])

    def test_enricher_failure_is_nonfatal(self):
        tracking = {
            "fps": 10,
            "source": "drop/m.mp4",
            "frames": _frames([True] * 30, fps=10),
            "events": [{"t": 1.0, "type": "serve"}],
        }

        def boom(*a):
            raise RuntimeError("nim down")

        manifest = highlights.build_manifest(tracking, tag_enricher=boom)
        self.assertEqual(manifest["clips"][0]["tags"], ["serve"])
        self.assertIn("warning", manifest["clips"][0])


class TestRenderClips(unittest.TestCase):
    def test_dry_run_lists_commands(self):
        manifest = {"clips": [{"id": "rally_001", "ffmpeg_cmd": ["ffmpeg", "x", "out.mp4"]}]}
        res = highlights.render_clips(manifest, dry_run=True)
        self.assertEqual(res[0]["status"], "dry-run")

    def test_no_source_skipped(self):
        manifest = {"clips": [{"id": "rally_001", "ffmpeg_cmd": None}]}
        res = highlights.render_clips(manifest)
        self.assertEqual(res[0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
