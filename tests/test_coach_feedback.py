import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import coach_feedback as cf  # noqa: E402


def _tracking(hand=2, leg=1, frames=100, detected=80, fps=10.0):
    events = ([{"type": "hand_strike"}] * hand) + ([{"type": "leg_strike"}] * leg)
    return {"fps": fps, "frame_count": frames, "detected_frames": detected, "events": events}


class TestSummarizeStats(unittest.TestCase):
    def test_counts_and_rates(self):
        text = cf.summarize_stats(_tracking(hand=3, leg=1, frames=100, fps=10.0),
                                  segment_count=4)
        self.assertIn("Strike attempts detected: 4 (hand 3, leg 1)", text)
        self.assertIn("Clip length: 10.0s", text)
        self.assertIn("Fighters in frame: 80%", text)
        self.assertIn("Exchanges (motion segments): 4", text)
        # 4 strikes over 10s -> 24/min
        self.assertIn("Strike rate: 24.0 per minute", text)

    def test_no_strikes_no_mix_line(self):
        text = cf.summarize_stats(_tracking(hand=0, leg=0))
        self.assertIn("Strike attempts detected: 0 (hand 0, leg 0)", text)
        self.assertNotIn("Hand/leg mix", text)

    def test_zero_frames_safe(self):
        text = cf.summarize_stats({"fps": 0, "frame_count": 0, "detected_frames": 0,
                                   "events": []})
        self.assertIn("0 frames", text)


class TestBuildMessages(unittest.TestCase):
    def test_text_then_images_then_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(2):
                p = os.path.join(d, f"s{i}.jpg")
                with open(p, "wb") as fh:
                    fh.write(b"\xff\xd8\xff\xe0fakejpeg")
                paths.append(p)
            msgs = cf.build_messages("STATS", paths)
            self.assertEqual(len(msgs), 1)
            content = msgs[0]["content"]
            # text, 2 images, closing text
            self.assertEqual(content[0]["type"], "text")
            self.assertIn("STATS", content[0]["text"])
            self.assertEqual([c["type"] for c in content[1:3]], ["image", "image"])
            self.assertEqual(content[-1]["type"], "text")
            self.assertEqual(content[1]["source"]["media_type"], "image/jpeg")

    def test_caps_at_max_stills(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(cf.MAX_STILLS + 3):
                p = os.path.join(d, f"s{i}.jpg")
                with open(p, "wb") as fh:
                    fh.write(b"x")
                paths.append(p)
            content = cf.build_messages("S", paths)[0]["content"]
            images = [c for c in content if c["type"] == "image"]
            self.assertEqual(len(images), cf.MAX_STILLS)


class TestGenerateFeedbackGate(unittest.TestCase):
    def test_no_key_returns_none(self):
        # Cost gate: without a key the step skips cleanly (no SDK import, no call).
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            self.assertIsNone(cf.generate_feedback(_tracking(), [], api_key=None))
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
