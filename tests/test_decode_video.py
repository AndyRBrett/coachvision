import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decode_video  # noqa: E402
import detect  # noqa: E402


def _pgm_stream(width, height, n_frames):
    """Concatenated P5 PGM frames (what ffmpeg image2pipe/pgm emits), as bytes."""
    header = b"P5\n%d %d\n255\n" % (width, height)
    out = bytearray()
    for i in range(n_frames):
        buf = bytearray(width * height)
        buf[(i % height) * width + (i % width)] = 255  # one moving bright pixel
        out += header + bytes(buf)
    return bytes(out)


class TestFfmpegCmd(unittest.TestCase):
    def test_cmd_shape(self):
        cmd = decode_video.ffmpeg_decode_cmd("in.mp4", fps=12, width=128)
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("in.mp4", cmd)
        joined = " ".join(cmd)
        self.assertIn("fps=12", joined)
        self.assertIn("scale=128:-2", joined)
        self.assertIn("format=gray", joined)
        self.assertIn("pgm", joined)


class TestDecodeRoundtrip(unittest.TestCase):
    def setUp(self):
        self._orig = decode_video._run_ffmpeg

    def tearDown(self):
        decode_video._run_ffmpeg = self._orig

    def test_decode_writes_parseable_clip(self):
        decode_video._run_ffmpeg = lambda cmd: _pgm_stream(8, 6, 5)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "clip.pgm.gz")
            summary = decode_video.decode_to_pgm_gz("anything.mp4", out, fps=10)
            self.assertEqual(summary["frame_count"], 5)
            self.assertEqual((summary["width"], summary["height"]), (8, 6))
            # The detector must be able to read exactly what we wrote.
            w, h, frames = detect.load_pgm_frames(out)
            self.assertEqual((w, h, len(frames)), (8, 6, 5))

    def test_empty_output_raises(self):
        decode_video._run_ffmpeg = lambda cmd: b""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                decode_video.decode_to_pgm_gz("bad.mp4", os.path.join(d, "x.pgm.gz"))


class TestResolveSource(unittest.TestCase):
    def test_missing_path_raises(self):
        with self.assertRaises(ValueError):
            decode_video.resolve_source(clip_path="/no/such/file.mp4")

    def test_requires_one_of(self):
        with self.assertRaises(ValueError):
            decode_video.resolve_source()


if __name__ == "__main__":
    unittest.main()
