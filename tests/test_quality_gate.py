import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decode_video  # noqa: E402
import quality_gate  # noqa: E402


def _ffprobe_json(width=320, height=240, duration="5.0", fps="30/1"):
    return json.dumps({
        "streams": [{"width": width, "height": height,
                     "r_frame_rate": fps, "duration": duration}],
        "format": {"duration": duration},
    }).encode()


def _pgm_stream(width, height, n_frames, fill=200):
    header = b"P5\n%d %d\n255\n" % (width, height)
    out = bytearray()
    for _ in range(n_frames):
        out += header + bytes([fill]) * (width * height)
    return bytes(out)


class TestProbeMetadata(unittest.TestCase):
    def setUp(self):
        self._orig = quality_gate._run_ffprobe

    def tearDown(self):
        quality_gate._run_ffprobe = self._orig

    def test_parses_resolution_duration_fps(self):
        quality_gate._run_ffprobe = lambda cmd: json.loads(_ffprobe_json())
        meta = quality_gate.probe_metadata("in.mp4")
        self.assertEqual((meta["width"], meta["height"]), (320, 240))
        self.assertEqual(meta["duration"], 5.0)
        self.assertEqual(meta["fps"], 30.0)

    def test_no_stream_raises(self):
        quality_gate._run_ffprobe = lambda cmd: {"streams": [], "format": {}}
        with self.assertRaises(ValueError):
            quality_gate.probe_metadata("bad.mp4")


class TestCheckClipQuality(unittest.TestCase):
    def setUp(self):
        self._orig_probe = quality_gate.probe_metadata
        self._orig_lum = quality_gate.sample_mean_luminance

    def tearDown(self):
        quality_gate.probe_metadata = self._orig_probe
        quality_gate.sample_mean_luminance = self._orig_lum

    def test_corrupt_file(self):
        quality_gate.probe_metadata = lambda src: (_ for _ in ()).throw(ValueError("no stream"))
        result = quality_gate.check_clip_quality("bad.mp4")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_CORRUPT_FILE)

    def test_resolution_too_low(self):
        quality_gate.probe_metadata = lambda src: {"width": 16, "height": 16, "duration": 5.0, "fps": 30.0}
        result = quality_gate.check_clip_quality("small.mp4")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_RESOLUTION_TOO_LOW)

    def test_too_short(self):
        quality_gate.probe_metadata = lambda src: {"width": 320, "height": 240, "duration": 0.2, "fps": 30.0}
        result = quality_gate.check_clip_quality("short.mp4")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_TOO_SHORT)

    def test_unreadable_frame_rate_is_corrupt(self):
        quality_gate.probe_metadata = lambda src: {"width": 320, "height": 240, "duration": 5.0, "fps": None}
        result = quality_gate.check_clip_quality("nofps.mp4")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_CORRUPT_FILE)

    def test_too_dark(self):
        quality_gate.probe_metadata = lambda src: {"width": 320, "height": 240, "duration": 5.0, "fps": 30.0}
        quality_gate.sample_mean_luminance = lambda src: 2.0
        result = quality_gate.check_clip_quality("dark.mp4")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_TOO_DARK)

    def test_passes_all_checks(self):
        quality_gate.probe_metadata = lambda src: {"width": 320, "height": 240, "duration": 5.0, "fps": 30.0}
        quality_gate.sample_mean_luminance = lambda src: 120.0
        result = quality_gate.check_clip_quality("good.mp4")
        self.assertTrue(result["ok"])
        self.assertEqual(result["metadata"]["width"], 320)


class TestSampleMeanLuminance(unittest.TestCase):
    def setUp(self):
        self._orig = decode_video._run_ffmpeg

    def tearDown(self):
        decode_video._run_ffmpeg = self._orig

    def test_computes_mean_of_frame_bytes(self):
        decode_video._run_ffmpeg = lambda cmd: _pgm_stream(4, 4, 2, fill=100)
        mean = quality_gate.sample_mean_luminance("anything.mp4")
        self.assertAlmostEqual(mean, 100.0)

    def test_empty_sample_returns_none(self):
        decode_video._run_ffmpeg = lambda cmd: b""
        self.assertIsNone(quality_gate.sample_mean_luminance("anything.mp4"))


class TestRecordRejection(unittest.TestCase):
    def test_accumulates_total_and_records_latest(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "quality_gate.json")
            quality_gate.record_rejection("too_dark", "mean luminance 2.0", status_path=path)
            record = quality_gate.record_rejection(
                "too_short", "duration 0.1s", metadata={"width": 10}, status_path=path)
            self.assertEqual(record["rejected_total"], 2)
            self.assertEqual(record["last_rejection"]["reason"], "too_short")
            self.assertEqual(record["last_rejection"]["metadata"]["width"], 10)
            with open(path) as fh:
                on_disk = json.load(fh)
            self.assertEqual(on_disk["rejected_total"], 2)


if __name__ == "__main__":
    unittest.main()
