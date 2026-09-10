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
    # Every module-level name these tests stub. Saved and restored as a SET
    # rather than one attribute at a time: the per-name form silently stopped
    # covering sampled_elapsed_seconds when it was added, leaving a stub in
    # place for the rest of the suite and making the run order-dependent
    # (Codex P3 on #44). Adding a name to this tuple is now the only thing a
    # new stub needs.
    PATCHED = ("probe_metadata", "sample_mean_luminance")

    def setUp(self):
        self._orig = {name: getattr(quality_gate, name) for name in self.PATCHED}

    def tearDown(self):
        for name, func in self._orig.items():
            setattr(quality_gate, name, func)

    def test_every_stubbed_name_is_restored(self):
        # The guard on the guard: a stub applied to a name missing from PATCHED
        # would leak, and the symptom is a DIFFERENT test failing later for no
        # visible reason.
        for name in self.PATCHED:
            self.assertTrue(hasattr(quality_gate, name), name)

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

    def test_an_unreadable_duration_is_refused_not_measured(self):
        # THE RECORDED DECISION, not an oversight. Four review rounds tried to
        # measure the length instead -- frame counts against a nominal rate,
        # packet timestamps, both together -- and each produced a new way to be
        # wrong about VFR or about streams carrying no PTS. See the block at the
        # top of quality_gate.py. A file whose duration ffprobe cannot read is
        # now refused with a reason naming what actually happened, and the cost
        # (some fragmented and raw streams need a remux) is visible in
        # results/quality_gate.json rather than hidden in a guess.
        quality_gate.probe_metadata = lambda src: {
            "width": 320, "height": 240, "duration": None, "fps": 30.0}
        result = quality_gate.check_clip_quality("no-duration.mp4")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_CORRUPT_FILE)
        self.assertIn("no duration", result["detail"])

    def test_the_gate_holds_no_length_measurement(self):
        # Pins the deletion. Reintroducing either helper means reopening the
        # four rounds on PR #44, so it should be a deliberate act rather than
        # something that creeps back in.
        for gone in ("sampled_elapsed_seconds", "sampled_frame_count"):
            self.assertFalse(hasattr(quality_gate, gone), gone)

    def test_an_empty_frame_sample_is_corrupt_not_acceptable(self):
        # Codex P2 on #44, and it corrects the symmetry the first pass reached
        # for. ffmpeg can exit 0 emitting no frames on a file ffprobe described
        # happily -- and decode_video.decode_to_pgm_gz raises "ffmpeg produced
        # no frames" for that same input. Waving it through here turns a
        # recordable rejection into a crashed run, which is the outcome this
        # gate exists to prevent.
        quality_gate.probe_metadata = lambda src: {
            "width": 320, "height": 240, "duration": 5.0, "fps": 30.0}
        quality_gate.sample_mean_luminance = lambda src: None
        result = quality_gate.check_clip_quality("no-frames.mp4")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_CORRUPT_FILE)

    def test_a_known_short_duration_is_still_rejected(self):
        # The other half of the change above: relaxing the unknown case must not
        # relax the case the check exists for.
        quality_gate.probe_metadata = lambda src: {
            "width": 320, "height": 240, "duration": 0.2, "fps": 30.0}
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
