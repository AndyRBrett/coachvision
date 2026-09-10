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
    PATCHED = ("probe_metadata", "sample_mean_luminance", "sampled_elapsed_seconds",
               "sampled_frame_count")

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

    def test_unknown_duration_is_measured_not_rejected(self):
        # A container that carries no duration is not a short clip, and this
        # gate must not discard footage on missing metadata. Fragmented MP4s
        # from a phone are the case -- and mobile quick-upload (#34) is one of
        # the ingest paths this gate was asked for. A truncated file never gets
        # this far: ffprobe fails on it and probe_metadata answers corrupt_file.
        quality_gate.probe_metadata = lambda src: {
            "width": 320, "height": 240, "duration": None, "fps": 30.0}
        quality_gate.sampled_elapsed_seconds = lambda src, seconds=2.0: 4.9
        quality_gate.sample_mean_luminance = lambda src: 120.0
        result = quality_gate.check_clip_quality("no-duration.mp4")
        self.assertTrue(result["ok"], result)

    def test_unknown_duration_still_enforces_the_minimum(self):
        # Codex P2 on #44. Admitting unknown-duration inputs must not leave the
        # too_short gate unenforced for exactly those inputs -- a raw elementary
        # stream can report dimensions and rate with no duration and still be a
        # fifth of a second long.
        quality_gate.probe_metadata = lambda src: {
            "width": 320, "height": 240, "duration": None, "fps": 30.0}
        quality_gate.sampled_elapsed_seconds = lambda src, seconds=2.0: 0.2
        quality_gate.sample_mean_luminance = lambda src: 120.0
        result = quality_gate.check_clip_quality("short-no-duration.mp4")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_TOO_SHORT)

    def test_a_variable_frame_rate_clip_is_not_rejected_on_its_nominal_rate(self):
        # Codex P2 on 45eaf97, and it was a regression in the same spirit as the
        # bug it followed. r_frame_rate is the stream's NOMINAL base rate, so a
        # real one-second VFR phone clip can carry far fewer frames than it
        # implies -- and counting frames against it rejected exactly the footage
        # this branch exists to admit. Length is measured from TIMESTAMPS now,
        # so a clip whose packets span a second passes whatever its frame count.
        quality_gate.probe_metadata = lambda src: {
            "width": 320, "height": 240, "duration": None, "fps": 30.0}
        quality_gate.sampled_elapsed_seconds = lambda src, seconds=2.0: 1.02
        quality_gate.sample_mean_luminance = lambda src: 120.0

        # And the frame count must not even be consulted here. That ordering is
        # the whole design: timestamps are exact and VFR-safe, frame counting is
        # only sound for the no-timestamp streams it falls back to. Reversing it
        # would reinstate the false rejection this test is named for.
        def _must_not_run(src, seconds=1.0):
            raise AssertionError("frame count consulted despite readable timestamps")
        quality_gate.sampled_frame_count = _must_not_run

        result = quality_gate.check_clip_quality("vfr-phone-clip.mp4")
        self.assertTrue(result["ok"], result)

    def test_no_timestamps_falls_back_to_counting_frames(self):
        # Codex P2 on 11c0332. A raw H.264 elementary stream derives dimensions
        # and rate from its SPS and carries no PTS, so the timestamp-only
        # measurement returned None and waived the check -- reopening the
        # original hole for exactly the input class that motivated it. 6 frames
        # at 30fps is 0.2s.
        quality_gate.probe_metadata = lambda src: {
            "width": 320, "height": 240, "duration": None, "fps": 30.0}
        quality_gate.sampled_elapsed_seconds = lambda src, seconds=2.0: None
        quality_gate.sampled_frame_count = lambda src, seconds=1.0: 6
        quality_gate.sample_mean_luminance = lambda src: 120.0
        result = quality_gate.check_clip_quality("raw-stream.h264")
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], quality_gate.REASON_TOO_SHORT)

    def test_no_timestamps_but_long_enough_passes(self):
        # The other half: the fallback must admit a real clip, not just reject.
        quality_gate.probe_metadata = lambda src: {
            "width": 320, "height": 240, "duration": None, "fps": 30.0}
        quality_gate.sampled_elapsed_seconds = lambda src, seconds=2.0: None
        quality_gate.sampled_frame_count = lambda src, seconds=1.0: 30
        quality_gate.sample_mean_luminance = lambda src: 120.0
        self.assertTrue(quality_gate.check_clip_quality("raw-stream.h264")["ok"])

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
