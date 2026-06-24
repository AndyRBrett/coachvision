import gzip
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detect  # noqa: E402


def _pgm_frame(width, height, bright_pixels):
    """Build one P5/PGM frame with the given (x, y) pixels set to 255."""
    buf = bytearray(width * height)
    for x, y in bright_pixels:
        buf[y * width + x] = 255
    return b"P5\n%d %d\n255\n" % (width, height) + bytes(buf)


def _write_clip(frames, gzipped=True):
    data = b"".join(frames)
    suffix = ".pgm.gz" if gzipped else ".pgm"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(gzip.compress(data) if gzipped else data)
    return path


class TestDetectBall(unittest.TestCase):
    def test_centroid_of_blob(self):
        # A 3x3 block centered at (5, 4) -> centroid exactly (5, 4).
        block = [(x, y) for x in range(4, 7) for y in range(3, 6)]
        raster = _pgm_frame(10, 8, block)
        _, _, frames = self._load_single(raster)
        ball = detect.detect_blob(frames[0], 10, 8)
        self.assertEqual(ball, [5.0, 4.0])

    def test_empty_frame_is_none(self):
        raster = _pgm_frame(10, 8, [])
        _, _, frames = self._load_single(raster)
        self.assertIsNone(detect.detect_blob(frames[0], 10, 8))

    def test_below_min_pixels_is_none(self):
        raster = _pgm_frame(10, 8, [(2, 2)])  # 1 bright pixel < default min 3
        _, _, frames = self._load_single(raster)
        self.assertIsNone(detect.detect_blob(frames[0], 10, 8))

    def _load_single(self, raster):
        path = _write_clip([raster])
        try:
            return detect.load_pgm_frames(path)
        finally:
            os.remove(path)


class TestLoadFrames(unittest.TestCase):
    def test_multi_frame_roundtrip(self):
        frames = [_pgm_frame(6, 4, [(1, 1)]), _pgm_frame(6, 4, [(2, 2)])]
        path = _write_clip(frames)
        try:
            w, h, loaded = detect.load_pgm_frames(path)
        finally:
            os.remove(path)
        self.assertEqual((w, h), (6, 4))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(len(loaded[0]), 24)

    def test_reads_uncompressed(self):
        path = _write_clip([_pgm_frame(5, 5, [(0, 0)])], gzipped=False)
        try:
            w, h, loaded = detect.load_pgm_frames(path)
        finally:
            os.remove(path)
        self.assertEqual((w, h), (5, 5))
        self.assertEqual(len(loaded), 1)

    def test_inconsistent_dims_raise(self):
        frames = [_pgm_frame(6, 4, []), _pgm_frame(8, 4, [])]
        path = _write_clip(frames)
        try:
            with self.assertRaises(ValueError):
                detect.load_pgm_frames(path)
        finally:
            os.remove(path)


class TestRunDetection(unittest.TestCase):
    def test_reference_clip_tracks_and_segments(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clip = os.path.join(here, "fixtures", "reference_clip.pgm.gz")
        events = os.path.join(here, "fixtures", "reference_clip.events.json")
        ev, fps, _ = detect._load_events_sidecar(events)
        tracking = detect.run_detection(clip, fps=fps, events=ev)
        self.assertEqual(tracking["frame_count"], 104)
        self.assertGreater(tracking["detected_frames"], 0)
        self.assertEqual((tracking["width"], tracking["height"]), (80, 45))
        self.assertEqual(tracking["events"], ev)


if __name__ == "__main__":
    unittest.main()
