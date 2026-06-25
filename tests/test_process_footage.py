import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decode_video  # noqa: E402
import detect  # noqa: E402
import process_footage  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSlugify(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(process_footage.slugify("drop/Spar Session 2.MOV"), "spar-session-2")
        self.assertEqual(process_footage.slugify("clip.mp4"), "clip")


class TestProcessEndToEnd(unittest.TestCase):
    """Exercise decode -> pipeline -> reports/index with the ffmpeg step faked.

    A committed fixture clip stands in for the decoder's output, so the real
    pipeline runs against real frames without needing ffmpeg in the test env.
    """

    def setUp(self):
        self._orig = decode_video.decode_to_pgm_gz

    def tearDown(self):
        decode_video.decode_to_pgm_gz = self._orig

    def _fake_decode(self, fixture):
        def fake(src, out_path, fps=10.0, width=160):
            shutil.copyfile(os.path.join(HERE, "fixtures", fixture), out_path)
            w, h, frames = detect.load_pgm_frames(out_path)
            return {"out_path": out_path, "frame_count": len(frames),
                    "width": w, "height": h, "fps": fps}
        return fake

    def _run(self, domain, fixture, label):
        decode_video.decode_to_pgm_gz = self._fake_decode(fixture)
        with tempfile.TemporaryDirectory() as reports:
            entry = process_footage.process(
                os.path.join(reports, "src_placeholder"),
                domain=domain, reports_dir=reports, source_label=label,
            )
            clip_dir = os.path.join(reports, entry["id"])
            self.assertTrue(os.path.isfile(os.path.join(clip_dir, "coaching", "report.json")))
            self.assertTrue(os.path.isfile(os.path.join(clip_dir, "coaching", "summary.txt")))
            manifest_path = os.path.join(clip_dir, "highlights", "manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))
            # Decoded frames are intermediate and must not be left behind.
            self.assertFalse(os.path.exists(os.path.join(clip_dir, f"{entry['id']}.pgm.gz")))

            # The catalog entry carries the per-segment clip list the PWA renders.
            self.assertIn("title", entry)
            self.assertEqual(len(entry["clips"]), entry["segment_count"])
            for c in entry["clips"]:
                self.assertIn("tags", c)
                self.assertIn("video", c)  # path when rendered, else None

            # The published manifest drops the internal ffmpeg argv and records
            # render status per clip.
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            for c in manifest["clips"]:
                self.assertNotIn("ffmpeg_cmd", c)
                self.assertIn("rendered", c)

            # The catalog the PWA reads has this clip.
            with open(os.path.join(reports, "index.json")) as fh:
                index = json.load(fh)
            ids = [c["id"] for c in index["clips"]]
            self.assertIn(entry["id"], ids)
            return entry

    def test_martial_arts(self):
        entry = self._run("martial_arts", "martialarts_clip.pgm.gz", "spar_session.mp4")
        self.assertEqual(entry["domain"], "martial_arts")
        self.assertGreaterEqual(entry["segment_count"], 1)

    def test_volleyball(self):
        entry = self._run("volleyball", "reference_clip.pgm.gz", "game1.mp4")
        self.assertEqual(entry["domain"], "volleyball")
        self.assertGreaterEqual(entry["segment_count"], 1)

    def test_generic_label_gets_timestamp_id(self):
        # A Drive "...//view" tail slugs to a useless "view" -> use a timestamp.
        decode_video.decode_to_pgm_gz = self._fake_decode("martialarts_clip.pgm.gz")
        with tempfile.TemporaryDirectory() as reports:
            entry = process_footage.process(
                os.path.join(reports, "x"), domain="martial_arts",
                reports_dir=reports, source_label="view")
            self.assertTrue(entry["id"].startswith("clip-"), entry["id"])

    def test_name_sets_title_and_id(self):
        decode_video.decode_to_pgm_gz = self._fake_decode("martialarts_clip.pgm.gz")
        with tempfile.TemporaryDirectory() as reports:
            entry = process_footage.process(
                os.path.join(reports, "x"), domain="martial_arts",
                reports_dir=reports, source_label="view", name="Sparring Tue")
            self.assertEqual(entry["title"], "Sparring Tue")
            self.assertEqual(entry["id"], "sparring-tue")


if __name__ == "__main__":
    unittest.main()
