import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ingest_watch  # noqa: E402


class TestIngestWatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.drop = os.path.join(self.tmp.name, "drop")
        os.makedirs(self.drop)
        self.state = os.path.join(self.tmp.name, "state.json")
        self.queue = os.path.join(self.tmp.name, "queue.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_clip(self, name, size=10):
        path = os.path.join(self.drop, name)
        with open(path, "wb") as fh:
            fh.write(b"x" * size)
        return path

    def test_is_video(self):
        self.assertTrue(ingest_watch.is_video("a.MP4"))
        self.assertTrue(ingest_watch.is_video("clip.mov"))
        self.assertFalse(ingest_watch.is_video("notes.txt"))

    def test_missing_drop_folder_is_idle_not_error(self):
        summary = ingest_watch.run_scan(
            drop_dir=os.path.join(self.tmp.name, "nope"),
            state_path=self.state,
            queue_path=self.queue,
        )
        self.assertEqual(summary["found"], 0)
        self.assertEqual(summary["pending"], 0)

    def test_new_clip_only_enqueued_after_stable_second_scan(self):
        self._write_clip("match1.mp4", size=100)
        # First scan: discovered, not yet enqueued (could still be copying).
        s1 = ingest_watch.run_scan(self.drop, self.state, self.queue)
        self.assertEqual(s1["found"], 1)
        self.assertEqual(s1["newly_enqueued"], 0)
        self.assertEqual(s1["pending"], 0)
        # Second scan, same size: now stable -> enqueued.
        s2 = ingest_watch.run_scan(self.drop, self.state, self.queue)
        self.assertEqual(s2["newly_enqueued"], 1)
        self.assertEqual(s2["pending"], 1)
        # Third scan, unchanged: no duplicate enqueue.
        s3 = ingest_watch.run_scan(self.drop, self.state, self.queue)
        self.assertEqual(s3["newly_enqueued"], 0)
        self.assertEqual(s3["pending"], 1)

    def test_growing_file_not_enqueued_until_size_settles(self):
        self._write_clip("grow.mp4", size=50)
        ingest_watch.run_scan(self.drop, self.state, self.queue)
        # File grew between scans -> still considered unstable.
        self._write_clip("grow.mp4", size=200)
        s = ingest_watch.run_scan(self.drop, self.state, self.queue)
        self.assertEqual(s["newly_enqueued"], 0)
        # Settles now.
        s2 = ingest_watch.run_scan(self.drop, self.state, self.queue)
        self.assertEqual(s2["newly_enqueued"], 1)

    def test_last_footage_at_reported(self):
        self._write_clip("a.mp4")
        s = ingest_watch.run_scan(self.drop, self.state, self.queue)
        self.assertIsNotNone(s["last_footage_at"])
        self.assertTrue(s["last_footage_at"].endswith("Z"))

    def test_diff_new_footage(self):
        found = {"a.mp4": {"path": "a.mp4", "size": 5}, "b.mp4": {"path": "b.mp4", "size": 9}}
        seen = {"a.mp4": {"path": "a.mp4", "size": 5}}
        new = ingest_watch.diff_new_footage(found, seen)
        self.assertEqual([r["path"] for r in new], ["b.mp4"])


if __name__ == "__main__":
    unittest.main()
