import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dispatch_queue  # noqa: E402


class TestTriggerWorkflow(unittest.TestCase):
    def test_returns_true_on_204(self):
        class FakeResp:
            status = 204
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("urllib.request.urlopen", return_value=FakeResp()):
            ok = dispatch_queue.trigger_workflow("o", "r", "wf.yml", "main", {}, "tok")
        self.assertTrue(ok)

    def test_returns_false_on_http_error(self):
        import urllib.error
        err = urllib.error.HTTPError(None, 422, "Unprocessable", {}, None)
        err.read = lambda: b"bad input"
        with patch("urllib.request.urlopen", side_effect=err):
            ok = dispatch_queue.trigger_workflow("o", "r", "wf.yml", "main", {}, "tok")
        self.assertFalse(ok)


class TestMain(unittest.TestCase):
    def _queue(self, tmpdir, pending):
        path = os.path.join(tmpdir, "ingest_queue.json")
        with open(path, "w") as fh:
            json.dump({"pending": pending, "updated_at": "2026-06-26T00:00:00Z"}, fh)
        return path

    def _env(self, tmpdir, queue_path):
        return {
            "GITHUB_TOKEN": "test-token",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_REF_NAME": "main",
            "COACHVISION_DOMAIN": "martial_arts",
            "COACHVISION_INGEST_QUEUE": queue_path,
        }

    def test_dispatches_pending_and_clears_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = self._queue(tmpdir, [
                {"path": "spar.mp4", "size": 100, "enqueued_at": "2026-06-26T00:00:00Z"}
            ])
            env = self._env(tmpdir, queue_path)
            with patch.dict(os.environ, env, clear=False):
                with patch.object(dispatch_queue, "trigger_workflow", return_value=True):
                    dispatch_queue.main()

            with open(queue_path) as fh:
                queue = json.load(fh)
            self.assertEqual(queue["pending"], [])

    def test_failed_dispatch_keeps_item_in_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = self._queue(tmpdir, [
                {"path": "spar.mp4", "size": 100, "enqueued_at": "2026-06-26T00:00:00Z"}
            ])
            env = self._env(tmpdir, queue_path)
            with patch.dict(os.environ, env, clear=False):
                with patch.object(dispatch_queue, "trigger_workflow", return_value=False):
                    dispatch_queue.main()

            with open(queue_path) as fh:
                queue = json.load(fh)
            self.assertEqual(len(queue["pending"]), 1)

    def test_empty_queue_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue_path = self._queue(tmpdir, [])
            env = self._env(tmpdir, queue_path)
            called = []
            with patch.dict(os.environ, env, clear=False):
                with patch.object(dispatch_queue, "trigger_workflow",
                                  side_effect=lambda *a, **kw: called.append(1) or True):
                    dispatch_queue.main()
            self.assertEqual(called, [])

    def test_no_token_is_noop(self):
        env = {"GITHUB_TOKEN": "", "GITHUB_REPOSITORY": "o/r"}
        called = []
        with patch.dict(os.environ, env, clear=False):
            with patch.object(dispatch_queue, "trigger_workflow",
                              side_effect=lambda *a, **kw: called.append(1) or True):
                dispatch_queue.main()
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
