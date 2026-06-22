import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cosmos_tagger  # noqa: E402


class TestParseResponse(unittest.TestCase):
    def _body(self, content):
        return {"choices": [{"message": {"content": content}}]}

    def test_plain_array(self):
        tags = cosmos_tagger.parse_response(self._body('["attack", "block"]'))
        self.assertEqual(tags, ["attack", "block"])

    def test_array_wrapped_in_prose(self):
        tags = cosmos_tagger.parse_response(self._body('Sure! ["dig", "Set"] are visible.'))
        self.assertEqual(tags, ["dig", "set"])

    def test_no_array_returns_none(self):
        self.assertIsNone(cosmos_tagger.parse_response(self._body("no events")))

    def test_malformed_body_returns_none(self):
        self.assertIsNone(cosmos_tagger.parse_response({}))


class TestMergeTags(unittest.TestCase):
    def test_union_in_canonical_order(self):
        merged = cosmos_tagger.merge_tags(["attack"], ["serve", "dig"])
        self.assertEqual(merged, ["serve", "attack", "dig"])

    def test_unknown_model_tags_dropped(self):
        merged = cosmos_tagger.merge_tags(["attack"], ["banana"])
        self.assertEqual(merged, ["attack"])

    def test_dedupe(self):
        merged = cosmos_tagger.merge_tags(["attack"], ["attack"])
        self.assertEqual(merged, ["attack"])


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("VOLLEYBALL_COSMOS_NIM_URL", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["VOLLEYBALL_COSMOS_NIM_URL"] = self._saved
        else:
            os.environ.pop("VOLLEYBALL_COSMOS_NIM_URL", None)

    def test_not_configured(self):
        self.assertFalse(cosmos_tagger.is_configured())
        with self.assertRaises(RuntimeError):
            cosmos_tagger.make_enricher()

    def test_configured(self):
        os.environ["VOLLEYBALL_COSMOS_NIM_URL"] = "http://localhost:8000/v1/chat/completions"
        self.assertTrue(cosmos_tagger.is_configured())
        enricher = cosmos_tagger.make_enricher()
        self.assertTrue(callable(enricher))

    def test_query_without_url_returns_none(self):
        self.assertIsNone(cosmos_tagger.query_cosmos(1.0, 2.0, [], url=None))


class TestEnricherGraceful(unittest.TestCase):
    def test_enricher_returns_base_on_failure(self):
        os.environ["VOLLEYBALL_COSMOS_NIM_URL"] = "http://127.0.0.1:9/v1/chat/completions"
        try:
            enrich = cosmos_tagger.make_enricher()
            # Endpoint is unreachable -> falls back to base tags, never raises.
            self.assertEqual(enrich("x.mp4", 1.0, 2.0, ["serve"]), ["serve"])
        finally:
            os.environ.pop("VOLLEYBALL_COSMOS_NIM_URL", None)


if __name__ == "__main__":
    unittest.main()
