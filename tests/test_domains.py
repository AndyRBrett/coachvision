import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import domains  # noqa: E402


class TestGetDomain(unittest.TestCase):
    def test_default_is_volleyball(self):
        # With no argument and no env override, the default preserves prior
        # behaviour (and the overseer's existing contract).
        os.environ.pop(domains.ENV_VAR, None)
        self.assertEqual(domains.get_domain().key, "volleyball")

    def test_resolves_canonical_and_aliases(self):
        self.assertEqual(domains.get_domain("martial_arts").key, "martial_arts")
        for alias in ("mma", "Martial-Arts", "martial arts", "FIGHTING"):
            self.assertEqual(domains.get_domain(alias).key, "martial_arts")
        self.assertEqual(domains.get_domain("vb").key, "volleyball")

    def test_env_var_selects_domain(self):
        os.environ[domains.ENV_VAR] = "martial_arts"
        try:
            self.assertEqual(domains.get_domain().key, "martial_arts")
        finally:
            os.environ.pop(domains.ENV_VAR, None)

    def test_passthrough_domain_instance(self):
        d = domains.MARTIAL_ARTS
        self.assertIs(domains.get_domain(d), d)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            domains.get_domain("badminton")


class TestDomainShape(unittest.TestCase):
    def test_detectors_are_known(self):
        for d in domains.REGISTRY.values():
            self.assertIn(d.detector, ("brightest_blob", "motion_energy"))
            self.assertTrue(d.tags, f"{d.key} has no tags")
            self.assertGreater(d.max_gap_s, 0)

    def test_domains_have_distinct_vocab(self):
        # The whole point of switching domains: the action vocabulary differs.
        self.assertNotEqual(set(domains.VOLLEYBALL.tags), set(domains.MARTIAL_ARTS.tags))
        self.assertEqual(domains.MARTIAL_ARTS.action_plural, "strikes")
        self.assertEqual(domains.MARTIAL_ARTS.segment_plural, "exchanges")


if __name__ == "__main__":
    unittest.main()
