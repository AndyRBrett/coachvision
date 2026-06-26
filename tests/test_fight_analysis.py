import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fight_analysis as fa  # noqa: E402


def _fighter(box, fid=1, kp=None):
    """A fighter with 17 zero-confidence keypoints, overriding kp={idx:(x,y,c)}."""
    kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
    for idx, val in (kp or {}).items():
        kpts[idx] = list(val)
    return {"id": fid, "box": list(box), "kpts": kpts}


class TestSelectFighters(unittest.TestCase):
    def test_drops_background_person(self):
        big = _fighter([400, 400, 600, 800], fid=1)         # large, central
        bystander = _fighter([10, 10, 40, 70], fid=2)       # tiny, corner
        picked = fa.select_fighters([big, bystander], 1000, 1000)
        self.assertEqual([p["id"] for p in picked], [1])

    def test_keeps_two_fighters(self):
        a = _fighter([350, 300, 520, 760], fid=1)
        b = _fighter([520, 300, 690, 760], fid=2)
        picked = fa.select_fighters([a, b], 1000, 1000)
        self.assertEqual(sorted(p["id"] for p in picked), [1, 2])

    def test_caps_at_two(self):
        people = [_fighter([400 + i, 400, 480 + i, 760], fid=i) for i in range(4)]
        self.assertEqual(len(fa.select_fighters(people, 1000, 1000)), 2)

    def test_excludes_central_referee(self):
        # Two fighters clashing (adjacent boxes) plus a same-size referee standing
        # apart. The engaged pair is the two fighters; the ref is dropped.
        a = _fighter([380, 350, 520, 850], fid=1)
        b = _fighter([500, 350, 640, 850], fid=2)
        ref = _fighter([790, 350, 910, 850], fid=3)
        picked = fa.select_fighters([a, b, ref], 1000, 1000)
        self.assertEqual(sorted(p["id"] for p in picked), [1, 2])

    def test_drops_short_background_row(self):
        # A row of small, short background people behind two big fighters.
        a = _fighter([300, 200, 470, 880], fid=1)
        b = _fighter([520, 200, 690, 880], fid=2)
        crowd = [_fighter([50 + i * 60, 100, 90 + i * 60, 180], fid=10 + i) for i in range(6)]
        picked = fa.select_fighters([a, b] + crowd, 1000, 1000)
        self.assertEqual(sorted(p["id"] for p in picked), [1, 2])

    def test_empty(self):
        self.assertEqual(fa.select_fighters([], 1000, 1000), [])


class TestDetectStrikes(unittest.TestCase):
    def _records(self, wrist_xs, idx=fa.L_WRIST, dt=0.1):
        # box height 100 so speed = dist/100/dt; a 40px wrist jump at dt=0.1 -> 4.0
        recs = []
        for i, x in enumerate(wrist_xs):
            recs.append({"t": round(i * dt, 3),
                         "fighters": [_fighter([0, 0, 100, 100], fid=1, kp={idx: (x, 50, 0.9)})]})
        return recs

    def test_fast_wrist_is_hand_strike(self):
        events = fa.detect_strikes(self._records([50, 90]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "hand_strike")

    def test_slow_wrist_no_strike(self):
        self.assertEqual(fa.detect_strikes(self._records([50, 55, 60, 65])), [])

    def test_fast_ankle_is_leg_strike(self):
        events = fa.detect_strikes(self._records([50, 95], idx=fa.L_ANKLE))
        self.assertTrue(events and events[0]["type"] == "leg_strike")

    def test_refractory_dedupes(self):
        # three fast frames in a row collapse to one strike within the window
        events = fa.detect_strikes(self._records([50, 90, 130, 170]))
        self.assertEqual(len(events), 1)


class TestBuildTracking(unittest.TestCase):
    def test_subject_centroid_and_gaps(self):
        recs = [
            {"t": 0.0, "fighters": [_fighter([0, 0, 100, 100], fid=1)]},
            {"t": 0.1, "fighters": []},
        ]
        tracking = fa.build_tracking(recs, 200, 200, 10.0)
        self.assertEqual(tracking["frames"][0]["subject"], [50.0, 50.0])
        self.assertIsNone(tracking["frames"][1]["subject"])
        self.assertEqual(tracking["detected_frames"], 1)
        self.assertEqual(tracking["domain"], "martial_arts")


if __name__ == "__main__":
    unittest.main()
