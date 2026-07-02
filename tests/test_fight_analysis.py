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

    def test_engagement_gates_subject_for_slot_records(self):
        # Slot records: subject exists only while the pair is within range, so
        # fighters resetting between exchanges read as segmentation gaps even
        # though both stay in frame.
        near = {"t": 0.0, "fighters": [], "slot_boxes": [[0, 0, 100, 200], [120, 0, 220, 200]]}
        far = {"t": 0.1, "fighters": [], "slot_boxes": [[0, 0, 100, 200], [700, 0, 800, 200]]}
        one = {"t": 0.2, "fighters": [], "slot_boxes": [[0, 0, 100, 200], None]}
        tracking = fa.build_tracking([near, far, one], 1000, 200, 10.0)
        self.assertEqual(tracking["frames"][0]["subject"], [110.0, 100.0])
        self.assertIsNone(tracking["frames"][1]["subject"])
        self.assertIsNone(tracking["frames"][2]["subject"])
        self.assertEqual(tracking["detected_frames"], 1)
        self.assertEqual(tracking["fighter_frames"], 3)


class TestPairEngaged(unittest.TestCase):
    def test_close_pair_engaged(self):
        self.assertTrue(fa.pair_engaged([[0, 0, 100, 200], [150, 0, 250, 200]]))

    def test_far_pair_not_engaged(self):
        self.assertFalse(fa.pair_engaged([[0, 0, 100, 200], [700, 0, 800, 200]]))

    def test_missing_slot_not_engaged(self):
        self.assertFalse(fa.pair_engaged([[0, 0, 100, 200], None]))


class TestTrackFighterSlots(unittest.TestCase):
    def _frames(self, per_frame_persons, dt=0.1):
        return [{"t": round(i * dt, 3), "persons": persons}
                for i, persons in enumerate(per_frame_persons)]

    def test_slots_keep_identity_across_id_churn(self):
        # Same two people drift right while the detector re-ids one of them;
        # proximity keeps each in its original slot.
        frames = []
        for i in range(4):
            a = _fighter([300 + 5 * i, 300, 470 + 5 * i, 800], fid=1)
            b = _fighter([520 + 5 * i, 300, 690 + 5 * i, 800], fid=(2 if i < 2 else 99))
            frames.append([a, b])
        records = fa.track_fighter_slots(self._frames(frames), 1000, 1000)
        for rec in records:
            self.assertEqual(sorted(f["id"] for f in rec["fighters"]), [0, 1])
        # slot 1 stays on person b even after its detector id churned to 99
        xs = [f["box"][0] for rec in records for f in rec["fighters"] if f["id"] == 1]
        self.assertEqual(xs, [520, 525, 530, 535])

    def test_lost_slot_coasts_then_empties(self):
        a = _fighter([300, 300, 470, 800], fid=1)
        b = _fighter([520, 300, 690, 800], fid=2)
        frames = [[a, b]] + [[a]] * 14   # b disappears for 1.4s at dt=0.1
        records = fa.track_fighter_slots(self._frames(frames), 1000, 1000)
        # coasts (box kept) within TRACK_MEMORY_S...
        self.assertIsNotNone(records[5]["slot_boxes"][1])
        # ...then empties once the memory expires
        self.assertIsNone(records[-1]["slot_boxes"][1])

    def test_static_person_excluded(self):
        # A large "spectator" who never moves must not fill the second slot,
        # even though they pass the size gates.
        mover = _fighter([300, 300, 470, 800], fid=1)
        statue = _fighter([600, 300, 770, 800], fid=2)
        frames = []
        for i in range(40):  # 4s at dt=0.1 -- beyond STATIC_WINDOW_S
            m = _fighter([300 + 8 * i, 300, 470 + 8 * i, 800], fid=1)
            frames.append([m, statue])
        records = fa.track_fighter_slots(self._frames(frames), 1000, 1000)
        # by the end of the clip the statue has been dropped from candidacy
        self.assertEqual([f["id"] for f in records[-1]["fighters"]], [0])
        self.assertIsNone(records[-1]["slot_boxes"][1])
        self.assertEqual(records[-1]["fighters"][0]["box"][0], 300 + 8 * 39)


if __name__ == "__main__":
    unittest.main()
