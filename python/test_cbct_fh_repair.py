"""Geometry checks for the frozen FH repair rule; no patient images needed."""
import unittest

import numpy as np

from cbct_fh_repair import PAIRS, local, repair
from cbct_eb_detect import midline_frame


class FrozenRepairTests(unittest.TestCase):
    def setUp(self):
        self.points = {k: np.array(v, dtype=float) for k, v in {
            "N": (0, 0, 0), "S": (0, 50, 20), "Ba": (0, 90, -30),
            "ANS": (0, -10, -50), "PNS": (0, 40, -50),
            "RPo": (-55, 80, -25), "LPo": (55, 80, -25),
            "ROr": (-35, 10, -25), "LOr": (35, 10, -25),
        }.items()}
        self.ref = {k: {"median": local(self.points, k)[0].tolist(), "mad": [2., 2., 2.]}
                    for pair in PAIRS for k in pair}

    def test_normal_points_are_unchanged(self):
        result, log = repair(self.points, self.ref)
        self.assertEqual(log, [])
        for key in self.points:
            np.testing.assert_array_equal(result[key], self.points[key])

    def test_single_bad_point_uses_opposite_reflection_without_mutation(self):
        bad = {k: v.copy() for k, v in self.points.items()}
        bad["RPo"] += [0, 60, 45]
        before = {k: v.copy() for k, v in bad.items()}
        result, log = repair(bad, self.ref)
        self.assertEqual([(k, action) for k, _, action in log], [("RPo", "mirrored")])
        origin, frame, _ = midline_frame(bad)
        expected = frame @ (bad["LPo"]-origin)
        expected[0] *= -1
        np.testing.assert_allclose(result["RPo"], origin+expected@frame)
        for key in bad:
            np.testing.assert_array_equal(bad[key], before[key])
            if key != "RPo":
                np.testing.assert_array_equal(result[key], before[key])

    def test_both_bad_sides_flagged_without_reflection(self):
        bad = {k: v.copy() for k, v in self.points.items()}
        for key in ("RPo", "LPo"):
            bad[key] += [0, 60, 45]
        result, log = repair(bad, self.ref)
        self.assertEqual({action for _, _, action in log}, {"both_outlying_not_changed"})
        self.assertEqual(len(log), 2)
        for key in bad:
            np.testing.assert_array_equal(result[key], bad[key])


if __name__ == "__main__":
    unittest.main()
