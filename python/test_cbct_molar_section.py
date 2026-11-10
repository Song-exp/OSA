import unittest

import numpy as np

import cbct_molar_section as ms


def tooth(apical_sign):
    """치관(반지름 5, 높이 0~8)과 그 아래 두 치근(높이 8~18). apical_sign=+1이면 치근이 위쪽(상악)."""
    g = np.arange(-8, 8.01, .4); L, A, H = np.meshgrid(g, g, np.arange(0, 18, .4), indexing="ij")
    crown = (L**2+A**2 < 25) & (H < 8)
    roots = (((L-3.5)**2+A**2 < 3) | ((L+3.5)**2+A**2 < 3)) & (H >= 8)
    m = crown | roots
    return np.c_[L[m], A[m], apical_sign*H[m]]


class Molar(unittest.TestCase):
    def test_furcation_at_crown_root_junction_both_arches(self):
        for sign, crown_low in ((1, True), (-1, False)):
            f, crown, root = ms.furcation(tooth(sign), crown_low, .4)
            self.assertLess(abs(abs(f[2])-8), 1); self.assertLess(abs(root-10), 1.5)

    def test_single_root_has_no_furcation(self):
        q = tooth(1); q = q[np.abs(q[:, 0]-3.5) < 2.5]
        self.assertIsNone(ms.furcation(q, True, .4))

    def test_buccal_run_stops_at_gap(self):
        s = np.arange(0, 10, .1); hit = (s < 3) | ((s > 3.5) & (s < 5)) | (s > 7)
        self.assertAlmostEqual(ms.buccal_run(hit, s), 4.9, places=1)


if __name__ == "__main__":
    unittest.main()
