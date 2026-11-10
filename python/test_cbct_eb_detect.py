import unittest

import numpy as np

import cbct_eb_detect as eb


class EbDetect(unittest.TestCase):
    def test_midline_frame_ignores_po_and_orients_axes(self):
        P = {"N": (0, -90, 60), "S": (0, -20, 40), "Ba": (0, 20, 0), "ANS": (0, -80, 0), "PNS": (0, -30, 0)}
        P = {k: np.array(v, float) for k, v in P.items()}
        o, R, resid = eb.midline_frame(P)
        np.testing.assert_allclose(o, P["PNS"]); np.testing.assert_allclose(R, [[1, 0, 0], [0, -1, 0], [0, 0, 1]], atol=1e-9)
        self.assertLess(resid, 1e-9)

    def test_vallecula_is_lowest_row_of_front_branch(self):
        ant, sup = np.arange(-30, 10, eb.STEP), np.arange(-60, -20, eb.STEP)
        air = np.zeros((len(ant), len(sup)), bool)
        air[(ant > -25) & (ant < -18), :] = True                    # 인두
        air[np.ix_((ant > -15) & (ant < -10), (sup > -48) & (sup < -30))] = True   # 후두개곡
        air[np.ix_((ant > -25) & (ant < -10), (sup > -32) & (sup < -30))] = True   # 후두개 끝 위에서 연결
        c, z, n = eb.vallecula(air, ant, sup, (20.0, -5.0, -50.0))
        self.assertAlmostEqual(z, -48, delta=eb.STEP*1.5); self.assertLess(abs(c+12.5), 1.0); self.assertGreater(n, 3)

    def test_hyoid_touching_window_edge_is_rejected(self):
        ant, sup = np.arange(*eb.ANT, eb.STEP), np.arange(*eb.SUP, eb.STEP)
        mx = np.full((len(ant), len(sup)), -500.0)
        mx[np.ix_((ant > -12) & (ant < -2), (sup > -75) & (sup < -70))] = 900
        self.assertIsNotNone(eb.hyoid(mx, ant, sup))
        mx[np.ix_((ant > -40) & (ant < -2), (sup > -75) & (sup < -70))] = 900
        self.assertIsNone(eb.hyoid(mx, ant, sup))


if __name__ == "__main__":
    unittest.main()
