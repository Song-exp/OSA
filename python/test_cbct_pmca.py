import unittest

import numpy as np

import cbct_pmca as pm


class Pmca(unittest.TestCase):
    def setUp(self):
        self.Ls = np.arange(-20, 20+1e-9, pm.STEP); self.ant = np.arange(-40, 0+1e-9, pm.STEP); self.sup = np.arange(-40, 20+1e-9, pm.STEP)

    def tube(self, radius):
        L, A, Z = np.meshgrid(self.Ls, self.ant, self.sup, indexing="ij")
        return L**2+(A+20)**2 < radius(Z)**2

    def test_areas_match_circle_and_find_narrowest_level(self):
        tube = self.tube(lambda z: 6+3*np.abs(z+15)/15)
        ax, pe, _ = pm.sections(tube, self.Ls, self.ant, self.sup)
        for area in (ax, pe):
            a, z = pm.pick(area, self.sup, -35, 0)
            self.assertLess(abs(a-np.pi*36)/(np.pi*36), .1); self.assertLess(abs(z+15), 1.5)

    def test_oblique_tube_uses_perpendicular_section(self):
        L, A, Z = np.meshgrid(self.Ls, self.ant, self.sup, indexing="ij")
        tube = L**2+(A+20-.5*(Z+10))**2 < 25                         # 약 26.6° 기울어진 관
        ax, pe, _ = pm.sections(tube, self.Ls, self.ant, self.sup)
        k = np.argmin(np.abs(self.sup+10))
        self.assertGreater(ax[k], pe[k]*1.05)
        self.assertLess(abs(pe[k]-np.pi*25*np.cos(np.arctan(.5)))/(np.pi*25), .12)

    def test_airway_keeps_only_region_behind_s_pns_line(self):
        vol = np.full((len(self.Ls), len(self.ant), len(self.sup)), 100.0)
        vol[self.tube(lambda z: 6+0*z)] = -900
        vol[np.ix_(np.abs(self.Ls) < 3, self.ant > -6, self.sup > -30)] = -900     # 앞쪽 공기
        m = pm.airway(vol, self.Ls, self.ant, self.sup, -20.0, 40.0)
        self.assertTrue(m.any()); self.assertFalse(m[np.ix_(np.abs(self.Ls) < 3, self.ant > -2, self.sup < -10)].any())


if __name__ == "__main__":
    unittest.main()
