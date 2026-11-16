import unittest
import numpy as np
import cbct_landmark_audit as e
import cbct_final_criteria85 as f


class LandmarkAudit(unittest.TestCase):
    def test_basis_orthonormal_and_rotation(self):
        eu,ev=e.basis((1,0),0.);self.assertTrue(np.allclose(eu,[1,0]) and np.allclose(ev,[0,1]))
        eu,ev=e.basis((0,-1),5.);self.assertAlmostEqual(eu@ev,0.);self.assertAlmostEqual(np.linalg.norm(eu),1.)

    def test_support_finds_disk_extreme(self):
        u=np.arange(-17,17.01,.5);v=np.arange(-10,10.01,.5)
        U,V=np.meshgrid(u,v,indexing='ij');disk=(U**2+V**2)<=36   # 반지름 6 mm, 최외곽 u=6, v=0
        front,c,status=e.support_candidates(disk,u,v)
        self.assertEqual(status,'candidate');self.assertAlmostEqual(c[0]['front'],6.);self.assertAlmostEqual(c[0]['sup'],0.)

    def test_support_rejects_flat_edge_at_window_end(self):
        u=np.arange(-17,17.01,.5);v=np.arange(-10,10.01,.5)
        U,V=np.meshgrid(u,v,indexing='ij');ramp=U<=V*.5      # 창 끝에서 최대
        self.assertEqual(e.support_candidates(ramp,u,v)[2],'extremum_at_search_boundary')

    def test_deviation_sign(self):
        run=dict(profile=dict(v=[-1,0,1],u=[2.,2.,2.]),basis=[[1,0],[0,1]])
        self.assertAlmostEqual(e.deviation(run,np.zeros(3),np.array([0,3.,0])),1.)   # 윤곽 바깥 +1
        self.assertIsNone(e.deviation(run,np.zeros(3),np.array([0,3.,5.])))

    def test_mandible_points_follow_seed_lateral(self):
        self.assertEqual(e.center_L(np.array([4.,0,0]),'N'),0.);self.assertEqual(e.center_L(np.array([4.,0,0]),'Me'),4.)

    def test_decide_offset_removes_convention(self):
        stab=dict(representative=None,status='insufficient_support',adequate_support=False)
        run=lambda name:dict(variant=name,profile=dict(v=[-5,0,5],u=[0.,0.,0.]),basis=[[1,0],[0,1]])
        res=[run('contour_base'),run('native_otsu_+0')];orig=np.array([0,2.,0])
        self.assertTrue(e.decide(orig,np.zeros(3),res,stab,'N')['triggered'])
        q=e.decide(orig,np.zeros(3),res,stab,'N',{'mask':2.,'native':2.})
        self.assertFalse(q['triggered']);self.assertTrue(q['independent_support'])


class FinalCriteria(unittest.TestCase):
    def test_fps_keeps_size_and_extremes(self):
        pts=np.c_[np.arange(20.),np.zeros(20),np.zeros(20)];s=f.fps(pts,6)
        self.assertEqual(len(s),6);self.assertIn(0.,s[:,0]);self.assertIn(19.,s[:,0])

    def test_support_from_deviation(self):
        off={'mask':1.,'native':1.}
        self.assertTrue(f.support_from_deviation({'mask':1.5,'native':None},off))
        self.assertFalse(f.support_from_deviation({'mask':3.,'native':3.5},off))

    def test_jung_metrics_do_not_use_LOr(self):
        m=f.metrics('jung');self.assertNotIn('LOr',m['Nperp-A'][2]);self.assertIn('LOr',f.metrics('yao4')['MP-FH'][2])


if __name__=='__main__':unittest.main()
