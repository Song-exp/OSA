import unittest
import numpy as np
import cbct_apog_axis_diag as d
from osa.ceph import measures

P={'N':(0,0,0),'Ba':(0,-90,-30),'RPo':(-55,-80,-25),'LPo':(55,-80,-26),'ROr':(-35,-10,-24),'LOr':(35,-10,-25),
   'A':(0,-2,-55),'B':(0,-6,-95),'Pog':(1,-5,-110),'Me':(0,-12,-122),'RGo':(-48,-85,-95),'LGo':(47,-84,-96)}
P={k:np.asarray(v,float)*np.array([-1,-1,1]) for k,v in P.items()}   # LPS


class AxisDiag(unittest.TestCase):
    def test_zero_pitch_reproduces_ceph(self):
        y=measures(P,'yao4');j=measures(P,'jung')
        oy,Ry=d.frame(P,'yao4');oj,Rj=d.frame(P,'jung')
        self.assertAlmostEqual(d.nperp(P,'A',oy,Ry),y['Nperp_A'],9)
        self.assertAlmostEqual(d.nperp(P,'Pog',oy,Ry),y['Nperp_Pog'],9)
        self.assertAlmostEqual(d.mpfh(P,oy,Ry),y['MPFH_GoMe'],9)
        self.assertAlmostEqual(d.anb(P,oj,Rj),j['ANB'],9)

    def test_anb_pitch_invariant_and_mpfh_shifts(self):
        o,R=d.frame(P,'jung');a0=d.anb(P,o,R);m0=d.mpfh(P,o,R)
        for deg in (-5,2.5,6):
            o,R2=d.frame(P,'jung',deg)
            self.assertAlmostEqual(d.anb(P,o,R2),a0,9)
            self.assertAlmostEqual(abs(d.mpfh(P,o,R2)-m0),abs(deg),9)
        self.assertTrue(np.allclose(R2@R2.T,np.eye(3)))

    def test_sign_convention(self):
        o,R=d.frame(P,'yao4');o,R2=d.frame(P,'yao4',2.)
        self.assertLess(d.nperp(P,'Pog',o,R2),d.nperp(P,'Pog',o,R))

    def test_common_pitch_recovers_delta(self):
        delta,res=d.common_pitch([1.,2.],[.5,1.]);self.assertAlmostEqual(delta,2.);self.assertTrue(np.allclose(res,0))
        delta,res=d.common_pitch([1.,np.nan],[.5,1.]);self.assertAlmostEqual(delta,2.)

    def test_reach_and_classify(self):
        self.assertEqual(d.reach([1,3],2),0.);self.assertEqual(d.reach([1,3],5),2.);self.assertTrue(np.isnan(d.reach([],1)))
        self.assertEqual(d.classify(.5,1,np.nan,False),'small_error')
        self.assertEqual(d.classify(3,1,0.,False),'point_candidate_reaches_reference')
        self.assertEqual(d.classify(3,1,2.,True),'common_pitch_consistent')
        self.assertEqual(d.classify(3,1,2.,False),'unexplained_by_point_or_pitch')
        self.assertEqual(d.classify(np.nan,1,0.,True),'no_reference')


if __name__=='__main__':unittest.main()
