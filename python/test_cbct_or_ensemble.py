import unittest
import numpy as np
from cbct_or_ensemble import rim_candidates
from cbct_fh_joint85 import required_points,variants_for_metric,calculate_metric


class OrbitalRimTests(unittest.TestCase):
    def sample(self,edge=True,contrast=True,height=2.):
        L=np.arange(14.,38.01,.5);A=np.arange(-15.,15.01,.5);S=np.arange(-14.,14.01,.5)
        l,a,s=np.meshgrid(L,A,S,indexing='ij')
        roof=height+.12*(l-26)**2
        if edge:roof=np.where(a<1.,roof,-10.)
        mask=s<=roof
        image=np.where(mask,1000.,-100.).astype(np.float32) if contrast else np.zeros(mask.shape,np.float32)
        return mask,image,[L,A,S],np.array([26.,0.,height])

    def detect(self,**kwargs):
        return rim_candidates(*self.sample(**kwargs),.5,1.,100.)

    def test_detects_front_rim_instead_of_posterior_floor(self):
        nodes=self.detect();self.assertTrue(nodes)
        self.assertTrue(all(n['point'][1]>-1.5 for n in nodes))
        self.assertTrue(any(abs(n['point'][0]-26)<1 for n in nodes))

    def test_posterior_floor_without_front_transition_is_not_rim(self):
        self.assertFalse(self.detect(edge=False))

    def test_requires_native_boundary_contrast(self):
        self.assertFalse(self.detect(contrast=False))

    def test_different_height_is_preserved_without_opposite_side(self):
        a=self.detect(height=2.);b=self.detect(height=-2.)
        self.assertTrue(a and b)
        self.assertAlmostEqual(min(x['point'][2] for x in a)-min(x['point'][2] for x in b),4.)

    def test_lor_uncertainty_does_not_block_three_point_anb(self):
        P={k:np.zeros(3) for k in ('RPo','LPo','ROr','LOr')}
        points={k:dict(triggered=False,stability={}) for k in P}
        points['LOr']=dict(triggered=True,stability=dict(candidates_lps=[],adequate_support=False))
        arrays,active,adequate=variants_for_metric(points,P,'ANB','joint')
        self.assertEqual(required_points('ANB'),('RPo','LPo','ROr'))
        self.assertFalse(active);self.assertTrue(adequate);self.assertEqual(len(arrays),3)
        arrays,active,adequate=variants_for_metric(points,P,'MP-FH','joint')
        self.assertEqual(active,['LOr']);self.assertFalse(adequate)

    def test_combination_cap_withholds_instead_of_subsampling(self):
        values,n=calculate_metric({},'MP-FH',[np.zeros((10,3))]*4,None,None,None)
        self.assertIsNone(values);self.assertEqual(n,10000)


if __name__=='__main__':unittest.main()
