import unittest
import numpy as np
from cbct_po_ensemble import VARIANTS,stability,metric_decision,crop_variant


class CandidateStabilityTests(unittest.TestCase):
    def results(self,mask_point=(1.,2.,3.),native_point=(1.,2.,3.),sparse=False,competitor=False):
        out=[]
        for i,v in enumerate(VARIANTS):
            point=np.array(mask_point if v['family']=='AMASSS' else native_point)
            exists=not sparse or i<2
            eligible=[dict(point=point,score=100)] if exists else []
            if competitor and exists:eligible.append(dict(point=point+[0,6,0],score=40))
            out.append(dict(variant=v['name'],family=v['family'],winner=point if exists else None,eligible=eligible))
        return out

    def test_agreement_is_operational_stability_not_anatomical_confirmation(self):
        r=stability(self.results())
        self.assertEqual(r['status'],'stable_single_candidate')
        self.assertNotIn('anatomically_confirmed',r)

    def test_consistent_but_different_families_do_not_confirm_one_point(self):
        r=stability(self.results(native_point=(1.,8.,3.)))
        self.assertEqual(r['status'],'candidate_region')
        self.assertAlmostEqual(r['family_distance_mm'],6.)

    def test_competing_distant_region_prevents_single_point(self):
        self.assertEqual(stability(self.results(competitor=True))['status'],'candidate_region')

    def test_two_successes_among_sixteen_are_insufficient(self):
        r=stability(self.results(sparse=True))
        self.assertEqual(r['status'],'insufficient_support')
        self.assertFalse(r['adequate_support'])

    def test_metric_can_be_stable_even_if_point_is_not_unique(self):
        r=metric_decision([1.,1.5,1.8],True,'ANB')
        self.assertEqual(r['status'],'candidate_stable_metric')
        self.assertAlmostEqual(r['value'],1.5)

    def test_wide_metric_range_returns_no_single_number(self):
        r=metric_decision([-2.,3.],True,'Nperp-A')
        self.assertEqual(r['status'],'candidate_interval_only')
        self.assertIsNone(r['value'])
        self.assertEqual((r['low'],r['high']),(-2.,3.))

    def test_insufficient_support_cannot_be_rescued_by_small_metric_range(self):
        self.assertEqual(metric_decision([1.,1.01],False,'ANB')['status'],'automatic_withhold')

    def test_roi_shift_and_coarse_grid_keep_physical_coordinates(self):
        axis=np.arange(-20.,20.01,.5)
        l,a,s=np.meshgrid(axis,axis,axis,indexing='ij')
        image=(l+2*a+3*s).astype(np.float32);mask=np.ones(image.shape,bool)
        variant=dict(family='AMASSS',step=.75,offset=[1.5,0.,-1.5])
        _,sampled,axes,_=crop_variant(mask,image,[axis,axis,axis],np.zeros(3),variant)
        ll,aa,ss=np.meshgrid(*axes,indexing='ij')
        np.testing.assert_allclose(sampled,ll+2*aa+3*ss,atol=1e-5)

    def test_no_candidate_combinations_do_not_create_a_measurement(self):
        from cbct_po_stability85 import metric_values
        result,n=metric_values({},[],[[0,0,0]],None,None,None)
        self.assertEqual(n,0)
        self.assertTrue(all(not v for v in result.values()))

    def test_excessive_combinations_withhold_instead_of_sampling(self):
        from cbct_po_stability85 import metric_values
        result,n=metric_values({},np.zeros((70,3)),np.zeros((70,3)),None,None,None)
        self.assertIsNone(result);self.assertEqual(n,4900)


if __name__=='__main__':unittest.main()
