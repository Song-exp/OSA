import unittest
from unittest.mock import patch
import numpy as np
import cbct_apog_ensemble as e
import cbct_apog85 as driver


class ContourTests(unittest.TestCase):
    def test_A_concavity_and_Pog_convexity(self):
        s=np.arange(-5.,5.01,.25)
        for key,front in [('A',5.+.2*s*s),('Pog',10.-.2*s*s)]:
            nodes,status=e.extrema_candidates(front,s,key,step=.25)
            self.assertEqual(status,'candidate')
            self.assertTrue(any(abs(n['sup'])<.3 for n in nodes))

    def test_monotonic_boundary_is_not_a_landmark(self):
        s=np.arange(-5.,5.01,.5)
        for key in ('A','Pog'):
            self.assertFalse(e.extrema_candidates(s,s,key)[0])

    def test_missing_contour_is_withheld(self):
        s=np.arange(-5.,5.01,.5);front=.2*s*s
        front[:7]=np.nan
        self.assertEqual(e.extrema_candidates(front,s,'A')[1],'insufficient_contour_coverage')

    def decision(self,mask_front,native_front):
        results=[dict(variant=name,profile=dict(sup=[-2.,0.,2.],front=[x]*3))
                 for name,x in [('contour_base',mask_front),('native_otsu_+0',native_front)]]
        return e.decide(np.zeros(3),np.zeros(3),results,
                        dict(representative=None,status='insufficient_support',adequate_support=False))

    def test_mask_mismatch_preserves_native_supported_point(self):
        r=self.decision(3.,.2)
        self.assertFalse(r['triggered'])
        self.assertEqual(r['route'],'mask_mismatch_original_retained')

    def test_both_sources_off_trigger_withhold_without_support(self):
        r=self.decision(3.,2.)
        self.assertTrue(r['triggered']);self.assertEqual(r['route'],'automatic_withhold')

    def test_reference_patient_cannot_supply_own_seed(self):
        def transfer(ref,target,anchor):
            return np.asarray(anchor),dict(local_transfer_residual_mm=.1,local_transfer_condition=2.)
        annotations={5:{'A':np.array([99.,99.,99.])},11:{'A':np.array([1.,2.,3.])},28:{'A':np.array([3.,4.,5.])}}
        with patch.object(driver,'physical_correspondences',return_value=np.zeros((20,3))),patch.object(driver,'local_transfer',side_effect=transfer):
            seed,info=driver.reference_seed('P_5','A',None,annotations)
        np.testing.assert_allclose(seed,[2.,3.,4.])
        self.assertEqual(info['reference_ids'],['P_11','P_28'])

    def test_measurement_dependencies(self):
        P={k:np.zeros(3) for k in ['A','Pog','RPo','LPo','ROr','LOr']}
        ap={k:dict(triggered=True,stability=dict(candidates_lps=[[1.,2.,3.]],adequate_support=True)) for k in ('A','Pog')}
        fh={k:dict(triggered=False) for k in ('RPo','LPo','ROr','LOr')}
        for metric,mode,expected in [('ANB','Pog_only',[]),('Nperp-A','Pog_only',[]),('Nperp-Pog','A_only',[]),('MP-FH','APog_only',[]),('ANB','A_only',['A'])]:
            self.assertEqual(driver.configurations(P,ap,fh,metric,mode)[2],expected)

    def test_cap_withholds_without_subsampling(self):
        values,n=driver.calculate({},'ANB',[],[np.zeros((10,3))]*4,[],None,None,None)
        self.assertIsNone(values);self.assertEqual(n,10000)


if __name__=='__main__':unittest.main()
