import unittest
import numpy as np
import cbct_apog_structure as e


class StructureTests(unittest.TestCase):
    def test_bone_and_canal_labels_are_not_teeth(self):
        np.testing.assert_array_equal(e.tooth_binary(np.array([0,1,32,33,52,53,54,55])),
                                      [False,True,True,True,True,False,False,False])

    def data(self):
        axes=[np.arange(-2.,2.01,.5),np.arange(-8.,8.01,.5),np.arange(-10.,10.01,.5)]
        l,a,s=np.meshgrid(*axes,indexing='ij');mask=(a>=-4)&(a<=0)&(s>=-6)
        return axes,mask,np.where(mask,1000.,-200.).astype(float)

    def check(self,key='A',point=(0.,0.,0.),tooth=False,domain=True,contrast=True,truncated=False):
        axes,mask,ct=self.data()
        if truncated:mask[:,:,:]=np.meshgrid(*axes,indexing='ij')[1]<=0
        teeth=np.zeros(mask.shape,bool)
        if tooth:teeth[len(axes[0])//2,len(axes[1])//2,len(axes[2])//2]=True
        data=e.prepare(mask,ct if contrast else np.zeros_like(ct),teeth,np.full(mask.shape,domain),axes)
        pr=dict(sup=axes[2],front=np.full(len(axes[2]),point[1]))
        return e.check(point,pr,key,data)

    def test_anterior_cortex_supported(self):
        self.assertTrue(self.check()['accepted'])
        self.assertTrue(self.check(key='Pog')['accepted'])

    def test_tooth_proximity_rejected(self):
        self.assertIn('tooth_or_root_proximity',self.check(tooth=True)['reasons'])

    def test_low_CT_contrast_rejected(self):
        self.assertIn('CT_cortical_contrast_insufficient',self.check(contrast=False)['reasons'])

    def test_posterior_inner_surface_rejected(self):
        self.assertIn('not_anterior_outer_cortex',self.check(point=(0.,-4.,0.))['reasons'])

    def test_inferior_border_not_pogonion(self):
        self.assertIn('too_close_to_inferior_border',self.check(key='Pog',point=(0.,0.,-5.))['reasons'])

    def test_truncated_lower_border_is_unknown(self):
        self.assertIn('inferior_boundary_truncated',self.check(key='Pog',truncated=True)['reasons'])

    def test_tooth_crop_background_outside_domain_is_unknown(self):
        self.assertIn('tooth_crop_unknown_at_candidate',self.check(domain=False)['reasons'])

    def test_filter_retains_order_and_does_not_invent_candidates(self):
        axes,mask,ct=self.data();data=e.prepare(mask,ct,np.zeros_like(mask),np.ones_like(mask),axes)
        run=dict(eligible=[dict(point=[0.,-4.,0.],score=2.),dict(point=[0.,0.,0.],score=1.)],
                 profile=dict(sup=axes[2],front=np.zeros(len(axes[2]))),status='candidate')
        filtered=e.filter_variant(run,'A',data)
        self.assertEqual(len(filtered['eligible']),1)
        self.assertEqual(filtered['winner'],[0.,0.,0.])
        self.assertEqual(len(run['eligible']),2)


if __name__=='__main__':unittest.main()
