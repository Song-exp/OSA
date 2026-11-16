import unittest
import numpy as np
from cbct_seg_landmarks85 import physical_correspondences,local_transfer,po_sections,or_sections,grid


class SegmentationLandmarkTests(unittest.TestCase):
    def test_inverse_stored_frame_recovers_patient_coordinates(self):
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        origin=np.array([12.,-8.,30.])
        world=np.array([[1.,3.,8.],[9.,2.,5.]])
        z=dict(ids=np.array([7]),frame=np.array([(world-origin)@rotation.T]),
               head_R=np.array([rotation]),head_origin=np.array([origin]))
        np.testing.assert_allclose(physical_correspondences(z,'P_7'),world)

    def test_roi_transfer_uses_correspondence_not_patient_landmark(self):
        rng=np.random.default_rng(4)
        template=rng.normal(size=(32,3))*3
        transform=np.array([[1.1,.02,0.],[0.,.9,.04],[.1,0.,1.]])
        shift=np.array([9.,-4.,2.]);anchor=np.array([.5,.2,.7])
        patient=template@transform+shift
        predicted,info=local_transfer(template,patient,anchor)
        np.testing.assert_allclose(predicted,anchor@transform+shift,atol=1e-10)
        self.assertLess(info['local_transfer_residual_mm'],1e-10)

    def test_po_mask_hole_has_stable_upper_rim(self):
        L=np.arange(45.,49.,.5);A=np.arange(-8.,8.1,.5);S=np.arange(-8.,8.1,.5)
        a,s=np.meshgrid(A,S,indexing='ij')
        ring=((a*a+s*s)>=9)&((a*a+s*s)<=25)
        mask=np.broadcast_to(ring,(len(L),*ring.shape)).copy()
        candidates,n=po_sections(mask,[L,A,S],'L')
        self.assertGreater(n,0);self.assertGreater(len(candidates),0)
        self.assertTrue(all(abs(c['local'][1])<1 for c in candidates))
        self.assertTrue(all(2<=c['local'][2]<=4 for c in candidates))

    def test_or_minimum_keeps_side_specific_height(self):
        L=np.arange(20.,32.1,.5);A=np.arange(-3.,3.1,.5);S=np.arange(-10.,14.1,.5)
        low,ant,sup=np.meshgrid(L,A,S,indexing='ij')
        mask=sup<=(2+.12*(low-26)**2)
        candidates,_=or_sections(mask,[L,A,S],np.array([26.,0.,2.]))
        self.assertTrue(candidates)
        self.assertTrue(any(abs(c['local'][0]-26)<1 for c in candidates))
        # 반대편을 사용하지 않고 해당 측의 높이가 다른 모양을 처리한다.
        other=sup<=(-2+.12*(low-26)**2)
        candidates2,_=or_sections(other,[L,A,S],np.array([26.,0.,-2.]))
        self.assertTrue(candidates2)
        self.assertAlmostEqual(min(c['local'][2] for c in candidates)-min(c['local'][2] for c in candidates2),4.)

    def test_roi_outside_fov_is_explicit(self):
        mask=np.ones((25,25,25),dtype=np.uint8)
        m,axes,inside=grid(mask,np.eye(3),np.zeros(3),np.zeros(3),np.eye(3),np.array([1.,12.,12.]),half=(4.,4.,4.))
        self.assertLess(inside.mean(),1.)
        self.assertEqual(m.shape,inside.shape)


if __name__=='__main__':unittest.main()
