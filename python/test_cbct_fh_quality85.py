import unittest
import numpy as np
from cbct_fh_quality85 import holes, sample, support, margin


class FHImageRulesTests(unittest.TestCase):
    def test_air_hole_requires_closed_region_and_roof_bone(self):
        axis=np.arange(-12,12.01,.5)
        a,z=np.meshgrid(axis,axis,indexing='ij')
        image=np.full(a.shape,1000.,dtype=np.float32)
        image[a*a+(z+3)**2<=9]=-900
        found=holes(image,axis,axis,-400)
        self.assertEqual(len(found),1)
        self.assertLess(abs(found[0]['ant']),.6)
        self.assertLess(abs(found[0]['sup']),.6)
        image[:,0:20]=-900 # 영상 바깥과 연결된 공기는 폐쇄 외이도 후보가 아니다.
        self.assertEqual(holes(image,axis,axis,-400),[])

    def test_soft_tissue_roof_is_not_accepted_as_bone(self):
        axis=np.arange(-12,12.01,.5)
        a,z=np.meshgrid(axis,axis,indexing='ij')
        image=np.zeros(a.shape,dtype=np.float32)
        image[a*a+(z+3)**2<=9]=-900
        self.assertEqual(holes(image,axis,axis,-400),[])

    def test_physical_sampling_with_anisotropic_spacing(self):
        z,y,x=np.indices((15,15,15))
        v=(x+10*y+100*z).astype(np.int16)
        dirs=np.diag([.5,1.,2.]);org=np.array([-3.,-5.,10.])
        p=org+np.array([4.5,6.,7.])@dirs
        self.assertAlmostEqual(float(sample(v,dirs,org,p[None])[0]),764.5)
        self.assertTrue(np.isnan(sample(v,dirs,org,np.array([[100.,100.,100.]]))[0]))

    def test_fov_and_bone_support_do_not_wrap_negative_indices(self):
        v=np.zeros((21,21,21),dtype=np.int16)
        v[:,:,-1]=2000
        p=np.array([-1.,10.,10.]);dirs=np.eye(3);org=np.zeros(3)
        self.assertLess(margin(v,dirs,org,p),0)
        s=support(v,dirs,org,p)
        self.assertTrue(np.isinf(s['nearest_dense_mm']))
        self.assertLess(s['patch_finite_fraction'],1.)


if __name__=='__main__':unittest.main()
