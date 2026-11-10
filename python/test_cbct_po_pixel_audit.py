import unittest
from unittest.mock import patch

import numpy as np

import cbct_po_pixel_audit as audit


class PixelAuditTests(unittest.TestCase):
    def test_frame_directions_and_po_independence(self):
        points={"N":np.array([0.,-10.,10.]),"S":np.array([0.,0.,10.]),
                "Ba":np.array([0.,10.,0.]),"ANS":np.array([0.,-10.,0.]),
                "PNS":np.array([0.,0.,0.]),"LPo":np.array([99.,99.,99.])}
        left=audit.frame(points,'LPo');right=audit.frame(points,'RPo')
        np.testing.assert_allclose(left[:,0],[1,0,0],atol=1e-10)
        np.testing.assert_allclose(left[:,1],[0,-1,0],atol=1e-10)
        np.testing.assert_allclose(left[:,2],[0,0,1],atol=1e-10)
        np.testing.assert_allclose(right[:,0],-left[:,0],atol=1e-10)
        np.testing.assert_allclose(right[:,1:],left[:,1:],atol=1e-10)
        points['LPo']*=100
        np.testing.assert_allclose(audit.frame(points,'LPo'),left)

    def test_native_voxels_not_interpolated_and_fov(self):
        volume=np.zeros((41,41,41),dtype=np.int16)
        volume[20,20,20]=-1000
        metrics,hist=audit.native_stats(volume,np.eye(3),np.zeros(3),np.array([20.,20.,20.]),(-1000.,1000.))
        self.assertEqual(metrics['native_voxels'],7153)
        self.assertAlmostEqual(metrics['airlike_fraction'],1/7153)
        self.assertAlmostEqual(hist.sum(),1.)
        self.assertTrue(metrics['roi_complete'])
        edge,_=audit.native_stats(volume,np.eye(3),np.zeros(3),np.array([1.,20.,20.]),(-1000.,1000.))
        self.assertFalse(edge['roi_complete'])

    def test_patch_interpolates_in_physical_space(self):
        z,y,x=np.indices((41,41,41))
        volume=(x+2*y+3*z).astype(np.int16)
        sampled=audit.aligned_patch(volume,np.eye(3),np.zeros(3),np.array([20.25,20.,20.]),np.eye(3))
        mid=len(audit.AXIS)//2
        self.assertAlmostEqual(float(sampled[mid,mid,mid]),120.25)
        self.assertAlmostEqual(float(sampled[mid,mid,mid+1]),120.75)
        self.assertEqual(sampled.dtype,np.float32)

    def test_excluded_patient_stops_before_artifact_read(self):
        with patch.object(audit,'included',return_value=[16]),patch.object(audit,'landmarks') as lm:
            with self.assertRaisesRegex(ValueError,'excluded'):
                audit.process(2)
            lm.assert_not_called()


if __name__=='__main__':unittest.main()
