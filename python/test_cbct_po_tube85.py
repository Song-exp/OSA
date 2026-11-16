import unittest
import numpy as np
from cbct_po_tube85 import choose_tube


class TubeRecoveryTests(unittest.TestCase):
    def sample(self,length=17,contrast=True,two=False):
        L=np.arange(40.,52.01,.5);A=np.arange(-15.,15.01,.5);S=np.arange(-10.,10.01,.5)
        a,s=np.meshgrid(A,S,indexing='ij')
        mask=np.zeros((len(L),len(A),len(S)),bool)
        for center in ([-6.,6.] if two else [0.]):
            radius=(a-center)**2+s*s
            ring=(radius>=9)&(radius<=25)
            mask[4:4+length]|=ring
        ct=np.where(mask,1000.,-200.).astype(np.float32) if contrast else np.zeros(mask.shape,np.float32)
        return mask,[L,A,S],ct,np.array([46.,0.,3.])

    def test_persistent_tube_with_native_rim_generates_candidate(self):
        r=choose_tube(*self.sample())
        self.assertEqual(r['status'],'tube_supported_candidate_requires_review')
        self.assertGreaterEqual(r['selected_track']['span_mm'],4.)
        self.assertIsNotNone(r['candidate_local'])

    def test_small_number_of_slices_is_not_enough(self):
        r=choose_tube(*self.sample(length=2))
        self.assertIsNone(r['candidate_local'])
        self.assertTrue(any('insufficient_lateral_extent' in t['reasons'] for t in r['tracks']))

    def test_no_native_contrast_blocks_shape_only_candidate(self):
        r=choose_tube(*self.sample(contrast=False))
        self.assertIsNone(r['candidate_local'])

    def test_two_equal_competing_tubes_require_review(self):
        r=choose_tube(*self.sample(two=True))
        self.assertEqual(r['status'],'multiple_competing_tubes')
        self.assertIsNone(r['candidate_local'])


if __name__=='__main__':unittest.main()
