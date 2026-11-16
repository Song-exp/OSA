import unittest
import numpy as np
from cbct_seg_landmarks_policy85 import decide
from cbct_seg_landmarks_confirmed85 import decide_confirmed


class SegmentationExceptionTests(unittest.TestCase):
    def record(self):
        return dict(transferred_seed_lps=[30.,0.,0.],original_to_mask_mm=.5,
                    registration_fit_flag=False,candidate=[30.,1.,2.],
                    transfer=dict(local_transfer_residual_mm=.8,local_transfer_condition=4.))

    def test_keep_unflagged_point_even_when_different_candidate_exists(self):
        r=decide(np.array([30.,8.,4.]),self.record(),(np.zeros(3),np.eye(3)))
        self.assertEqual(r['route'],'original_retained')
        np.testing.assert_allclose(r['proposed_point'],[30.,8.,4.])

    def test_far_point_uses_same_side_segmentation_candidate(self):
        rec=self.record();rec['original_to_mask_mm']=4.
        r=decide(np.array([30.,0.,0.]),rec,(np.zeros(3),np.eye(3)))
        self.assertEqual(r['route'],'segmentation_recovery_proposal')
        self.assertEqual(r['proposed_point'],rec['candidate'])

    def test_bad_correspondence_blocks_recovery(self):
        rec=self.record();rec['original_to_mask_mm']=4.;rec['registration_fit_flag']=True
        r=decide(np.array([30.,0.,0.]),rec,(np.zeros(3),np.eye(3)))
        self.assertEqual(r['route'],'needs_review_no_reliable_candidate')
        self.assertIsNone(r['proposed_point'])

    def test_no_segmentation_around_point_and_no_candidate_is_not_success(self):
        rec=self.record();rec['original_to_mask_mm']=None;rec['candidate']=None
        r=decide(np.array([30.,0.,0.]),rec,(np.zeros(3),np.eye(3)))
        self.assertIsNone(r['proposed_point']);self.assertFalse(r['changed'])

    def test_point_outside_roi_triggers_even_if_it_is_on_other_bone(self):
        rec=self.record();rec['original_to_mask_mm']=0.
        r=decide(np.array([30.,20.,0.]),rec,(np.zeros(3),np.eye(3)))
        self.assertTrue(r['changed'])
        self.assertIn('original_outside_transferred_anatomical_roi',r['reasons'])

    def test_mask_disagreement_alone_does_not_override_image_supported_point(self):
        rec=self.record();rec['original_to_mask_mm']=4.
        r=decide_confirmed(np.array([30.,0.,0.]),rec,(np.zeros(3),np.eye(3)),.5)
        self.assertEqual(r['route'],'mask_mismatch_original_retained')
        self.assertFalse(r['changed'])

    def test_far_from_mask_and_image_remains_recovery_proposal(self):
        rec=self.record();rec['original_to_mask_mm']=4.
        r=decide_confirmed(np.array([30.,0.,0.]),rec,(np.zeros(3),np.eye(3)),3.9)
        self.assertEqual(r['route'],'segmentation_recovery_proposal')


if __name__=='__main__':unittest.main()
