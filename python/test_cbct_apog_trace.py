import unittest
import numpy as np
import cbct_apog_trace as e


class TraceTests(unittest.TestCase):
    def test_discards_detached_front_island(self):
        ant=np.arange(-10.,10.01,.5);sup=np.arange(-6.,6.01,.5)
        a,s=np.meshgrid(ant,sup,indexing='ij')
        binary=(a>=-8)&(a<=2-.1*s*s)
        binary|=(a>=7)&(a<=8)&(abs(s)<.6)
        front,status=e.trace_front(binary,ant,sup)
        self.assertEqual(status,'tracked');self.assertLess(np.nanmax(front),3.)

    def test_retains_broad_supported_extrema(self):
        sup=np.arange(-8.,8.01,.5)
        for key,front in [('A',.05*sup**2),('Pog',-.05*sup**2)]:
            nodes,status=e.contour_candidates(front,sup,np.zeros(3),key,0)
            self.assertEqual(status,'candidate')
            self.assertTrue(any(abs(c['sup'])<=.5 for c in nodes))

    def test_monotonic_trace_is_not_forced_to_candidate(self):
        sup=np.arange(-8.,8.01,.5)
        for key in ('A','Pog'):
            self.assertFalse(e.contour_candidates(sup,sup,np.zeros(3),key,0)[0])

    def test_axis_changes_candidate_but_coordinates_stay_in_original_frame(self):
        sup=np.arange(-8.,8.01,.5);front=-.1*sup**2
        left,_=e.contour_candidates(front,sup,np.zeros(3),'Pog',-5)
        right,_=e.contour_candidates(front,sup,np.zeros(3),'Pog',5)
        self.assertTrue(left and right)
        self.assertLess(np.mean([c['sup'] for c in left]),np.mean([c['sup'] for c in right]))
        for c in left+right:self.assertEqual(c['front'],front[c['index']])

    def test_long_missing_segment_is_rejected(self):
        ant=np.arange(-10.,10.01,.5);sup=np.arange(-6.,6.01,.5)
        a,s=np.meshgrid(ant,sup,indexing='ij');binary=(a<=0)&(a>=-8)&(abs(s)>2)
        front,status=e.trace_front(binary,ant,sup)
        self.assertEqual(status,'insufficient_contour_coverage')

    def test_variants_form_complete_factorial(self):
        self.assertEqual(len(e.VARIANTS),24)
        for angle in (-5.,0.,5.):
            for window in ('base','expanded'):
                group=[v for v in e.VARIANTS if v['angle_deg']==angle and v['window']==window]
                self.assertEqual(len(group),4)
                self.assertEqual(sum(v['family']=='AMASSS' for v in group),1)

    def test_raw_ct_without_contrast_does_not_generate_point(self):
        axes=[np.arange(-3.,3.01,.5),np.arange(-18.,18.01,.5),np.arange(-16.,16.01,.5)]
        shape=tuple(len(x) for x in axes)
        v=next(v for v in e.VARIANTS if v['name']=='native_otsu_+0')
        result=e.detect_variant(np.ones(shape,bool),np.zeros(shape),axes,np.zeros(3),8.,'Pog',v)
        self.assertIsNone(result['winner']);self.assertEqual(result['status'],'no_image_contrast')


if __name__=='__main__':unittest.main()
