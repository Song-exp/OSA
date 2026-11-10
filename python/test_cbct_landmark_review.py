"""Review geometry checks: physical orientation and cohort boundary matter."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import cbct_landmark_review as review


class ReviewGeometryTests(unittest.TestCase):
    def test_reflection_uses_midline_not_bad_po_or_excel(self):
        points={"N":np.array([0.,0.,10.]),"S":np.array([0.,5.,8.]),
                "Ba":np.array([0.,10.,0.]),"ANS":np.array([0.,-10.,0.]),
                "PNS":np.array([0.,-3.,2.]),"LPo":np.array([50.,6.,7.]),
                "RPo":np.array([-20.,-30.,40.])}
        predicted,_=review.reflection(points,"RPo")
        np.testing.assert_allclose(predicted,[-50.,6.,7.],atol=1e-10)
        points['RPo']=np.array([999.,999.,999.])
        np.testing.assert_allclose(review.reflection(points,"RPo")[0],predicted)

    def test_sagittal_slice_and_physical_overlay(self):
        volume=np.arange(3*4*5).reshape(3,4,5)
        fig,ax=plt.subplots()
        try:
            review.panel(ax,volume,np.diag([2.,3.,4.]),np.array([10.,20.,30.]),
                         np.array([12.,26.,34.]),"sagittal",
                         {"p":(np.array([12.,26.,34.]),"red","+")})
            np.testing.assert_array_equal(ax.images[0].get_array(),volume[:,:,1])
            np.testing.assert_allclose(ax.images[0].get_extent(),[18.5,30.5,28.,40.])
            np.testing.assert_allclose(ax.lines[0].get_xdata(),[26.])
            np.testing.assert_allclose(ax.lines[0].get_ydata(),[34.])
        finally:
            plt.close(fig)

    def test_excluded_patient_stops_before_image_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'cohort.csv').write_text('Patient,Include,Reason\nP_2,false,inquiry\n',encoding='utf-8')
            with patch.object(review,'OUT',root),patch.object(review,'landmarks') as landmarks:
                with self.assertRaisesRegex(ValueError,'not included'):
                    review.review(2)
                landmarks.assert_not_called()


if __name__=='__main__':
    unittest.main()
