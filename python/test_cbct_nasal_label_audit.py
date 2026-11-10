"""Check physical-coordinate sampling independently of patient data."""
import unittest

import numpy as np
import SimpleITK as sitk

from cbct_nasal_label_audit import sample


class SamplingTests(unittest.TestCase):
    def test_physical_origin_spacing_and_outside_are_respected(self):
        array = np.zeros((3, 4, 5), dtype=np.uint8)
        array[1, 2, 3] = 14
        array[2, 1, 2] = 15
        array[0, 0, 0] = 11
        image = sitk.GetImageFromArray(array)
        image.SetOrigin((10., 20., 30.))
        image.SetSpacing((.5, 1., 2.))
        indices = [(3, 2, 1), (2, 1, 2), (0, 0, 0)]
        points = [image.TransformIndexToPhysicalPoint(ix) for ix in indices]+[(100., 100., 100.)]
        np.testing.assert_array_equal(sample(array, image, np.asarray(points)), [True, True, False, False])

    def test_rotated_image_direction(self):
        array = np.zeros((3, 4, 5), dtype=np.uint8)
        array[1, 2, 3] = 15
        image = sitk.GetImageFromArray(array)
        image.SetOrigin((-5., 10., 20.))
        image.SetSpacing((.4, .6, .8))
        image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
        point = image.TransformIndexToPhysicalPoint((3, 2, 1))
        np.testing.assert_array_equal(sample(array, image, np.asarray([point])), [True])


if __name__ == "__main__":
    unittest.main()
