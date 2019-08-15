import unittest

import cv2
import numpy as np

from imagelib import filters
from samplelib import SampleLoader, SampleType


class FiltersTest(unittest.TestCase):
    def test_high_pass_filter(self):
        src_samples = SampleLoader.load(SampleType.FACE, './test_src', None)
        # dst_samples = SampleLoader.load(SampleType.FACE, './test_dst', None)

        results = []
        for src_sample in src_samples:
            print(src_sample.filename)
            src_img = src_sample.load_bgr()
            # src_mask = src_sample.load_mask()

            # Toggle to see masks
            # show_masks = False

            hpf_img = filters.high_pass_filter(src_img)
            sbx_img = filters.sobel_x(src_img)
            sby_img = filters.sobel_y(src_img)
            results.append(np.concatenate((src_img, hpf_img, sbx_img, sby_img), axis=1))

        results = np.concatenate(results, axis=0)

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        cv2.imshow('test output', results)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
