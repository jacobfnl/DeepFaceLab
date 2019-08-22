import unittest

import cv2
import numpy as np

from mainscripts.Extractor import ExtractSubprocessor
from nnlib import nnlib
from facelib import LandmarksExtractor, S3FDExtractor


class LandmarkExtractorTest(unittest.TestCase):
    def test_extract(self):
        im = cv2.imread('../../imagelib/test/test_src/rami/rami.png')
        h, w, _ = im.shape

        device_config = nnlib.DeviceConfig(cpu_only=True)
        nnlib.import_all(device_config)
        landmark_extractor = LandmarksExtractor(nnlib.keras)
        s3fd_extractor = S3FDExtractor()

        rects = s3fd_extractor.extract(input_image=im, is_bgr=True)
        print('rects:', rects)
        l, t, r, b = rects[0]

        landmark_extractor.__enter__()
        # landmarks = landmark_extractor.extract(input_image=im, rects=rects, second_pass_extractor=None,
        #                                        is_bgr=True)
        s3fd_extractor.__enter__()
        landmarks = landmark_extractor.extract(input_image=im, rects=rects, second_pass_extractor=s3fd_extractor,
                                               is_bgr=True)[-1]
        s3fd_extractor.__exit__()
        landmark_extractor.__exit__()

        # print('landmarks', list(landmarks))

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        cv2.imshow('test output', im)
        cv2.waitKey(0)

        cv2.rectangle(im, (l, t), (r, b), (255, 255, 0))
        cv2.imshow('test output', im)
        cv2.waitKey(0)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5

        def pt(arr=None, x=None, y=None):
            if x and y:
                return int(x), int(y)
            else:
                return int(arr[0]), int(arr[1])

        for i, m in enumerate(landmarks):
            print(i, m)
            cv2.circle(im, pt(m), 3, (0, 255, 0))
            cv2.putText(im, str(i), pt(m), font_face, font_scale, (0, 255, 0), thickness=1)
        cv2.imshow('test output', im)
        cv2.waitKey(0)

        cv2.line(im, pt(landmarks[8]), pt(landmarks[27]), (0, 0, 255), thickness=4)
        l_eyebrow = np.mean(landmarks[17:22, :], axis=0)
        r_eyebrow = np.mean(landmarks[22:27, :], axis=0)
        print(l_eyebrow, r_eyebrow)
        cv2.line(im, pt(l_eyebrow), pt(r_eyebrow), (0, 0, 255), thickness=4)

        brow_slope = (r_eyebrow[1] - l_eyebrow[1]) / (r_eyebrow[0] - l_eyebrow[0])
        print(brow_slope)
        nose = np.mean([landmarks[31], landmarks[35]], axis=0)
        l_nose_line = nose - (10 * brow_slope, 10)
        r_nose_line = nose + (10 * brow_slope, 10)
        print(l_nose_line, r_nose_line)
        cv2.line(im, pt(l_nose_line), pt(r_nose_line), (0, 0, 255), thickness=2)

        cv2.imshow('test output', im)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
