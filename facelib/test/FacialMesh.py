import unittest

import cv2
import numpy as np

from facelib.FacialMesh import _predict_3d_mesh, get_mesh_landmarks
from nnlib import nnlib
from facelib import LandmarksExtractor, S3FDExtractor

import eos

class MyTestCase(unittest.TestCase):
    def test_something(self):
        im = cv2.imread('../../imagelib/test/test_src/carrey/carrey.jpg')

        device_config = nnlib.DeviceConfig(cpu_only=True)
        nnlib.import_all(device_config)
        landmark_extractor = LandmarksExtractor(nnlib.keras)
        s3fd_extractor = S3FDExtractor()

        rects = s3fd_extractor.extract(input_image=im, is_bgr=True)
        print('rects:', rects)
        bbox = rects[0]  # bounding box

        landmark_extractor.__enter__()
        # landmarks = landmark_extractor.extract(input_image=im, rects=rects, second_pass_extractor=None,
        #                                        is_bgr=True)
        s3fd_extractor.__enter__()
        landmarks = landmark_extractor.extract(input_image=im, rects=rects, second_pass_extractor=s3fd_extractor,
                                               is_bgr=True)[-1]
        s3fd_extractor.__exit__()
        landmark_extractor.__exit__()
        print('landmarks shape:', np.shape(landmarks))

        mesh_points = get_mesh_landmarks(landmarks, bbox)
        print('mesh_points:', mesh_points)

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)
        for i, pt in enumerate(mesh_points):
            cv2.circle(im, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), thickness=-1)
        cv2.imshow('test output', im)
        cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()
