import unittest

import cv2
import numpy as np

from facelib.FacialMesh import _predict_3d_mesh, get_mesh_landmarks, get_texture
from nnlib import nnlib
from facelib import LandmarksExtractor, S3FDExtractor


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
        l, t, r, b = bbox

        landmark_extractor.__enter__()
        s3fd_extractor.__enter__()

        landmarks = landmark_extractor.extract(input_image=im, rects=rects, second_pass_extractor=s3fd_extractor,
                                               is_bgr=True)[-1]
        s3fd_extractor.__exit__()
        landmark_extractor.__exit__()
        print('landmarks shape:', np.shape(landmarks))

        mesh_points, isomap, rendered = get_mesh_landmarks(landmarks, im)
        print('mesh_points:', np.shape(mesh_points))

        cv2.namedWindow('test output', cv2.WINDOW_NORMAL)

        # Draw the bounding box
        cv2.rectangle(im, (l, t), (r, b), (0, 0, 255), thickness=2)

        # Draw the landmarks
        for i, pt in enumerate(landmarks):
            cv2.circle(im, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), thickness=-1)

        # Draw the 3D mesh
        for i, pt in enumerate(mesh_points):
            cv2.circle(im, (int(pt[0]), int(pt[1])), 1, (255, 255, 255), thickness=-1)
        cv2.imshow('test output', im)
        cv2.waitKey(0)

        cv2.imshow('test output', isomap.transpose([1, 0, 2]))
        cv2.waitKey(0)

        cv2.imshow('test output', rendered)
        cv2.waitKey(0)

        # cv2.imshow('test output', iso_points)

        cv2.waitKey(0)

        cv2.destroyAllWindows()





if __name__ == '__main__':
    unittest.main()
