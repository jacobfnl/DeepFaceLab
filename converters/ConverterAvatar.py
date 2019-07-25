import time

import cv2
import numpy as np

from facelib import FaceType, LandmarksProcessor
from joblib import SubprocessFunctionCaller
from utils.pickle_utils import AntiPickler

from .Converter import Converter

class ConverterAvatar(Converter):

    #override
    def __init__(self,  predictor_func,
                        predictor_input_size=0):

        super().__init__(predictor_func, Converter.TYPE_FACE_AVATAR)

        self.predictor_input_size = predictor_input_size
        
        #dummy predict and sleep, tensorflow caching kernels. If remove it, conversion speed will be x2 slower
        predictor_func ( np.zeros ( (predictor_input_size,predictor_input_size,3), dtype=np.float32 ) )
        time.sleep(2)

        predictor_func_host, predictor_func = SubprocessFunctionCaller.make_pair(predictor_func)
        self.predictor_func_host = AntiPickler(predictor_func_host)
        self.predictor_func = predictor_func

    #overridable
    def on_host_tick(self):
        self.predictor_func_host.obj.process_messages()
        
    #override
    def cli_convert_face (self, img_bgr, img_face_landmarks, debug, **kwargs):
        

        img_size = img_bgr.shape[1], img_bgr.shape[0]

        inp_size = self.predictor_input_size        
        face_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, inp_size, face_type=FaceType.FULL_NO_ROTATION)

        inp_face_bgr = cv2.warpAffine( img_bgr, face_mat, (inp_size, inp_size), flags=cv2.INTER_LANCZOS4 )
        if debug:
            debugs = [inp_face_bgr.copy()]
        prd_face_bgr = self.predictor_func ( inp_face_bgr )

        out_img = np.clip(prd_face_bgr, 0.0, 1.0)
        
        if debug:
            debugs += [out_img.copy()]
        
        return debugs if debug else out_img
