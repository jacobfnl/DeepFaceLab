"""
from https://github.com/jacobfnl/warriors_code
"""
import numpy as np
import cv2


def image_statistics(luma32):
    mean, std = cv2.meanStdDev(luma32)
    return mean, std


def auto_gamma_image(image_input, brightness_power=1.0, debug=False):
    """
    Produces an image that is more balance in gamma and contrast.
    @param image_input: 8-bits per channel BGR Color OpenCV image
    @param brightness_power: Extra brightness pressure on the shadow areas of the image.
    @param debug: print statements on/off
    @return: 8bit per channel BGR OpenCV image
    """
    brightness_power = min(max(0.0, brightness_power), 1.9)  # verify brightness_power within acceptable range
    i8 = 1./255.0
    i16 = 1./65535.0
    # convert to floating point
    im32 = np.float32(image_input)
    im32 *= i8
    ycc = cv2.cvtColor(im32, cv2.COLOR_BGR2YCrCb)
    l32 = ycc[:, :, 0]
    mean, std = image_statistics(l32)
    clip_limit = 26.0 - (mean[0][0] + std[0][0]) * 45.0
    if debug:
        print("mean: {}\nstdd: {}".format(mean[0][0], std[0][0]))
        print("contrast clip: {}".format(clip_limit))
        print("brightness_power: {}".format(brightness_power))
    if brightness_power != 0.0:
        l32 = np.power(l32, 1.0 - brightness_power*0.35)
    l32 *= 65535
    luma = np.uint16(l32)
    # adjusting the
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(5, 3))
    luma = clahe.apply(luma)
    l32 = np.float32(luma)
    l32 *= i16
    image_statistics(l32)
    ycc[:, :, 0] = l32
    im32 = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    im32 = cv2.cvtColor(im32, cv2.COLOR_BGR2HSV)
    l32 = im32[:, :, 1]
    if brightness_power != 0.0:
        l32 = np.power(l32, 1.0-brightness_power*0.2)
    im32[:, :, 1] = l32
    im32 = cv2.cvtColor(im32, cv2.COLOR_HSV2BGR)
    im32 *= 255
    return np.uint8(im32)
