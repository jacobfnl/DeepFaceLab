import cv2


def high_pass_filter(img, ksize=3):
    return cv2.Laplacian(img, cv2.CV_32F, ksize=ksize)


def sobel_x(img, ksize=3):
    return cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=ksize)


def sobel_y(img, ksize=3):
    return cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=ksize)
