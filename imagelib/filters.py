import cv2


def high_pass_filter(im, ksize=3):
    im = cv2.Laplacian(im, cv2.CV_32F, ksize=ksize)
    norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image


def sobel_x(im, ksize=3):
    im = cv2.Sobel(im, cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image


def sobel_y(im, ksize=3):
    im = cv2.Sobel(im, cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image
