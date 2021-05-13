import numpy as np
from skimage import io, color
from algorithm import *


def imSegment(im, r, feature_type, c):
    height = np.size(im, 0)
    width = np.size(im, 1)
    rgb = im
    im_lab = color.rgb2lab(rgb)
    im_l = im_lab[..., 0]
    im_a = im_lab[..., 1]
    im_b = im_lab[..., 2]

    if feature_type == '3D':
        im_space = [im_l, im_a, im_b]
    elif feature_type == '5D':
        xx, yy = np.meshgrid(range(1, width), range(1, height))
        x = np.reshape(xx, (height, 1))
        y = np.reshape(yy, (width, 1))
        im_space = [im_l, im_a, im_b, x, y]
    else:
        print('Incorrect dimension')

    labels, peaks = meanshift_opt(im_space, r, c)
    label_pixel = np.reshape(labels, height, width)

    # for i in range(1, len(peaks)):
    segmIm = np.reshape(im_space[..., 1:3], height, width, 3)
    segmIm = color.lab2rgb(segmIm)
    return segmIm, labels, peaks