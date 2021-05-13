import numpy as np
from skimage import io, color
from algorithm import *


def imSegment(im, r, c, feature_type):
    height = np.size(im, 0)
    width = np.size(im, 1)
    image_2d = np.reshape(im, (-1, 3))
    im_lab = color.rgb2lab(image_2d)
    im_l = im_lab[..., 0]
    im_a = im_lab[..., 1]
    im_b = im_lab[..., 2]
    im_space = np.array([im_l, im_a, im_b]).transpose()

    if feature_type == '3D':
        labels, peaks = meanshift_opt(im_space, r, c)
    elif feature_type == '5D':
        xx, yy = np.meshgrid(range(1, width+1), range(1, height+1))
        x = np.reshape(xx, (height*width, 1))
        y = np.reshape(yy, (height*width, 1))
        im_space_x = np.append(im_space, x, axis=1)
        im_space_xy = np.append(im_space_x, y, axis=1)
        labels, peaks = meanshift_opt(im_space_xy, r, c)
    else:
        print('Incorrect dimension')

    segmIm = np.zeros((np.size(image_2d, 0), np.size(image_2d, 1)))
    for label, peak in enumerate(peaks):
        label = label + 1
        pixel_assigned = np.where(labels == label)[0]
        segmIm[pixel_assigned, :] = peak
    segmIm = np.reshape(segmIm, (height, width, 3))
    segmIm = color.lab2rgb(segmIm)
    return segmIm, labels, peaks