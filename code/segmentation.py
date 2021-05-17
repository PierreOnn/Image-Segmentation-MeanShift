import numpy as np
from skimage import color
from algorithm import *


def imSegment(im, r, c, feature_type):
    # Dimensions of the image
    height = np.size(im, 0)
    width = np.size(im, 1)
    # Flattened image for manipulation
    image_2d = np.reshape(im, (-1, 3))
    im_lab = color.rgb2lab(image_2d)
    # Splitting of the color channels
    im_l = im_lab[..., 0]
    im_a = im_lab[..., 1]
    im_b = im_lab[..., 2]
    im_space = np.array([im_l, im_a, im_b]).transpose()

    if feature_type == '3D':
        # Labels and peaks are retrieved only taking into account the color space
        labels, peaks = meanshift_opt(im_space, r, c)
    elif feature_type == '5D':
        # Meshgrid is created for mimicking the pixel coordinate system
        # which is flattened and added by axis to the feature space
        xx, yy = np.meshgrid(range(1, width + 1), range(1, height + 1))
        x = np.reshape(xx, (height * width, 1))
        y = np.reshape(yy, (height * width, 1))
        im_space_x = np.append(im_space, x, axis=1)
        im_space_xy = np.append(im_space_x, y, axis=1)
        # Labels and peaks are retrieved taking into account the color space + pixel coordinates
        labels, peaks_xy = meanshift_opt(im_space_xy, r, c)
        # Peaks were determined on color space + pixel coordinates,
        # but only the color space is of relevance for converting cluster centers back to RGB
        peaks = peaks_xy[:, 0:3]
        im_space = im_space_xy[:, 0:3]
    else:
        print('Incorrect dimension')

    # Every peak ant its label is run through to assign it with the associated pixels
    segmIm = np.zeros((len(im_space), len(im_space[0])))
    for label, peak in enumerate(peaks, start=1):
        segmIm[np.where(labels == label)[0]] = peak
    segmIm = np.reshape(segmIm, (height, width, 3))
    segmIm = color.lab2rgb(segmIm)

    return segmIm, labels, peaks
