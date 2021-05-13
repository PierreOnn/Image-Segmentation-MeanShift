from scipy.io import loadmat
from skimage import io
from segmentation import *
from algorithm import *
from plotclusters3D import *
import matplotlib.pyplot as plt


def main():
    # matfile = loadmat('../data/pts.mat')
    # data = np.array(matfile['data'])
    # data = data.transpose()
    # labels, peaks = meanshift_opt(data, 2, 4)
    # plotclusters3D(data, labels, peaks)
    # plt.show()

    # image = io.imread('../images/55075.jpg')
    # image_2d = np.reshape(image, (-1, 3))
    # labels, peaks = meanshift_opt(image_2d, 30, 4)
    # plotclusters3D(image_2d, labels, peaks)
    # plt.show()

    image = io.imread('../images/181091.jpg')
    segmIm, labels, peaks = imSegment(image, 30, 4, '5D')
    io.imshow(segmIm)
    io.show()


if __name__ == "__main__":
    main()
