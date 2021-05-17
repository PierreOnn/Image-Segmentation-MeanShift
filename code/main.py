from scipy.io import loadmat
from skimage import io
from segmentation import *
from algorithm import *
from plotclusters3D import *
import matplotlib.pyplot as plt
import time


def main():
    # matfile = loadmat('../data/pts.mat')
    # data = np.array(matfile['data'])
    # data = data.transpose()
    # labels, peaks = meanshift_opt(data, 2, 4)
    # plotclusters3D(data, labels, peaks)
    # plt.savefig('../experiments/plots3D/pts3dplot.png')
    # plt.show()

    # image = io.imread('../images/181091.jpg')
    # image_2d = np.reshape(image, (-1, 3))
    # labels, peaks = meanshift_opt(image_2d, 30, 2)
    # plotclusters3D(image_2d, labels, peaks)
    # plt.savefig('../experiments/plots3D/181091_r10_c4.png')
    # plt.show()

    t0 = time.time()
    image = io.imread('../images/55075.jpg')
    segmIm, labels, peaks = imSegment(image, 30, 2, '3D')
    t1 = time.time()
    print(t1 - t0)
    io.imshow(segmIm)
    plt.savefig('../experiments/55075_r30_c2_3d.png')
    io.show()


if __name__ == "__main__":
    main()
