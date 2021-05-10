from scipy.io import loadmat
from skimage import io
from segmentation import *
from algorithm import *
from plotclusters3D import *
import cv2


def main():
    # matfile = loadmat('../data/pts.mat')
    # data = np.array(matfile['data'])
    # data = data.transpose()
    #
    # labels, peaks = meanshift_opt(data, 2, 4)
    # plotclusters3D(data, labels, peaks)
    # plt.show()

    image = io.imread('../images/55075.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_2d = image.reshape(-1, 3)

    labels, peaks = meanshift_opt(image_2d, 2, 4)
    plotclusters3D(image_2d, labels, peaks)
    plt.show()


if __name__ == "__main__":
    main()
