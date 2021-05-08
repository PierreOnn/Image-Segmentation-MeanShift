from scipy.io import loadmat
from algorithm import *
from plotclusters3D import *


def main():
    matfile = loadmat('../data/pts.mat')
    data = np.array(matfile['data'])
    data = data.transpose()

    labels, peaks = meanshift_opt(data, 2, 4)
    plotclusters3D(data, labels, peaks)
    plt.show()


if __name__ == "__main__":
    main()
