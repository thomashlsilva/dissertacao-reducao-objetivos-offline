import numpy as np
from numpy.linalg import eig


def main():
    x = np.array([[9893, 564, 17689], [8776, 389, 17359], [13572, 1103, 18597], [6455, 743, 8745],
                  [5129, 203, 14397], [5432, 215, 3467], [3807, 385, 4679], [3423, 187, 6754],
                  [3708, 127, 2275], [3294, 297, 6754], [5433, 432, 5589], [6287, 451, 8972]])

    std_x = np.std(x, axis=0, ddof=1)
    # print(std_x)
    mean_x = np.mean(x, axis=0)
    # print(mean_x)
    z = (x - mean_x)/std_x
    # print(z)
    p = np.cov(z, rowvar=False)
    print(p)

    eigenValues, eigenVectors = eig(p)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    print(eigenValues)
    print(eigenVectors)

    y = np.matmul(eigenVectors.transpose(), z.transpose())
    print(y.transpose())

if __name__ == '__main__':
    main()