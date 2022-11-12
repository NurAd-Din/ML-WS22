import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)

    num_samples, dim = data.shape
    x = np.ones((num_samples, 1))
    x = np.concatenate([x, data], axis=1)
    weight = np.linalg.pinv(x).dot(label)
    bias = weight[0]
    weight = weight[1:(dim + 1)]

    return weight, bias
