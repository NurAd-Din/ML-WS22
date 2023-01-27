import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####

    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector
    N, = samples.shape
    ones = np.ones((N, N))
    pos_mat = pos * ones
    pos_mat = np.transpose(pos_mat)
    diff_mat = pos_mat - samples
    diff_mat = np.transpose(diff_mat)
    diff_mat = np.absolute(diff_mat)
    diff_mat = np.sort(diff_mat, axis=0)
    estDensity = 2 * diff_mat[k - 1, :]
    estDensity = k / (N * estDensity)
    estDensity = np.column_stack((pos, estDensity))

    # Compute the number of the samples created
    return estDensity
