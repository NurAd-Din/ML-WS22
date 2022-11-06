import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####

    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector
    ones = np.ones((100, 100))
    pos_mat = pos * ones
    pos_mat = np.transpose(pos_mat)
    diff_mat = pos_mat - samples
    diff_mat = np.transpose(diff_mat)
    diff_mat = np.absolute(diff_mat)
    bool_mat = diff_mat < h
    estDensity = np.count_nonzero(bool_mat, axis=0)
    estDensity = estDensity / (100 * 2 * h)
    estDensity = np.column_stack((pos, estDensity))

    # Compute the number of samples created
    return estDensity
