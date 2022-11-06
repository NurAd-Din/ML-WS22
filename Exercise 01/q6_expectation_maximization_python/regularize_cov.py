import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    #####Insert your code here for subtask 6d#####

    d, x = covariance.shape
    id = np.eye(d)
    regularized_cov = covariance + epsilon * id

    return regularized_cov
