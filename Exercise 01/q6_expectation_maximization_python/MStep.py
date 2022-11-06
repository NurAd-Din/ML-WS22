import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####

    n, d = X.shape
    n, k = gamma.shape
    n_tild = np.sum(gamma, axis=0)
    weights = n_tild / n
    means = np.zeros((k, d))
    for j in range(k):

        sum_inner = 0
        for i in range(n):

            sum_inner += gamma[i, j] * X[i, :]

        means[j, :] = sum_inner / n_tild[j]

    covariances = np.zeros((d, d, k))
    for j in range(k):

        sum_inner = 0
        for i in range(n):
            x = np.reshape((X[i, :] - means[j, :]), (d, 1))
            y = np.reshape((X[i, :] - means[j, :]), (1, d))
            sum_inner += gamma[i, j] * np.matmul(x, y)

        covariances[:, :, j] = sum_inner / n_tild[j]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood
