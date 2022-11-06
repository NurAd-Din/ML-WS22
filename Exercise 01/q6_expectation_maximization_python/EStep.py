import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####

    weights = np.asarray(weights)
    means = np.asarray(means)

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    n, d = X.shape
    k, d = means.shape

    sum_inner = 0
    gamma = np.zeros((n,k))

    for i in range(n):
        sum_inner = 0
        for j in range(k):
            scale = (1 / (np.sqrt(2 * np.pi) ** d) * np.sqrt(np.linalg.det(covariances[:, :, j])))
            sum_inner += weights[j] * scale * np.exp(- 0.5 * np.matmul((X[i, :] - means[j, :]),
                                                                       np.matmul(np.linalg.inv(covariances[:, :, j]),
                                                                                 np.transpose(X[i, :] - means[j, :]))))

        for j in range(k):
            scale = (1 / (np.sqrt(2 * np.pi) ** d) * np.sqrt(np.linalg.det(covariances[:, :, j])))
            gamma[i, j] = weights[j] * scale * np.exp(- 0.5 * np.matmul((X[i, :] - means[j, :]),
                                                                       np.matmul(np.linalg.inv(covariances[:, :, j]),
                                                                                 np.transpose(X[i, :] - means[j, :]))))
            gamma[i, j] = gamma[i, j] / sum_inner

    return [logLikelihood, gamma]
