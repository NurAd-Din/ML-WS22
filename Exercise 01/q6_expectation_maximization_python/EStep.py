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

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    weights = np.asarray(weights)
    means = np.asarray(means)
    n, d = X.shape
    k, d = means.shape
    gamma = np.zeros((n, k))

    for i in range(n):
        sum_inner = 0
        for j in range(k):
            x_m = X[i, :] - means[j, :]
            scale = (1 / (np.sqrt(2 * np.pi) ** d) * np.sqrt(np.linalg.det(covariances[:, :, j])))
            sum_inner += weights[j] * scale * np.exp(- 0.5 * np.linalg.solve(covariances[:, :, j], x_m).T.dot(x_m))
        for j in range(k):
            x_m = X[i, :] - means[j, :]
            scale = (1 / (np.sqrt(2 * np.pi) ** d) * np.sqrt(np.linalg.det(covariances[:, :, j])))
            gamma[i, j] = weights[j] * scale * np.exp(- 0.5 * np.linalg.solve(covariances[:, :, j], x_m).T.dot(x_m))
            gamma[i, j] = gamma[i, j] / sum_inner

    return [logLikelihood, gamma]
