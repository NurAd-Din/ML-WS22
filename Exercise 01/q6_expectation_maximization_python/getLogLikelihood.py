import numpy as np


def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####

    weights = np.asarray(weights)
    means = np.asarray(means)
    n, d = X.shape
    k, = weights.shape
    logLikelihood = 0

    for i in range(n):
        sum_inner = 0
        for j in range(k):
            x_m = X[i, :] - means[j, :]
            scale = 1 / ((np.sqrt(2 * np.pi) ** d) * np.sqrt(np.linalg.det(covariances[:, :, j])))
            sum_inner += weights[j] * scale * np.exp(- 0.5 * np.linalg.solve(covariances[:, :, j], x_m).T.dot(x_m))
        logLikelihood += np.log(sum_inner)

    return logLikelihood
