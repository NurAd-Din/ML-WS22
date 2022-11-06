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
    # one = np.ones((K, D, K))
    # means = means * one
    # means = np.swapaxes(means, 0, 1)
    # scale = (1 / (np.sqrt(2 * np.pi) ** D) * np.sqrt(np.linalg.det(covariances)))
    # gaussian = scale * np.exp(- 0.5 * np.matmul(np.transpose((X - means), axes=(0, 2, 1)), np.matmul(np.linalg.inv(covariances), (X - means))))

    logLikelihood = 0
    sum_inner = 0

    for i in range(n):
        sum_inner = 0
        for j in range(k):
            scale = 1 / ((np.sqrt(2 * np.pi) ** d) * np.sqrt(np.linalg.det(covariances[:, :, j])))
            sum_inner += weights[j] * scale * np.exp(- 0.5 * np.matmul((X[i, :] - means[j, :]), np.matmul(np.linalg.inv(covariances[:, :, j]), np.transpose(X[i, :] - means[j, :]))))

        logLikelihood += np.log(sum_inner)

    return logLikelihood

