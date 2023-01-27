import numpy as np
from numpy.random import choice

from simpleClassifier import simpleClassifier
def adaboostSimple(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar) 
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK 	: voting weights (K x 1) - for each round
    # para		: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta

    #####Insert your code here for subtask 1c#####

    numSamples, numDim = X.shape
    alphaK = np.zeros(K)
    para = np.zeros((K, 2))

    weights = (1 / numSamples) * np.ones(numSamples)

    for k in range(K):
        idx = np.random.choice(numSamples, nSamples, p=weights)
        x_train = X[idx, :]
        y_train = Y[idx]
        j, theta = simpleClassifier(x_train, y_train)
        result = np.where(X[:, j] < theta, -1, 1)
        p = 1
        yy = np.abs((result - Y.reshape((numSamples,))))
        result1 = weights.dot(yy)
        result2 = weights.dot(np.abs((result + Y.reshape((numSamples,)))))
        error = result1
        if result2 < result1:
            result = -result
            error = result2
            p = -1
        error = error / ((np.sum(weights)) * 2)
        if error == 0:
            break
        para[k, 0] = j
        para[k, 1] = theta
        alphaK[k] = p * np.log((1 - error) / error) * 0.5
        weights = weights * np.exp(- alphaK[k] * result * Y.reshape((numSamples,)))
        weights = weights / (np.sum(weights))

    return alphaK, para
