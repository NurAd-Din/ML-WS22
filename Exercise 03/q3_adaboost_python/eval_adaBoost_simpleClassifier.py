import numpy as np


def eval_adaBoost_simpleClassifier(X, alphaK, para):
    # INPUT:
    # para	: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (numSamples x 1)

    #####Insert your code here for subtask 1c#####

    result = np.where(X[:, para[:, 0].astype(int)] < para[:, 1], -1, 1)
    result = alphaK.dot(result.T)
    classLabels = np.sign(result)

    return classLabels, result
