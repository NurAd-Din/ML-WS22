import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
import cvxopt


def svmlin(X, t, C):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)


    #####Insert your code here for subtask 2a#####

    N,dim = np.shape(X)
    P = cvxopt.matrix(t.reshape(N, 1).dot(t.reshape(1, N)) * X.dot(X.T))
    q = cvxopt.matrix((-1) * np.ones(N))
    G = cvxopt.matrix(np.vstack([-np.eye(N), np.eye(N)]))
    h = cvxopt.matrix(np.hstack([np.zeros(N), C * np.ones(N)]))
    A = cvxopt.matrix(t.reshape((1, N)))
    b = cvxopt.matrix(np.zeros(1))

    alpha = cvxopt.solvers.qp(P, q, G, h, A, b)

    alpha = np.asarray(alpha['x'])
    alpha = alpha.reshape(N,)
    sv = alpha > 1e-6
    sv = sv.reshape((N,))
    w = (alpha[sv] * t[sv]).dot(X[sv])
    b = np.mean(t[sv] - w.dot(X[sv].T))
    result = w.dot(X.T) + b
    slack = np.abs(t - result) > 1e-6
    slack[sv == False] = False

    return alpha, sv, w, b, result, slack
