import numpy as np
from kern import kern
import cvxopt


def svmkern(X, t, C, p):
    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                        (num_samples x dim)
    # t        : labeling                           (num_samples x 1)
    # C        : penalty factor the slack variables (scalar)
    # p        : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv       : support vectors (boolean)          (1 x num_samples)
    # b        : bias of the classifier             (scalar)
    # slack    : points inside the margin (boolean) (1 x num_samples)

    #####Insert your code here for subtask 2d#####

    N, dim = np.shape(X)
    P = cvxopt.matrix(t.reshape(N, 1).dot(t.reshape(1, N)) * kern(X.T, X.T, p))
    q = cvxopt.matrix((-1) * np.ones(N))
    G = cvxopt.matrix(np.vstack([-np.eye(N), np.eye(N)]))
    h = cvxopt.matrix(np.hstack([np.zeros(N), C * np.ones(N)]))
    A = cvxopt.matrix(t.reshape((1, N)))
    b = cvxopt.matrix(np.zeros(1))

    alpha = cvxopt.solvers.qp(P, q, G, h, A, b)

    alpha = np.asarray(alpha['x'])
    alpha = alpha.reshape(N, )
    sv = alpha > 1e-8
    sv = sv.reshape((N,))
    b = np.mean(t[sv] - np.sum(t[sv].reshape((np.sum(sv), 1)) * alpha[sv].reshape((np.sum(sv), 1)) * kern(X[sv].T, X[sv].T, p), axis=0))
    result = np.sign(np.sum(t[sv].reshape((np.sum(sv), 1)) * alpha[sv].reshape((np.sum(sv), 1)) * kern(X.T, X[sv].T, p), axis=0) + b)
    slack = alpha > C - 1e-8

    return alpha, sv, b, result, slack
