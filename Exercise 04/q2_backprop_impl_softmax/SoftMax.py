import numpy as np

class SoftMax:

    def __init__(self):
        self.cache = None

    def fprop(self, z):
        self.cache = np.exp(z) / np.sum(np.exp(z), axis=0).reshape((1, z.shape[1]))
        return self.cache

    def bprop(self, dE):
        batch, n_out = dE.shape
        zeft = np.empty((batch, 1))
        for i in range(batch):
            zeft[i, :] = np.dot(dE[i, :], self.cache[:, i])
        return np.multiply(self.cache.transpose(), (dE - np.broadcast_to(zeft, (batch, n_out))))

    def update(self, rate):
        pass