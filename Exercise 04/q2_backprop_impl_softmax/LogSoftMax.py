import numpy as np

class LogSoftMax:

    def __init__(self):
        self.cache = None

    def fprop(self, z):
        z -= np.amax(z, axis=0).reshape((1, z.shape[1]))
        self.cache = np.exp(z) / np.sum(np.exp(z), axis=0).reshape((1, z.shape[1]))
        return np.log(self.cache)

    def bprop(self, dE):
        return dE - self.cache.transpose() * np.broadcast_to(np.sum(dE, axis=1).reshape(dE.shape[0], 1), dE.shape)

    def update(self, rate):
        pass