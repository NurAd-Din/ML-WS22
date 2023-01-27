import numpy as np

class CrossEntropyLog:

    def __init__(self):
        self.cache = None

    def fprop(self, y, t):
        _, batch = y.shape
        self.cache = np.zeros(y.shape)
        self.cache[t, np.arange(batch)] = - 1
        return - y[t, np.arange(batch)]

    def bprop(self, dE):
        return np.multiply(self.cache.transpose(), dE)

    def update(self, rate):
        pass