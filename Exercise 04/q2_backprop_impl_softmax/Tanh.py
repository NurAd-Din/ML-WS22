import numpy as np

class Tanh:

    def __init__(self):
        self.cache = None

    def fprop(self, z):
        self.cache = 1 - np.power(np.tanh(z), 2)
        return np.tanh(z)

    def bprop(self, dE):
        return np.multiply(dE, self.cache.transpose())

    def update(self, rate):
        pass