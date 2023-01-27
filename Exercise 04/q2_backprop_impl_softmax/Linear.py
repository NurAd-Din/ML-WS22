import numpy as np

class Linear:

    def __init__(self, input, output):
        self.W = np.random.normal(0, np.sqrt(2 / (input + output)), (input, output))
        self.b = np.zeros((output, 1))
        self.cache = None
        self.dW = None
        self.db = None

    def fprop(self, x):
        self.cache = x
        return np.matmul(self.W.transpose(), x) + self.b

    def bprop(self, dE):
        self.dW = np.matmul(self.cache, dE)
        self.db = np.sum(dE, axis=0).reshape(self.b.shape)
        return np.matmul(dE, self.W.transpose())

    def update(self, rate):
        self.W = self.W - rate * self.dW
        self.b = self.b - rate * self.db