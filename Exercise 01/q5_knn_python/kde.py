import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####

    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector

    N, = samples.shape
    norm = N * np.sqrt(2 * np.pi) * h
    estDensity = np.sum(np.exp(-(((pos[np.newaxis, :] - samples[:, np.newaxis]) ** 2)/ (2 * (h ** 2)))), axis=0) / norm

    estDensity = np.stack((pos, estDensity), axis=1)

    # Compute the number of samples created
    return estDensity
