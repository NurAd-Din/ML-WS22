import numpy as np

def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    pos = np.arange(-5, 5.0, 0.1)
    estDensity = np.zeros((100, 2))

    # estDensity[:, 1] = pos[:]
    # for i in np.range(100):
    #     center = pos[i]
    #     k = 0
    #     for x in samples:
    #         if np.abs(x-center) <= h:
    #             k = k + 1
    #     estDensity[i, 2] = k / 100 / h  
          



    return estDensity
