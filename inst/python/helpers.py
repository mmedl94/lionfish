import numpy as np


def gram_schmidt(v1, v2):
    return v2-np.multiply((np.dot(v2, v1)/np.dot(v1, v1)), v1)
