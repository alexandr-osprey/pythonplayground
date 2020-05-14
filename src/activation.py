import numpy as np

def tanh(Z):
    e2x = np.exp(2 * Z)
    t = (e2x - 1) / (e2x + 1)
    return t

def tanh_der(Z):
    output = tanh(Z)
    der = 1 - output * output
    return der