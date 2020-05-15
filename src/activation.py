import numpy as np

def tanh(Z):
    e2x = np.exp(2 * Z)
    t = (e2x - 1) / (e2x + 1)
    return t

def tanh_der(Z):
    output = tanh(Z)
    der = 1 - output * output
    return der

def relu(Z):
    positive = (Z > 0).astype(int)
    output = positive * Z
    return output

def relu_der(Z):
    m = (Z > 0).astype(int)
    der = (1 * m).reshape(Z.shape)
    return der

def sigmoid(Z):
    m = 1 + np.exp(-Z)
    output = 1 / m
    return output

def sigmoid_der(Z):
    m = sigmoid(Z)
    output = m * (1 - m)
    return output
