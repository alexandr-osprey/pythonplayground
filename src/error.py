import numpy as np

def square_loss(AL, Y):
    loss = np.sum(0.5 * np.power(AL - Y, 2))
    return loss

def square_cost(AL, Y):
    loss = np.sum(square_loss(AL, Y))
    cost = loss / Y.shape[1]
    return cost

def square_loss_der(AL, Y):
    return AL - Y

def square_cost_der(AL, Y):
    return square_loss_der(AL, Y) / Y.shape[1]
