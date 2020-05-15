import sys
import numpy as np
import math as math
import matplotlib.pyplot as plt
import xor
import activation as ac
import error as er
from network import Network
import visualization as vs

class FunctionContainer:
    pass

def find_fittest():
    problem = get_data_generator()
    data_generator, _ = problem
    layer_dims = get_layer_dims()
    networks_num = get_networks_num()
    activation = get_activation()
    output_activation = get_output_activation()
    error = get_error()
    epochs = get_epochs()
    learning_rate = get_learning_rate()
    data_size, test_percentage = get_data_size()
    network_history = []
    for n in range(networks_num):
        problem_data = data_generator(data_size, test_percentage)
        network = Network(activation, output_activation, error, layer_dims, problem_data, epochs, learning_rate)
        network.build_and_train()
        network_history.append(network)

    m =  min(network_history, key=lambda n: n.test_cost)
    print('min cost: {0:f}'.format(m.test_cost))
    plt.subplot(2, 1, 1)
    plt.title('best weights')
    vs.test_propagation(np.argmin, problem, network_history, data_size)
    plt.subplot(2, 1, 2)
    plt.title('worst weights')
    vs.test_propagation(np.argmax,  problem, network_history, data_size)
    plt.show()

def get_activation():
    return __get_activation(sys.argv[2])

def get_output_activation():
    return __get_activation(sys.argv[3])

def __get_activation(name):
    tanh = (ac.tanh, ac.tanh_der)
    relu = (ac.relu, ac.relu_der)
    sigmoid = (ac.sigmoid, ac.sigmoid_der)
    functions = { 'tanh': tanh, 'relu': relu, 'sigmoid': sigmoid }
    ft = functions.get(name, tanh)
    function = FunctionContainer()
    function.func, function.der = ft
    return function

def get_data_generator():
    name = sys.argv[1]
    xor_problem = (xor.generate_data, xor.get_wrong_points)
    problems = { 'xor': xor_problem }
    problem = problems.get(name, xor_problem)
    return problem

def get_layer_dims():
    value = sys.argv[5]
    return tuple(map(int, value.split(',')))

def get_networks_num():
    value = sys.argv[6]
    return int(value)

def get_epochs():
    value = sys.argv[7]
    return int(value)

def get_data_size():
    value = sys.argv[8]
    return tuple(map(int, value.split(',')))

def get_learning_rate():
    value = sys.argv[9]
    return float(value)

def get_error():
    square = (er.square_loss, er.square_loss_der, er.square_cost, er.square_cost_der)
    value = sys.argv[4]
    errors = { 'square': square }
    e = errors.get(value, square)
    error = FunctionContainer()
    error.loss, error.loss_der, error.cost, error.cost_der = e
    return error

find_fittest()
