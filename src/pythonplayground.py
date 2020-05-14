import sys
import numpy as np
import math as math
import matplotlib.pyplot as plt
import xor
import activation as ac
import error as er
from network import Network

class FunctionContainer:
    pass

def find_fittest():
    data_generator = get_data_generator()
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
    test_propagation(np.argmin, data_generator, network_history, data_size)
    plt.subplot(2, 1, 2)
    plt.title('worst weights')
    test_propagation(np.argmax,  data_generator, network_history, data_size)
    plt.show()

def get_activation():
    return __get_activation(sys.argv[2])

def get_output_activation():
    return __get_activation(sys.argv[3])

def __get_activation(name):
    tanh = (ac.tanh, ac.tanh_der)
    functions = { 'tanh': tanh }
    ft = functions.get(name, tanh)
    function = FunctionContainer()
    function.func, function.der = ft
    return function

def get_data_generator():
    name = sys.argv[1]
    problems = { 'xor': xor.generate_data }
    problem = problems.get(name, xor.generate_data)
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

def test_propagation(cost_selector, data_generator, network_history, data_size):
    i = cost_selector([o.test_cost for o in network_history])
    n = network_history[i]
    train_data = n.problem_data[0]
    _, test_data = data_generator(data_size, 99)
    AL, cost = n.get_cost(test_data[0], test_data[1])
    tr = { 'x': train_data[0][0], 'y': train_data[0][1], 'labels': train_data[1][0]}
    ao = { 'x': test_data[0][0], 'y': test_data[0][1], 'labels': test_data[1][0]}
    print('test propagation cost {0:f}'.format(cost))
    plot_data(tr, ao, AL[0])

def plot_data(train_data, test_data, actual_output):
    X_p, Y_p, X_n, Y_n = split_data_by_sign(train_data['x'], train_data['y'], train_data['labels'])
    plt.plot(X_p, Y_p, 'mo')
    plt.plot(X_n, Y_n, 'co')

    X_p, Y_p, X_n, Y_n = split_data_by_sign(test_data['x'], test_data['y'], test_data['labels'])
    plt.plot(X_p, Y_p, 'ro')
    plt.plot(X_n, Y_n, 'bo')

    X_wrong, Y_wrong = get_wrong_points(test_data, actual_output)
    plt.plot(X_wrong, Y_wrong, 'ko')

def split_data_by_sign(X, Y, labels):
    X_p, Y_p, X_n, Y_n = [], [], [], []

    for l in range(len(labels)):
        if labels[l] > 0:
            X_p.append(X[l])
            Y_p.append(Y[l])
        else:
            X_n.append(X[l])
            Y_n.append(Y[l])

    return X_p, Y_p, X_n, Y_n

def get_wrong_points(test_data, actual_output):
    x, y = [], []
    tl = test_data['labels']

    for p in range(len(tl)):
        if tl[p] * actual_output[p] < 0:
            x.append(test_data['x'][p])
            y.append(test_data['y'][p])
    
    return x, y


find_fittest()
