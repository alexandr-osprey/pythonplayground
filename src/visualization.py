import numpy as np
import matplotlib.pyplot as plt

def test_propagation(cost_selector, problem, network_history, data_size):
    data_generator, wrong_points_selector = problem
    i = cost_selector([o.test_cost for o in network_history])
    n = network_history[i]
    train_data = n.problem_data[0]
    _, test_data = data_generator(data_size, 99)
    AL, cost = n.get_cost(test_data[0], test_data[1])
    tr = { 'x': train_data[0][0], 'y': train_data[0][1], 'labels': train_data[1][0]}
    ao = { 'x': test_data[0][0], 'y': test_data[0][1], 'labels': test_data[1][0]}
    print('test propagation cost {0:f}'.format(cost))
    plot_data(wrong_points_selector, tr, ao, AL[0])

def plot_data(wrong_points_selector, train_data, test_data, actual_output):
    X_p, Y_p, X_n, Y_n = split_data_by_sign(train_data['x'], train_data['y'], train_data['labels'])
    plt.plot(X_p, Y_p, 'mo')
    plt.plot(X_n, Y_n, 'co')

    X_p, Y_p, X_n, Y_n = split_data_by_sign(test_data['x'], test_data['y'], test_data['labels'])
    plt.plot(X_p, Y_p, 'ro')
    plt.plot(X_n, Y_n, 'bo')

    X_wrong, Y_wrong = wrong_points_selector(test_data, actual_output)
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
