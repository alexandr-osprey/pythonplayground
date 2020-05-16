import numpy as np
import math as math
import datasets as ds

def generate_data(size, test_part):
    points = ds.get_random_uniform(2, size, -5, 5)
    points = _add_padding(points, 0.3)
    labels = _get_xor_labels(points)
    train, test = ds.split_data(points, labels, test_part)
    return train, test

def _add_padding(points, padding):
    p = (points <= 0).astype(int)
    p = padding * pow(-1, p)
    points = points + p
    return points

def _get_xor_labels(points):
    size = points.shape[1]
    x1 = points[0, :]
    x2 = points[1, :]
    l = ((x1 * x2) > 0).astype(int)
    labels = pow(-1, l).reshape(1, size)
    return labels

def get_wrong_points(test_data, actual_output):
    x, y = [], []
    tl = test_data['labels']

    for p in range(len(tl)):
        if tl[p] * actual_output[p] < 0:
            x.append(test_data['x'][p])
            y.append(test_data['y'][p])
    
    return x, y

# def normalize_rows(self, x):
#     norm = np.linalg.norm(x, axis=1, keepdims=True)
#     normalized = x / norm
#     return normalized