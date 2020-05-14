import numpy as np
import math as math

def generate_data(size, test_part):
    points = _get_rand_uniform(2, size, -5, 5)
    points = _add_padding(points, 0.3)
    labels = _get_xor_labels(points)
    train, test = _split_data(points, labels, test_part)
    return train, test

def _get_rand_uniform(r, c, a, b):
    points = np.random.rand(r, c)
    points = points * (b - a) + a
    return points

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

def _split_data(points, labels, test_part):
    size = points.shape[1]
    part = math.floor(size * test_part / 100)
    train = points[:, part:], labels[:, part:]
    test = points[:, :part], labels[:, :part]
    return train, test


# def normalize_rows(self, x):
#     norm = np.linalg.norm(x, axis=1, keepdims=True)
#     normalized = x / norm
#     return normalized