import numpy as np
import math

def get_random_uniform(r, c, a, b):
    points = np.random.rand(r, c)
    points = points * (b - a) + a
    return points

def split_data(points, labels, test_part):
    size = points.shape[1]
    part = math.floor(size * test_part / 100)
    train = points[:, part:], labels[:, part:]
    test = points[:, :part], labels[:, :part]
    return train, test