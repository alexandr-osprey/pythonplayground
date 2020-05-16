import numpy as np
import datasets as ds
import math

def generate_data(size, test_part):
    radius = 5
    noise = 0
    positive_num = int(size / 2)
    positive_data = _generate_region(positive_num, radius, noise, 0, radius * 0.5)
    negative_data = _generate_region(positive_num, radius, noise, radius, radius * 0.7)
    return positive_data, negative_data


def _generate_region(size, radius, noise, a, b):
    r = ds.get_random_uniform(1, size, a, b)
    angle = ds.get_random_uniform(1, size, 0, 2 * math.pi)
    x = r * np.sin(angle)
    y = r * np.cos(angle)
    noise_x = ds.get_random_uniform(1, size, -radius, radius) * noise
    noise_y = ds.get_random_uniform(1, size, -radius, radius) * noise
    labels = _get_label((x + noise_x, y + noise_y), (0, 0), radius)
    d = np.vstack((x, y))
    return d, labels

def _get_label(point, center, radius):
    v = (_dist(point, center) >= (radius * 0.5)).astype(int)
    l = pow(-1, v)
    return l

def _dist(point_a, point_b):
    ax, ay = point_a
    bx, by = point_b
    dx = ax - bx
    dy = ay - by
    return np.sqrt(dx * dx + dy * dy)

def get_wrong_points(test_data, actual_output):
    x, y = [], []
    tl = test_data['labels']

    for p in range(len(tl)):
        if tl[p] * actual_output[p] < 0:
            x.append(test_data['x'][p])
            y.append(test_data['y'][p])
    
    return x, y