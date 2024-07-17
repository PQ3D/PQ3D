import math
import torch


def rotate_around_z_axis(points, theta):
    # Rotate point around the z axis
    results = torch.zeros_like(points)
    results[..., 2] = points[..., 2]
    results[..., 0] = math.cos(theta) * points[..., 0] - math.sin(theta) * points[..., 1]
    results[..., 1] = math.sin(theta) * points[..., 0] + math.cos(theta) * points[..., 1]
    return results