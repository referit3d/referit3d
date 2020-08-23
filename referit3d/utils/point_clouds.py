import numpy as np


def rotate_z_axis_by_degrees(pointcloud, theta, clockwise=True):
    theta = np.deg2rad(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_matrix = np.array([[cos_t, -sin_t, 0],
                           [sin_t, cos_t, 0],
                           [0, 0, 1]], pointcloud.dtype)
    if not clockwise:
        rot_matrix = rot_matrix.T
    return pointcloud.dot(rot_matrix)


def zero_mean_in_unit_sphere(pc, in_place=True):
    if not in_place:
        pc = pc.copy()
    center_of_mass = np.mean(pc, axis=0)
    pc -= center_of_mass
    largest_distance = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc /= largest_distance
    return pc


def center_in_unit_sphere(pc, in_place=True):
    if not in_place:
        pc = pc.copy()

    for axis in range(3):  # center around each axis
        r_max = np.max(pc[:, axis])
        r_min = np.min(pc[:, axis])
        gap = (r_max + r_min) / 2.0
        pc[:, axis] -= gap

    largest_distance = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc /= largest_distance
    return pc


def uniform_sample(points, n_samples, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    replace = False
    if n_samples > len(points):
        replace = True
    idx = np.random.choice(len(points), n_samples, replace=replace)

    if random_seed is not None:
        np.random.seed(None)

    return points[idx]
