import numpy as np


def xy_distance(x0, y0, x1=None, y1=None):
    """
    Output x and y distance matrices.

    If x1 and y1 are not supplied we calculate the auto-distance matrices.
    """

    if x1 is None and y1 is None:
        x1 = x0
        y1 = y0

    dx = x0.ravel()[:, np.newaxis] - x1.ravel()[np.newaxis, :]
    dy = y0.ravel()[:, np.newaxis] - y1.ravel()[np.newaxis, :]

    return dx, dy


def r_distance(x0, y0, x1=None, y1=None):
    """
    Distance matrix.

    If x1 and y1 are not supplied we calculate the auto-distance matrix.
    """
    dx, dy = xy_distance(x0, y0, x1, y1)
    return np.sqrt(dx**2 + dy**2)