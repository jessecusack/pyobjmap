import numpy as np

from . import utils


def tile_position(x0, y0, x1=None, y1=None):
    """Need doc string..."""

    if x1 is None and y1 is None:
        x1 = x0
        y1 = y0

    if (x0.size != y0.size) or (x1.size != y1.size):
        raise ValueError("x0 and y0 or x1 and y1 size do not match.")

    x0g = np.tile(x0.ravel()[:, np.newaxis], (1, x1.size))
    y0g = np.tile(y0.ravel()[:, np.newaxis], (1, x1.size))
    x1g = np.tile(x1.ravel()[np.newaxis, :], (x0.size, 1))
    y1g = np.tile(y1.ravel()[np.newaxis, :], (x0.size, 1))

    return x0g, y0g, x1g, y1g


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


def r_distance(x0, y0, x1=None, y1=None, coords="cartesian"):
    """
    Distance matrix.

    If x1 and y1 are not supplied we calculate the auto-distance matrix.
    """
    if coords == "cartesian":
        dx, dy = xy_distance(x0, y0, x1, y1)
        r = np.sqrt(dx ** 2 + dy ** 2)
    elif coords == "latlon":
        r = utils.haversine_distance(*tile_position(x0, y0, x1, y1))

    return r
