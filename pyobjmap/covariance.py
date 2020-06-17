import numpy as np
import scipy.optimize as opt
import scipy.stats as stats

from . import matrix as mat


# class Gauss(object):
#     def __init__(self, A, l):
#         self.A = A
#         self.l = l
#     def Cr(self, r)
#         return self.A*np.exp(-0.5*(r/self.l)**2)
#     def


def gauss(r, A, l):
    """Gaussian"""
    return A * np.exp(-0.5 * (r / l) ** 2)


def gauss2d(x, y, A, lx, ly, theta=0, x0=0, y0=0):
    """2D Gaussian with rotation of axis. Rotation in degrees 0 - 360."""
    thetar = np.deg2rad(theta)
    a = np.cos(thetar) ** 2 / (2 * lx ** 2) + np.sin(thetar) ** 2 / (2 * ly ** 2)
    b = -np.sin(2 * thetar) / (4 * lx ** 2) + np.sin(2 * thetar) / (4 * ly ** 2)
    c = np.sin(thetar) ** 2 / (2 * lx ** 2) + np.cos(thetar) ** 2 / (2 * ly ** 2)
    return A * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )


def marko(r, A, l):
    """Exponential"""
    ra = np.abs(r) / l
    return A * (1 + ra) * np.exp(-ra)


def letra(r, A, l):
    ra = np.abs(r) / l
    rsq = ra ** 2
    return A * np.exp(-ra) * (1 + ra + rsq / 6 - ra * rsq / 6)


def funccheck(func):
    if callable(func):
        cfunc = func
    elif func == "gauss":
        cfunc = gauss
    elif func == "marko":
        cfunc = marko
    elif func == "letra":
        cfunc = letra
    elif func == "gauss2d":
        cfunc = gauss2d
    else:
        raise ValueError("func = {} not supported.".format(cov_func))

    return cfunc


def bincovr(x, y, z, bins=10, origin="mean"):

    if origin is None:
        pass
    elif origin == "mean":
        x = x - x.mean()
        y = y - y.mean()
    else:
        raise ValueError("Origin can be mean only for now.")

    # Construct distance matrix.
    R = mat.r_distance(x, y)
    itri, jtri = np.triu_indices_from(R)

    # remove mean before calculating covariance
    zdetrend = z - z.mean()

    # Covariance matrix
    C = np.outer(zdetrend, zdetrend)

    Cr, rbins, _ = stats.binned_statistic(
        R[itri, jtri], C[itri, jtri], statistic="mean", bins=bins
    )

    return rbins, Cr


def bincovxy(x, y, z, bins=10):

    xdist, ydist = mat.xy_distance(x, y)

    # remove mean before calculating covariance
    zdetrend = z - z.mean()

    # Covariance matrix
    C = np.outer(zdetrend, zdetrend)
    itri, jtri = np.triu_indices_from(C)

    Cxy, xbins, ybins, _ = stats.binned_statistic_2d(
        xdist[itri, jtri], ydist[itri, jtri], C[itri, jtri], statistic="mean", bins=bins
    )

    return xbins, ybins, Cxy.T


def bincovxyabs(x, y, z, bins=10):

    xdist, ydist = mat.xy_distance(x, y)

    # remove mean before calculating covariance
    zdetrend = z - z.mean()

    # Covariance matrix
    C = np.outer(zdetrend, zdetrend)
    itri, jtri = np.triu_indices_from(C)

    Cxy, xbins, ybins, _ = stats.binned_statistic_2d(
        xdist[itri, jtri], ydist[itri, jtri], C[itri, jtri], statistic="mean", bins=bins
    )

    return xbins, ybins, Cxy.T


def bincovxyuv(x, y, u, v, bins=10):

    xdist, ydist = mat.xy_distance(x, y)

    # remove mean before calculating covariance
    udetrend = u - u.mean()
    vdetrend = v - v.mean()

    # Covariance matrix
    C = np.outer(udetrend, vdetrend)
    itri, jtri = np.triu_indices_from(C)

    Cxy, xbins, ybins, _ = stats.binned_statistic_2d(
        xdist[itri, jtri], ydist[itri, jtri], C[itri, jtri], statistic="mean", bins=bins
    )

    return xbins, ybins, Cxy.T


def covfit(x, y, z, bins=10, cfunc="gauss", p0=[1, 1], rfitmax=None):

    cfunc = funccheck(cfunc)

    rbins, Cr = bincovr(x, y, z, bins=bins)
    r = 0.5 * (rbins[1:] + rbins[:-1])

    if rfitmax is None:
        raise ValueError("rfitmax cannot be None.")

    infit = r <= rfitmax

    popt, _ = opt.curve_fit(cfunc, r[infit], Cr[infit], p0=p0)

    return popt


# Gaussian covariance functions for velocity and streamfunction
def Cuu(x, y, A, l):
    r = np.sqrt(x ** 2 + y ** 2)
    return A * (l ** 2 - r ** 2 + x ** 2) * np.exp(-0.5 * r ** 2 / l ** 2) / l ** 4


def Cvv(x, y, A, l):
    r = np.sqrt(x ** 2 + y ** 2)
    return A * (l ** 2 - r ** 2 + y ** 2) * np.exp(-0.5 * r ** 2 / l ** 2) / l ** 4


def Cuv(x, y, A, l):
    r = np.sqrt(x ** 2 + y ** 2)
    return A * x * y * np.exp(-0.5 * r ** 2 / l ** 2) / l ** 4


def Cpsiu(x, y, A, l):
    r = np.sqrt(x ** 2 + y ** 2)
    return A * y * np.exp(-0.5 * r ** 2 / l ** 2) / l ** 2


def Cpsiv(x, y, A, l):
    r = np.sqrt(x ** 2 + y ** 2)
    return -A * x * np.exp(-0.5 * r ** 2 / l ** 2) / l ** 2


# def C(x, y, A, l):
#     r = np.sqrt(x**2 + y**2)
#     return A*np.exp(-0.5*r**2/l**2)

# def R(x, y, A, l):
#     r = np.sqrt(x**2 + y**2)
#     return A*np.exp(-0.5*r**2/l**2)/l**2

# def S(x, y, A, l):
#     r = np.sqrt(x**2 + y**2)
#     return A*(l**2 - r**2)*np.exp(-0.5*r**2/l**2)/l**4
