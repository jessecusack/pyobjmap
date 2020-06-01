import numpy as np
import scipy.io as io
import scipy.spatial as spat
import scipy.stats as stats
import scipy.optimize as opt


def gauss(r, a, l):
    return a*np.exp(-0.5*(r/l)**2)


def marko(r, a, l):
    ra = np.abs(r)/l
    return a*(1 + ra)*np.exp(-ra)


def letra(r, a, l):
    ra = np.abs(r)/l
    rsq = ra**2
    return a*np.exp(-ra)*(1 + ra + rsq/6 - ra*rsq/6)


def funccheck(func):
    if callable(func):
        cfunc = func
    elif func == 'gauss':
        cfunc = gauss
    elif func == 'marko':
        cfunc = marko
    elif func == 'letra':
        cfunc = letra
    else:
        raise ValueError('func = {} not supported.'.format(cov_func))

    return cfunc


def bincovr(x, y, z, bins=10, origin='mean'):

    if origin == 'mean':
        x = x - x.mean()
        y = y - y.mean()
    else:
        raise ValueError('Origin can be mean only for now.')

    # Construct distance matrix.
    xy = np.stack((x, y), 1)
    R = spat.distance.cdist(xy, xy)
    itri, jtri = np.triu_indices_from(R)

    # remove mean before calculating covariance
    zdetrend = z - z.mean()

    # Covariance matrix
    C = np.outer(zdetrend, zdetrend)

    # TODO: This seems unncessary because binned_statistic must formulate its own bin sizes.
    # Make bins
    if isinstance(bins, int):
        rmax = R.max()
        rbins = np.linspace(0, rmax, bins+1)
    elif np.iterable(bins):
        rbins = np.asarray(bins)
    else:
        raise ValueErrror('Bins not specified correctly.')

    Cr, _, _ = stats.binned_statistic(R[itri, jtri], C[itri, jtri], statistic='mean', bins=rbins)

    return rbins, Cr


def covfit(x, y, z, bins=10, cfunc='gauss', p0=[1, 1], rfitmax=None):

    cfunc = funccheck(cfunc)

    rbins, Cr = bincovr(x, y, z, bins=bins)
    r = 0.5*(rbins[1:] + rbins[:-1])

    if rfitmax is None:
        raise ValueError("rfitmax cannot be None.")

    infit = r <= rfitmax

    popt, _ = opt.curve_fit(cfunc, r[infit], Cr[infit], p0=p0)

    return popt


def objmap(xd, yd, zd, xm, ym, a, l, cfunc='gauss', detrend='mean'):
    """Needs docstring."""

    # Use the covariance function specified.
    cfunc = funccheck(cfunc)

    # Detrend the data.
    if detrend == "mean":
        ztrend = zd.mean()
    else:
        raise ValueError("detrend = {}, is not available.".format(detrend))

    zdetrend = zd - ztrend

    # Data - data covariance matrix.
    C = np.outer(zdetrend, zdetrend)

    # Construct data - data distance matrix in coordinate system where
    # zero is at the centre of the data.
    # TODO: is it really necessary to move coordinates to origin? Probably not...
    xdmid = np.mean(xd)
    ydmid = np.mean(yd)
    xyd = np.stack((xd - xdmid, yd - ydmid), 1)
    Rdd = spat.distance.cdist(xyd, xyd)

    # Data - data covarance matrix using the function.
    Cdd0 = cfunc(Rdd, 1, l)
    # Add variance back in.
    lam = (C.diagonal().mean() - a)/a
    Cdd = Cdd0 + np.eye(*Cdd0.shape)*lam

    # Construct model - data distance matrix.
    xym = np.stack((xm.ravel() - xdmid, ym.ravel() - ydmid), 1)
    Rmd = spat.distance.cdist(xym, xyd)

    # Construct the model - data covariance matrix.
    Cmd = cfunc(Rmd, 1, l)

    # Do the objective mapping.
    A, _, _, _ = np.linalg.lstsq(Cdd, zdetrend, rcond=None)

    zmg = zdmean + (Cmd @ A).reshape(xm.shape)

    return zmg


def fit_plane(x, y, z):
    # Credit: amroamroamro gist on github
    A = np.c_[x, y, np.ones(z.size)]
    C, _, _, _ = scipy.linalg.lstsq(A, z)    # coefficients

#     # evaluate it on grid
#     Z = C[0]*X + C[1]*Y + C[2]
    return C