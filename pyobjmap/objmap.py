import numpy as np
import scipy.spatial as spat
import scipy.stats as stats
import scipy.optimize as opt


def gauss(r, A, l):
    """Gaussian"""
    return A*np.exp(-0.5*(r/l)**2)


def gauss2d(x, y, A, lx, ly, theta=0, x0=0, y0=0):
    """2D Gaussian with rotation of axis. Rotation in degrees 0 - 360."""
    thetar = np.deg2rad(theta)
    a = np.cos(thetar)**2/(2*lx**2) + np.sin(thetar)**2/(2*ly**2)
    b = -np.sin(2*thetar)/(4*lx**2) + np.sin(2*thetar)/(4*ly**2)
    c = np.sin(thetar)**2/(2*lx**2) + np.cos(thetar)**2/(2*ly**2)
    return A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))


def marko(r, A, l):
    ra = np.abs(r)/l
    return A*(1 + ra)*np.exp(-ra)


def letra(r, A, l):
    ra = np.abs(r)/l
    rsq = ra**2
    return A*np.exp(-ra)*(1 + ra + rsq/6 - ra*rsq/6)


def funccheck(func):
    if callable(func):
        cfunc = func
    elif func == 'gauss':
        cfunc = gauss
    elif func == 'marko':
        cfunc = marko
    elif func == 'letra':
        cfunc = letra
    elif func == 'gauss2d':
        cfunc = gauss2d
    else:
        raise ValueError('func = {} not supported.'.format(cov_func))

    return cfunc


def bincovr(x, y, z, bins=10, origin='mean'):

    if origin is None:
        pass
    elif origin == 'mean':
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

    Cr, rbins, _ = stats.binned_statistic(R[itri, jtri], C[itri, jtri], statistic='mean', bins=bins)

    return rbins, Cr


def bincovxy(x, y, z, bins=10):

    # x distance matrix
    xdist = x[:, np.newaxis] - x[np.newaxis, :]
    # y distance matrix
    ydist = y[:, np.newaxis] - y[np.newaxis, :]

    # remove mean before calculating covariance
    zdetrend = z - z.mean()

    # Covariance matrix
    C = np.outer(zdetrend, zdetrend)
    itri, jtri = np.triu_indices_from(C)

    Cxy, xbins, ybins, _ = stats.binned_statistic_2d(xdist[itri, jtri], ydist[itri, jtri], C[itri, jtri], statistic='mean', bins=bins)

    return xbins, ybins, Cxy.T


def bincovxyabs(x, y, z, bins=10):

    # x distance matrix
    xdist = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
    # y distance matrix
    ydist = np.abs(y[:, np.newaxis] - y[np.newaxis, :])

    # remove mean before calculating covariance
    zdetrend = z - z.mean()

    # Covariance matrix
    C = np.outer(zdetrend, zdetrend)
    itri, jtri = np.triu_indices_from(C)

    Cxy, xbins, ybins, _ = stats.binned_statistic_2d(xdist[itri, jtri], ydist[itri, jtri], C[itri, jtri], statistic='mean', bins=bins)

    return xbins, ybins, Cxy.T


def bincovxyuv(x, y, u, v, bins=10):

    # x distance matrix
    xdist = x[:, np.newaxis] - x[np.newaxis, :]
    # y distance matrix
    ydist = y[:, np.newaxis] - y[np.newaxis, :]

    # remove mean before calculating covariance
    udetrend = u - u.mean()
    vdetrend = v - v.mean()

    # Covariance matrix
    C = np.outer(udetrend, vdetrend)
    itri, jtri = np.triu_indices_from(C)

    Cxy, xbins, ybins, _ = stats.binned_statistic_2d(xdist[itri, jtri], ydist[itri, jtri], C[itri, jtri], statistic='mean', bins=bins)

    return xbins, ybins, Cxy.T


def covfit(x, y, z, bins=10, cfunc='gauss', p0=[1, 1], rfitmax=None):

    cfunc = funccheck(cfunc)

    rbins, Cr = bincovr(x, y, z, bins=bins)
    r = 0.5*(rbins[1:] + rbins[:-1])

    if rfitmax is None:
        raise ValueError("rfitmax cannot be None.")

    infit = r <= rfitmax

    popt, _ = opt.curve_fit(cfunc, r[infit], Cr[infit], p0=p0)

    return popt


def objmap(xd, yd, zd, xm, ym, SNR, l, cfunc='gauss', detrend='mean'):
    """Needs docstring."""

    # Use the covariance function specified.
    cfunc = funccheck(cfunc)

    # Detrend the data.
    if detrend is None:
        ztrend = 0
    elif detrend == "mean":
        ztrend = zd.mean()
    elif detrend == "plane":
        pcoef = plane_coeff(xd, yd, zd)
        ztrend = pcoef[0]*xd + pcoef[1]*yd + pcoef[2]
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
    Cdd = Cdd0 + np.eye(*Cdd0.shape)/SNR

    # Construct model - data distance matrix.
    xym = np.stack((xm.ravel() - xdmid, ym.ravel() - ydmid), 1)
    Rmd = spat.distance.cdist(xym, xyd)

    # Construct the model - data covariance matrix.
    Cmd = cfunc(Rmd, 1, l)

    # Do the objective mapping.
    A, _, _, _ = np.linalg.lstsq(Cdd, zdetrend, rcond=None)

    zmg = (Cmd @ A).reshape(xm.shape)

    # Add trend back to result.
    if detrend == "mean":
        zmg += ztrend
    elif detrend == "plane":
        zmg += pcoef[0]*xm + pcoef[1]*ym + pcoef[2]

    return zmg


def plane_coeff(x, y, z):
    # Credit: amroamroamro gist on github
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    A = np.c_[x.ravel(), y.ravel(), np.ones(z.ravel().size)]
    C, _, _, _ = np.linalg.lstsq(A, z.ravel(), rcond=None)  # coefficients
    return C


def objmap2(xd, yd, zd, xm, ym, SNR, lx, ly, theta=0):
    """Needs docstring."""

    # Use the covariance function specified.
    cfunc = gauss2d

    ztrend = zd.mean()

    zdetrend = zd - ztrend

    # Data - data covariance matrix.
    C = np.outer(zdetrend, zdetrend)

    # Construct data - data distance matrix in coordinate system where
    # zero is at the centre of the data.
    # TODO: is it really necessary to move coordinates to origin? Probably not...
    xdmid = np.mean(xd)
    ydmid = np.mean(yd)
    xd = xd - xdmid
    yd = yd - ydmid

    xdist = xd[:, np.newaxis] - xd[np.newaxis, :]
    ydist = yd[:, np.newaxis] - yd[np.newaxis, :]

    # Data - data covarance matrix using the function.
    Cdd0 = cfunc(xdist, ydist, 1, lx, ly, theta)
    # Add variance back in.
    Cdd = Cdd0 + np.eye(*Cdd0.shape)/SNR

    # Construct model - data distance matrix.
    xm = xm - xdmid
    ym = ym - ydmid
    xmddist = xm.ravel()[:, np.newaxis] - xd.ravel()[np.newaxis, :]
    ymddist = ym.ravel()[:, np.newaxis] - yd.ravel()[np.newaxis, :]

    # Construct the model - data covariance matrix.
    Cmd = cfunc(xmddist, ymddist, 1, lx, ly, theta)

    # Do the objective mapping.
    A, _, _, _ = np.linalg.lstsq(Cdd, zdetrend, rcond=None)

    zmg = (Cmd @ A).reshape(xm.shape)

    zmg += ztrend

    return zmg


def objmap_streamfunc(xd, yd, ud, vd, xm, ym, l, SNR):
    """Map velocity observations to non-divergent streamfunction"""
    input_shape = xm.shape

    ud = ud - ud.mean()
    vd = vd - vd.mean()

    phi_obs = np.hstack((ud, vd))[:, np.newaxis]  # Column vector...

    # Data data distances
    # TODO: I don't think subtracting the mean is strictly necessary if we're
    # always working with distance differences...
    xdmid = np.mean(xd)
    ydmid = np.mean(yd)
    xd = xd - xdmid
    yd = yd - ydmid

    xdist = xd[:, np.newaxis] - xd[np.newaxis, :]
    ydist = yd[:, np.newaxis] - yd[np.newaxis, :]

    # Data - data covarance matrix
    Muu = Cuu(xdist, ydist, 1, l) + np.eye(*xdist.shape)/SNR
    Mvv = Cvv(xdist, ydist, 1, l) + np.eye(*xdist.shape)/SNR
    Muv = Cuv(xdist, ydist, 1, l)

    Cdd = np.vstack((np.hstack((Muu, Muv)), np.hstack((Muv, Mvv))))

    xm = xm.ravel() - xdmid
    ym = ym.ravel() - ydmid
    xmddist = xm[:, np.newaxis] - xd[np.newaxis, :]
    ymddist = ym[:, np.newaxis] - yd[np.newaxis, :]

    Mpsiu = Cpsiu(xmddist, ymddist, 1, l)
    Mpsiv = Cpsiv(xmddist, ymddist, 1, l)

    Cmd = np.hstack((Mpsiu, Mpsiv))

    A, _, _, _ = np.linalg.lstsq(Cdd, phi_obs, rcond=None)

    psi = Cmd @ A

    return psi.reshape(input_shape)

# Gaussian covariance functions for velocity and streamfunction
def Cuu(x, y, A, l):
    r = np.sqrt(x**2 + y**2)
    return A*(l**2 - r**2 + x**2)*np.exp(-0.5*r**2/l**2)/l**4


def Cvv(x, y, A, l):
    r = np.sqrt(x**2 + y**2)
    return A*(l**2 - r**2 + y**2)*np.exp(-0.5*r**2/l**2)/l**4


def Cuv(x, y, A, l):
    r = np.sqrt(x**2 + y**2)
    return A*x*y*np.exp(-0.5*r**2/l**2)/l**4


def Cpsiu(x, y, A, l):
    r = np.sqrt(x**2 + y**2)
    return A*y*np.exp(-0.5*r**2/l**2)/l**2


def Cpsiv(x, y, A, l):
    r = np.sqrt(x**2 + y**2)
    return -A*x*np.exp(-0.5*r**2/l**2)/l**2


# def C(x, y, A, l):
#     r = np.sqrt(x**2 + y**2)
#     return A*np.exp(-0.5*r**2/l**2)

# def R(x, y, A, l):
#     r = np.sqrt(x**2 + y**2)
#     return A*np.exp(-0.5*r**2/l**2)/l**2

# def S(x, y, A, l):
#     r = np.sqrt(x**2 + y**2)
#     return A*(l**2 - r**2)*np.exp(-0.5*r**2/l**2)/l**4