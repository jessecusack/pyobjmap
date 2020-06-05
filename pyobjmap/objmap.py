import numpy as np
import scipy.spatial as spat
from . import covariance as cov


def objmap(xd, yd, zd, xm, ym, SNR, l, cfunc='gauss', detrend='mean'):
    """Needs docstring."""

    # Use the covariance function specified.
    cfunc = cov.funccheck(cfunc)

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


def objmap2(xd, yd, zd, xm, ym, SNR, lx, ly, theta=0):
    """Needs docstring."""

    # Use the covariance function specified.
    cfunc = cov.gauss2d

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
    Muu = cov.Cuu(xdist, ydist, 1, l) + np.eye(*xdist.shape)/SNR
    Mvv = cov.Cvv(xdist, ydist, 1, l) + np.eye(*xdist.shape)/SNR
    Muv = cov.Cuv(xdist, ydist, 1, l)

    Cdd = np.vstack((np.hstack((Muu, Muv)), np.hstack((Muv, Mvv))))

    xm = xm.ravel() - xdmid
    ym = ym.ravel() - ydmid
    xmddist = xm[:, np.newaxis] - xd[np.newaxis, :]
    ymddist = ym[:, np.newaxis] - yd[np.newaxis, :]

    Mpsiu = cov.Cpsiu(xmddist, ymddist, 1, l)
    Mpsiv = cov.Cpsiv(xmddist, ymddist, 1, l)

    Cmd = np.hstack((Mpsiu, Mpsiv))

    A, _, _, _ = np.linalg.lstsq(Cdd, phi_obs, rcond=None)

    psi = Cmd @ A

    return psi.reshape(input_shape)