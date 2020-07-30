import numpy as np

from . import covariance as cov
from . import matrix as mat
from . import utils


def relerr(Cdd, Cmd):
    """Relative error."""
    A, _, _, _ = np.linalg.lstsq(Cdd, Cmd.T, rcond=None)
    return 1 - np.diag(Cmd @ A)


def objmap(
    xd,
    yd,
    zd,
    xm,
    ym,
    SNR,
    l,
    cfunc="gauss",
    detrend="mean",
    coords="cartesian",
    return_err=False,
):
    """Needs docstring."""

    xd = np.asarray(xd).ravel()
    yd = np.asarray(yd).ravel()
    zd = np.asarray(zd).ravel()
    xm = np.asarray(xm)
    ym = np.asarray(ym)

    # Use the covariance function specified.
    cfunc = cov.funccheck(cfunc)

    # Detrend the data.
    if detrend is None:
        ztrend = 0
    elif detrend == "mean":
        ztrend = zd.mean()
    elif detrend == "plane":
        pcoef = utils.plane_coeff(xd, yd, zd)
        ztrend = pcoef[0] * xd + pcoef[1] * yd + pcoef[2]
    else:
        raise ValueError("detrend = {}, is not available.".format(detrend))

    zdetrend = zd - ztrend

    # Data - data covariance matrix.
    C = np.outer(zdetrend, zdetrend)

    # Construct data - data distance matrix in coordinate system where
    # zero is at the centre of the data.
    Rdd = mat.r_distance(xd, yd, coords=coords)

    # Data - data covarance matrix using the function.
    Cdd0 = cfunc(Rdd, 1, l)
    # Add variance back in.
    Cdd = Cdd0 + np.eye(*Cdd0.shape) / SNR

    # Construct model - data distance matrix.
    Rmd = mat.r_distance(xm, ym, xd, yd, coords=coords)

    # Construct the model - data covariance matrix.
    Cmd = cfunc(Rmd, 1, l)

    # Do the objective mapping.
    A, _, _, _ = np.linalg.lstsq(Cdd, zdetrend, rcond=None)

    zmg = (Cmd @ A).reshape(xm.shape)

    # Add trend back to result.
    if detrend == "mean":
        zmg += ztrend
    elif detrend == "plane":
        zmg += pcoef[0] * xm + pcoef[1] * ym + pcoef[2]

    if return_err:
        err = relerr(Cdd, Cmd).reshape(xm.shape)
        return zmg, err
    else:
        return zmg


def objmap2(xd, yd, zd, xm, ym, SNR, lx, ly, theta=0, return_err=False):
    """Needs docstring."""

    xd = np.asarray(xd).ravel()
    yd = np.asarray(yd).ravel()
    zd = np.asarray(zd).ravel()
    xm = np.asarray(xm)
    ym = np.asarray(ym)

    # Use the covariance function specified.
    cfunc = cov.gauss2d

    ztrend = zd.mean()

    zdetrend = zd - ztrend

    # Data - data covariance matrix.
    C = np.outer(zdetrend, zdetrend)

    # Construct data - data distance matrix in coordinate system where
    # zero is at the centre of the data.
    xdist, ydist = mat.xy_distance(xd, yd)

    # Data - data covarance matrix using the function.
    Cdd0 = cfunc(xdist, ydist, 1, lx, ly, theta)
    # Add variance back in.
    Cdd = Cdd0 + np.eye(*Cdd0.shape) / SNR

    # Construct model - data distance matrix.
    xmddist, ymddist = mat.xy_distance(xm, ym, xd, yd)

    # Construct the model - data covariance matrix.
    Cmd = cfunc(xmddist, ymddist, 1, lx, ly, theta)

    # Do the objective mapping.
    A, _, _, _ = np.linalg.lstsq(Cdd, zdetrend, rcond=None)

    zmg = (Cmd @ A).reshape(xm.shape)

    zmg += ztrend

    if return_err:
        err = relerr(Cdd, Cmd).reshape(xm.shape)
        return zmg, err
    else:
        return zmg


def objmap_streamfunc(xd, yd, ud, vd, xm, ym, l, SNR, return_err=False):
    """Map velocity observations to non-divergent streamfunction"""
    xd = np.asarray(xd).ravel()
    yd = np.asarray(yd).ravel()
    ud = np.asarray(ud).ravel()
    vd = np.asarray(vd).ravel()
    xm = np.asarray(xm)
    ym = np.asarray(ym)

    input_shape = xm.shape

    udmean = ud.mean()
    vdmean = vd.mean()
    ud = ud - udmean
    vd = vd - vdmean

    # Data vector, should be a column vector.
    uvobs = np.hstack((ud, vd))[:, np.newaxis]

    # Data data distances
    xdist, ydist = mat.xy_distance(xd, yd)

    # Data - data covarance matrix plus the noise.
    # The diagonal of the normalised covariance of uu and vv is not equal to 1 even
    # though A = 1. This is because A represents the streamfunction variance
    # which must be scaled to get the velocity variance. The scaling factor for
    # a Gaussian covariance function is 2/(3*l**2) i.e. it scales as l**-2.
    # This is why we multiply by cov.Cuu(0, 0, 1, l).
    Muu = cov.Cuu(xdist, ydist, 1, l) + cov.Cuu(0, 0, 1, l) * np.eye(*xdist.shape) / SNR
    Mvv = cov.Cvv(xdist, ydist, 1, l) + cov.Cvv(0, 0, 1, l) * np.eye(*xdist.shape) / SNR
    Muv = cov.Cuv(xdist, ydist, 1, l)

    Cdd = np.vstack((np.hstack((Muu, Muv)), np.hstack((Muv, Mvv))))

    xmddist, ymddist = mat.xy_distance(xm, ym, xd, yd)

    Mpsiu = cov.Cpsiu(xmddist, ymddist, 1, l)
    Mpsiv = cov.Cpsiv(xmddist, ymddist, 1, l)

    Cmd = np.hstack((Mpsiu, Mpsiv))

    A, _, _, _ = np.linalg.lstsq(Cdd, uvobs, rcond=None)

    psi = Cmd @ A

    # Reshape and add back the mean velocity. Also add a minus sign because we want to 
    # follow the oceanographic convention whereby dpsi/dx = v and dpsi/dy = -u.
    psi = -psi.reshape(input_shape) - udmean * ym + vdmean * xm

    if return_err:
        err = relerr(Cdd, Cmd).reshape(input_shape)
        return psi, err
    else:
        return psi
