import numpy as np
from . import covariance as cov
from . import matrix as mat


def objmap(xd, yd, zd, xm, ym, SNR, l, cfunc='gauss', detrend='mean', coords='cartesian'):
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
    Rdd = mat.r_distance(xd, yd, coords=coords)

    # Data - data covarance matrix using the function.
    Cdd0 = cfunc(Rdd, 1, l)
    # Add variance back in.
    Cdd = Cdd0 + np.eye(*Cdd0.shape)/SNR

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
    xdist, ydist = mat.xy_distance(xd, yd)

    # Data - data covarance matrix using the function.
    Cdd0 = cfunc(xdist, ydist, 1, lx, ly, theta)
    # Add variance back in.
    Cdd = Cdd0 + np.eye(*Cdd0.shape)/SNR

    # Construct model - data distance matrix.
    xmddist, ymddist = mat.xy_distance(xm, ym, xd, yd)

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

    udmean = ud.mean()
    vdmean = vd.mean()
    ud = ud - udmean
    vd = vd - vdmean

    uvobs = np.hstack((ud, vd))[:, np.newaxis]  # Column vector...

    # Data data distances
    xdist, ydist = mat.xy_distance(xd, yd)

    # Data - data covarance matrix
    Muu = cov.Cuu(xdist, ydist, 1, l) + np.eye(*xdist.shape)/SNR
    Mvv = cov.Cvv(xdist, ydist, 1, l) + np.eye(*xdist.shape)/SNR
    Muv = cov.Cuv(xdist, ydist, 1, l)

    Cdd = np.vstack((np.hstack((Muu, Muv)), np.hstack((Muv, Mvv))))

    xmddist, ymddist = mat.xy_distance(xm, ym, xd, yd)

    Mpsiu = cov.Cpsiu(xmddist, ymddist, 1, l)
    Mpsiv = cov.Cpsiv(xmddist, ymddist, 1, l)

    Cmd = np.hstack((Mpsiu, Mpsiv))

    A, _, _, _ = np.linalg.lstsq(Cdd, uvobs, rcond=None)

    psi = Cmd @ A

    # Reshape and add back the mean velocity...
    psi = psi.reshape(input_shape) + udmean*ym - vdmean*xm

    return psi