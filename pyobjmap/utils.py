import numpy as np


def plane_coeff(x, y, z):
    # Credit: amroamroamro gist on github
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    A = np.c_[x.ravel(), y.ravel(), np.ones(z.ravel().size)]
    C, _, _, _ = np.linalg.lstsq(A, z.ravel(), rcond=None)  # coefficients
    return C


def cartesian_gradient(f, x, y):
    """
    f : 2d array
    x : 1d array (colums)
    y : 1d array (rows)

    """
    f = np.asarray(f)
    x = np.asarray(x)
    y = np.asarray(y)

    nr, nc = f.shape
    if (nr != len(y)) or (nc != len(x)):
        raise ValueError("y and x are expected to be rows and columns respectively")

    dfdy = np.gradient(f, y, axis=0)
    dfdx = np.gradient(f, x, axis=1)

    return dfdx, dfdy


def spherical_polar_gradient(f, lon, lat, r=6371000.0):
    """
    f : scalar array
    lon : 1d array -180 to 180
    lat : 1d array -90 to 90
    Doesn't deal with the dateline...?
    """
    f = np.asarray(f)
    lon = np.deg2rad(np.asarray(lon))
    lat = np.deg2rad(np.asarray(lat))

    nr, nc = f.shape
    if (nr != len(lat)) or (nc != len(lon)):
        raise ValueError(
            "Latitude and longitude are expected to be rows and columns respectively"
        )

    _, latg = np.meshgrid(lon, lat)

    # Cosine because latitude from -pi/2 to pi/2. Not 0 to pi.
    dfdlat = np.gradient(f, lat, axis=0) / r
    dfdlon = np.gradient(f, lon, axis=1) / (r * np.cos(latg))

    return dfdlon, dfdlat


def haversine(theta):
    return 0.5 * (1 - np.cos(theta))


def archaversine(y):
    return np.arccos(1 - 2 * y)


def haversine_distance(lon0, lat0, lon1, lat1, r=6371000.0):
    """Calculates the distance between longitude and latitude coordinates on a
    spherical earth with radius using the Haversine formula.

    Parameters
    ----------
    lon0 : 1d numpy array
        Longitude values. [degrees]
    lat0 : 1d numpy array
        Latitude values. [degrees]
    lon1 : 1d numpy array
        Longitude values. [degrees]
    lat1 : 1d numpy array
        Latitude values. [degrees]

    Returns
    -------
    dist : 1d numpy array
        Distance between lon and lat positions. [m]

    """

    lon0 = np.deg2rad(lon0)
    lat0 = np.deg2rad(lat0)
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)

    dist = r * archaversine(
        haversine(lat1 - lat0) + np.cos(lat1) * np.cos(lat2) * haversine(lon1 - lon0)
    )

    return dist


# Projecting to UTM with pyproj: https://gist.github.com/twpayne/4409500
