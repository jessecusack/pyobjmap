import numpy as np


def plane_coeff(x, y, z):
    # Credit: amroamroamro gist on github
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    A = np.c_[x.ravel(), y.ravel(), np.ones(z.ravel().size)]
    C, _, _, _ = np.linalg.lstsq(A, z.ravel(), rcond=None)  # coefficients
    return C


def spherical_polar_gradient(f, lon, lat, r=6371000.0):
    """
    f - scalar
    lon - -180 to 180
    lat - -90 to 90
    Doesn't deal with the dateline...
    """
    nr, nc = f.shape
    if (nr != len(lat)) or (nc != len(lon)):
        raise ValueError(
            "Latitude and longitude are expected to be rows and" "columns respectively"
        )

    _, latg = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))

    # Cosine because latitude from -90 to 90. Not 0 to pi.
    dfdlat = np.gradient(f, lat, axis=0) / r
    dfdlon = np.gradient(f, lon, axis=1) / (r * np.cos(latg))

    return dfdlon, dfdlat