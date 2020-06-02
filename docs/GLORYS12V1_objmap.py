# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: pyobjmap
#     language: python
#     name: pyobjmap
# ---

# %% [markdown]
# # Download reanalysis data and test out optimal interpolation
#
# We require a portion of the GLORYS 12 V1 dataset. To acquire it, one must make an account on https://resources.marine.copernicus.eu/ and then download the dataset with (WARNING: 1.2GB file):
#
# `wget --user=USERNAME --password=PASSWORD ftp://my.cmems-du.eu/Core/GLOBAL_REANALYSIS_PHY_001_030/global-reanalysis-phy-001-030-daily/2015/06/mercatorglorys12v1_gl12_mean_20150617_R20150624.nc`
#
# not forgetting to fill in `USERNAME` and `PASSWORD` appropriately. Download the dataset to `../data`. 
#
# The dataset contains a single daily mean output from the GLORYS 12 V1 (1/12th degree) global ocean reanalysis. More information on the reanalysis is available here:
#
# https://resources.marine.copernicus.eu/?option=com_csw&task=results?option=com_csw&view=details&product_id=GLOBAL_REANALYSIS_PHY_001_030

# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pyobjmap import objmap

# %% [markdown]
# We don't care about sea ice variables or bottom temperature so drop them.

# %%
ds = xr.open_dataset("../data/mercatorglorys12v1_gl12_mean_20150617_R20150624.nc", drop_variables=["usi", "vsi", "sithick", "siconc", "bottomT"])

# %%
ds

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ds.zos.plot(ax=ax)

# %% [markdown]
# Select a smaller region that is easier to work with.

# %%
west = 30
east = 80
south = -50
north = -30
idepth = 0

def nearestidx(value, arr):
    return np.argmin(np.abs(arr - value))

iwest = nearestidx(west, ds.longitude.values)
ieast = nearestidx(east, ds.longitude.values)
isouth = nearestidx(south , ds.latitude.values)
inorth = nearestidx(north , ds.latitude.values)
ilon = np.arange(iwest, ieast, dtype=int)
ilat = np.arange(isouth, inorth, dtype=int)

dsbox = ds.isel(longitude=ilon, latitude=ilat, depth=idepth, time=0)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
dsbox.zos.plot(ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
dsbox.thetao.plot(ax=ax)

step = 5  # reduce for plotting
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.quiver(dsbox.longitude[::step], dsbox.latitude[::step], dsbox.uo[::step, ::step], dsbox.vo[::step, ::step])

# %% [markdown]
# Save box to netcdf...

# %%
dsbox.to_netcdf("../data/small_glorys_region.nc")


# %% [markdown]
# Now lets estimate some useful quantities such as the vorticity and divergence. We'll need to apply the vector operators in spherical co-orrdinates since the data are on a lon - lat grid. 

# %%
def spherical_polar_gradient(f, lon, lat, r=6371000.0):
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


# %% [markdown]
# Gradients in sea surface height.

# %%
dhdlon, dhdlat = spherical_polar_gradient(dsbox.zos, dsbox.longitude, dsbox.latitude)

# %%
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
axs[0].pcolormesh(dsbox.longitude, dsbox.latitude, dhdlon)
axs[0].set_title('dhdlon')
axs[1].pcolormesh(dsbox.longitude, dsbox.latitude, dhdlat)
axs[1].set_title('dhdlat')

# %% [markdown]
# Gradients in velocity.

# %%
dudlon, dudlat = spherical_polar_gradient(dsbox.uo, dsbox.longitude, dsbox.latitude)
dvdlon, dvdlat = spherical_polar_gradient(dsbox.vo, dsbox.longitude, dsbox.latitude)
div = dudlon + dvdlat
vort = dvdlon - dudlat

# %%
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
axs[0].pcolormesh(dsbox.longitude, dsbox.latitude, div, cmap='coolwarm', vmin=-8e-7, vmax=8e-7)
axs[0].set_title('div')
axs[1].pcolormesh(dsbox.longitude, dsbox.latitude, vort, cmap='coolwarm', vmin=-8e-7, vmax=8e-7)
axs[1].set_title('vort')

# %% [markdown]
# Rossby number

# %%
# Earth rotates once per day (to a very good approximation)
Tearth = 86400  # seconds
fcor = (4*np.pi/Tearth)*np.sin(np.deg2rad(dsbox.latitude))  # Don't forget to change to radians

Ro = np.abs(vort/fcor.values[:, np.newaxis])

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
CP = ax.pcolormesh(dsbox.longitude, dsbox.latitude, Ro)
ax.set_title('Ro')
plt.colorbar(CP)

# %% [markdown]
# As expected for daily mean, 1/12th degree ($~\sim 10$ km) model output, the Rossby number is very small. As such, we would expect the velocity to be in geostrophic balance to a good approximation and not horizontally divergent. 

# %% [markdown]
# ## Now lets try running the objective mapping...
#
# First we subsample the region, to simulate 'real' observations which are sparse in space. To be even more realistic we might subsample in time too, but that would require downloading a lot more data. 

# %%
np.random.seed(1895099999)
frac = 0.01  # Fraction of data to sample, in this case 1 %.

# Grid the longitude and latitude data.
londg, latdg = np.meshgrid(dsbox.longitude.values, dsbox.latitude.values)

notnan = np.isfinite(dsbox.thetao.values.ravel())

t_ = dsbox.thetao.values.ravel()[notnan]
lons_ = londg.ravel()[notnan]
lats_ = latdg.ravel()[notnan]

n = int(frac*t_.size)  # number of samples.

# Choose samples without replacement.
sidx = np.random.choice(t_.size, size=(n), replace=False)
lond = lons_[sidx]
latd = lats_[sidx]
t = t_[sidx]

# Roughly convert data to x - y coordinates on sphere, with origin at mid of data.
rearth = 6370800 # metres
lonmid = 0.5*(dsbox.longitude[0] + dsbox.longitude[-1]).values
latmid = 0.5*(dsbox.latitude[0] + dsbox.latitude[-1]).values
xd = rearth*(np.deg2rad(lond - lonmid))*np.cos(np.deg2rad(latd))
yd = rearth*np.deg2rad(latd - latmid)

fig, ax = plt.subplots(1, 1)
CS = ax.scatter(xd, yd, s=5, c=t)
plt.colorbar(CS)

# %% [markdown]
# Investigate data - data covariance.

# %%
bins = 30 #np.linspace(0, 2e6, 10)

rbins, Cr = objmap.bincovr(xd, yd, t, bins=bins)
rmid = 0.5*(rbins[1:] + rbins[:-1])

popt = objmap.covfit(xd, yd, t, bins=bins, cfunc='gauss', p0=[30, 5e5], rfitmax=3e6)
a, l = popt

fig, ax = plt.subplots(1, 1)
ax.plot(rmid, Cr, 'x')
ax.plot(rmid, objmap.gauss(rmid, a, l))
ax.set_title('l = {:1.0f} km'.format(l/1000))

# %% [markdown]
# Objectively map the field.

# %%
nlon = 50
nlat = 100
l = 700e3

lonm = np.linspace(dsbox.longitude[0], dsbox.longitude[-1], nlon, retstep=False)
latm = np.linspace(dsbox.latitude[0], dsbox.latitude[-1], nlat, retstep=False)

latmg, lonmg = np.meshgrid(latm, lonm, indexing='ij')
ymg = rearth*np.deg2rad(latmg - latmid)
xmg = rearth*(np.deg2rad(lonmg - lonmid))*np.cos(np.deg2rad(latmg))

tg = objmap.objmap(xd, yd, t, xmg, ymg, a, [l, l, theta], cfunc='gauss2d', detrend="mean")

# %%
clevs = np.linspace(4, 24, 11)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
C = ax.contourf(lonm, latm, tg, clevs, extend='both')
ax.scatter(lond, latd, s=2)
plt.colorbar(C)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
C = ax.contourf(dsbox.longitude, dsbox.latitude, dsbox.thetao, clevs, extend='both')
ax.scatter(lond, latd, s=2)
plt.colorbar(C)

# %% [markdown]
# Is isotropy breaking down?

# %%
# bins = [np.linspace(-3e6, 3e6, 31), np.linspace(-1.5e6, 1.5e6, 21)]
bins = [np.linspace(0, 3e6, 31), np.linspace(0, 1.5e6, 21)]

# xbins, ybins, Cxy = objmap.bincovxy(xd, yd, t, bins=bins)
xbins, ybins, Cxy = objmap.bincovxyabs(xd, yd, t, bins=bins)
xmid = 0.5*(xbins[1:] + xbins[:-1])
ymid = 0.5*(ybins[1:] + ybins[:-1])

# %%
clev = np.linspace(0, 40, 11)
l = 500000

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
CC = ax.contourf(xmid, ymid, Cxy, clev, extend='both')
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
CC = ax.contourf(xmidg, ymidg, objmap.gauss2d(xmidg, ymidg, a, 6*l, l, 2), clev, extend='both')
plt.colorbar(CC)


# %%
def objmap2(xd, yd, zd, xm, ym, a, lx, ly, theta=0, cfunc='gauss2d'):
    """Needs docstring."""

    # Use the covariance function specified.
    cfunc = objmap.funccheck(cfunc)

    ztrend = zd.mean()

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

    zmg = (Cmd @ A).reshape(xm.shape)

    zmg += ztrend

    return zmg
