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


def nearestidx(value, arr):
    return np.argmin(np.abs(arr - value))


# %% [markdown]
# We don't care about sea ice variables or bottom temperature so drop them.

# %%
ds = xr.open_dataset("../data/mercatorglorys12v1_gl12_mean_20150617_R20150624.nc", drop_variables=["usi", "vsi", "sithick", "siconc", "bottomT"])

# %% [markdown]
# ## Investigate the dataset

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
# ## Objective mapping with an isotropic covariance function
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
l = 500e3

lonm = np.linspace(dsbox.longitude[0], dsbox.longitude[-1], nlon, retstep=False)
latm = np.linspace(dsbox.latitude[0], dsbox.latitude[-1], nlat, retstep=False)

latmg, lonmg = np.meshgrid(latm, lonm, indexing='ij')
ymg = rearth*np.deg2rad(latmg - latmid)
xmg = rearth*(np.deg2rad(lonmg - lonmid))*np.cos(np.deg2rad(latmg))

tg = objmap.objmap(xd, yd, t, xmg, ymg, a, l, cfunc='gauss', detrend="mean")

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
# ## Anisotropic covariance function

# %%
# bins = [np.linspace(-3e6, 3e6, 31), np.linspace(-1.5e6, 1.5e6, 21)]
bins = [np.linspace(0, 3e6, 31), np.linspace(0, 1.5e6, 21)]

# xbins, ybins, Cxy = objmap.bincovxy(xd, yd, t, bins=bins)
xbins, ybins, Cxy = objmap.bincovxyabs(xd, yd, t, bins=bins)
xmid = 0.5*(xbins[1:] + xbins[:-1])
ymid = 0.5*(ybins[1:] + ybins[:-1])

# %%
clev = np.linspace(0, 40, 11)
lx = 1200e3
ly = 250e3

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
CC = ax.contourf(xmid, ymid, Cxy, clev, extend='both')
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
CC = ax.contourf(xmidg, ymidg, objmap.gauss2d(xmidg, ymidg, a, lx, ly, 2), clev, extend='both')
plt.colorbar(CC)

# %%
lx = 1200e3
ly = 250e3
theta = 2
SNR = 4

tg = objmap.objmap2(xd, yd, t, xmg, ymg, SNR, lx, ly, theta)

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
# ## Mapping velocity to a stream function
#
# Define all the various functions (Wilkins 2002)

# %%
def C(x, y, A, l):
    r = np.sqrt(x**2 + y**2)
    return A*np.exp(-0.5*r**2/l**2)

def R(x, y, A, l):
    r = np.sqrt(x**2 + y**2)
    return A*np.exp(-0.5*r**2/l**2)/l**2

def S(x, y, A, l):
    r = np.sqrt(x**2 + y**2)
    return A*(l**2 - r**2)*np.exp(-0.5*r**2/l**2)/l**4

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


# %% [markdown]
# Make a micro box for velocity...

# %%
west = 50
east = 60
south = -50
north = -35
idepth = 0

iwest = nearestidx(west, ds.longitude.values)
ieast = nearestidx(east, ds.longitude.values)
isouth = nearestidx(south , ds.latitude.values)
inorth = nearestidx(north , ds.latitude.values)
ilon = np.arange(iwest, ieast, dtype=int)
ilat = np.arange(isouth, inorth, dtype=int)

dmbox = ds.isel(longitude=ilon, latitude=ilat, depth=idepth, time=0)

step = 3  # reduce for plotting
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect('equal')
ax.quiver(dmbox.longitude[::step], dmbox.latitude[::step], dmbox.uo[::step, ::step], dmbox.vo[::step, ::step])

# %%
np.random.seed(82615811)
frac = 0.02  # Fraction of data to sample, in this case 5 %.

# Grid the longitude and latitude data.
londg, latdg = np.meshgrid(dmbox.longitude.values, dmbox.latitude.values)

notnan = np.isfinite(dmbox.thetao.values.ravel())

u_ = dmbox.uo.values.ravel()[notnan]
v_ = dmbox.vo.values.ravel()[notnan]
lons_ = londg.ravel()[notnan]
lats_ = latdg.ravel()[notnan]

n = int(frac*u_.size)  # number of samples.

# Choose samples without replacement.
sidx = np.random.choice(u_.size, size=(n), replace=False)
lond = lons_[sidx]
latd = lats_[sidx]
u = u_[sidx]
v = v_[sidx]

# Roughly convert data to x - y coordinates on sphere, with origin at mid of data.
rearth = 6370800 # metres
lonmid = 0.5*(dmbox.longitude[0] + dmbox.longitude[-1]).values
latmid = 0.5*(dmbox.latitude[0] + dmbox.latitude[-1]).values
xd = rearth*(np.deg2rad(lond - lonmid))*np.cos(np.deg2rad(latd))
yd = rearth*np.deg2rad(latd - latmid)

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect('equal')
ax.quiver(xd, yd, u, v)

# %% [markdown]
# Lets investigate the covariance.

# %%
l = 100e3
A = 5e8

# %%
bins = [np.linspace(-250e3, 250e3, 11), np.linspace(-250e3, 250e3, 11)]

xbins, ybins, Cxy = objmap.bincovxyuv(xd, yd, u, v, bins=bins)
xmid = 0.5*(xbins[1:] + xbins[:-1])
ymid = 0.5*(ybins[1:] + ybins[:-1])

clev = np.linspace(-0.015, 0.015, 11)

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmid, ymid, Cxy, clev, cmap="coolwarm", extend='both')
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmidg, ymidg, Cuv(xmidg, ymidg, A, l), clev, cmap="coolwarm", extend='both')
plt.colorbar(CC)

# %%
bins = [np.linspace(-250e3, 250e3, 11), np.linspace(-250e3, 250e3, 11)]

xbins, ybins, Cxy = objmap.bincovxyuv(xd, yd, u, u, bins=bins)
xmid = 0.5*(xbins[1:] + xbins[:-1])
ymid = 0.5*(ybins[1:] + ybins[:-1])

clev = np.linspace(-0.015, 0.015, 11)

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmid, ymid, Cxy, clev, cmap="coolwarm", extend='both')
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmidg, ymidg, Cuu(xmidg, ymidg, A, l), clev, cmap="coolwarm", extend='both')
plt.colorbar(CC)

# %%
bins = [np.linspace(-250e3, 250e3, 11), np.linspace(-250e3, 250e3, 11)]

xbins, ybins, Cxy = objmap.bincovxyuv(xd, yd, v, v, bins=bins)
xmid = 0.5*(xbins[1:] + xbins[:-1])
ymid = 0.5*(ybins[1:] + ybins[:-1])

clev = np.linspace(-0.015, 0.015, 11)

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmid, ymid, Cxy, clev, cmap="coolwarm", extend='both')
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmidg, ymidg, Cvv(xmidg, ymidg, A, l), clev, cmap="coolwarm", extend='both')
plt.colorbar(CC)

# %% [markdown]
# Construct the matrices for the optimal interpolation.
#
# \begin{equation}
# \psi^\text{est}(x) = \mathbf{C} \mathbf{A}^{-1} \phi^\text{obs}
# \end{equation}
#
# \begin{equation}
# \mathbf{A} = 
#   \begin{bmatrix}
#   \mathbf{C}_{uu} + \sigma_u \mathbf{I} & \mathbf{C}_{uv} \\
#   \mathbf{C}_{uv} & \mathbf{C}_{vv} + \sigma_u \mathbf{I}
#   \end{bmatrix}
# \end{equation}
#
# \begin{equation}
# \mathbf{C} = 
#   \begin{bmatrix}
#   \mathbf{C}_{\psi u} & \mathbf{C}_{\psi v}
#   \end{bmatrix}
# \end{equation}
#
# \begin{equation}
# \phi^\text{obs} =
#   \begin{bmatrix}
#   \mathbf{u} \\ 
#   \mathbf{v}
#   \end{bmatrix}
# \end{equation}
#
# <!-- \begin{gather}
#  \begin{bmatrix} \Phi_{11} & \Phi_{12} \\ \Phi_{21} & \Phi_{22} \end{bmatrix}
#  =
#  \frac{1}{\det(X)}
#   \begin{bmatrix}
#    X_{22} Y_{11} - X_{12} Y_{21} &
#    X_{22} Y_{12} - X_{12} Y_{22} \\
#    X_{11} Y_{21} - X_{21} Y_{11} &
#    X_{11} Y_{22} - X_{21} Y_{12} 
#    \end{bmatrix}
# \end{gather} -->
#

# %%
SNR = np.inf
l = 100e3
# A = 5e8

ud = u - u.mean()
vd = v - v.mean()

phi_obs = np.hstack((ud, vd))[:, np.newaxis]  # Column vector...

# Data data distances
xdmid = np.mean(xd)
ydmid = np.mean(yd)
x_ = xd - xdmid
y_ = yd - ydmid

xdist = x_[:, np.newaxis] - x_[np.newaxis, :]
ydist = y_[:, np.newaxis] - y_[np.newaxis, :]

# Data - data covarance matrix
Muu = Cuu(xdist, ydist, 1, l) + np.eye(*xdist.shape)/SNR
Mvv = Cvv(xdist, ydist, 1, l) + np.eye(*xdist.shape)/SNR
Muv = Cuv(xdist, ydist, 1, l)

M = np.vstack((np.hstack((Muu, Muv)), np.hstack((Muv, Mvv))))


# %%
nlon = 25
nlat = 50

lonm = np.linspace(dmbox.longitude[0], dmbox.longitude[-1], nlon, retstep=False)
latm = np.linspace(dmbox.latitude[0], dmbox.latitude[-1], nlat, retstep=False)

lonmg, latmg = np.meshgrid(lonm, latm)
ymg = rearth*np.deg2rad(latmg - latmid)
xmg = rearth*(np.deg2rad(lonmg - lonmid))*np.cos(np.deg2rad(latmg))

# %%
xm = xmg.ravel() - xdmid
ym = ymg.ravel() - ydmid
xmddist = xm[:, np.newaxis] - xd[np.newaxis, :]
ymddist = ym[:, np.newaxis] - yd[np.newaxis, :]

Mpsiu = Cpsiu(xmddist, ymddist, 1, l)
Mpsiv = Cpsiv(xmddist, ymddist, 1, l)

Cmd = np.hstack((Mpsiu, Mpsiv))

# %%
A, _, _, _ = np.linalg.lstsq(M, phi_obs, rcond=None)

# %%
psi = (Cmd @ A).reshape(xmg.shape)

# %%
fig, ax = plt.subplots(1, 1)
ax.contour(lonmg, latmg, psi)

# %%
