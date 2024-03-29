# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: pyobjmap
#     language: python
#     name: pyobjmap
# ---

# %% [markdown]
# # Optimal interpolation demo

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as itpl
import xarray as xr

from pyobjmap import covariance as cov
from pyobjmap import objmap, utils

# %% [markdown]
# Load the small GLORYS region

# %%
dsbox = xr.open_dataset("../data/small_glorys_region.nc")

# %% [markdown]
# ## Examine the region and its properties
#
# Now lets estimate some useful quantities such as the vorticity and divergence. We'll need to apply the vector operators in spherical co-orrdinates since the data are on a lon - lat grid. 

# %%
dhdlon, dhdlat = utils.spherical_polar_gradient(
    dsbox.zos, dsbox.longitude, dsbox.latitude
)

# %%
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
axs[0].pcolormesh(dsbox.longitude, dsbox.latitude, dhdlon)
axs[0].set_title("dhdlon")
axs[1].pcolormesh(dsbox.longitude, dsbox.latitude, dhdlat)
axs[1].set_title("dhdlat")

# %% [markdown]
# Gradients in velocity.

# %%
dudlon, dudlat = utils.spherical_polar_gradient(
    dsbox.uo, dsbox.longitude, dsbox.latitude
)
dvdlon, dvdlat = utils.spherical_polar_gradient(
    dsbox.vo, dsbox.longitude, dsbox.latitude
)
div = dudlon + dvdlat
vort = dvdlon - dudlat

# %%
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
axs[0].pcolormesh(
    dsbox.longitude, dsbox.latitude, div, cmap="coolwarm", vmin=-2e-5, vmax=2e-5
)
axs[0].set_title("div")
CP = axs[1].pcolormesh(
    dsbox.longitude, dsbox.latitude, vort, cmap="coolwarm", vmin=-2e-5, vmax=2e-5
)
axs[1].set_title("vort")
plt.colorbar(CP)

# %% [markdown]
# ### Rossby number

# %%
# Earth rotates once per day (to a very good approximation)
Tearth = 86400  # seconds
fcor = (4 * np.pi / Tearth) * np.sin(
    np.deg2rad(dsbox.latitude)
)  # Don't forget to change to radians

Ro = np.abs(vort / fcor.values[:, np.newaxis])

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
CP = ax.contourf(dsbox.longitude, dsbox.latitude, Ro, np.linspace(0, 1, 5))
ax.set_title("Ro")
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

n = int(frac * t_.size)  # number of samples.

# Choose samples without replacement.
sidx = np.random.choice(t_.size, size=(n), replace=False)
lond = lons_[sidx]
latd = lats_[sidx]
t = t_[sidx]

# Roughly convert data to x - y coordinates on sphere, with origin at mid of data.
rearth = 6370800  # metres
lonmid = 0.5 * (dsbox.longitude[0] + dsbox.longitude[-1]).values
latmid = 0.5 * (dsbox.latitude[0] + dsbox.latitude[-1]).values
xd = rearth * (np.deg2rad(lond - lonmid)) * np.cos(np.deg2rad(latd))
yd = rearth * np.deg2rad(latd - latmid)

fig, ax = plt.subplots(1, 1)
CS = ax.scatter(xd, yd, s=5, c=t)
plt.colorbar(CS)

# %% [markdown]
# Investigate data - data covariance.

# %%
bins = 30  # np.linspace(0, 2e6, 10)

rbins, Cr = cov.bincovr(xd, yd, t, bins=bins)
rmid = 0.5 * (rbins[1:] + rbins[:-1])

popt = cov.covfit(xd, yd, t, bins=bins, cfunc="gauss", p0=[30, 5e5], rfitmax=3e6)
a, l = popt

fig, ax = plt.subplots(1, 1)
ax.plot(rmid, Cr, "x")
ax.plot(rmid, cov.gauss(rmid, a, l))
ax.set_title("l = {:1.0f} km".format(l / 1000))

# %% [markdown]
# Objectively map the field.

# %%
nlon = 50
nlat = 100
l = 500e3

lonm = np.linspace(dsbox.longitude[0], dsbox.longitude[-1], nlon, retstep=False)
latm = np.linspace(dsbox.latitude[0], dsbox.latitude[-1], nlat, retstep=False)

latmg, lonmg = np.meshgrid(latm, lonm, indexing="ij")
ymg = rearth * np.deg2rad(latmg - latmid)
xmg = rearth * (np.deg2rad(lonmg - lonmid)) * np.cos(np.deg2rad(latmg))

tg, tgerr = objmap.objmap(
    xd, yd, t, xmg, ymg, a, l, cfunc="gauss", detrend="mean", return_err=True
)

# %%
clevs = np.linspace(4, 24, 11)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
C = ax.contourf(lonm, latm, tg, clevs, extend="both")
ax.scatter(lond, latd, s=2)
plt.colorbar(C)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
C = ax.contourf(dsbox.longitude, dsbox.latitude, dsbox.thetao, clevs, extend="both")
ax.scatter(lond, latd, s=2)
plt.colorbar(C)

# %% [markdown]
# Errors...

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
C = ax.contourf(lonm, latm, tgerr)
ax.scatter(lond, latd, s=2, color='k')
plt.colorbar(C)

# %% [markdown]
# ## Anisotropic covariance function

# %%
# bins = [np.linspace(-3e6, 3e6, 31), np.linspace(-1.5e6, 1.5e6, 21)]
bins = [np.linspace(0, 3e6, 31), np.linspace(0, 1.5e6, 21)]

# xbins, ybins, Cxy = objmap.bincovxy(xd, yd, t, bins=bins)
xbins, ybins, Cxy = cov.bincovxyabs(xd, yd, t, bins=bins)
xmid = 0.5 * (xbins[1:] + xbins[:-1])
ymid = 0.5 * (ybins[1:] + ybins[:-1])

# %%
clev = np.linspace(0, 40, 11)
lx = 1200e3
ly = 250e3

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1)
ax.set_aspect("equal")
CC = ax.contourf(xmid, ymid, Cxy, clev, extend="both")
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1)
ax.set_aspect("equal")
CC = ax.contourf(
    xmidg, ymidg, cov.gauss2d(xmidg, ymidg, a, lx, ly, 2), clev, extend="both"
)
plt.colorbar(CC)

# %%
lx = 1200e3
ly = 250e3
theta = 2
SNR = 4

tg, tgerr = objmap.objmap2(xd, yd, t, xmg, ymg, SNR, lx, ly, theta, return_err=True)

# %%
clevs = np.linspace(4, 24, 11)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
C = ax.contourf(lonm, latm, tg, clevs, extend="both")
ax.scatter(lond, latd, s=2)
plt.colorbar(C)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
C = ax.contourf(dsbox.longitude, dsbox.latitude, dsbox.thetao, clevs, extend="both")
ax.scatter(lond, latd, s=2)
plt.colorbar(C)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
C = ax.contourf(lonm, latm, tgerr)
ax.scatter(lond, latd, s=2, color='k')
plt.colorbar(C)

# %% [markdown]
# ## Mapping velocity to a stream function
#
# Make a micro box for velocity...

# %%
west = 50
east = 60
south = -50
north = -35

dmbox = dsbox.sel(longitude=slice(west, east), latitude=slice(south, north))

step = 3  # reduce for plotting
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
ax.quiver(
    dmbox.longitude[::step],
    dmbox.latitude[::step],
    dmbox.uo[::step, ::step],
    dmbox.vo[::step, ::step],
)

# %%
np.random.seed(82615811)
frac = 0.04  # Fraction of data to sample, in this case 4 %.

# Grid the longitude and latitude data.
londg, latdg = np.meshgrid(dmbox.longitude.values, dmbox.latitude.values)

notnan = np.isfinite(dmbox.thetao.values.ravel())

u_ = dmbox.uo.values.ravel()[notnan]
v_ = dmbox.vo.values.ravel()[notnan]
lons_ = londg.ravel()[notnan]
lats_ = latdg.ravel()[notnan]

n = int(frac * u_.size)  # number of samples.

# Choose samples without replacement.
sidx = np.random.choice(u_.size, size=(n), replace=False)
lond = lons_[sidx]
latd = lats_[sidx]
u = u_[sidx]
v = v_[sidx]

# Roughly convert data to x - y coordinates on sphere, with origin at mid of data.
rearth = 6370800  # metres
lonmid = 0.5 * (dmbox.longitude[0] + dmbox.longitude[-1]).values
latmid = 0.5 * (dmbox.latitude[0] + dmbox.latitude[-1]).values
xd = rearth * (np.deg2rad(lond - lonmid)) * np.cos(np.deg2rad(latd))
yd = rearth * np.deg2rad(latd - latmid)

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
ax.quiver(xd, yd, u, v)

# %% [markdown]
# Lets investigate the covariance.

# %%
l = 100e3
A = 5e8

# %%
bins = [np.linspace(-250e3, 250e3, 11), np.linspace(-250e3, 250e3, 11)]

xbins, ybins, Cxy = cov.bincovxyuv(xd, yd, u, v, bins=bins)
xmid = 0.5 * (xbins[1:] + xbins[:-1])
ymid = 0.5 * (ybins[1:] + ybins[:-1])

clev = np.linspace(-0.015, 0.015, 11)

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmid, ymid, Cxy, clev, cmap="coolwarm", extend="both")
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(
    xmidg, ymidg, cov.Cuv(xmidg, ymidg, A, l), clev, cmap="coolwarm", extend="both"
)
plt.colorbar(CC)

# %%
bins = [np.linspace(-250e3, 250e3, 11), np.linspace(-250e3, 250e3, 11)]

xbins, ybins, Cxy = cov.bincovxyuv(xd, yd, u, u, bins=bins)
xmid = 0.5 * (xbins[1:] + xbins[:-1])
ymid = 0.5 * (ybins[1:] + ybins[:-1])

clev = np.linspace(-0.015, 0.015, 11)

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmid, ymid, Cxy, clev, cmap="coolwarm", extend="both")
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(
    xmidg, ymidg, cov.Cuu(xmidg, ymidg, A, l), clev, cmap="coolwarm", extend="both"
)
plt.colorbar(CC)

# %%
bins = [np.linspace(-250e3, 250e3, 11), np.linspace(-250e3, 250e3, 11)]

xbins, ybins, Cxy = cov.bincovxyuv(xd, yd, v, v, bins=bins)
xmid = 0.5 * (xbins[1:] + xbins[:-1])
ymid = 0.5 * (ybins[1:] + ybins[:-1])

clev = np.linspace(-0.015, 0.015, 11)

xmidg, ymidg = np.meshgrid(xmid, ymid)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(xmid, ymid, Cxy, clev, cmap="coolwarm", extend="both")
plt.colorbar(CC)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# ax.set_aspect('equal')
CC = ax.contourf(
    xmidg, ymidg, cov.Cvv(xmidg, ymidg, A, l), clev, cmap="coolwarm", extend="both"
)
plt.colorbar(CC)

# %% [markdown]
# The matricies used for optimal interpolation.
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
SNR = 1000
l = 200e3
A = 5e8
nlon = 50
nlat = 100

lonm = (
    dmbox.longitude.values
)  # np.linspace(dmbox.longitude[0], dmbox.longitude[-1], nlon, retstep=False)
latm = (
    dmbox.latitude.values
)  # np.linspace(dmbox.latitude[0], dmbox.latitude[-1], nlat, retstep=False)

lonmg, latmg = np.meshgrid(lonm, latm)
ymg = rearth * np.deg2rad(latmg - latmid)
xmg = rearth * (np.deg2rad(lonmg - lonmid)) * np.cos(np.deg2rad(latmg))

psi = objmap.objmap_streamfunc(xd, yd, u, v, xmg, ymg, l, SNR, return_err=False)

# %% [markdown]
# Lets look at the mapped field.

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
ax.contour(lonmg, latmg, psi, 10, colors="k")
ax.quiver(lond, latd, u, v)

# fig, ax = plt.subplots(1, 1, figsize=(9, 6))
# C = ax.contourf(lonmg, latmg, psierr)
# plt.colorbar(C)

# %%
vm, um = utils.spherical_polar_gradient(psi, lonm, latm)
um *= -1

fum = itpl.RectBivariateSpline(latm, lonm, um)
fvm = itpl.RectBivariateSpline(latm, lonm, vm)

umd = fum(latd, lond, grid=False)
vmd = fvm(latd, lond, grid=False)

dumdx, dumdy = utils.spherical_polar_gradient(um, lonm, latm)
dvmdx, dvmdy = utils.spherical_polar_gradient(vm, lonm, latm)

divm = dumdx + dvmdy
vortm = dvmdx - dumdy

# %%
step = 3
scale = 30
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
ax.quiver(
    lonmg[::step, ::step],
    latmg[::step, ::step],
    um[::step, ::step],
    vm[::step, ::step],
    color="r",
    scale=scale,
)
ax.quiver(lond, latd, u, v, scale=scale)

# %% [markdown]
# Divergence.

# %%
c = 4e-5

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
CF = ax.contourf(
    lonmg, latmg, divm, np.linspace(-c, c, 11), cmap="coolwarm", extend="both"
)
plt.colorbar(CF)
ax.set_title('Divergence')

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
CF = ax.contourf(
    lonmg, latmg, vortm, np.linspace(-c, c, 11), cmap="coolwarm", extend="both"
)
plt.colorbar(CF)
ax.set_title('Vorticity')

# %% [markdown]
# ## Mapping velocity and height to a geostrophic streamfunction

# %%
np.random.seed(58194851)
frac = 0.04  # Fraction of data to sample, in this case 4 %.

# Grid the longitude and latitude data.
londg, latdg = np.meshgrid(dmbox.longitude.values, dmbox.latitude.values)

notnan = np.isfinite(dmbox.thetao.values.ravel())

u_ = dmbox.uo.values.ravel()[notnan]
v_ = dmbox.vo.values.ravel()[notnan]
h_ = dmbox.zos.values.ravel()[notnan]
lons_ = londg.ravel()[notnan]
lats_ = latdg.ravel()[notnan]

n = int(frac * u_.size)  # number of samples.

# Choose samples without replacement.
sidx = np.random.choice(u_.size, size=(n), replace=False)
lond = lons_[sidx]
latd = lats_[sidx]
u = u_[sidx]
v = v_[sidx]
h = h_[sidx]

# Roughly convert data to x - y coordinates on sphere, with origin at mid of data.
rearth = 6370800  # metres
lonmid = 0.5 * (dmbox.longitude[0] + dmbox.longitude[-1]).values
latmid = 0.5 * (dmbox.latitude[0] + dmbox.latitude[-1]).values
xd = rearth * (np.deg2rad(lond - lonmid)) * np.cos(np.deg2rad(latd))
yd = rearth * np.deg2rad(latd - latmid)

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
Q = ax.quiver(xd, yd, u, v, h)
plt.colorbar(Q)

# %%
SNR = 100000
l = 200e3
A = 5e8
nlon = 50
nlat = 100
g = 9.91
Tearth = 86400  # seconds
fcor = (4 * np.pi / Tearth) * np.sin(
    np.deg2rad(latd.mean())
)  # Don't forget to change to radians

lonm = (
    dmbox.longitude.values
)  # np.linspace(dmbox.longitude[0], dmbox.longitude[-1], nlon, retstep=False)
latm = (
    dmbox.latitude.values
)  # np.linspace(dmbox.latitude[0], dmbox.latitude[-1], nlat, retstep=False)

lonmg, latmg = np.meshgrid(lonm, latm)
ymg = rearth * np.deg2rad(latmg - latmid)
xmg = rearth * (np.deg2rad(lonmg - lonmid)) * np.cos(np.deg2rad(latmg))

psi = objmap.objmap_streamfunc_uvh(xd, yd, u, v, h, xmg, ymg, l, SNR, fcor, g, return_err=False)

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
ax.contour(lonmg, latmg, psi, 10, colors="k")
ax.quiver(lond, latd, u, v)

# %%
step = 3
scale = 30

vm, um = utils.spherical_polar_gradient(psi, lonm, latm)
um *= -1

fum = itpl.RectBivariateSpline(latm, lonm, um)
fvm = itpl.RectBivariateSpline(latm, lonm, vm)

umd = fum(latd, lond, grid=False)
vmd = fvm(latd, lond, grid=False)

dumdx, dumdy = utils.spherical_polar_gradient(um, lonm, latm)
dvmdx, dvmdy = utils.spherical_polar_gradient(vm, lonm, latm)

divm = dumdx + dvmdy
vortm = dvmdx - dumdy

fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
ax.quiver(
    lonmg[::step, ::step],
    latmg[::step, ::step],
    um[::step, ::step],
    vm[::step, ::step],
    color="r",
    scale=scale,
)
ax.quiver(lond, latd, u, v, scale=scale)
