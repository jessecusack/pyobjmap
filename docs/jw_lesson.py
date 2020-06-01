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
# # Recreate Wilkins OI lecture in python

# %%
import matplotlib.pyplot as plt
import munch
import numpy as np
import scipy.io as io
import scipy.spatial as spat
import scipy.stats as stats
import scipy.optimize as opt

dat = munch.munchify(io.loadmat("mercator_temperature.mat", squeeze_me=True, chars_as_strings=True))

# %%
dat.keys()

# %%
k = 9

fig, ax = plt.subplots(1, 1)
CC = ax.contourf(dat.x, dat.y, dat.temp[k, ...], 20)
ax.set_title("Depth {:1.0f} m".format(dat.depth[k]))
plt.colorbar(CC)

# %% [markdown]
# Randomly select some data.

# %%
np.random.seed(94818511)
k = 9
frac = 0.1

tg = dat.temp[k, ...]
latdg, londg = np.meshgrid(dat.y, dat.x, indexing='ij')

notnan = np.isfinite(tg.ravel())

t_ = tg.ravel()[notnan]
lons = londg.ravel()[notnan]
lats = latdg.ravel()[notnan]

n = int(frac*t_.size)

idx = np.random.choice(t_.size, size=(n), replace=False)

lond = lons[idx]
latd = lats[idx]
t = t_[idx]

fig, ax = plt.subplots(1, 1)
CS = ax.scatter(lond, latd, s=5, c=t)
plt.colorbar(CS)

# %% [markdown]
# Data covariance function estimate.

# %%
## Assuming isotropic homogeneous variability, make an estimate of the
#  data covariance function by computing binned-lagged covariance for all
#  possible data-data pairs

# Get distances separating every pair of stations using approximate
# spherical geometry
rearth = 6370800 # metres

# xkm, ykm will be the coordinates of the data (converted from lon/lat)
latmid = 0.5*(dat.ax[2] + dat.ax[3])
lonmid = 0.5*(dat.ax[0] + dat.ax[1])
yd = rearth*np.deg2rad(latd - latmid)/1000
xd = rearth*(np.deg2rad(lond - lonmid))*np.cos(np.deg2rad(latd))/1000

fig, ax = plt.subplots(1, 1)
CS = ax.scatter(xd, yd, s=5, c=t)
plt.colorbar(CS)

# %% [markdown]
# Add noise to data.

# %%
np.random.seed(475829851)
err = 0.4
te = t + err*np.random.randn(t.size)

# %% [markdown]
# Fit to covariance. 

# %%
# Compute the binnned lagged covariance function
nbins = 40
rmin = 0
rmax = 1000
rfitmax = 600

# Construct distance matrix.
xyd = np.stack((xd, yd), 1)
Rdd = spat.distance.cdist(xyd, xyd)
# Upper triangle indices.
itri, jtri = np.triu_indices_from(Rdd)

# remove mean before calculating covariance
td = te - te.mean()

# Covariance matrix
C = np.outer(td, td)

# Covariance at different distances
rbins = np.linspace(rmin, rmax, nbins+1)
r = 0.5*(rbins[:-1] + rbins[1:])  # mid points
# cf = np.zeros_like(r)
cf, _, _ = stats.binned_statistic(Rdd[itri, jtri], C[itri, jtri], statistic='mean', bins=rbins)

# Now fit an analytical covariance function to the data.
# In this example I limit the fit to values in r < 600 km. Don't fit to
# the zero lag data because it has the error variance included, and don't
# go out to very long lags because the covariance there is based on very
# few estimates)
infit = r < rfitmax

def gauss(r, a, l):
    return a*np.exp(-0.5*(r/l)**2)

def marko(r, a, l):
    ra = np.abs(r)/l
    return a*(1 + ra)*np.exp(-ra)

def letra(r, a, l):
    ra = np.abs(r)/l
    rsq = ra**2
    return a*np.exp(-ra)*(1 + ra + rsq/6 - ra*rsq/6)

def cost(func, x, y, args):
    return np.mean((func(x, *args) - y)**2)


p0 = (20, 400)
popt_gauss, _ = opt.curve_fit(gauss, r[infit], cf[infit], p0=p0)
popt_marko, _ = opt.curve_fit(marko, r[infit], cf[infit], p0=p0)
popt_letra, _ = opt.curve_fit(letra, r[infit], cf[infit], p0=p0)

cost_gauss = cost(gauss, r[infit], cf[infit], popt_gauss)
cost_marko = cost(marko, r[infit], cf[infit], popt_marko)
cost_letra = cost(letra, r[infit], cf[infit], popt_letra)

print("cost gauss = {:1.2f}".format(cost_gauss))
print("cost marko = {:1.2f}".format(cost_marko))
print("cost letra = {:1.2f}".format(cost_letra))

fig, ax = plt.subplots(1, 1)
ax.plot(r, cf, 'x')
ax.plot(r, gauss(r, *popt_gauss), label='gauss')
ax.plot(r, gauss(r, *popt_marko), label='marko')
ax.plot(r, gauss(r, *popt_letra), label='letra')

ax.legend()

# %% [markdown]
# # Now for the Optimal Interpolation
#
# Data - data covariance matrix.

# %%
cfunc = gauss
args = np.array([1, popt_gauss[1]])

Cdd0 = cfunc(Rdd, *args)

# Add variance back in.
lam = (C.diagonal().mean() - popt_gauss[0])/popt_gauss[0]
Cdd = Cdd0 + np.eye(*Cdd0.shape)*lam

# %% [markdown]
# Model - data covariance matrix.

# %% Build the model-data covariance matrix Cmd
nlon = 50
nlat = 100
args = np.array([1, popt_gauss[1]])

lonm, dlon = np.linspace(dat.ax[0], dat.ax[1], nlon, retstep=True)
latm, dlat = np.linspace(dat.ax[2], dat.ax[3], nlat, retstep=True)

latmg, lonmg = np.meshgrid(latm, lonm, indexing='ij')

ygm = rearth*np.deg2rad(latmg - latmid)/1000
xgm = rearth*(np.deg2rad(lonmg - lonmid))*np.cos(np.deg2rad(latmg))/1000

xym = np.stack((xgm.ravel(), ygm.ravel()), 1)
Rmd = spat.distance.cdist(xym, xyd)

Cmd = cfunc(Rmd, *args)

# %% [markdown]
# Do the optimal interpolation.

# %%
A, _, _, _ = np.linalg.lstsq(Cdd, td, rcond=None)
D = te.mean() + (Cmd @ A).reshape(latmg.shape)

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
CC = ax.contourf(lonm, latm, D, 20)
plt.colorbar(CC)
