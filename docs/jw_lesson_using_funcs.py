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

# %%
import matplotlib.pyplot as plt
import munch
import numpy as np
import scipy.io as io
from pyobjmap import objmap

dat = munch.munchify(io.loadmat("../data/mercator_temperature.mat", squeeze_me=True, chars_as_strings=True))


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

rearth = 6370800 # metres
lonmid = 0.5*(dat.ax[0] + dat.ax[1])
latmid = 0.5*(dat.ax[2] + dat.ax[3])
xd = rearth*(np.deg2rad(lond - lonmid))*np.cos(np.deg2rad(latd))/1000
yd = rearth*np.deg2rad(latd - latmid)/1000

fig, ax = plt.subplots(1, 1)
CS = ax.scatter(xd, yd, s=5, c=t)
plt.colorbar(CS)

# %%
popt = objmap.covfit(xd, yd, t, bins=20, cfunc='gauss', p0=[20, 400], rfitmax=600)
a, l = popt

# %%
nlon = 50
nlat = 100
l = 200

lonm, _ = np.linspace(dat.ax[0], dat.ax[1], nlon, retstep=True)
latm, _ = np.linspace(dat.ax[2], dat.ax[3], nlat, retstep=True)

latmg, lonmg = np.meshgrid(latm, lonm, indexing='ij')
ymg = rearth*np.deg2rad(latmg - latmid)/1000
xmg = rearth*(np.deg2rad(lonmg - lonmid))*np.cos(np.deg2rad(latmg))/1000

tg = objmap.objmap(xd, yd, t, xmg, ymg, a, l, cfunc='gauss', detrend="mean")

# %%
clevs = np.linspace(14, 24, 11)

fig, ax = plt.subplots(1, 1)
C = ax.contourf(lonm, latm, tg, clevs, extend='both')
plt.colorbar(C)

fig, ax = plt.subplots(1, 1)
C = ax.contourf(dat.x, dat.y, dat.temp[k, ...], clevs, extend='both')
plt.colorbar(C)

# %%
