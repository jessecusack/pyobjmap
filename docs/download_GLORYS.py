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
# # Download reanalysis data from Copernicus
#
# You need an account with Copernicus marine.
#
# The dataset: https://resources.marine.copernicus.eu/product-detail/GLOBAL_MULTIYEAR_PHY_001_030/INFORMATION
#

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import getpass

import copernicusmarine as cm

# %% [markdown]
# Provide Copernicus Marine login details.

# %%
USERNAME = 'jcusack1'
PASSWORD = getpass.getpass('Enter your password: ')

# %%
DATASET_ID = "cmems_mod_glo_phy_my_0.083_P1D-m"

# Drop sea ice variables
ds = xr.open_dataset(cm.copernicusmarine_datastore(DATASET_ID, USERNAME, PASSWORD), drop_variables=["sithick", "siconc", "usi", "vsi"])
ds

# %% [markdown]
# ## Investigate the dataset

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ds.zos.isel(time=-1).sel(longitude=slice(-100, 40), latitude=slice(0, 80)).plot(ax=ax)

# %% [markdown]
# ## Save a small region

# %%
west = 30
east = 80
south = -50
north = -30
time = "2015-06-17"
idepth = 0

dsbox = ds.isel(depth=idepth).sel(longitude=slice(west, east), latitude=slice(south, north)).sel(time=time, method="nearest")

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
dsbox.zos.plot(ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
dsbox.thetao.plot(ax=ax)

step = 5  # reduce for plotting
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.quiver(
    dsbox.longitude[::step],
    dsbox.latitude[::step],
    dsbox.uo[::step, ::step],
    dsbox.vo[::step, ::step],
)

# %% [markdown]
# Save box to netcdf...

# %%
dsbox.to_netcdf("../data/small_glorys_region.nc")
