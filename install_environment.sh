#!/usr/bin/env bash
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate pyobjmap 
python -m ipykernel install --user --name pyobjmap
jupytext --to=ipynb docs/*.py