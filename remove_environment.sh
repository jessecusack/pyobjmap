#!/usr/bin/env bash
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate pyobjmap && jupyter kernelspec uninstall pyobjmap && conda deactivate
conda remove --name pyobjmap --all
