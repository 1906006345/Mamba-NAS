#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]] || ! grep -qi microsoft /proc/version; then
  echo "Run this script inside Ubuntu 22.04 on WSL2." >&2
  echo "From an elevated Windows PowerShell: wsl --install -d Ubuntu-22.04" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "The WSL NVIDIA driver is unavailable. Install/update the Windows NVIDIA driver." >&2
  exit 3
fi

if ! command -v conda >/dev/null 2>&1; then
  installer="/tmp/miniconda-mamba-nas.sh"
  curl -fsSL -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$installer" -b -p "$HOME/miniconda3"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda env create -f environment-mamba-nas.yml || conda env update -f environment-mamba-nas.yml --prune
conda activate mamba-tsc-nas
python -m pip install causal-conv1d==1.4.0 --no-build-isolation
python -m pip install mamba-ssm==2.1.0 --no-build-isolation
python -m mamba_nas.cli verify-environment
python -m mamba_nas.cli synthetic-smoke --budget smoke --device cuda

