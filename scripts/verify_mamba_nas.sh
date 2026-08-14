#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba-tsc-nas
python -m mamba_nas.cli verify-environment
pytest -q mamba_nas/tests

