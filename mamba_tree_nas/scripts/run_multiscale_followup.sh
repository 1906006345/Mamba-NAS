#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# cuBLAS reads this before CUDA work begins; keep explicit caller settings.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
python -m mamba_tree_nas.followup "$@"
