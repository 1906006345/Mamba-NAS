#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
"${PYTHON:-python}" -m mamba_tree_nas.deployment_nas "$@"
