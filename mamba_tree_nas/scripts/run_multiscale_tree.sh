#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m mamba_tree_nas run-multiscale "$@"
