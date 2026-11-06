#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m mamba_tree_nas run-multiscale --profile pilot --dataset Heartbeat \
  --search-seed 2021 --run-id multiscale-gpu-acceptance --multiscale-gpu-check --with-refit \
  --set search.population=2 --set search.budget=3 --set train.epochs=2 --set train.min_epochs=1 \
  --set calibration.epochs=2 --set calibration.min_epochs=1 "$@"
