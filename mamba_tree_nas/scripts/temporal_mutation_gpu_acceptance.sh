#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
bash mamba_tree_nas/scripts/run_temporal_mutation.sh search --profile pilot --dataset Heartbeat --seed 2021 \
  --output mamba_tree_nas/work/temporal_mutation/gpu-acceptance --execute \
  --set search.population=3 --set search.budget=6 --set train.epochs=2 --set train.min_epochs=1 "$@"
