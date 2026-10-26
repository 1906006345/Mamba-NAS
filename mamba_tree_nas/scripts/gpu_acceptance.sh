#!/usr/bin/env bash
# Run only after activating your Conda environment. Real GPU acceptance is small
# and explicitly launched by the user; no downloads or formal suite here.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m mamba_tree_nas verify-environment --gpu-test
python -m pytest mamba_tree_nas/tests -q
python -m mamba_tree_nas search --profile pilot --dataset Heartbeat --variant tree_direction --run-id gpu-acceptance \
  --set search.population=2 --set search.budget=3 --set train.epochs=2 --set train.min_epochs=1 \
  --set calibration.epochs=2 --set calibration.min_epochs=1 --with-refit "$@"
python -m mamba_tree_nas report --run mamba_tree_nas/runs_pilot/Heartbeat/tree_direction/seed_2021/gpu-acceptance \
  --output mamba_tree_nas/paper_exports/gpu-acceptance
