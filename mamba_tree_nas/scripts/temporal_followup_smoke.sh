#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
out="${1:-mamba_tree_nas/work/temporal_followup/smoke_acceptance}"
py="${PYTHON:-python}"
"$py" -m pytest mamba_tree_nas/tests/test_temporal_followup.py -q --basetemp=mamba_tree_nas/work/pytest_followup
for stage in audit census diagnose; do
  "$py" -m mamba_tree_nas.temporal_followup "$stage" --profile smoke --dataset Heartbeat --output "$out" --execute
done
"$py" -m mamba_tree_nas.temporal_followup search --profile smoke --dataset Heartbeat --output "$out" --with-refit --execute
# A second invocation validates completed-run reuse, including shared refits.
"$py" -m mamba_tree_nas.temporal_followup search --profile smoke --dataset Heartbeat --output "$out" --with-refit --execute
"$py" -m mamba_tree_nas.temporal_followup report --profile smoke --output "$out" --execute
