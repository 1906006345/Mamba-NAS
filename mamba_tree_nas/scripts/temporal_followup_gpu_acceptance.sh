#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
py="${PYTHON:-python}"
out="${1:-mamba_tree_nas/work/temporal_followup/gpu_acceptance}"
data="${2:-data/UEA}"
source="${3:-mamba_tree_nas/work/temporal_mutation/search/paper}"
"$py" -m mamba_tree_nas verify-environment --profile pilot
common=(--profile pilot --dataset Heartbeat --output "$out" --source-root "$source" --data-root "$data")
for stage in audit census diagnose; do
  "$py" -m mamba_tree_nas.temporal_followup "$stage" "${common[@]}" --execute
done
"$py" -m mamba_tree_nas.temporal_followup search "${common[@]}" --with-refit --execute
"$py" -m mamba_tree_nas.temporal_followup report "${common[@]}" --execute
