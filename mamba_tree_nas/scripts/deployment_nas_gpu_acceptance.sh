#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
runner=mamba_tree_nas/scripts/run_deployment_nas.sh
out="${1:-mamba_tree_nas/work/deployment_nas/gpu_acceptance}"
data="${2:-data/UEA}"
common=(--profile pilot --dataset Heartbeat --output "$out" --data-root "$data")
for stage in benchmark objectives rank compare report; do
  bash "$runner" "$stage" "${common[@]}" --execute
done
"${PYTHON:-python}" -m mamba_tree_nas.deployment_nas_acceptance "$out"
