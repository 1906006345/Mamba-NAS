#!/usr/bin/env bash
# Compare baseline NSGA-II, scan-order NSGA-II and scan-order + dual-space crowding.
# This script intentionally does not enable Diversity Archive or Local Search.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NAS_DEVICE="${NAS_DEVICE:-cuda}"
NAS_BUDGET="${NAS_BUDGET:-paper}"
NAS_DATA_DIR="${NAS_DATA_DIR:-data/UEA}"
NAS_OUTPUT_DIR="${NAS_OUTPUT_DIR:-runs}"
NAS_DATASETS="${NAS_DATASETS:-Heartbeat SpokenArabicDigits PEMS-SF}"
NAS_SEEDS="${NAS_SEEDS:-2021 2022 2023}"
NAS_REFIT="${NAS_REFIT:-1}"
NAS_RESUME="${NAS_RESUME:-0}"
NAS_DRY_RUN="${NAS_DRY_RUN:-0}"
NAS_SKIP_COMPLETED="${NAS_SKIP_COMPLETED:-1}"
NAS_RUN_TAG="${NAS_RUN_TAG:--scan-dual-crowding}"
NAS_OBJECTIVE_WEIGHT="${NAS_OBJECTIVE_WEIGHT:-0.70}"
NAS_DECISION_WEIGHT="${NAS_DECISION_WEIGHT:-0.30}"

METHODS=(baseline scan_order scan_order_dual_crowding)

is_complete() {
  local run_id="$1" dataset="$2" seed="$3" variant="$4"
  [[ "${NAS_SKIP_COMPLETED}" == "1" ]] || return 1
  "${PYTHON_BIN}" - "${NAS_OUTPUT_DIR}" "${run_id}" "${dataset}" "${seed}" "${variant}" <<'PY'
import json
import sys
from pathlib import Path

output, run_id, dataset, seed, variant = sys.argv[1:]
root = Path(output) / run_id / dataset
manifest_path = root / "manifest.json"
if not manifest_path.exists() or not (root / "search" / "pareto_front.csv").exists():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config = manifest.get("config", {})
if manifest.get("search_completed") is not True:
    raise SystemExit(1)
if config.get("search_algorithm", "nsga2") != "nsga2":
    raise SystemExit(1)
if config.get("search_variant", "baseline") != variant:
    raise SystemExit(1)
if config.get("validation_strategy", "holdout") != "holdout":
    raise SystemExit(1)
if int(config.get("seed", 2021)) != int(seed):
    raise SystemExit(1)
if variant in {"scan_order", "scan_order_dual_crowding"}:
    if not config.get("use_scan_order", False):
        raise SystemExit(1)
    if config.get("hypervolume_space", "active") != "scan_order":
        raise SystemExit(1)
if variant == "scan_order_dual_crowding":
    if not config.get("use_decision_crowding", False):
        raise SystemExit(1)
    if config.get("use_diversity_archive", False) or config.get("use_pareto_local_search", False):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_one() {
  local dataset="$1" seed="$2" variant="$3"
  local run_id="${dataset,,}-holdout${NAS_RUN_TAG}-${variant}-seed${seed}"
  local command=(
    "${PYTHON_BIN}" -m mamba_nas.cli search
    --dataset "${dataset}" --budget "${NAS_BUDGET}" --seed "${seed}"
    --data-dir "${NAS_DATA_DIR}" --output-dir "${NAS_OUTPUT_DIR}"
    --device "${NAS_DEVICE}" --validation-strategy holdout
    --search-algorithm nsga2 --search-variant "${variant}"
    --objective-weight "${NAS_OBJECTIVE_WEIGHT}"
    --decision-weight "${NAS_DECISION_WEIGHT}"
    --hypervolume-space scan_order
    --run-id "${run_id}"
  )
  [[ "${NAS_RESUME}" == "1" ]] && command+=(--resume)
  echo
  echo "[NAS] dataset=${dataset} seed=${seed} variant=${variant} run_id=${run_id}"
  if is_complete "${run_id}" "${dataset}" "${seed}" "${variant}"; then
    echo "[NAS] completed search; skip ${run_id}/${dataset}"
    if [[ "${NAS_REFIT}" == "1" && ! -f "${NAS_OUTPUT_DIR}/${run_id}/${dataset}/final/summary.csv" && "${NAS_DRY_RUN}" != "1" ]]; then
      "${PYTHON_BIN}" -m mamba_nas.cli refit --run "${run_id}" --output-dir "${NAS_OUTPUT_DIR}"
    fi
    return
  fi
  printf '[NAS] command:'; printf ' %q' "${command[@]}"; printf '\n'
  [[ "${NAS_DRY_RUN}" == "1" ]] && return
  "${command[@]}"
  if [[ "${NAS_REFIT}" == "1" ]]; then
    "${PYTHON_BIN}" -m mamba_nas.cli refit --run "${run_id}" --output-dir "${NAS_OUTPUT_DIR}"
  fi
}

echo "[NAS] scan-order dual-crowding ablation datasets=${NAS_DATASETS} seeds=${NAS_SEEDS}"
for seed in ${NAS_SEEDS}; do
  for variant in "${METHODS[@]}"; do
    for dataset in ${NAS_DATASETS}; do
      run_one "${dataset}" "${seed}" "${variant}"
    done
  done
done
echo "[NAS] scan-order dual-crowding ablation completed."
