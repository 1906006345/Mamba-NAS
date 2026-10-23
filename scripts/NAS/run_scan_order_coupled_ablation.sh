#!/usr/bin/env bash
# Four-group scan-order and coupled-local-search ablation.

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
NAS_LOCAL_SEARCH_RATIO="${NAS_LOCAL_SEARCH_RATIO:-0.20}"
NAS_COUPLED_NEIGHBOR_RATIO="${NAS_COUPLED_NEIGHBOR_RATIO:-0.30}"
NAS_RUN_TAG="${NAS_RUN_TAG:--scan-ablation}"

METHODS=(baseline scan_order scan_order_local_search scan_order_coupled_local_search)

is_complete() {
  local run_id="$1" dataset="$2" seed="$3" variant="$4"
  [[ "${NAS_SKIP_COMPLETED}" == "1" ]] || return 1
  "${PYTHON_BIN}" - "${NAS_OUTPUT_DIR}" "${run_id}" "${dataset}" "${seed}" "${variant}" "${NAS_LOCAL_SEARCH_RATIO}" "${NAS_COUPLED_NEIGHBOR_RATIO}" <<'PY'
import json
import math
import sys
from pathlib import Path

output, run_id, dataset, seed, variant, local_ratio, coupled_ratio = sys.argv[1:]
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
if config.get("hypervolume_space", "active") != "scan_order":
    raise SystemExit(1)
if int(config.get("seed", 2021)) != int(seed):
    raise SystemExit(1)
scan_variants = {"scan_order", "scan_order_local_search", "scan_order_coupled_local_search"}
if bool(config.get("use_scan_order", False)) != (variant in scan_variants):
    raise SystemExit(1)
if variant in {"scan_order_local_search", "scan_order_coupled_local_search"}:
    if not math.isclose(float(config.get("local_search_ratio", 0.20)), float(local_ratio), abs_tol=1e-9):
        raise SystemExit(1)
if variant == "scan_order_coupled_local_search":
    if not config.get("use_coupled_local_search", False):
        raise SystemExit(1)
    if not math.isclose(float(config.get("coupled_neighbor_ratio", 0.30)), float(coupled_ratio), abs_tol=1e-9):
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
    --hypervolume-space scan_order
    --run-id "${run_id}"
  )
  if [[ "${variant}" == "scan_order_local_search" || "${variant}" == "scan_order_coupled_local_search" ]]; then
    command+=(--local-search-ratio "${NAS_LOCAL_SEARCH_RATIO}")
  fi
  if [[ "${variant}" == "scan_order_coupled_local_search" ]]; then
    command+=(--coupled-neighbor-ratio "${NAS_COUPLED_NEIGHBOR_RATIO}")
  fi
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

echo "[NAS] scan-order ablation datasets=${NAS_DATASETS} seeds=${NAS_SEEDS}"
for seed in ${NAS_SEEDS}; do
  for variant in "${METHODS[@]}"; do
    for dataset in ${NAS_DATASETS}; do
      run_one "${dataset}" "${seed}" "${variant}"
    done
  done
done
echo "[NAS] scan-order coupled-local-search ablation completed."
