#!/usr/bin/env bash
# Sensitivity study for archive_local_search's local_search_ratio.
# Default datasets intentionally exclude UWaveGestureLibrary.

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
NAS_RATIOS="${NAS_RATIOS:-0.10 0.20 0.30 0.40}"
NAS_REFIT="${NAS_REFIT:-0}"
NAS_RESUME="${NAS_RESUME:-0}"
NAS_DRY_RUN="${NAS_DRY_RUN:-0}"
NAS_SKIP_COMPLETED="${NAS_SKIP_COMPLETED:-1}"
NAS_RUN_TAG="${NAS_RUN_TAG:--ratio-sensitivity}"
NAS_OBJECTIVE_WEIGHT="${NAS_OBJECTIVE_WEIGHT:-0.70}"
NAS_DECISION_WEIGHT="${NAS_DECISION_WEIGHT:-0.30}"

ratio_tag() {
  echo "${1//./p}"
}

is_complete() {
  local run_id="$1" dataset="$2" seed="$3" ratio="$4"
  [[ "${NAS_SKIP_COMPLETED}" == "1" ]] || return 1
  "${PYTHON_BIN}" - "${NAS_OUTPUT_DIR}" "${run_id}" "${dataset}" "${seed}" "${ratio}" <<'PY'
import json
import math
import sys
from pathlib import Path

output, run_id, dataset, seed, ratio = sys.argv[1:]
root = Path(output) / run_id / dataset
manifest_path = root / "manifest.json"
front_path = root / "search" / "pareto_front.csv"
if not manifest_path.exists() or not front_path.exists():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config = manifest.get("config", {})
if manifest.get("search_completed") is not True:
    raise SystemExit(1)
if config.get("search_algorithm", "nsga2") != "nsga2":
    raise SystemExit(1)
if config.get("search_variant", "baseline") != "archive_local_search":
    raise SystemExit(1)
if int(config.get("seed", 2021)) != int(seed):
    raise SystemExit(1)
if not math.isclose(float(config.get("local_search_ratio", 0.30)), float(ratio), rel_tol=0, abs_tol=1e-9):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_one() {
  local dataset="$1" seed="$2" ratio="$3"
  local tag="${NAS_RUN_TAG}"
  local ratio_suffix="$(ratio_tag "${ratio}")"
  local run_id="${dataset,,}-holdout${tag}-archive_local_search-ratio${ratio_suffix}-seed${seed}"
  local command=(
    "${PYTHON_BIN}" -m mamba_nas.cli search
    --dataset "${dataset}" --budget "${NAS_BUDGET}" --seed "${seed}"
    --data-dir "${NAS_DATA_DIR}" --output-dir "${NAS_OUTPUT_DIR}"
    --device "${NAS_DEVICE}" --validation-strategy holdout
    --search-algorithm nsga2 --search-variant archive_local_search
    --local-search-ratio "${ratio}"
    --objective-weight "${NAS_OBJECTIVE_WEIGHT}"
    --decision-weight "${NAS_DECISION_WEIGHT}"
    --run-id "${run_id}"
  )
  [[ "${NAS_RESUME}" == "1" ]] && command+=(--resume)
  echo
  echo "[NAS] dataset=${dataset} seed=${seed} local_search_ratio=${ratio} run_id=${run_id}"
  if is_complete "${run_id}" "${dataset}" "${seed}" "${ratio}"; then
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

echo "[NAS] local_search_ratio sensitivity"
echo "[NAS] datasets=${NAS_DATASETS} seeds=${NAS_SEEDS} ratios=${NAS_RATIOS}"
for seed in ${NAS_SEEDS}; do
  for ratio in ${NAS_RATIOS}; do
    for dataset in ${NAS_DATASETS}; do
      run_one "${dataset}" "${seed}" "${ratio}"
    done
  done
done
echo "[NAS] local_search_ratio sensitivity runs completed."
