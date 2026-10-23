#!/usr/bin/env bash
# Compare independent Random Search, Vanilla NSGA-II and archive_local_search.
# The default protocol uses holdout for all datasets. Set NAS_KFOLD_DATASETS to
# a space-separated list only when a K-fold comparison is explicitly desired.

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
NAS_CV_FOLDS="${NAS_CV_FOLDS:-5}"
NAS_REFIT="${NAS_REFIT:-0}"
NAS_RESUME="${NAS_RESUME:-0}"
NAS_DRY_RUN="${NAS_DRY_RUN:-0}"
NAS_SKIP_COMPLETED="${NAS_SKIP_COMPLETED:-1}"
NAS_KFOLD_DATASETS="${NAS_KFOLD_DATASETS:-}"
# Use a new default tag because older archive_local_search runs were produced
# before the unified unique-offspring protocol and lack comparable timing
# metadata. Set NAS_RUN_TAG explicitly to reuse a different naming scheme.
NAS_RUN_TAG="${NAS_RUN_TAG:--algcmp-v2}"
NAS_OBJECTIVE_WEIGHT="${NAS_OBJECTIVE_WEIGHT:-0.70}"
NAS_DECISION_WEIGHT="${NAS_DECISION_WEIGHT:-0.30}"

METHODS=(random nsga2 archive_local_search)

validation_for() {
  local dataset="$1"
  for candidate in ${NAS_KFOLD_DATASETS}; do
    [[ "${candidate}" == "${dataset}" ]] && { echo kfold; return; }
  done
  echo holdout
}

is_complete() {
  local run_id="$1" dataset="$2" algorithm="$3" variant="$4" validation="$5" seed="$6"
  [[ "${NAS_SKIP_COMPLETED}" == "1" ]] || return 1
  "${PYTHON_BIN}" - "${NAS_OUTPUT_DIR}" "${run_id}" "${dataset}" "${algorithm}" "${variant}" "${validation}" "${seed}" "${NAS_CV_FOLDS}" <<'PY'
import json
import sys
from pathlib import Path

output, run_id, dataset, algorithm, variant, validation, seed, folds = sys.argv[1:]
root = Path(output) / run_id / dataset
manifest_path = root / "manifest.json"
if not (manifest_path.exists() and (root / "search" / "pareto_front.csv").exists()):
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config = manifest.get("config", {})
if manifest.get("search_completed") is not True:
    raise SystemExit(1)
if config.get("search_algorithm", "nsga2") != algorithm:
    raise SystemExit(1)
if config.get("search_variant", "baseline") != variant:
    raise SystemExit(1)
if config.get("validation_strategy", "holdout") != validation:
    raise SystemExit(1)
if validation == "kfold" and int(config.get("cv_folds", 5)) != int(folds):
    raise SystemExit(1)
if int(config.get("seed", 2021)) != int(seed):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_one() {
  local dataset="$1" algorithm="$2" seed="$3"
  local validation
  validation="$(validation_for "${dataset}")"
  local variant="baseline"
  [[ "${algorithm}" == "archive_local_search" ]] && variant="archive_local_search"
  local engine="${algorithm}"
  [[ "${algorithm}" == "archive_local_search" ]] && engine="nsga2"
  local tag="${NAS_RUN_TAG}"
  local run_id="${dataset,,}-${validation}${tag}-${algorithm}-seed${seed}"
  local command=(
    "${PYTHON_BIN}" -m mamba_nas.cli search
    --dataset "${dataset}" --budget "${NAS_BUDGET}" --seed "${seed}"
    --data-dir "${NAS_DATA_DIR}" --output-dir "${NAS_OUTPUT_DIR}"
    --device "${NAS_DEVICE}" --validation-strategy "${validation}"
    --search-algorithm "${engine}" --search-variant "${variant}"
    --objective-weight "${NAS_OBJECTIVE_WEIGHT}"
    --decision-weight "${NAS_DECISION_WEIGHT}"
    --run-id "${run_id}"
  )
  [[ "${validation}" == "kfold" ]] && command+=(--cv-folds "${NAS_CV_FOLDS}")
  [[ "${NAS_RESUME}" == "1" ]] && command+=(--resume)
  echo
  echo "[NAS] dataset=${dataset} validation=${validation} algorithm=${algorithm} seed=${seed}"
  if is_complete "${run_id}" "${dataset}" "${engine}" "${variant}" "${validation}" "${seed}"; then
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

for seed in ${NAS_SEEDS}; do
  for algorithm in "${METHODS[@]}"; do
    for dataset in ${NAS_DATASETS}; do
      run_one "${dataset}" "${algorithm}" "${seed}"
    done
  done
done
echo "[NAS] search algorithm comparison completed."
