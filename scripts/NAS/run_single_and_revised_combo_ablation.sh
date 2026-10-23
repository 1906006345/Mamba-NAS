#!/usr/bin/env bash
# Run baseline, single mechanisms, original combo and revised combo.
# By default Heartbeat uses 5-fold K-fold; other datasets use holdout.
# Override NAS_KFOLD_DATASETS to run K-fold for additional datasets.
# Completed runs are skipped by default; set NAS_SKIP_COMPLETED=0 to rerun.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NAS_DEVICE="${NAS_DEVICE:-cuda}"
NAS_BUDGET="${NAS_BUDGET:-paper}"
NAS_DATA_DIR="${NAS_DATA_DIR:-data/UEA}"
NAS_OUTPUT_DIR="${NAS_OUTPUT_DIR:-runs}"
NAS_SEEDS="${NAS_SEEDS:-2021 2022 2023}"
NAS_DATASETS="${NAS_DATASETS:-Heartbeat SpokenArabicDigits PEMS-SF}"
NAS_REFIT="${NAS_REFIT:-1}"
NAS_DRY_RUN="${NAS_DRY_RUN:-0}"
NAS_RESUME="${NAS_RESUME:-0}"
NAS_OBJECTIVE_WEIGHT="${NAS_OBJECTIVE_WEIGHT:-0.70}"
NAS_DECISION_WEIGHT="${NAS_DECISION_WEIGHT:-0.30}"
NAS_LOCAL_SEARCH_RATIO="${NAS_LOCAL_SEARCH_RATIO:-0.20}"
NAS_ARCHIVE_SEED_RATIO="${NAS_ARCHIVE_SEED_RATIO:-0.30}"
NAS_PARETO_SEED_RATIO="${NAS_PARETO_SEED_RATIO:-0.50}"
NAS_POPULATION_SEED_RATIO="${NAS_POPULATION_SEED_RATIO:-0.20}"
NAS_SKIP_COMPLETED="${NAS_SKIP_COMPLETED:-1}"
NAS_V2_RUN_TAG="${NAS_V2_RUN_TAG:-slotsfix}"
NAS_KFOLD_DATASETS="${NAS_KFOLD_DATASETS:-Heartbeat}"
NAS_CV_FOLDS="${NAS_CV_FOLDS:-5}"
NAS_RUN_TAG="${NAS_RUN_TAG:-}"

METHODS=(baseline diversity_archive local_search archive_local_search archive_local_search_v2)

search_is_complete() {
  local dataset="$1" validation="$2" variant="$3" seed="$4" run_id="$5"
  [[ "${NAS_SKIP_COMPLETED}" == "1" ]] || return 1
  "${PYTHON_BIN}" - "${NAS_OUTPUT_DIR}" "${run_id}" "${dataset}" "${validation}" "${variant}" "${seed}" "${NAS_LOCAL_SEARCH_RATIO}" "${NAS_CV_FOLDS}" <<'PY'
import json
import math
import sys
from pathlib import Path

output, run_id, dataset, validation, variant, seed, revised_ratio, cv_folds = sys.argv[1:]
root = Path(output) / run_id / dataset
manifest_path = root / "manifest.json"
front_path = root / "search" / "pareto_front.csv"
if not manifest_path.exists() or not front_path.exists():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("search_completed") is not True:
    raise SystemExit(1)
config = manifest.get("config", {})
if config.get("validation_strategy", "holdout") != validation:
    raise SystemExit(1)
if validation == "kfold" and int(config.get("cv_folds", 5)) != int(cv_folds):
    raise SystemExit(1)
if int(config.get("seed", 2021)) != int(seed):
    raise SystemExit(1)
if config.get("search_variant", "baseline") != variant:
    raise SystemExit(1)
if variant == "archive_local_search_v2":
    if not config.get("use_macro_micro_archive", False):
        raise SystemExit(1)
    if not math.isclose(float(config.get("local_search_ratio", 0.30)), float(revised_ratio), rel_tol=0, abs_tol=1e-9):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_one() {
  local dataset="$1" validation="$2" variant="$3" seed="$4"
  local run_variant="${variant}"
  local tag="${NAS_RUN_TAG}"
  if [[ "${variant}" == "archive_local_search_v2" && -n "${NAS_V2_RUN_TAG}" ]]; then
    tag="${tag}-${NAS_V2_RUN_TAG}"
  fi
  local run_id="${dataset,,}-${validation}${tag}-${run_variant}-seed${seed}"
  local local_ratio="0.30"
  if [[ "${variant}" == "archive_local_search_v2" ]]; then
    local_ratio="${NAS_LOCAL_SEARCH_RATIO}"
  fi
  local command=(
    "${PYTHON_BIN}" -m mamba_nas.cli search
    --dataset "${dataset}" --budget "${NAS_BUDGET}" --seed "${seed}"
    --data-dir "${NAS_DATA_DIR}" --output-dir "${NAS_OUTPUT_DIR}"
    --device "${NAS_DEVICE}" --validation-strategy "${validation}"
    --search-variant "${variant}"
    --objective-weight "${NAS_OBJECTIVE_WEIGHT}"
    --decision-weight "${NAS_DECISION_WEIGHT}"
    --local-search-ratio "${local_ratio}"
    --archive-seed-ratio "${NAS_ARCHIVE_SEED_RATIO}"
    --pareto-seed-ratio "${NAS_PARETO_SEED_RATIO}"
    --population-seed-ratio "${NAS_POPULATION_SEED_RATIO}"
    --run-id "${run_id}"
  )
  if [[ "${validation}" == "kfold" ]]; then command+=(--cv-folds "${NAS_CV_FOLDS}"); fi
  if [[ "${NAS_RESUME}" == "1" ]]; then command+=(--resume); fi
  echo
  echo "[NAS] dataset=${dataset} validation=${validation} variant=${variant} seed=${seed}"
  if search_is_complete "${dataset}" "${validation}" "${variant}" "${seed}" "${run_id}"; then
    echo "[NAS] search already completed; skip: ${run_id}/${dataset}"
    if [[ "${NAS_REFIT}" == "1" && -f "${NAS_OUTPUT_DIR}/${run_id}/${dataset}/final/summary.csv" ]]; then
      echo "[NAS] refit already completed; skip: ${run_id}/${dataset}"
    elif [[ "${NAS_REFIT}" == "1" && "${NAS_DRY_RUN}" != "1" ]]; then
      echo "[NAS] search exists but refit is missing; run refit: ${run_id}/${dataset}"
      "${PYTHON_BIN}" -m mamba_nas.cli refit --run "${run_id}" --output-dir "${NAS_OUTPUT_DIR}"
    fi
    return
  fi
  printf '[NAS] command:'; printf ' %q' "${command[@]}"; printf '\n'
  if [[ "${NAS_DRY_RUN}" == "1" ]]; then return; fi
  "${command[@]}"
  if [[ "${NAS_REFIT}" == "1" ]]; then
    "${PYTHON_BIN}" -m mamba_nas.cli refit --run "${run_id}" --output-dir "${NAS_OUTPUT_DIR}"
  fi
}

echo "[NAS] methods=${METHODS[*]} seeds=${NAS_SEEDS} budget=${NAS_BUDGET} refit=${NAS_REFIT}"
echo "[NAS] kfold_datasets=${NAS_KFOLD_DATASETS} cv_folds=${NAS_CV_FOLDS} run_tag=${NAS_RUN_TAG}"
validation_for() {
  local dataset="$1"
  for candidate in ${NAS_KFOLD_DATASETS}; do
    [[ "${candidate}" == "${dataset}" ]] && { echo kfold; return; }
  done
  echo holdout
}
for seed in ${NAS_SEEDS}; do
  for variant in "${METHODS[@]}"; do
    for dataset in ${NAS_DATASETS}; do
      run_one "${dataset}" "$(validation_for "${dataset}")" "${variant}" "${seed}"
    done
  done
done
echo "[NAS] single/revised-combination ablation completed."
