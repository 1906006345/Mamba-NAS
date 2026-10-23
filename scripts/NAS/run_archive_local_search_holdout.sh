#!/usr/bin/env bash
# Run Diversity Archive + Pareto Local Search on four datasets with holdout validation.
#
# Example:
#   NAS_DEVICE=cuda NAS_SEEDS="2021" bash scripts/NAS/run_archive_local_search_holdout.sh
#   NAS_DRY_RUN=1 NAS_DATASETS="Heartbeat" bash scripts/NAS/run_archive_local_search_holdout.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NAS_DEVICE="${NAS_DEVICE:-cuda}"
NAS_BUDGET="${NAS_BUDGET:-paper}"
NAS_DATA_DIR="${NAS_DATA_DIR:-data/UEA}"
NAS_OUTPUT_DIR="${NAS_OUTPUT_DIR:-runs}"
NAS_DATASETS="${NAS_DATASETS:-Heartbeat SpokenArabicDigits UWaveGestureLibrary PEMS-SF}"
NAS_SEEDS="${NAS_SEEDS:-2021}"
NAS_DRY_RUN="${NAS_DRY_RUN:-0}"
NAS_RESUME="${NAS_RESUME:-0}"
NAS_REFIT="${NAS_REFIT:-0}"
NAS_OBJECTIVE_WEIGHT="${NAS_OBJECTIVE_WEIGHT:-0.70}"
NAS_DECISION_WEIGHT="${NAS_DECISION_WEIGHT:-0.30}"

for dataset in ${NAS_DATASETS}; do
  for seed in ${NAS_SEEDS}; do
    run_id="${dataset,,}-holdout-archive_local_search-seed${seed}"
    command=(
      "${PYTHON_BIN}" -m mamba_nas.cli search
      --dataset "${dataset}"
      --budget "${NAS_BUDGET}"
      --seed "${seed}"
      --data-dir "${NAS_DATA_DIR}"
      --output-dir "${NAS_OUTPUT_DIR}"
      --device "${NAS_DEVICE}"
      --validation-strategy holdout
      --search-variant archive_local_search
      --objective-weight "${NAS_OBJECTIVE_WEIGHT}"
      --decision-weight "${NAS_DECISION_WEIGHT}"
      --run-id "${run_id}"
    )
    if [[ "${NAS_RESUME}" == "1" ]]; then
      command+=(--resume)
    fi
    echo
    echo "[NAS] dataset=${dataset} seed=${seed} run_id=${run_id} variant=archive_local_search"
    printf '[NAS] command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "${NAS_DRY_RUN}" == "1" ]]; then
      continue
    fi
    "${command[@]}"
    if [[ "${NAS_REFIT}" == "1" ]]; then
      "${PYTHON_BIN}" -m mamba_nas.cli refit \
        --run "${run_id}" \
        --output-dir "${NAS_OUTPUT_DIR}"
    fi
  done
done

echo "[NAS] archive_local_search holdout runs completed."
