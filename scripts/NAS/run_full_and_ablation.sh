#!/usr/bin/env bash
# Run the proposed full Mamba-NAS method and its fixed ablation suite.
#
# Usage:
#   bash scripts/NAS/run_full_and_ablation.sh [full|ablation|all]
#
# Default (all) runs the six search variants below, sequentially, on the three
# target datasets and NAS seeds 2021/2022/2023.  Override runtime settings with
# environment variables, for example:
#   NAS_DEVICE=cuda NAS_BUDGET=paper bash scripts/NAS/run_full_and_ablation.sh all
#   NAS_SEEDS="2021" NAS_DATASETS="Heartbeat" NAS_DRY_RUN=1 \
#     bash scripts/NAS/run_full_and_ablation.sh full
#   NAS_RESUME=1 bash scripts/NAS/run_full_and_ablation.sh full
#
# Set NAS_REFIT_FULL=1 to refit the final representatives of the full method
# after every full-method search.  It is disabled by default because refit
# accesses TEST and is normally performed only after selecting a final method.

set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  full|ablation|all) ;;
  *)
    echo "Usage: $0 [full|ablation|all]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NAS_DEVICE="${NAS_DEVICE:-cuda}"
NAS_BUDGET="${NAS_BUDGET:-paper}"
NAS_VALIDATION_STRATEGY="${NAS_VALIDATION_STRATEGY:-holdout}"
NAS_CV_FOLDS="${NAS_CV_FOLDS:-5}"
NAS_DATA_DIR="${NAS_DATA_DIR:-data/UEA}"
NAS_OUTPUT_DIR="${NAS_OUTPUT_DIR:-runs}"
NAS_DATASETS="${NAS_DATASETS:-Heartbeat SpokenArabicDigits UWaveGestureLibrary PEMS-SF}"
NAS_SEEDS="${NAS_SEEDS:-2021 2022 2023}"
NAS_REFIT_FULL="${NAS_REFIT_FULL:-0}"
NAS_DRY_RUN="${NAS_DRY_RUN:-0}"
NAS_RESUME="${NAS_RESUME:-0}"
NAS_OBJECTIVE_WEIGHT="${NAS_OBJECTIVE_WEIGHT:-0.70}"
NAS_DECISION_WEIGHT="${NAS_DECISION_WEIGHT:-0.30}"

# Fixed ablation protocol from the method plan:
# A0 baseline; A1 decision crowding; A2 archive; A3 local search;
# A4 dual space; A5 archive + local search; A6 full method.
ABLATION_VARIANTS=(
  baseline
  dual_crowding
  diversity_archive
  local_search
  dual_space
  archive_local_search
)
FULL_VARIANT="full"

if [[ "${MODE}" == "full" ]]; then
  VARIANTS=("${FULL_VARIANT}")
elif [[ "${MODE}" == "ablation" ]]; then
  VARIANTS=("${ABLATION_VARIANTS[@]}")
else
  VARIANTS=("${ABLATION_VARIANTS[@]}" "${FULL_VARIANT}")
fi

if [[ "${NAS_VALIDATION_STRATEGY}" != "holdout" && "${NAS_VALIDATION_STRATEGY}" != "kfold" ]]; then
  echo "NAS_VALIDATION_STRATEGY must be holdout or kfold, got: ${NAS_VALIDATION_STRATEGY}" >&2
  exit 2
fi

run_search() {
  local dataset="$1"
  local variant="$2"
  local seed="$3"
  local weight_suffix=""
  if [[ "${NAS_OBJECTIVE_WEIGHT}" != "0.70" || "${NAS_DECISION_WEIGHT}" != "0.30" ]]; then
    local objective_tag="${NAS_OBJECTIVE_WEIGHT//./p}"
    local decision_tag="${NAS_DECISION_WEIGHT//./p}"
    weight_suffix="-ow${objective_tag}-dw${decision_tag}"
  fi
  local run_id="${dataset,,}-${NAS_VALIDATION_STRATEGY}-${variant}${weight_suffix}-seed${seed}"
  local command=(
    "${PYTHON_BIN}" -m mamba_nas.cli search
    --dataset "${dataset}"
    --budget "${NAS_BUDGET}"
    --seed "${seed}"
    --data-dir "${NAS_DATA_DIR}"
    --output-dir "${NAS_OUTPUT_DIR}"
    --device "${NAS_DEVICE}"
    --validation-strategy "${NAS_VALIDATION_STRATEGY}"
    --search-variant "${variant}"
    --objective-weight "${NAS_OBJECTIVE_WEIGHT}"
    --decision-weight "${NAS_DECISION_WEIGHT}"
    --run-id "${run_id}"
  )

  if [[ "${NAS_VALIDATION_STRATEGY}" == "kfold" ]]; then
    command+=(--cv-folds "${NAS_CV_FOLDS}")
  fi
  if [[ "${NAS_RESUME}" == "1" ]]; then
    command+=(--resume)
  fi

  echo
  echo "[NAS] dataset=${dataset} variant=${variant} seed=${seed} run_id=${run_id}"
  printf '[NAS] command:'
  printf ' %q' "${command[@]}"
  printf '\n'

  if [[ "${NAS_DRY_RUN}" == "1" ]]; then
    return
  fi
  "${command[@]}"

  if [[ "${variant}" == "${FULL_VARIANT}" && "${NAS_REFIT_FULL}" == "1" ]]; then
    echo "[NAS] refit final Pareto representatives for ${run_id}"
    "${PYTHON_BIN}" -m mamba_nas.cli refit \
      --run "${run_id}" \
      --output-dir "${NAS_OUTPUT_DIR}"
  fi
}

echo "[NAS] mode=${MODE}, budget=${NAS_BUDGET}, device=${NAS_DEVICE}, validation=${NAS_VALIDATION_STRATEGY}"
echo "[NAS] crowding_weights=objective:${NAS_OBJECTIVE_WEIGHT},decision:${NAS_DECISION_WEIGHT}"
echo "[NAS] datasets=${NAS_DATASETS}"
echo "[NAS] seeds=${NAS_SEEDS}"
echo "[NAS] variants=${VARIANTS[*]}"

for dataset in ${NAS_DATASETS}; do
  for seed in ${NAS_SEEDS}; do
    for variant in "${VARIANTS[@]}"; do
      run_search "${dataset}" "${variant}" "${seed}"
    done
  done
done

echo
echo "[NAS] completed mode=${MODE}."
