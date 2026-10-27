#!/usr/bin/env bash
# Uses the active Conda Python; all budgets/seeds/options come from config.yaml.
# Default: traditional serial NSGA-II + Random Search on the primary datasets.
# Add --include-tree only to report existing functional-tree reference runs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
exec "${PYTHON_BIN:-python}" mamba_tree_nas/scripts/run.py traditional-comparison "$@"
