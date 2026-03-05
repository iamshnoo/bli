#!/usr/bin/env bash
set -euo pipefail

BLI_ROOT="${BLI_ROOT:-/scratch/$USER/bli}"
NANOTRON_ENV="${NANOTRON_ENV:-$HOME/nanotron-env}"
source "${NANOTRON_ENV}/bin/activate"
cd "${BLI_ROOT}"

python src/pipeline/submit_train_convert_analysis.py "$@"
