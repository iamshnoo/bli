#!/usr/bin/env bash
set -euo pipefail

source /home/amukher6/nanotron-env/bin/activate
cd /scratch/amukher6/bli

python src/pipeline/submit_train_convert_analysis.py "$@"
