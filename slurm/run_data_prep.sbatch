#!/bin/bash
#SBATCH --job-name=bli2_prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/%u/bli/logs/slurm_logs/pipeline_prep-%j.out
#SBATCH --error=/scratch/%u/bli/logs/slurm_logs/pipeline_prep-%j.err

set -euo pipefail

BLI_ROOT="${BLI_ROOT:-/scratch/$USER/bli}"
NANOTRON_ENV="${NANOTRON_ENV:-$HOME/nanotron-env}"
source "${NANOTRON_ENV}/bin/activate"
cd "${BLI_ROOT}"

set +x
set -a; source "${BLI_ROOT}/.env"; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HF_HOME="${HF_HOME:-/scratch/$USER/cache/hf_cache}"
set -x

python src/pipeline/prepare_babylm_data.py \
  --tokenizer meta-llama/Llama-3.2-1B \
  --n-tasks 16 \
  --skip-translations
