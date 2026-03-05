#!/bin/bash
#SBATCH --job-name=bli2_prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/amukher6/bli/logs/slurm_logs/pipeline_prep-%j.out
#SBATCH --error=/scratch/amukher6/bli/logs/slurm_logs/pipeline_prep-%j.err

set -euo pipefail

source /home/amukher6/nanotron-env/bin/activate
cd /scratch/amukher6/bli

set +x
set -a; source /scratch/amukher6/bli/.env; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HF_HOME=/scratch/amukher6/cache/hf_cache
set -x

python src/pipeline/prepare_babylm_data.py \
  --tokenizer meta-llama/Llama-3.2-1B \
  --n-tasks 16 \
  --skip-translations
