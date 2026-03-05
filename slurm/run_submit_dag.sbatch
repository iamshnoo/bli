#!/bin/bash
#SBATCH --job-name=bli2_submit
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/%u/bli/logs/slurm_logs/pipeline_submit-%j.out
#SBATCH --error=/scratch/%u/bli/logs/slurm_logs/pipeline_submit-%j.err

set -euo pipefail

BLI_ROOT="${BLI_ROOT:-/scratch/$USER/bli}"
NANOTRON_ENV="${NANOTRON_ENV:-$HOME/nanotron-env}"
source "${NANOTRON_ENV}/bin/activate"
cd "${BLI_ROOT}"

python src/pipeline/submit_train_convert_analysis.py
