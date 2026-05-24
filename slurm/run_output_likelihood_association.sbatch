#!/bin/bash
#SBATCH --job-name=bli_output_assoc
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=64G
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/bli/logs/slurm_logs/output_assoc-%j.out
#SBATCH --error=/scratch/%u/bli/logs/slurm_logs/output_assoc-%j.err

set -euo pipefail

BLI_ROOT="${BLI_ROOT:-/scratch/$USER/bli}"
NANOTRON_ENV="${NANOTRON_ENV:-$HOME/nanotron-env}"
source "${NANOTRON_ENV}/bin/activate"
cd "${BLI_ROOT}"

export HF_HOME="${HF_HOME:-/scratch/$USER/cache/hf_cache}"
export PYTHONPATH="${BLI_ROOT}/src/bli_analysis:${PYTHONPATH:-}"

python src/bli_analysis/run_output_likelihood_association.py \
  --cases-csv data/probes/output_likelihood_association_cases_expanded.csv \
  --out-csv outputs/revision/en_ablation/bli_output_likelihood_association.csv \
  --template-out-csv outputs/revision/en_ablation/bli_output_likelihood_association_templates.csv \
  --summary-csv outputs/revision/en_ablation/bli_output_likelihood_association_summary.csv \
  --device cuda
