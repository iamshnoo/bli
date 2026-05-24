#!/bin/bash
#SBATCH --job-name=bli_wvb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=64G
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/bli/logs/slurm_logs/wvb_bli-%j.out
#SBATCH --error=/scratch/%u/bli/logs/slurm_logs/wvb_bli-%j.err

set -euo pipefail

BLI_ROOT="${BLI_ROOT:-/scratch/$USER/bli}"
NANOTRON_ENV="${NANOTRON_ENV:-$HOME/nanotron-env}"
WORLDVALUESBENCH_ROOT="${WORLDVALUESBENCH_ROOT:-${BLI_ROOT}/external/WorldValuesBench}"
source "${NANOTRON_ENV}/bin/activate"
cd "${BLI_ROOT}"

export HF_HOME="${HF_HOME:-/scratch/$USER/cache/hf_cache}"
export PYTHONPATH="${BLI_ROOT}/src/bli_analysis:${PYTHONPATH:-}"

python src/bli_analysis/run_worldvaluebench_eval.py \
  --worldvaluesbench-root "${WORLDVALUESBENCH_ROOT}" \
  --out-dir outputs/revision/en_ablation/worldvaluebench_bli \
  --device cuda
