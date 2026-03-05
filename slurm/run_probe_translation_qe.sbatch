#!/bin/bash
#SBATCH --job-name=bli2_probe_qe
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=80G
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/amukher6/bli/logs/slurm_logs/probe_qe-%j.out
#SBATCH --error=/scratch/amukher6/bli/logs/slurm_logs/probe_qe-%j.err

set -euo pipefail

source /home/amukher6/nanotron-env/bin/activate
cd /scratch/amukher6/bli

set +x
set -a; source /scratch/amukher6/bli/.env; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HF_HOME=/scratch/amukher6/cache/hf_cache
set -x

python src/probes/build_multilingual_probes.py \
  --probe-set /scratch/amukher6/bli/data/probes/probe_sets.json \
  --output-dir /scratch/amukher6/bli/data/probes \
  --model facebook/nllb-200-distilled-600M \
  --comet-model Unbabel/wmt22-cometkiwi-da \
  --batch-size 32 \
  --comet-batch-size 64 \
  --device cuda
