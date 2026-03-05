#!/bin/bash
#SBATCH --job-name=bli2_submit
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/amukher6/bli/logs/slurm_logs/pipeline_submit-%j.out
#SBATCH --error=/scratch/amukher6/bli/logs/slurm_logs/pipeline_submit-%j.err

set -euo pipefail

source /home/amukher6/nanotron-env/bin/activate
cd /scratch/amukher6/bli

python src/pipeline/submit_train_convert_analysis.py
