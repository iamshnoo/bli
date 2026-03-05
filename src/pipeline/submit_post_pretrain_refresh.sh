#!/usr/bin/env bash
set -euo pipefail

cd /scratch/amukher6/bli
mkdir -p logs/slurm_logs

probe_job=$(sbatch slurm/run_probe_translation_qe.sbatch | awk '{print $4}')
submit_job=$(sbatch --dependency=afterok:${probe_job} slurm/run_submit_dag.sbatch | awk '{print $4}')

echo "Submitted probe translation/QE job: ${probe_job}"
echo "Submitted DAG submission job: ${submit_job} (afterok:${probe_job})"
