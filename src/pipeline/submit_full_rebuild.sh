#!/usr/bin/env bash
set -euo pipefail

BLI_ROOT="${BLI_ROOT:-/scratch/$USER/bli}"
cd "${BLI_ROOT}"
mkdir -p logs/slurm_logs

prep_job=$(sbatch slurm/run_data_prep.sbatch | awk '{print $4}')
probe_job=$(sbatch --dependency=afterok:${prep_job} slurm/run_probe_translation_qe.sbatch | awk '{print $4}')
submit_job=$(sbatch --dependency=afterok:${probe_job} slurm/run_submit_dag.sbatch | awk '{print $4}')

echo "Submitted data prep job: ${prep_job}"
echo "Submitted probe translation/QE job: ${probe_job} (afterok:${prep_job})"
echo "Submitted DAG submission job: ${submit_job} (afterok:${probe_job})"
