# BLI

Anonymous reproducibility snapshot for multilingual BabyLM pretraining analysis.
The repository contains the code, configs, and small probe inputs needed to
prepare data, train/convert checkpoints, run BLI analyses, and reproduce the
tracked add-on evaluations.

## Layout

- `src/`: pipeline, training utilities, probe builders, and analysis code.
- `slurm/`: Slurm entrypoints for data prep, probe translation/QE, DAG submit,
  output-likelihood analysis, and WorldValuesBench evaluation.
- `config/` and `configs/`: model registries and stage JSONs.
- `data/probes/`: probe inventory, multilingual translations, and small
  association-case CSVs.
- `requirements.txt`: Python package manifest for the analysis environment.
- `REPRODUCIBILITY.md`: full setup, data, and rerun notes.

Large or identifying local state is ignored: `.env`, `external/`, `data/raw/`,
`models/`, `logs/`, `outputs/`, `output/`, and `latex/`.

## Quick Setup

Use Python 3.11. Install a CUDA-compatible PyTorch wheel first on GPU systems,
then install the tracked dependencies:

```bash
python -m venv "$HOME/nanotron-env"
source "$HOME/nanotron-env/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Copy `.env.example` to `.env` or export the same variables in your shell:

```bash
export BLI_ROOT=/path/to/bli
export NANOTRON_ROOT=/path/to/nanotron
export NANOTRON_ENV=$HOME/nanotron-env
export HF_HOME=/path/to/hf_cache
export HF_TOKEN=<huggingface-token-for-gated-datasets>
export BLI_ENABLE_WANDB=0
```

Install Nanotron separately in `NANOTRON_ROOT`; the conversion and Slurm launcher
code imports it from that checkout. Flash-attention is optional and should be
installed with a wheel matching the local CUDA/PyTorch stack when needed.

## Datasets

Hugging Face datasets used by the pipeline:

- `BabyLM-community/babylm-eng`
- `BabyLM-community/babylm-zho`
- `BabyLM-community/babylm-fra`
- `BabyLM-community/babylm-fas`
- `BabyLM-community/babylm-nld`
- `BabyLM-community/babylm-ukr`
- `BabyLM-community/babylm-bul`
- `BabyLM-community/babylm-ind`
- `BabyLM-community/babylm-deu`

English controls `eng_shared` and `eng_disjoint` are deterministic partitions
from `BabyLM-community/babylm-eng`.

## End-to-End Run

```bash
cd "$BLI_ROOT"
bash src/pipeline/submit_full_rebuild.sh
```

This submits:

1. `slurm/run_data_prep.sbatch`: partition English, tokenize datasets, rebuild probes.
2. `slurm/run_probe_translation_qe.sbatch`: translate probes and run COMETKiwi QE.
3. `slurm/run_submit_dag.sbatch`: launch train -> convert -> analysis -> postprocess jobs.

## Manual Run Order

```bash
python src/pipeline/prepare_babylm_data.py \
  --tokenizer meta-llama/Llama-3.2-1B \
  --n-tasks 16 \
  --probe-anchor-target 3000 \
  --probe-cultural-target 1000 \
  --probe-axis-target 50 \
  --probe-negative-target 100

python src/probes/build_multilingual_probes.py \
  --probe-set data/probes/probe_sets.json \
  --output-dir data/probes \
  --model facebook/nllb-200-distilled-600M \
  --comet-model Unbabel/wmt22-cometkiwi-da \
  --batch-size 32 \
  --comet-batch-size 64 \
  --device cuda

python src/pipeline/submit_train_convert_analysis.py
```

## Add-On Analyses

Regenerate the expanded output-likelihood case set:

```bash
python src/bli_analysis/build_output_likelihood_association_cases_expanded.py
```

Run output-likelihood association:

```bash
sbatch slurm/run_output_likelihood_association.sbatch
```

Run WorldValuesBench evaluation after placing the benchmark checkout under
`$BLI_ROOT/external/WorldValuesBench` or setting `WORLDVALUESBENCH_ROOT`:

```bash
sbatch slurm/run_worldvaluebench_bli.sbatch
```

## Key Generated Outputs

- `outputs/revision/en_ablation/`
- `outputs/revision/zh_shared_language/`
- `outputs/revision/fr_shared_language/`
- `outputs/multilingual_expansion/`
- `outputs/validation/exposure_overlap_report.json`
- `data/probes/probe_sets.json`
- `data/probes/translations_*.csv`
- `data/probes/translation_summary.json`

Generated outputs and checkpoints are intentionally ignored by git.

## Monitoring

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
ls -lt "$BLI_ROOT"/logs/slurm_logs
```
