# Reproducibility Guide

This anonymous snapshot tracks source code, small probe inputs, configs, and
Slurm entrypoints. It intentionally does not track tokens, local checkouts,
model checkpoints, raw corpora, logs, generated outputs, or the local manuscript
workspace.

## Tracked Inputs

- `config/` and `configs/`: model and stage registries.
- `data/language_metadata.csv`: language metadata.
- `data/probes/`: probe sets, translated probe CSVs, and output-likelihood cases.
- `src/`: data prep, training submission, checkpoint conversion, probing, and analysis.
- `slurm/`: cluster entrypoints for the end-to-end pipeline and add-on analyses.

## Ignored Local State

- `.env`: private tokens and local paths.
- `external/`: external benchmark repositories such as WorldValuesBench.
- `data/raw/`, `models/`, `logs/`, `outputs/`, `output/`: regenerated data,
  checkpoints, job logs, and analysis products.
- `latex/`: local manuscript workspace.

## Environment

Use Python 3.11. On GPU systems, install the CUDA-compatible PyTorch wheel first,
then install the remaining requirements:

```bash
python -m venv "$HOME/nanotron-env"
source "$HOME/nanotron-env/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For training and checkpoint conversion, install Nanotron separately and point
`NANOTRON_ROOT` at that checkout. Copy `.env.example` to `.env` or export the
same variables in the shell. Keep `BLI_ENABLE_WANDB=0` for anonymous review runs.

## End-to-End Pipeline

```bash
export BLI_ROOT=/path/to/bli
export NANOTRON_ROOT=/path/to/nanotron
export NANOTRON_ENV=$HOME/nanotron-env
export HF_HOME=/path/to/hf_cache
export HF_TOKEN=<huggingface-token-for-gated-datasets>

cd "$BLI_ROOT"
bash src/pipeline/submit_full_rebuild.sh
```

The pipeline submits data prep, multilingual probe translation/QE, training,
checkpoint conversion, core analyses, and postprocessing through Slurm.

## Regenerating Small Probe Inputs

```bash
python src/bli_analysis/build_output_likelihood_association_cases_expanded.py \
  --probe-set data/probes/probe_sets.json \
  --out-csv data/probes/output_likelihood_association_cases_expanded.csv
```

## Add-On Analyses

Output-likelihood association:

```bash
sbatch slurm/run_output_likelihood_association.sbatch
```

WorldValuesBench target-country evaluation requires a local WorldValuesBench
checkout:

```bash
export WORLDVALUESBENCH_ROOT=$BLI_ROOT/external/WorldValuesBench
sbatch slurm/run_worldvaluebench_bli.sbatch
```

Use `--dry-run` with `src/bli_analysis/run_worldvaluebench_eval.py` to verify
that the benchmark files are discoverable before scoring models.
