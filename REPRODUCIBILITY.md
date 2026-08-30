# Reproducibility guide

The top-level README gives the shortest path through the repository. This guide
records the operational details needed for a full rebuild.

## Environment

Use Python 3.11 and a CUDA-compatible PyTorch build. Install the remaining
packages from `requirements.txt`. Training and checkpoint conversion also need
a separate Nanotron checkout.

Copy `.env.example` to `.env` and set:

```bash
export BLI_ROOT=/path/to/bli
export NANOTRON_ROOT=/path/to/nanotron
export NANOTRON_ENV=/path/to/python-environment
export HF_HOME=/path/to/huggingface-cache
export HF_TOKEN=<token-for-gated-datasets>
export BLI_ENABLE_WANDB=0
```

Do not commit `.env`, tokens, local paths, or experiment-tracking credentials.

## Sources of truth

- `configs/train_manifest.json`: 40 training runs and all model/training fields.
- `configs/data_manifest.json`: corpora, splits, probe assets, translations, and
  source citations.
- `configs/test_manifest.json`: evaluation entrypoints, inputs, model registries,
  and outputs.
- `configs/models/models_hub.json`: all 40 verified public model IDs.
- `data/probes/evaluation_data_lineage.csv`: normalized term-level lineage.
- `artifacts/manifest.json`: checksums and generated origins for released results.

The manifests are generated deterministically from
`src/pipeline/build_research_manifests.py`. Rebuild them with:

```bash
python src/pipeline/build_research_manifests.py
```

Validate the manifests, stage schedules, lineage row counts, artifact checksums,
and all public model IDs with:

```bash
python src/pipeline/validate_release.py --check-hub
```

## End-to-end rebuild

```bash
cd "$BLI_ROOT"
bash src/pipeline/submit_full_rebuild.sh
```

This submits data preparation, multilingual concept translation and COMETKiwi
quality estimation, pretraining, checkpoint conversion, core comparisons, and
postprocessing through Slurm. Use `--dry-run` with
`src/pipeline/submit_train_convert_analysis.py` to inspect the generated job DAG
without submitting it.

## Public checkpoints

All checkpoints can be used directly with Transformers through the suite-specific
`*_hub.json` registries in `configs/models/`. Local registries point to
`models/hf/` and are regenerated after local checkpoint conversion.

## Optional external evaluations

Output-likelihood association:

```bash
sbatch slurm/run_output_likelihood_association.sbatch
```

WorldValuesBench requires a local benchmark checkout:

```bash
export WORLDVALUESBENCH_ROOT="$BLI_ROOT/external/WorldValuesBench"
sbatch slurm/run_worldvaluebench_bli.sbatch
```

Use `--dry-run` with `src/bli_analysis/run_worldvaluebench_eval.py` to verify
benchmark discovery before scoring models.

## Regenerated and excluded state

Git excludes raw and tokenized corpora, model checkpoints, representation
arrays, logs, external repositories, generated output trees, and the local
manuscript workspace. The compact files needed to inspect the reported results
are tracked under `artifacts/`; their larger parents can be recreated from the
manifests and public models.
