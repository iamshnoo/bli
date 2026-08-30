# Double Trouble: Bilingual Pretraining Leaves Language-Conditioned Effects in Shared-Language Representations

This repository contains the training pipeline, model and data manifests, probe
inventory, evaluation code, released results, and paper figures for a controlled
study of bilingual pretraining. We ask whether an English concept is represented
differently when an otherwise matched model learns English with a different
second language. The comparison covers token embeddings, contextual states,
nearest neighbors, and 50 signed semantic axes across eight second languages.

## Setup and layout

Use Python 3.11. On a GPU system, install the appropriate PyTorch wheel before
the remaining dependencies:

```bash
git clone https://github.com/iamshnoo/bli.git
cd bli
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the local paths and Hugging Face token used
by the data-preparation and training jobs. Nanotron is installed separately and
referenced with `NANOTRON_ROOT`.

- `configs/`: one canonical configuration tree containing model registries,
  alternating-language schedules, and the train/test/data manifests.
- `data/probes/`: canonical probes, translations, translation QC, and the
  normalized evaluation-data lineage CSV.
- `src/`: data preparation, training submission, checkpoint conversion, probe
  construction, analysis, and release utilities.
- `slurm/`: cluster entrypoints for the full pipeline and external evaluations.
- `artifacts/`: the 24 paper figures and compact result tables, each covered by
  a SHA-256 manifest.
- `REPRODUCIBILITY.md`: environment, storage, and rerun details.

Large raw corpora, checkpoints, representation arrays, job logs, and local
manuscript files are intentionally excluded from git.

## 1. Pretraining

The [training manifest](configs/train_manifest.json) records every one of the
40 runs, including its data sources, exact step-derived token budget, seed,
architecture, optimization settings, stage schedule, local export path, and
public Hugging Face ID. The models are initialized from scratch;
`meta-llama/Llama-3.2-1B` supplies the tokenizer and vocabulary, not pretrained
weights.

All 40 checkpoints are public in the
[`iamshnoo` Hugging Face namespace](https://huggingface.co/iamshnoo). Their exact
IDs are listed in [the Hub registry](configs/models/models_hub.json), with
suite-specific registries alongside it.

Prepare the English partitions, tokenize all nine corpora, and rebuild the probe
inventory:

```bash
python src/pipeline/prepare_babylm_data.py \
  --tokenizer meta-llama/Llama-3.2-1B \
  --n-tasks 16 \
  --probe-anchor-target 3000 \
  --probe-cultural-target 1000 \
  --probe-axis-target 50 \
  --probe-negative-target 100
```

Submit the complete data → translation/QC → train → convert → evaluate DAG:

```bash
bash src/pipeline/submit_full_rebuild.sh
```

The deterministic English split uses MD5 parity of `doc-id`: even hashes form
the shared partition and odd hashes form the disjoint partition. Dataset IDs,
splits, columns, processed paths, and external translation resources are in the
[data manifest](configs/data_manifest.json).

## 2. Alignment and evaluation

The main analysis fits an orthogonal map on 3,000 neutral English anchors, then
compares 1,000 English concepts across model spaces. A public-model example for
one matched-compute comparison is:

```bash
python src/bli_analysis/run_bli_pipeline.py \
  --models-json configs/models/models_en_ablation_hub.json \
  --pair en_100m,en_fas_a \
  --probe-set data/probes/probe_sets.json \
  --output-dir outputs/example_en_fas_c3
```

When explicit `--pair` arguments are supplied, only the requested models are
loaded. The [test manifest](configs/test_manifest.json) maps every core,
statistical, robustness, validation, and scope-extension suite to its entrypoint,
inputs, model registry, and generated artifacts.

The target-language validation compares the same English concept inventory in
second-language-only and bilingual model spaces. The translated concept files
support translation-quality stratification; they are not substituted into the
model prompts.

## 3. Experiment, alignment, and evaluation map

| Setup | What it tests | Models or data | Manifest suite / artifact |
|---|---|---|---|
| C1: matched English documents | Effect of adding a second language while English documents are fixed | `en_50m` vs. `en_{L2}_a` | `english_control_matrix` |
| C2: disjoint English documents | Same comparison without shared English documents | `en_50m` vs. `en_{L2}_b` | `english_control_matrix` |
| C3: matched compute | Language mixture at the same number of training steps | `en_100m` vs. `en_{L2}_a` | `english_control_matrix` |
| C4: matched compute + disjoint | Matched compute without shared English documents | `en_100m` vs. `en_{L2}_b` | `english_control_matrix` |
| Default alignment | Rigid rotation/reflection fit on neutral English embeddings | 3,000 anchors; orthogonal Procrustes | `orthogonal_embedding` |
| Contextual alignment | Whether the result depends on fitting the map to contextual states | 3,000 prompted anchor states | `contextual_alignment_variant` |
| Affine alignment | Whether allowing translation and scaling changes the result | 3,000 embedding anchors | `alignment_method_comparison` |
| Layerwise alignment | Where differences appear through the network | one orthogonal map per layer | `layerwise_analysis` |
| Second-language validation | Whether the pattern also appears in each second-language model space | eight L2-only/bilingual model groups | `target_language_validation` |
| Same-language null | Variation caused by random seed alone | six English-only controls | `same_language_seed_controls` |
| Anchor, framework, and k checks | Dependence on anchor count, held-out categories, and neighborhood size | repeated subsets; 10 categories; k = 5–100 | `anchor_sensitivity`, `framework_holdout`, `knn_sensitivity` |
| Training progress | Whether the difference persists across checkpoints | steps 500–3,000 | `training_progress` |
| Data/model validation | Exposure overlap, tokenizer identity, and mixed-language corpus audit | corpus and checkpoint metadata | `tokenizer_audit`, `mixed_language_corpus_audit` |
| Behavioral extensions | Output likelihood and WorldValuesBench associations | controlled cases and external benchmark | `output_likelihood_association`, `worldvaluesbench` |

The complete evaluation inventory, including exact output paths, is in
[`configs/test_manifest.json`](configs/test_manifest.json).

### Data lineage

[`evaluation_data_lineage.csv`](data/probes/evaluation_data_lineage.csv) has
13,000 normalized rows covering the 1,000 concepts in nine languages, both
endpoints of all 50 axes, 3,000 alignment anchors, and 100 negative controls. It
records language names, categories, endpoint direction, source citations,
translation scores and flags, availability, actual model-input status, and the
model-language conditions in which each English term was evaluated.

Endpoint 1 is encoded as `-1` and endpoint 2 as `+1` only to orient signed
differences; the signs are not value judgments. Rebuild and validate the CSV
with:

```bash
python src/probes/build_eval_data_lineage.py
```

## 4. Plotting and released artifacts

Generate the manuscript figures and tables from local analysis outputs:

```bash
python latex/scripts/generate_artifacts.py \
  --output-root outputs/revision \
  --multilingual-output-root outputs/multilingual_expansion \
  --latex-root latex/emnlp/anon
```

Refresh the tracked compact snapshot and its checksums:

```bash
python src/pipeline/export_release_artifacts.py
```

The tracked [artifact manifest](artifacts/manifest.json) links every released
figure or result to its generated source path. Cached arrays and large per-word
tables are omitted from git but can be regenerated through the test manifest.
