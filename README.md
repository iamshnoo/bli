# BLI

Reproducible pipeline for BLI-style alignment analysis across multilingual BabyLM runs, from dataset preparation through training, model conversion, probing/analysis, artifact generation, and paper PDF build.

## What Is In This Repo

Pushed content (recommended):
- `src/` pipeline, training, probes, and analysis code
- `slurm/` Slurm entry scripts
- `config/` model/stage config JSONs
- `data/probes/` probe inventories and multilingual probe translation CSVs
- `data/language_metadata.csv`
- `README.md`

Large artifacts that should **not** be pushed:
- `latex/`
- `AGENT.md`, `CLAUDE_PLAN.md`
- `logs/` (training checkpoints and job logs)
- `models/` (HF exported model weights)
- `data/raw/`, `data/processed/` (downloaded/tokenized corpora)
- `outputs/**/representations/*.npy` and other bulky intermediates

## External Dependencies

### 1) Pretraining library (Nanotron)
This project expects your Nanotron fork:
- Repo: `https://github.com/iamshnoo/nanotron`
- Used local checkout in this run: `/scratch/amukher6/pretrain/nanotron`
- Observed commit: `737ccd8854dd32ed99f9ad9189b09a3aa1b62b20`

### 2) Python environment
Use one environment for CPU + GPU jobs (example):
```bash
python -m venv ~/nanotron-env
source ~/nanotron-env/bin/activate
pip install --upgrade pip

# Install nanotron (editable)
pip install -e /path/to/nanotron

# BLI extras
pip install datasets transformers pandas numpy scipy matplotlib seaborn scikit-learn tqdm pillow wordfreq sentencepiece
pip install "unbabel-comet>=2.0.0"
```

### 3) LaTeX toolchain
- `pdflatex` + `bibtex` (TinyTeX/TeXLive is fine)

### 4) Hugging Face auth
BabyLM-community datasets are gated. Before running data prep:
```bash
export HF_TOKEN=...
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=/scratch/amukher6/cache/hf_cache  # optional but recommended
```

## Hugging Face Dataset IDs Used

- `BabyLM-community/babylm-eng`
- `BabyLM-community/babylm-zho`
- `BabyLM-community/babylm-fra`
- `BabyLM-community/babylm-fas`
- `BabyLM-community/babylm-nld`
- `BabyLM-community/babylm-ukr`
- `BabyLM-community/babylm-bul`
- `BabyLM-community/babylm-ind`
- `BabyLM-community/babylm-deu`

English controls:
- `eng_shared` and `eng_disjoint` are deterministic splits from `babylm-eng` created in `src/pipeline/prepare_babylm_data.py`.

## End-to-End Reproduction

### Option A: Full Slurm pipeline (recommended)
```bash
cd /scratch/amukher6/bli
bash src/pipeline/submit_full_rebuild.sh
```

This submits:
1. `slurm/run_data_prep.sbatch` (CPU, `normal`)
   - builds EN shared/disjoint partitions
   - tokenizes all 9 BabyLM datasets
   - writes stage configs
   - validates overlap controls
   - rebuilds probe sets (`3000` anchors, `1000` cultural probes, `50` semantic axes, `100` negative controls)
2. `slurm/run_probe_translation_qe.sbatch` (GPU, `contrib-gpuq`)
   - translates probes to `zh, fr, fas, nld, ukr, bul, ind, deu`
   - computes COMETKiwi QE with `Unbabel/wmt22-cometkiwi-da`
3. `slurm/run_submit_dag.sbatch` (CPU, `normal`)
   - submits training jobs via Nanotron launcher
   - submits checkpoint-to-HF conversion jobs
   - submits BLI analysis jobs
   - submits postprocess job (language-ratio summary, artifact generation, figure combine, PDF build)

### Option B: Post-pretraining refresh only
If models are already trained/converted and you only changed probes/analysis/paper:
```bash
cd /scratch/amukher6/bli
bash src/pipeline/submit_post_pretrain_refresh.sh
```

## Manual Step-by-Step Commands

### 1) Data prep + probe inventory
```bash
cd /scratch/amukher6/bli
python src/pipeline/prepare_babylm_data.py \
  --tokenizer meta-llama/Llama-3.2-1B \
  --n-tasks 16 \
  --probe-anchor-target 3000 \
  --probe-cultural-target 1000 \
  --probe-axis-target 50 \
  --probe-negative-target 100
```

### 2) Multilingual probe translation + QE
```bash
python src/probes/build_multilingual_probes.py \
  --probe-set /scratch/amukher6/bli/data/probes/probe_sets.json \
  --output-dir /scratch/amukher6/bli/data/probes \
  --model facebook/nllb-200-distilled-600M \
  --comet-model Unbabel/wmt22-cometkiwi-da \
  --batch-size 32 \
  --comet-batch-size 64 \
  --device cuda
```

### 3) Training -> conversion -> analysis DAG submission
```bash
python src/pipeline/submit_train_convert_analysis.py
```

### 4) Artifact generation + PDF
```bash
python src/pipeline/build_language_ratio_summary.py \
  --revision-root /scratch/amukher6/bli/outputs/revision \
  --multilingual-root /scratch/amukher6/bli/outputs/multilingual_expansion \
  --output /scratch/amukher6/bli/outputs/multilingual_expansion/language_ratio_summary.csv

python latex/scripts/generate_artifacts.py \
  --output-root /scratch/amukher6/bli/outputs/revision \
  --multilingual-output-root /scratch/amukher6/bli/outputs/multilingual_expansion \
  --latex-root /scratch/amukher6/bli/latex

python src/pipeline/combine_multilingual_figures.py \
  --left /scratch/amukher6/bli/latex/figures/main_multilingual_regression.png \
  --right /scratch/amukher6/bli/latex/figures/appendix_multilingual_overview.png \
  --output /scratch/amukher6/bli/latex/figures/combined_multilingual_fig5_fig11.png

bash latex/scripts/sync_report.sh
```

## Main Outputs

- Final paper PDF: `latex/main.pdf`
- Core analysis: `outputs/revision/en_ablation/`
- Multilingual expansion: `outputs/multilingual_expansion/`
- Overlap validation report: `outputs/validation/exposure_overlap_report.json`
- Probe inventory: `data/probes/probe_sets.json`
- Probe translations + QE: `data/probes/translations_*.csv`, `data/probes/translation_summary.json`

## Monitoring
```bash
squeue -u $USER
sacct -u $USER --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
ls -lt /scratch/amukher6/bli/logs/slurm_logs
```
