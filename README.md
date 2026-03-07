# BLI

Reproducible pipeline for multilingual BabyLM pretraining analysis: dataset prep, tokenization, train/convert, probing, metrics, plots, and paper PDF.

## Repo Layout

- `src/`: pipeline, training utilities, probes, and analysis code
- `slurm/`: cluster entry scripts (data prep, probe translation/QE, DAG submit)
- `config/`: model registries and stage JSONs
- `data/probes/`: probe inventory and multilingual translation CSV outputs
- `latex/`: paper source (`main.tex`, `references.bib`, `tables/`, `figures/`, `scripts/`)

## Prerequisites (Cluster Setup)

Set variables once:

```bash
export HF_USER=<your-hf-username>
export BLI_ROOT=/scratch/$USER/bli
export NANOTRON_ROOT=/scratch/$USER/nanotron
export NANOTRON_ENV=$HOME/nanotron-env
```

Environment bootstrap (CPU + GPU tasks use the same env):

```bash
echo "export HF_HOME=/scratch/$USER/cache/hf_cache" >> ~/.bashrc
cat ~/.bashrc

ml load gnu12/12.3.0
ml load python/3.11.7-cx
ml load cuda/12.4.0
ml load git

python -m venv ~/nanotron-env
source ~/nanotron-env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124

if [ ! -d "$NANOTRON_ROOT/.git" ]; then
  git clone https://github.com/$HF_USER/nanotron.git "$NANOTRON_ROOT"
fi
cd "$NANOTRON_ROOT"
pip install -e .
pip install datasets==3.6.0 transformers numba wandb ninja triton datatrove==0.3.0

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
rm flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install psutil pybind11
pip install trl bitsandbytes peft liger-kernel rich
pip install "unbabel-comet>=2.0.0"

echo "export VIRTUAL_ENV=$HOME/nanotron-env" >> ~/.bashrc
echo "export VIRTUAL_ENV" >> ~/.bashrc
echo "export PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc

hf auth login
wandb login
```

Flash-attention `.so` workaround note:
- Follow: `https://github.com/Dao-AILab/flash-attention/issues/1708#issuecomment-3283420504`

## TinyTeX / LaTeX Toolchain

Install TinyTeX (local to your user):

```bash
curl -fsSL https://yihui.org/tinytex/install-bin-unix.sh | sh
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

tlmgr install latexmk collection-latexrecommended collection-fontsrecommended \
  collection-latexextra natbib url hyperref booktabs multirow siunitx xcolor
```

Paper source tracked in `latex/` for Overleaf sync:
- `latex/main.tex`
- `latex/references.bib`
- `latex/tables/*.tex`
- `latex/figures/*` (generated/copied figure assets)
- `latex/scripts/generate_artifacts.py`
- `latex/scripts/sync_report.sh`

## Hugging Face Datasets Used

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
- `eng_shared` and `eng_disjoint` are deterministic partitions from `BabyLM-community/babylm-eng`.

## End-to-End Run

```bash
cd "$BLI_ROOT"
bash src/pipeline/submit_full_rebuild.sh
```

This submits:
1. `slurm/run_data_prep.sbatch` (CPU, `normal`): partition EN, tokenize datasets, rebuild probes.
2. `slurm/run_probe_translation_qe.sbatch` (GPU): translate probes + COMETKiwi QE (`Unbabel/wmt22-cometkiwi-da`).
3. `slurm/run_submit_dag.sbatch` (CPU): launch train -> convert -> analysis -> postprocess chain.

## Manual Step Order

1. Data prep + tokenization + probe inventory:
```bash
cd "$BLI_ROOT"
python src/pipeline/prepare_babylm_data.py \
  --tokenizer meta-llama/Llama-3.2-1B \
  --n-tasks 16 \
  --probe-anchor-target 3000 \
  --probe-cultural-target 1000 \
  --probe-axis-target 50 \
  --probe-negative-target 100
```

2. Multilingual probe translation + QE:
```bash
python src/probes/build_multilingual_probes.py \
  --probe-set data/probes/probe_sets.json \
  --output-dir data/probes \
  --model facebook/nllb-200-distilled-600M \
  --comet-model Unbabel/wmt22-cometkiwi-da \
  --batch-size 32 \
  --comet-batch-size 64 \
  --device cuda
```

3. Submit training/conversion/analysis DAG:
```bash
python src/pipeline/submit_train_convert_analysis.py
```

4. Build paper artifacts + PDF:
```bash
python src/pipeline/build_language_ratio_summary.py \
  --revision-root outputs/revision \
  --multilingual-root outputs/multilingual_expansion \
  --output outputs/multilingual_expansion/language_ratio_summary.csv

python latex/scripts/generate_artifacts.py \
  --output-root outputs/revision \
  --multilingual-output-root outputs/multilingual_expansion \
  --latex-root latex

python src/pipeline/combine_multilingual_figures.py \
  --left latex/figures/main_multilingual_regression.png \
  --right latex/figures/appendix_multilingual_overview.png \
  --output latex/figures/combined_multilingual_fig5_fig11.png

bash latex/scripts/sync_report.sh
```

## Key Outputs

- `latex/main.pdf`
- `outputs/revision/en_ablation/`
- `outputs/revision/zh_shared_language/`
- `outputs/revision/fr_shared_language/`
- `outputs/multilingual_expansion/`
- `outputs/validation/exposure_overlap_report.json`
- `data/probes/probe_sets.json`
- `data/probes/translations_*.csv`
- `data/probes/translation_summary.json`

## Monitoring

```bash
squeue -u $USER
sacct -u $USER --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
ls -lt "$BLI_ROOT"/logs/slurm_logs
```
