#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NANOTRON_ENV="${NANOTRON_ENV:-$HOME/nanotron-env}"
if [ -f "${NANOTRON_ENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${NANOTRON_ENV}/bin/activate"
fi

export NANOTRON_ROOT="${NANOTRON_ROOT:-/scratch/${USER}/pretrain/nanotron_full}"

TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/models/hf/en_100m}"
PROGRESS_MODEL_ROOT="${PROGRESS_MODEL_ROOT:-$ROOT_DIR/models/hf/progress_dense_all}"
MODELS_JSON="${MODELS_JSON:-$ROOT_DIR/configs/models_dense_progress_all.json}"
RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-$ROOT_DIR/outputs/revision/en_progress_trajectory_all}"
RAW_SUMMARY_CSV="${RAW_SUMMARY_CSV:-$RAW_OUTPUT_DIR/bli_summary_metrics.csv}"
DERIVED_CSV="${DERIVED_CSV:-$ROOT_DIR/outputs/revision/en_ablation/bli_dense_progress_trajectory.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STEPS=(500 1000 1500 2000 2500 3000)
LANGS_STR="${LANGS:-zh fr fas nld ukr bul ind deu}"
export LANGS_STR
read -r -a LANGS <<< "$LANGS_STR"

mkdir -p "$PROGRESS_MODEL_ROOT" "$(dirname "$MODELS_JSON")" "$RAW_OUTPUT_DIR" "$ROOT_DIR/configs"

convert_if_missing() {
  local checkpoint_path="$1"
  local save_path="$2"
  if [ -f "$save_path/config.json" ]; then
    echo "[skip] HF model already exists at $save_path"
    return 0
  fi
  python src/training/convert_checkpoint_to_hf.py \
    --checkpoint-path "$checkpoint_path" \
    --save-path "$save_path" \
    --tokenizer-name "$TOKENIZER_PATH"
}

for step in "${STEPS[@]}"; do
  convert_if_missing "$ROOT_DIR/logs/checkpoints/babylm_160m_en_100m/$step" \
    "$PROGRESS_MODEL_ROOT/en_${step}"
  for lang in "${LANGS[@]}"; do
    convert_if_missing "$ROOT_DIR/logs/checkpoints/babylm_160m_en_${lang}_setup_a/$step" \
      "$PROGRESS_MODEL_ROOT/en_${lang}_a_${step}"
  done
done

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path("models/hf/progress_dense_all")
langs = os.environ.get("LANGS_STR", "zh fr fas nld ukr bul ind deu").split()
payload = {}
for step in [500, 1000, 1500, 2000, 2500, 3000]:
    payload[f"en_{step}"] = str(root / f"en_{step}")
    for lang in langs:
        payload[f"en_{lang}_a_{step}"] = str(root / f"en_{lang}_a_{step}")
Path("configs/models_dense_progress_all.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("Wrote configs/models_dense_progress_all.json")
PY

python src/bli_analysis/run_dense_progress_summary.py \
  --models-json "$MODELS_JSON" \
  --output-csv "$RAW_SUMMARY_CSV" \
  --derived-csv "$DERIVED_CSV" \
  --device cuda \
  --batch-size "$BATCH_SIZE"
