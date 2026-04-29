#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NANOTRON_ENV="${NANOTRON_ENV:-$HOME/nanotron-env}"
if [ -f "${NANOTRON_ENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${NANOTRON_ENV}/bin/activate"
fi

TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/models/hf/en_100m}"
PROGRESS_MODEL_ROOT="${PROGRESS_MODEL_ROOT:-$ROOT_DIR/models/hf/progress_dense_zh_fr}"
MODELS_JSON="${MODELS_JSON:-$ROOT_DIR/configs/models_dense_progress_zh_fr.json}"
RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-$ROOT_DIR/outputs/revision/en_progress_trajectory}"
DERIVED_CSV="${DERIVED_CSV:-$ROOT_DIR/outputs/revision/en_ablation/bli_dense_progress_trajectory.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STEPS=(500 1000 1500 2000 2500 3000)

mkdir -p "$PROGRESS_MODEL_ROOT" "$(dirname "$MODELS_JSON")" "$RAW_OUTPUT_DIR"

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
  convert_if_missing "$ROOT_DIR/logs/checkpoints/babylm_160m_en_zh_setup_a/$step" \
    "$PROGRESS_MODEL_ROOT/en_zh_a_${step}"
  convert_if_missing "$ROOT_DIR/logs/checkpoints/babylm_160m_en_fr_setup_a/$step" \
    "$PROGRESS_MODEL_ROOT/en_fr_a_${step}"
done

python3 - <<'PY'
import json
from pathlib import Path

root = Path("models/hf/progress_dense_zh_fr")
payload = {}
for step in [500, 1000, 1500, 2000, 2500, 3000]:
    payload[f"en_{step}"] = str(root / f"en_{step}")
    payload[f"en_zh_a_{step}"] = str(root / f"en_zh_a_{step}")
    payload[f"en_fr_a_{step}"] = str(root / f"en_fr_a_{step}")
Path("configs/models_dense_progress_zh_fr.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("Wrote configs/models_dense_progress_zh_fr.json")
PY

RUN_CMD=(
  python src/bli_analysis/run_bli_pipeline.py
  --models-json "$MODELS_JSON"
  --output-dir "$RAW_OUTPUT_DIR"
  --device cuda
  --batch-size "$BATCH_SIZE"
)
for step in "${STEPS[@]}"; do
  RUN_CMD+=(--pair "en_${step},en_zh_a_${step}")
  RUN_CMD+=(--pair "en_${step},en_fr_a_${step}")
done
"${RUN_CMD[@]}"

python src/bli_analysis/build_dense_progress_trajectory.py \
  --summary-csv "$RAW_OUTPUT_DIR/bli_summary_metrics.csv" \
  --out-csv "$DERIVED_CSV"
