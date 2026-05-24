#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NANOTRON_ENV="${NANOTRON_ENV:-$HOME/nanotron-env}"
if [ -f "${NANOTRON_ENV}/bin/activate" ]; then
  source "${NANOTRON_ENV}/bin/activate"
fi

# TinyTeX preference order: repo local -> user local -> shared scratch
if [ -x "$ROOT_DIR/.tinytex/bin/x86_64-linux/pdflatex" ]; then
  export PATH="$ROOT_DIR/.tinytex/bin/x86_64-linux:$PATH"
elif [ -x "$HOME/.TinyTeX/bin/x86_64-linux/pdflatex" ]; then
  export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
elif [ -x "/scratch/$USER/tinytex/bin/x86_64-linux/pdflatex" ]; then
  export PATH="/scratch/$USER/tinytex/bin/x86_64-linux:$PATH"
fi

if [ "${REGENERATE_ARTIFACTS:-0}" = "1" ]; then
  REQUIRED=(
    "$ROOT_DIR/outputs/revision/en_ablation/bli_summary_metrics.csv"
    "$ROOT_DIR/outputs/revision/zh_shared_language/bli_summary_metrics.csv"
    "$ROOT_DIR/outputs/revision/fr_shared_language/bli_summary_metrics.csv"
    "$ROOT_DIR/outputs/multilingual_expansion/language_ratio_summary.csv"
  )
  MISSING=0
  for f in "${REQUIRED[@]}"; do
    if [ ! -f "$f" ]; then
      echo "Missing required artifact input: $f"
      MISSING=1
    fi
  done
  if [ "$MISSING" -eq 0 ]; then
    python latex/scripts/generate_artifacts.py \
      --output-root "$ROOT_DIR/outputs/revision" \
      --multilingual-output-root "$ROOT_DIR/outputs/multilingual_expansion" \
      --latex-root "$ROOT_DIR/latex" \
      --probe-set "$ROOT_DIR/data/probes/probe_sets.json" \
      --translations-dir "$ROOT_DIR/data/probes"
  else
    echo "Skipping artifact regeneration: required inputs are not present in this minimal checkout."
  fi
else
  echo "Skipping artifact regeneration (set REGENERATE_ARTIFACTS=1 to enable)."
fi

if command -v pdflatex >/dev/null 2>&1; then
  pushd latex >/dev/null
  BUILD_DIR=".build"
  mkdir -p "$BUILD_DIR"

  pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" main.tex

  BIBTEX_BIN=""
  if command -v bibtex >/dev/null 2>&1; then
    BIBTEX_BIN="$(command -v bibtex)"
  elif [ -x "$(dirname "$(command -v pdflatex)")/bibtex" ]; then
    BIBTEX_BIN="$(dirname "$(command -v pdflatex)")/bibtex"
  fi

  if [ -z "$BIBTEX_BIN" ]; then
    echo "ERROR: bibtex not found; cannot resolve bibliography references."
    exit 1
  fi

  (cd "$BUILD_DIR" && BSTINPUTS=".:..:${ROOT_DIR}/latex:${BSTINPUTS:-}" BIBINPUTS=".:..:${ROOT_DIR}/latex:${BIBINPUTS:-}" "$BIBTEX_BIN" main)
  pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" main.tex
  pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" main.tex
  pdflatex -interaction=nonstopmode -output-directory="$BUILD_DIR" main.tex

  cp "$BUILD_DIR/main.pdf" main.pdf
  rm -f "$BUILD_DIR"/main.aux "$BUILD_DIR"/main.bbl "$BUILD_DIR"/main.blg "$BUILD_DIR"/main.log "$BUILD_DIR"/main.out "$BUILD_DIR"/main.pdf
  rmdir "$BUILD_DIR" 2>/dev/null || true

  rm -f main.aux main.bbl main.blg main.log main.out
  popd >/dev/null
else
  echo "pdflatex not found; skipping PDF build"
fi

echo "report sync complete."
