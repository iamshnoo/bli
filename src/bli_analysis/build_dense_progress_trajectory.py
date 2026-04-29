#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd


STEP_RE = re.compile(r"_(\d{3,4})$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dense step-matched progress summaries from raw BLI metrics.")
    p.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Raw bli_summary_metrics.csv from the dense progress evaluation run.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV with one row per language, step, and representation family.",
    )
    return p.parse_args()


def infer_step(model_name: str) -> int:
    m = STEP_RE.search(model_name)
    if not m:
        raise ValueError(f"Could not infer checkpoint step from model name '{model_name}'.")
    return int(m.group(1))


def infer_language(model_b: str) -> str:
    parts = model_b.split("_")
    if len(parts) < 4 or parts[0] != "en":
        raise ValueError(f"Could not infer language from model name '{model_b}'.")
    return parts[1].upper()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)

    rows = []
    for _, row in df.iterrows():
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])
        step_a = infer_step(model_a)
        step_b = infer_step(model_b)
        if step_a != step_b:
            raise ValueError(f"Step mismatch in pair {model_a} vs {model_b}.")
        rows.append(
            {
                "step": step_a,
                "language": infer_language(model_b),
                "repr_type": str(row["repr_type"]),
                "axis_abs_projection_diff_mean": float(row["axis_abs_projection_diff_mean"]),
                "jaccard_at_k_mean": float(row["jaccard_at_k_mean"]),
                "frobenius_cultural_similarity": float(row["frobenius_cultural_similarity"]),
                "procrustes_anchor_residual_per_anchor": float(row["procrustes_anchor_residual_per_anchor"]),
            }
        )

    out = pd.DataFrame(rows).sort_values(["language", "repr_type", "step"]).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
