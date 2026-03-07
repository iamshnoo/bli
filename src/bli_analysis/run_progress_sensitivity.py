#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute 50M->100M sensitivity summaries for EN-centered C3 pairs.")
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/revision/en_ablation/bli_summary_metrics.csv"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/revision/en_ablation/bli_progress_sensitivity.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    df = df[df["model_b"].astype(str).str.endswith("_a")].copy()

    rows = []
    langs = sorted({mb.split("_")[1] for mb in df["model_b"].astype(str)})
    for lang in langs:
        m50 = f"en_{lang}_a"
        for repr_type in ["embedding_matrix", "pre_lmhead_contextual"]:
            r50 = df[
                (df["repr_type"] == repr_type)
                & (df["model_a"] == "en_50m")
                & (df["model_b"] == m50)
            ]
            r100 = df[
                (df["repr_type"] == repr_type)
                & (df["model_a"] == "en_100m")
                & (df["model_b"] == m50)
            ]
            if r50.empty or r100.empty:
                continue
            a50 = float(r50["axis_abs_projection_diff_mean"].iloc[0])
            a100 = float(r100["axis_abs_projection_diff_mean"].iloc[0])
            rows.append(
                {
                    "language": lang.upper(),
                    "repr_type": repr_type,
                    "daxis_50m": a50,
                    "daxis_100m": a100,
                    "delta_100m_minus_50m": a100 - a50,
                    "relative_change_pct": 100.0 * (a100 - a50) / max(1e-12, a50),
                }
            )

    out = pd.DataFrame(rows).sort_values(["language", "repr_type"])

    # Add per-language contextual/embedding ratio change summary.
    ratio_rows = []
    for lang in sorted(out["language"].unique()):
        e = out[(out["language"] == lang) & (out["repr_type"] == "embedding_matrix")]
        c = out[(out["language"] == lang) & (out["repr_type"] == "pre_lmhead_contextual")]
        if e.empty or c.empty:
            continue
        e50, e100 = float(e["daxis_50m"].iloc[0]), float(e["daxis_100m"].iloc[0])
        c50, c100 = float(c["daxis_50m"].iloc[0]), float(c["daxis_100m"].iloc[0])
        ratio_rows.append(
            {
                "language": lang,
                "repr_type": "ratio_ctx_over_emb",
                "daxis_50m": c50 / max(1e-12, e50),
                "daxis_100m": c100 / max(1e-12, e100),
                "delta_100m_minus_50m": (c100 / max(1e-12, e100)) - (c50 / max(1e-12, e50)),
                "relative_change_pct": 100.0
                * (((c100 / max(1e-12, e100)) - (c50 / max(1e-12, e50))) / max(1e-12, (c50 / max(1e-12, e50)))),
            }
        )

    if ratio_rows:
        out = pd.concat([out, pd.DataFrame(ratio_rows)], ignore_index=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()

