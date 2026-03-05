#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EN_IDV = 91

LANG_SPECS = [
    {
        "code": "zh",
        "language": "Chinese",
        "family": "Sino-Tibetan",
        "idv": 20,
        "summary_path": "revision/zh_shared_language/bli_summary_metrics.csv",
        "ci_path": "revision/zh_shared_language/bli_bootstrap_ci.csv",
        "mono_prefix": "zh",
        "bi_prefix": "en_zh",
    },
    {
        "code": "fr",
        "language": "French",
        "family": "Indo-Eur (Romance)",
        "idv": 71,
        "summary_path": "revision/fr_shared_language/bli_summary_metrics.csv",
        "ci_path": "revision/fr_shared_language/bli_bootstrap_ci.csv",
        "mono_prefix": "fr",
        "bi_prefix": "en_fr",
    },
    {
        "code": "fas",
        "language": "Persian",
        "family": "Indo-Eur (Iranian)",
        "idv": 41,
        "summary_path": "multi/fas_shared_language/bli_summary_metrics.csv",
        "ci_path": "multi/fas_shared_language/bli_bootstrap_ci.csv",
        "mono_prefix": "fas",
        "bi_prefix": "en_fas",
    },
    {
        "code": "nld",
        "language": "Dutch",
        "family": "Indo-Eur (Germanic)",
        "idv": 80,
        "summary_path": "multi/nld_shared_language/bli_summary_metrics.csv",
        "ci_path": "multi/nld_shared_language/bli_bootstrap_ci.csv",
        "mono_prefix": "nld",
        "bi_prefix": "en_nld",
    },
    {
        "code": "ukr",
        "language": "Ukrainian",
        "family": "Indo-Eur (Slavic)",
        "idv": 25,
        "summary_path": "multi/ukr_shared_language/bli_summary_metrics.csv",
        "ci_path": "multi/ukr_shared_language/bli_bootstrap_ci.csv",
        "mono_prefix": "ukr",
        "bi_prefix": "en_ukr",
    },
    {
        "code": "bul",
        "language": "Bulgarian",
        "family": "Indo-Eur (Slavic)",
        "idv": 30,
        "summary_path": "multi/bul_shared_language/bli_summary_metrics.csv",
        "ci_path": "multi/bul_shared_language/bli_bootstrap_ci.csv",
        "mono_prefix": "bul",
        "bi_prefix": "en_bul",
    },
    {
        "code": "ind",
        "language": "Indonesian",
        "family": "Austronesian",
        "idv": 14,
        "summary_path": "multi/ind_shared_language/bli_summary_metrics.csv",
        "ci_path": "multi/ind_shared_language/bli_bootstrap_ci.csv",
        "mono_prefix": "ind",
        "bi_prefix": "en_ind",
    },
    {
        "code": "deu",
        "language": "German",
        "family": "Indo-Eur (Germanic)",
        "idv": 67,
        "summary_path": "multi/deu_shared_language/bli_summary_metrics.csv",
        "ci_path": "multi/deu_shared_language/bli_bootstrap_ci.csv",
        "mono_prefix": "deu",
        "bi_prefix": "en_deu",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build language_ratio_summary.csv for multilingual main-text artifacts")
    p.add_argument("--revision-root", type=Path, default=Path("outputs/revision"))
    p.add_argument(
        "--multilingual-root",
        type=Path,
        default=Path("outputs/multilingual_expansion"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/multilingual_expansion/language_ratio_summary.csv"),
    )
    return p.parse_args()


def load_paths(spec: dict, revision_root: Path, multilingual_root: Path) -> tuple[Path, Path]:
    if spec["summary_path"].startswith("revision/"):
        summary_path = revision_root / spec["summary_path"].replace("revision/", "")
        ci_path = revision_root / spec["ci_path"].replace("revision/", "")
    else:
        summary_path = multilingual_root / spec["summary_path"].replace("multi/", "")
        ci_path = multilingual_root / spec["ci_path"].replace("multi/", "")
    return summary_path, ci_path


def axis_value(summary: pd.DataFrame, repr_type: str, model_a: str, model_b: str) -> float:
    sub = summary[
        (summary["repr_type"] == repr_type)
        & (summary["model_a"] == model_a)
        & (summary["model_b"] == model_b)
    ]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0]["axis_abs_projection_diff_mean"])


def ratio_ci(ci: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float]:
    sub = ci[
        (ci["metric"] == "axis_ratio_contextual_over_embedding")
        & (ci["model_a"] == model_a)
        & (ci["model_b"] == model_b)
    ]
    if sub.empty:
        return float("nan"), float("nan")
    return float(sub.iloc[0]["ci_low"]), float(sub.iloc[0]["ci_high"])


def main() -> None:
    args = parse_args()
    rows = []

    for spec in LANG_SPECS:
        summary_path, ci_path = load_paths(spec, args.revision_root, args.multilingual_root)
        if not summary_path.exists():
            print(f"[warn] Missing summary: {summary_path}")
            continue

        summary = pd.read_csv(summary_path)
        ci = pd.read_csv(ci_path) if ci_path.exists() else pd.DataFrame()

        mono = spec["mono_prefix"]
        bi = spec["bi_prefix"]

        e50 = axis_value(summary, "embedding_matrix", f"{mono}_50m", f"{bi}_a")
        c50 = axis_value(summary, "pre_lmhead_contextual", f"{mono}_50m", f"{bi}_a")
        e100 = axis_value(summary, "embedding_matrix", f"{mono}_100m", f"{bi}_a")
        c100 = axis_value(summary, "pre_lmhead_contextual", f"{mono}_100m", f"{bi}_a")

        ratio_50m = c50 / max(1e-12, e50)
        ratio_100m = c100 / max(1e-12, e100)
        ci_50m_low, ci_50m_high = ratio_ci(ci, f"{mono}_50m", f"{bi}_a")
        ci_100m_low, ci_100m_high = ratio_ci(ci, f"{mono}_100m", f"{bi}_a")

        rows.append(
            {
                "language": spec["language"],
                "lang_code": spec["code"],
                "family": spec["family"],
                "hofstede_idv": spec["idv"],
                "hofstede_dist": abs(EN_IDV - spec["idv"]),
                "ratio_50m": ratio_50m,
                "ratio_100m": ratio_100m,
                "avg_ratio": float(np.nanmean([ratio_50m, ratio_100m])),
                "ci_50m_low": ci_50m_low,
                "ci_50m_high": ci_50m_high,
                "ci_100m_low": ci_100m_low,
                "ci_100m_high": ci_100m_high,
            }
        )

    out = pd.DataFrame(rows).sort_values("language")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
