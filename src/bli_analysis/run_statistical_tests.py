#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical tests for overlap ablations")
    p.add_argument("--word-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--translations-dir", type=Path, default=Path("data/probes"))
    p.add_argument("--quality-zh-csv", type=Path, default=None)
    p.add_argument("--quality-fr-csv", type=Path, default=None)
    return p.parse_args()


def infer_family_from_pair(pair: str) -> str | None:
    if "__vs__" not in pair:
        return None
    _base, rhs = pair.split("__vs__", 1)
    if not (rhs.startswith("en_") and rhs.endswith(("_a", "_b"))):
        return None
    return rhs[:-2]


def load_quality_map(qpath: Path) -> dict[str, str]:
    qdf = pd.read_csv(qpath)
    if "source_term_en" not in qdf.columns:
        return {}
    qdf["source_term_en"] = qdf["source_term_en"].astype(str).str.lower().str.strip()
    score_col = None
    if "comet_kiwi_score" in qdf.columns:
        qdf["comet_kiwi_score"] = pd.to_numeric(qdf["comet_kiwi_score"], errors="coerce")
        score_col = "comet_kiwi_score"
    elif "back_similarity" in qdf.columns:
        qdf["back_similarity"] = pd.to_numeric(qdf["back_similarity"], errors="coerce")
        score_col = "back_similarity"
    if score_col is None:
        return {}

    quality: dict[str, str] = {}
    for _, r in qdf.iterrows():
        w = str(r["source_term_en"])
        s_raw = r[score_col]
        s = float(s_raw) if np.isfinite(s_raw) else np.nan
        if not np.isfinite(s):
            continue
        if score_col == "comet_kiwi_score":
            if s >= 0.80:
                quality[w] = "high"
            elif s >= 0.60:
                quality[w] = "medium"
            else:
                quality[w] = "low"
        else:
            if s > 0.80:
                quality[w] = "high"
            elif s >= 0.55:
                quality[w] = "medium"
            else:
                quality[w] = "low"
    return quality


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.word_csv)

    families = sorted(
        {
            fam
            for fam in df["pair"].astype(str).map(infer_family_from_pair)
            if fam is not None
        }
    )

    fam_quality: dict[str, dict[str, str]] = {}
    if args.translations_dir.exists():
        for qpath in sorted(args.translations_dir.glob("translations_*.csv")):
            lang = qpath.stem.replace("translations_", "").strip().lower()
            if not lang:
                continue
            fam_quality[f"en_{lang}"] = load_quality_map(qpath)

    # Backward-compatible explicit overrides.
    for fam, qpath in [("en_zh", args.quality_zh_csv), ("en_fr", args.quality_fr_csv)]:
        if qpath is not None and qpath.exists():
            fam_quality[fam] = load_quality_map(qpath)

    rows: list[dict] = []
    for repr_type in sorted(df["repr_type"].unique()):
        for base in ["en_50m", "en_100m"]:
            for fam in families:
                pa = f"{base}__vs__{fam}_a"
                pb = f"{base}__vs__{fam}_b"
                a = df[(df["repr_type"] == repr_type) & (df["pair"] == pa)][["word", "jaccard_divergence"]].rename(columns={"jaccard_divergence": "a"})
                b = df[(df["repr_type"] == repr_type) & (df["pair"] == pb)][["word", "jaccard_divergence"]].rename(columns={"jaccard_divergence": "b"})
                m = a.merge(b, on="word", how="inner")
                if m.empty:
                    continue
                quality_map = fam_quality.get(fam, {})
                m["quality_tier"] = m["word"].astype(str).str.lower().map(lambda w: quality_map.get(w, "unknown"))

                for tier in ["all", "high", "medium", "low"]:
                    if tier == "all":
                        mt = m
                    else:
                        mt = m[m["quality_tier"] == tier]
                    if mt.empty:
                        continue
                    d = (mt["a"] - mt["b"]).to_numpy()
                    if len(d) < 10:
                        continue

                    # Wilcoxon signed-rank test on per-word divergence deltas.
                    stat, pval = wilcoxon(
                        d, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto"
                    )
                    rows.append(
                        {
                            "repr_type": repr_type,
                            "baseline": base,
                            "family": fam,
                            "quality_tier": tier,
                            "n_words": int(len(d)),
                            "mean_delta_jaccard": float(np.mean(d)),
                            "median_delta_jaccard": float(np.median(d)),
                            "wilcoxon_stat": float(stat),
                            "p_value": float(pval),
                        }
                    )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
