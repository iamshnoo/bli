#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute seed-matched same-language control metrics from cached representations.")
    p.add_argument("--probe-set", type=Path, default=Path("data/probes/probe_sets.json"))
    p.add_argument(
        "--rep-dir",
        type=Path,
        default=Path("outputs/revision/en_seed_null/representations"),
        help="Directory containing cached representation .npy files from run_bli_pipeline.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/revision/en_ablation/bli_same_language_controls.csv"),
    )
    return p.parse_args()


def compute_metrics(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    neutral_idx: list[int],
    cultural_idx: list[int],
    axis_idx: list[tuple[int, int]],
) -> dict[str, float]:
    w, _ = orthogonal_procrustes(mat_a[neutral_idx], mat_b[neutral_idx])
    aligned_a = mat_a @ w

    axis_abs = []
    axis_signed = []
    for i, j in axis_idx:
        va = aligned_a[j] - aligned_a[i]
        vb = mat_b[j] - mat_b[i]
        nva = np.linalg.norm(va)
        nvb = np.linalg.norm(vb)
        if nva < 1e-12 or nvb < 1e-12:
            continue
        va = va / nva
        vb = vb / nvb
        pa = aligned_a[cultural_idx] @ va
        pb = mat_b[cultural_idx] @ vb
        d = pa - pb
        axis_abs.append(float(np.mean(np.abs(d))))
        axis_signed.append(float(np.mean(d)))

    return {
        "axis_abs_projection_diff_mean": float(np.mean(axis_abs)) if axis_abs else float("nan"),
        "axis_signed_projection_diff_mean": float(np.mean(axis_signed)) if axis_signed else float("nan"),
    }


def main() -> None:
    args = parse_args()

    probe = json.loads(args.probe_set.read_text(encoding="utf-8"))
    words = probe["all_probe_words"]
    w2i = {w: i for i, w in enumerate(words)}
    neutral_idx = [w2i[w] for w in probe["neutral_anchor_words"] if w in w2i]
    cultural_idx = [w2i[w] for w in probe["cultural_probe_words"] if w in w2i]
    axis_idx = [(w2i[a], w2i[b]) for a, b in probe["semantic_axes"] if a in w2i and b in w2i]

    rep_dir = args.rep_dir
    if not rep_dir.exists():
        raise FileNotFoundError(f"Representation directory not found: {rep_dir}")

    def _discover_models(eval_repr: str) -> list[str]:
        out = []
        suffix = f"__{eval_repr}.npy"
        for p in rep_dir.glob(f"*{suffix}"):
            name = p.name[: -len(suffix)]
            out.append(name)
        return sorted(set(out))

    # Preferred true seed-matched null groups.
    groups: list[tuple[str, str]] = [
        ("EN-50M", r"^en_50m_s\d+$"),
        ("EN-100M", r"^en_100m_s\d+$"),
    ]

    rows: list[dict[str, float | str]] = []
    for group_label, pattern in groups:
        compiled = re.compile(pattern)
        available_emb = _discover_models("embedding_matrix")
        members = sorted([m for m in available_emb if compiled.match(m)])
        if len(members) < 2:
            continue

        for model_a, model_b in combinations(members, 2):
            for eval_repr in ["embedding_matrix", "pre_lmhead_contextual"]:
                pa = rep_dir / f"{model_a}__{eval_repr}.npy"
                pb = rep_dir / f"{model_b}__{eval_repr}.npy"
                if not pa.exists() or not pb.exists():
                    continue
                mat_a = np.load(pa)
                mat_b = np.load(pb)
                metrics = compute_metrics(
                    mat_a=mat_a,
                    mat_b=mat_b,
                    neutral_idx=neutral_idx,
                    cultural_idx=cultural_idx,
                    axis_idx=axis_idx,
                )
                rows.append(
                    {
                        "language": group_label,
                        "model_a": model_a,
                        "model_b": model_b,
                        "eval_repr": eval_repr,
                        **metrics,
                    }
                )

    # Fallback for older caches: single EN checkpoint mismatch pair if present.
    if not rows:
        model_a = "en_50m"
        model_b = "en_100m"
        for eval_repr in ["embedding_matrix", "pre_lmhead_contextual"]:
            pa = rep_dir / f"en_50m__{eval_repr}.npy"
            pb = rep_dir / f"en_100m__{eval_repr}.npy"
            if not pa.exists() or not pb.exists():
                continue
            mat_a = np.load(pa)
            mat_b = np.load(pb)
            metrics = compute_metrics(
                mat_a=mat_a,
                mat_b=mat_b,
                neutral_idx=neutral_idx,
                cultural_idx=cultural_idx,
                axis_idx=axis_idx,
            )
            rows.append(
                {
                    "language": "EN-CHECKPOINT-MISMATCH",
                    "model_a": model_a,
                    "model_b": model_b,
                    "eval_repr": eval_repr,
                    **metrics,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["language", "model_a", "model_b", "eval_repr"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
