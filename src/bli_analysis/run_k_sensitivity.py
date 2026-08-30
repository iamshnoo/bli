#!/usr/bin/env python3
"""Evaluate nearest-neighbor disagreement across several neighborhood sizes.

This reuses the cached EN-centered representations from the main C3 comparison
and the matched English-only seed controls. Orthogonal alignment is omitted
here because it preserves all within-space cosine neighborhoods exactly.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from shared_utils import jaccard_divergence, l2_normalize, load_probe_set, load_repr


LANGS = ["zh", "fr", "fas", "nld", "ukr", "bul", "ind", "deu"]
REPRESENTATIONS = ["embedding_matrix", "pre_lmhead_contextual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor k sensitivity for the EN-centered C3 pairs."
    )
    parser.add_argument(
        "--probe-set", type=Path, default=Path("data/probes/probe_sets.json")
    )
    parser.add_argument(
        "--main-root",
        type=Path,
        default=Path("outputs/revision/en_ablation/representations"),
    )
    parser.add_argument(
        "--seed-root",
        type=Path,
        default=Path("outputs/revision/en_seed_null/representations"),
    )
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def top_neighbors(
    matrix: np.ndarray, evaluation_idx: list[int], max_k: int
) -> np.ndarray:
    normalized = l2_normalize(matrix)
    eval_array = np.asarray(evaluation_idx, dtype=np.int64)
    similarities = normalized[eval_array] @ normalized.T
    similarities[np.arange(len(eval_array)), eval_array] = -np.inf
    candidates = np.argpartition(-similarities, kth=max_k - 1, axis=1)[:, :max_k]
    candidate_scores = np.take_along_axis(similarities, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def mean_disagreement(left: np.ndarray, right: np.ndarray, k: int) -> float:
    values = [
        jaccard_divergence(set(a[:k].tolist()), set(b[:k].tolist()))
        for a, b in zip(left, right, strict=True)
    ]
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    if any(k <= 0 for k in args.k):
        raise ValueError("All k values must be positive")

    probe = load_probe_set(args.probe_set)
    evaluation_idx = probe["_cultural_idx"]
    max_k = max(args.k)
    rows: list[dict[str, float | int | str]] = []

    main_models = ["en_100m"] + [
        f"en_{lang}_{variant}" for lang in LANGS for variant in ["a", "b"]
    ]
    bilingual_pairs = [
        ("en_100m", f"en_{lang}_{variant}")
        for lang in LANGS
        for variant in ["a", "b"]
    ]
    seed_models = ["en_100m_s1", "en_100m_s2", "en_100m_s3"]
    # Keep the six ordered pairings used by the paper's expanded seed null.
    seed_pairs = list(permutations(seed_models, 2))

    for representation in REPRESENTATIONS:
        neighbors: dict[str, np.ndarray] = {}
        for model in main_models:
            matrix = load_repr(model, representation, [args.main_root])
            neighbors[model] = top_neighbors(matrix, evaluation_idx, max_k)
        for model in seed_models:
            matrix = load_repr(model, representation, [args.seed_root])
            neighbors[model] = top_neighbors(matrix, evaluation_idx, max_k)

        for k in args.k:
            seed_values = np.asarray(
                [mean_disagreement(neighbors[a], neighbors[b], k) for a, b in seed_pairs]
            )
            bilingual_values = np.asarray(
                [
                    mean_disagreement(neighbors[a], neighbors[b], k)
                    for a, b in bilingual_pairs
                ]
            )
            centered = bilingual_values - seed_values.mean()
            rows.append(
                {
                    "k": k,
                    "representation": (
                        "Embedding"
                        if representation == "embedding_matrix"
                        else "Contextual"
                    ),
                    "seed_mean": float(seed_values.mean()),
                    "bilingual_mean": float(bilingual_values.mean()),
                    "bilingual_median": float(np.median(bilingual_values)),
                    "centered_mean": float(centered.mean()),
                    "centered_median": float(np.median(centered)),
                    "centered_min": float(centered.min()),
                    "centered_max": float(centered.max()),
                    "n_bilingual_pairs": len(bilingual_pairs),
                    "n_seed_pairs": len(seed_pairs),
                }
            )

    output = pd.DataFrame(rows).sort_values(["k", "representation"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out_csv, index=False)
    print(output.to_string(index=False))
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
