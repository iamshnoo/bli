#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_utils import (
    align_source_to_target,
    cosine_similarity_matrix,
    jaccard_divergence,
    load_probe_set,
    load_repr,
    l2_normalize,
    topk_neighbor_indices,
)

CORE_LANGS = ['zh', 'fr', 'fas', 'nld', 'ukr', 'bul', 'ind', 'deu']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Anchor subset sensitivity for EN-centered comparisons.')
    p.add_argument('--probe-set', type=Path, default=Path('data/probes/probe_sets.json'))
    p.add_argument('--out-csv', type=Path, required=True)
    p.add_argument('--subset-sizes', type=int, nargs='+', default=[500, 1000, 2000, 3000])
    p.add_argument('--draws', type=int, default=5)
    p.add_argument('--topk', type=int, default=25)
    p.add_argument('--seed', type=int, default=17)
    return p.parse_args()


def rep_roots() -> list[Path]:
    roots = [Path('outputs/revision/en_ablation/representations')]
    for lang in CORE_LANGS:
        roots.append(Path(f'outputs/multilingual_expansion/{lang}_shared_language/representations'))
        roots.append(Path(f'outputs/revision/{lang}_shared_language/representations'))
    return [p for p in roots if p.exists()]


def compute_metrics(aligned_a: np.ndarray, mat_b: np.ndarray, cultural_idx: list[int], axis_idx: list[tuple[int, int]], topk: int) -> tuple[float, float, float]:
    norm_a = l2_normalize(aligned_a)
    norm_b = l2_normalize(mat_b)
    jaccards = []
    for idx in cultural_idx:
        na = topk_neighbor_indices(norm_a, idx, topk)
        nb = topk_neighbor_indices(norm_b, idx, topk)
        jaccards.append(jaccard_divergence(set(na.tolist()), set(nb.tolist())))
    c_a = cosine_similarity_matrix(aligned_a[cultural_idx])
    c_b = cosine_similarity_matrix(mat_b[cultural_idx])
    frob = float(np.linalg.norm(c_a - c_b, ord='fro') / max(1, len(cultural_idx)))
    axis_vals = []
    for i, j in axis_idx:
        axis_source = aligned_a[j] - aligned_a[i]
        axis_target = mat_b[j] - mat_b[i]
        nax = np.linalg.norm(axis_source)
        nbx = np.linalg.norm(axis_target)
        if nax < 1e-12 or nbx < 1e-12:
            continue
        axis_source = axis_source / nax
        axis_target = axis_target / nbx
        pa = aligned_a[cultural_idx] @ axis_source
        pb = mat_b[cultural_idx] @ axis_target
        axis_vals.append(float(np.mean(np.abs(pa - pb))))
    axis_mean = float(np.mean(axis_vals)) if axis_vals else float('nan')
    return float(np.mean(jaccards)), frob, axis_mean


def main() -> None:
    args = parse_args()
    probe = load_probe_set(args.probe_set)
    neutral_idx = probe['_neutral_idx']
    cultural_idx = probe['_cultural_idx']
    axis_idx = probe['_axis_idx']
    roots = rep_roots()
    rng = np.random.default_rng(args.seed)

    rows = []
    for base in ['en_50m', 'en_100m']:
        for lang in CORE_LANGS:
            model_b = f'en_{lang}_a'
            for repr_type in ['embedding_matrix', 'pre_lmhead_contextual']:
                mat_a = load_repr(base, repr_type, roots)
                mat_b = load_repr(model_b, repr_type, roots)
                neutral_arr = np.array(neutral_idx, dtype=np.int64)
                for subset_size in args.subset_sizes:
                    if subset_size > len(neutral_arr):
                        continue
                    for draw in range(args.draws):
                        local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                        subset = np.sort(local_rng.choice(neutral_arr, size=subset_size, replace=False))
                        aligned_a, _w, resid_per = align_source_to_target(mat_a, mat_b, subset.tolist())
                        d_nn, d_struct, d_axis = compute_metrics(aligned_a, mat_b, cultural_idx, axis_idx, args.topk)
                        rows.append(
                            {
                                'baseline': base,
                                'language': lang.upper(),
                                'model_b': model_b,
                                'repr_type': repr_type,
                                'subset_size': int(subset_size),
                                'draw': int(draw),
                                'jaccard_at_k_mean': d_nn,
                                'frobenius_cultural_similarity': d_struct,
                                'axis_abs_projection_diff_mean': d_axis,
                                'anchor_residual_per_anchor': resid_per,
                            }
                        )
    out = pd.DataFrame(rows).sort_values(['baseline', 'repr_type', 'language', 'subset_size', 'draw'])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f'Wrote: {args.out_csv}')


if __name__ == '__main__':
    main()
