#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_utils import align_source_to_target, load_probe_set, load_repr

CORE_LANGS = ['zh', 'fr', 'fas', 'nld', 'ukr', 'bul', 'ind', 'deu']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Specificity bridge for theory axes vs controls.')
    p.add_argument('--probe-set', type=Path, default=Path('data/probes/probe_sets.json'))
    p.add_argument('--out-csv', type=Path, required=True)
    p.add_argument('--summary-csv', type=Path, required=True)
    p.add_argument('--random-draws', type=int, default=5)
    p.add_argument('--seed', type=int, default=23)
    return p.parse_args()


def rep_roots() -> list[Path]:
    roots = [Path('outputs/revision/en_ablation/representations')]
    for lang in CORE_LANGS:
        roots.append(Path(f'outputs/multilingual_expansion/{lang}_shared_language/representations'))
        roots.append(Path(f'outputs/revision/{lang}_shared_language/representations'))
    return [p for p in roots if p.exists()]


def build_random_axis_draws(probe: dict, n_draws: int, seed: int) -> list[list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    w2i = probe['_w2i']
    left_words = [a for a, _b in probe['semantic_axes'] if a in w2i]
    right_words = [b for _a, b in probe['semantic_axes'] if b in w2i]
    originals = list(zip(left_words, right_words))
    draws = []
    for _ in range(n_draws):
        perm = right_words.copy()
        for _attempt in range(1000):
            rng.shuffle(perm)
            if all((l != r) and ((l, r) not in originals) for l, r in zip(left_words, perm)):
                break
        draws.append([(w2i[l], w2i[r]) for l, r in zip(left_words, perm)])
    return draws


def compute_axis_bundle(aligned_a: np.ndarray, mat_b: np.ndarray, probe_idx: list[int], axis_idx: list[tuple[int, int]]) -> tuple[float, float]:
    axis_abs = []
    coherence_parts = []
    for i, j in axis_idx:
        va = aligned_a[j] - aligned_a[i]
        vb = mat_b[j] - mat_b[i]
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            continue
        va = va / na
        vb = vb / nb
        d = (aligned_a[probe_idx] @ va) - (mat_b[probe_idx] @ vb)
        mean_abs = float(np.mean(np.abs(d)))
        mean_signed = float(np.mean(d))
        axis_abs.append(mean_abs)
        if mean_abs > 1e-12:
            coherence_parts.append(abs(mean_signed) / mean_abs)
    return (
        float(np.mean(axis_abs)) if axis_abs else float('nan'),
        float(np.mean(coherence_parts)) if coherence_parts else float('nan'),
    )


def main() -> None:
    args = parse_args()
    probe = load_probe_set(args.probe_set)
    roots = rep_roots()
    random_axes = build_random_axis_draws(probe, args.random_draws, args.seed)

    rows = []
    for base in ['en_50m', 'en_100m']:
        for lang in CORE_LANGS:
            model_b = f'en_{lang}_a'
            for repr_type in ['embedding_matrix', 'pre_lmhead_contextual']:
                mat_a = load_repr(base, repr_type, roots)
                mat_b = load_repr(model_b, repr_type, roots)
                aligned_a, _w, resid = align_source_to_target(mat_a, mat_b, probe['_neutral_idx'])
                cultural_axis, cultural_coh = compute_axis_bundle(aligned_a, mat_b, probe['_cultural_idx'], probe['_axis_idx'])
                negative_axis, negative_coh = compute_axis_bundle(aligned_a, mat_b, probe['_negative_idx'], probe['_axis_idx'])
                rows.append({
                    'baseline': base,
                    'language': lang.upper(),
                    'repr_type': repr_type,
                    'condition': 'cultural_theory',
                    'draw': 0,
                    'axis_abs_projection_diff_mean': cultural_axis,
                    'axis_sign_coherence': cultural_coh,
                    'anchor_residual_per_anchor': resid,
                })
                rows.append({
                    'baseline': base,
                    'language': lang.upper(),
                    'repr_type': repr_type,
                    'condition': 'negative_theory',
                    'draw': 0,
                    'axis_abs_projection_diff_mean': negative_axis,
                    'axis_sign_coherence': negative_coh,
                    'anchor_residual_per_anchor': resid,
                })
                for draw_id, axis_draw in enumerate(random_axes, start=1):
                    random_axis, random_coh = compute_axis_bundle(aligned_a, mat_b, probe['_cultural_idx'], axis_draw)
                    rows.append({
                        'baseline': base,
                        'language': lang.upper(),
                        'repr_type': repr_type,
                        'condition': 'cultural_random',
                        'draw': draw_id,
                        'axis_abs_projection_diff_mean': random_axis,
                        'axis_sign_coherence': random_coh,
                        'anchor_residual_per_anchor': resid,
                    })

    out = pd.DataFrame(rows).sort_values(['baseline', 'repr_type', 'condition', 'language', 'draw']).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary_rows = []
    for (baseline, condition), sub in out.groupby(['baseline', 'condition']):
        emb = sub[sub['repr_type'] == 'embedding_matrix']['axis_abs_projection_diff_mean'].to_numpy(dtype=float)
        ctx = sub[sub['repr_type'] == 'pre_lmhead_contextual']['axis_abs_projection_diff_mean'].to_numpy(dtype=float)
        coh_emb = sub[sub['repr_type'] == 'embedding_matrix']['axis_sign_coherence'].to_numpy(dtype=float)
        coh_ctx = sub[sub['repr_type'] == 'pre_lmhead_contextual']['axis_sign_coherence'].to_numpy(dtype=float)
        summary_rows.append({
            'baseline': baseline,
            'condition': condition,
            'embedding_axis_mean': float(np.mean(emb)),
            'contextual_axis_mean': float(np.mean(ctx)),
            'contextual_over_embedding_ratio': float(np.mean(ctx) / max(1e-12, np.mean(emb))),
            'embedding_sign_coherence_mean': float(np.mean(coh_emb)),
            'contextual_sign_coherence_mean': float(np.mean(coh_ctx)),
            'n_rows': int(len(sub)),
        })
    pd.DataFrame(summary_rows).sort_values(['baseline', 'condition']).to_csv(args.summary_csv, index=False)
    print(f'Wrote: {args.out_csv}')
    print(f'Wrote: {args.summary_csv}')


if __name__ == '__main__':
    main()
