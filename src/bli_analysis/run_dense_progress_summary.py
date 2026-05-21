#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from shared_utils import (
    DEFAULT_PROMPTS,
    extract_contextual_pre_lmhead_reprs,
    extract_embedding_matrix_reprs,
    load_probe_set,
    load_tokenizer_and_model,
    resolve_device,
)

STEP_RE = re.compile(r'_(\d{3,4})$')
LANGS = ['zh', 'fr', 'fas', 'nld', 'ukr', 'bul', 'ind', 'deu']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Lean dense progress scorer for all languages.')
    p.add_argument('--models-json', type=Path, required=True)
    p.add_argument('--probe-set', type=Path, default=Path('data/probes/probe_sets.json'))
    p.add_argument('--output-csv', type=Path, required=True)
    p.add_argument('--derived-csv', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, default=Path('outputs/revision/en_progress_trajectory_all/representations'))
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--topk', type=int, default=25)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return p.parse_args()


def infer_step(name: str) -> int:
    m = STEP_RE.search(name)
    if not m:
        raise ValueError(f'Could not infer step from {name}')
    return int(m.group(1))


def load_or_extract(model_name: str, model_path: str, words: list[str], cache_dir: Path, batch_size: int, device) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_cache = cache_dir / f'{model_name}__embedding_matrix.npy'
    ctx_cache = cache_dir / f'{model_name}__pre_lmhead_contextual.npy'
    if emb_cache.exists() and ctx_cache.exists():
        return np.load(emb_cache), np.load(ctx_cache)
    tok, model = load_tokenizer_and_model(model_path, device)
    try:
        emb = extract_embedding_matrix_reprs(model, tok, words)
        ctx = extract_contextual_pre_lmhead_reprs(model, tok, words, DEFAULT_PROMPTS, batch_size, device)
    finally:
        del model
        if device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()
    np.save(emb_cache, emb)
    np.save(ctx_cache, ctx)
    return emb, ctx


def compute_summary(
    model_a: str,
    model_b: str,
    repr_type: str,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    neutral_idx: list[int],
    cultural_idx: list[int],
    axis_idx: list[tuple[int, int]],
) -> dict:
    w, _ = orthogonal_procrustes(mat_a[list(neutral_idx)], mat_b[list(neutral_idx)])
    resid = (mat_a[list(neutral_idx)] @ w) - mat_b[list(neutral_idx)]
    resid_fro = float(np.linalg.norm(resid, ord='fro'))
    resid_per = float(resid_fro / max(1, len(neutral_idx)))

    aligned_cultural = mat_a[cultural_idx] @ w
    target_cultural = mat_b[cultural_idx]
    endpoint_idx = sorted({i for pair in axis_idx for i in pair})
    aligned_endpoints = {idx: mat_a[idx] @ w for idx in endpoint_idx}
    target_endpoints = {idx: mat_b[idx] for idx in endpoint_idx}

    axis_vals = []
    axis_signed = []
    axis_rel = []
    for i, j in axis_idx:
        va = aligned_endpoints[j] - aligned_endpoints[i]
        vb = target_endpoints[j] - target_endpoints[i]
        nax = np.linalg.norm(va)
        nbx = np.linalg.norm(vb)
        if nax < 1e-12 or nbx < 1e-12:
            continue
        va = va / nax
        vb = vb / nbx
        pa = aligned_cultural @ va
        pb = target_cultural @ vb
        d = pa - pb
        axis_vals.append(float(np.mean(np.abs(d))))
        denom = np.maximum(np.abs(pa) + np.abs(pb), 1e-12)
        axis_rel.append(float(np.mean(np.abs(d) / denom)))
        axis_signed.append(float(np.mean(d)))
    return {
        'repr_type': repr_type,
        'model_a': model_a,
        'model_b': model_b,
        'jaccard_at_k_mean': float('nan'),
        'jaccard_at_k_std': float('nan'),
        'frobenius_cultural_similarity': float('nan'),
        'procrustes_anchor_residual_fro': resid_fro,
        'procrustes_anchor_residual_per_anchor': resid_per,
        'axis_abs_projection_diff_mean': float(np.mean(axis_vals)),
        'axis_abs_projection_diff_rel_mean': float(np.mean(axis_rel)),
        'axis_abs_projection_diff_max': float(np.max(axis_vals)),
        'axis_signed_projection_diff_mean': float(np.mean(axis_signed)),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    probe = load_probe_set(args.probe_set)
    words = probe['_words']
    models = json.loads(args.models_json.read_text(encoding='utf-8'))

    reprs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name, model_path in models.items():
        reprs[model_name] = load_or_extract(model_name, model_path, words, args.cache_dir, args.batch_size, device)

    rows = []
    for step in [500, 1000, 1500, 2000, 2500, 3000]:
        base_name = f'en_{step}'
        for lang in LANGS:
            bi_name = f'en_{lang}_a_{step}'
            emb_a, ctx_a = reprs[base_name]
            emb_b, ctx_b = reprs[bi_name]
            rows.append(compute_summary(base_name, bi_name, 'embedding_matrix', emb_a, emb_b, probe['_neutral_idx'], probe['_cultural_idx'], probe['_axis_idx']))
            rows.append(compute_summary(base_name, bi_name, 'pre_lmhead_contextual', ctx_a, ctx_b, probe['_neutral_idx'], probe['_cultural_idx'], probe['_axis_idx']))

    out = pd.DataFrame(rows).sort_values(['model_b', 'repr_type']).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    derived_rows = []
    for _, row in out.iterrows():
        derived_rows.append({
            'step': infer_step(str(row['model_a'])),
            'language': str(row['model_b']).split('_')[1].upper(),
            'repr_type': str(row['repr_type']),
            'axis_abs_projection_diff_mean': float(row['axis_abs_projection_diff_mean']),
            'jaccard_at_k_mean': float(row['jaccard_at_k_mean']),
            'frobenius_cultural_similarity': float(row['frobenius_cultural_similarity']),
            'procrustes_anchor_residual_per_anchor': float(row['procrustes_anchor_residual_per_anchor']),
        })
    pd.DataFrame(derived_rows).sort_values(['language', 'repr_type', 'step']).to_csv(args.derived_csv, index=False)
    print(f'Wrote: {args.output_csv}')
    print(f'Wrote: {args.derived_csv}')


if __name__ == '__main__':
    main()
