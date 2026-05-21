#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shared_utils import (
    align_source_to_target,
    case_signed_shift,
    load_probe_set,
    load_repr,
    load_tokenizer_and_model,
    resolve_device,
    safe_mean,
    score_completion,
)

CORE_LANGS = ['zh', 'fr', 'fas', 'nld', 'ukr', 'bul', 'ind', 'deu']
TEMPLATES = [
    'If one had to choose between "{left}" and "{right}" for "{probe}", the closer association is',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Bridge signed representation shift to output preferences.')
    p.add_argument('--probe-set', type=Path, default=Path('data/probes/probe_sets.json'))
    p.add_argument('--cases-csv', type=Path, default=Path('data/probes/behavioral_bridge_cases.csv'))
    p.add_argument('--out-csv', type=Path, required=True)
    p.add_argument('--summary-csv', type=Path, required=True)
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return p.parse_args()


def rep_roots() -> list[Path]:
    roots = [Path('outputs/revision/en_ablation/representations')]
    for lang in CORE_LANGS:
        roots.append(Path(f'outputs/multilingual_expansion/{lang}_shared_language/representations'))
        roots.append(Path(f'outputs/revision/{lang}_shared_language/representations'))
    return [p for p in roots if p.exists()]


def parse_cases(path: Path) -> list[tuple[str, str, str]]:
    df = pd.read_csv(path)
    out = []
    seen = set()
    if {'probe', 'left_endpoint', 'right_endpoint'}.issubset(df.columns):
        case_iter = df[['probe', 'left_endpoint', 'right_endpoint']].itertuples(index=False, name=None)
    elif 'probe_axis' in df.columns:
        case_iter = []
        for raw in df['probe_axis'].astype(str):
            probe, axis = [x.strip() for x in raw.split('|', 1)]
            left, right = [x.strip() for x in axis.split('->', 1)]
            case_iter.append((probe, left, right))
    else:
        raise ValueError(f'{path} must contain either probe/left_endpoint/right_endpoint columns or probe_axis')
    for probe, left, right in case_iter:
        probe = str(probe).strip()
        left = str(left).strip()
        right = str(right).strip()
        key = (probe, left, right)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def build_random_controls(probe: dict, cases: list[tuple[str, str, str]], seed: int) -> list[tuple[str, str, str]]:
    rng = np.random.default_rng(seed)
    axis_words = sorted({w for axis in probe['semantic_axes'] for w in axis})
    random_cases = []
    for probe_word, left, right in cases:
        for _ in range(1000):
            rand_left, rand_right = rng.choice(axis_words, size=2, replace=False).tolist()
            if (rand_left, rand_right) != (left, right):
                random_cases.append((probe_word, rand_left, rand_right))
                break
        else:
            random_cases.append((probe_word, left, right))
    return random_cases


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    probe = load_probe_set(args.probe_set)
    w2i = probe['_w2i']
    roots = rep_roots()
    theory_cases = parse_cases(args.cases_csv)
    random_cases = build_random_controls(probe, theory_cases, args.seed)

    rep_cache: dict[tuple[str, str], np.ndarray] = {}
    def get_repr(model_name: str, repr_type: str) -> np.ndarray:
        key = (model_name, repr_type)
        if key not in rep_cache:
            rep_cache[key] = load_repr(model_name, repr_type, roots)
        return rep_cache[key]

    rows = []
    for base in ['en_50m', 'en_100m']:
        en_tok, en_model = load_tokenizer_and_model(Path('models/hf') / base, device)
        try:
            for lang in CORE_LANGS:
                model_b = f'en_{lang}_a'
                bi_tok, bi_model = load_tokenizer_and_model(Path('models/hf') / model_b, device)
                try:
                    emb_en = get_repr(base, 'embedding_matrix')
                    emb_bi = get_repr(model_b, 'embedding_matrix')
                    ctx_en = get_repr(base, 'pre_lmhead_contextual')
                    ctx_bi = get_repr(model_b, 'pre_lmhead_contextual')
                    emb_aligned, _emb_w, _ = align_source_to_target(emb_en, emb_bi, probe['_neutral_idx'])
                    ctx_aligned, _ctx_w, _ = align_source_to_target(ctx_en, ctx_bi, probe['_neutral_idx'])

                    for condition, cases in [('theory', theory_cases), ('random', random_cases)]:
                        for probe_word, left, right in cases:
                            if probe_word not in w2i or left not in w2i or right not in w2i:
                                continue
                            pidx = w2i[probe_word]
                            lidx = w2i[left]
                            ridx = w2i[right]
                            repr_signed_emb = case_signed_shift(emb_aligned, emb_bi, pidx, lidx, ridx)
                            repr_signed_ctx = case_signed_shift(ctx_aligned, ctx_bi, pidx, lidx, ridx)

                            per_template = []
                            for tmpl in TEMPLATES:
                                prompt = tmpl.format(probe=probe_word, left=left, right=right)
                                pref_en = score_completion(en_model, en_tok, prompt, f' {left}', device) - score_completion(en_model, en_tok, prompt, f' {right}', device)
                                pref_bi = score_completion(bi_model, bi_tok, prompt, f' {left}', device) - score_completion(bi_model, bi_tok, prompt, f' {right}', device)
                                per_template.append(float(pref_bi - pref_en))
                            delta_out = safe_mean(per_template)
                            rows.append(
                                {
                                    'baseline': base,
                                    'language': lang.upper(),
                                    'model_b': model_b,
                                    'condition': condition,
                                    'probe': probe_word,
                                    'left_endpoint': left,
                                    'right_endpoint': right,
                                    'probe_axis': f'{probe_word} | {left}->{right}',
                                    'repr_signed_embedding': repr_signed_emb,
                                    'repr_signed_contextual': repr_signed_ctx,
                                    'output_shift_left_minus_right': delta_out,
                                    'template_shift_mean': delta_out,
                                    'template_shift_std': float(np.std(per_template)),
                                    'sign_agree_contextual': int(np.sign(repr_signed_ctx) == np.sign(delta_out)) if np.isfinite(repr_signed_ctx) and np.isfinite(delta_out) and abs(delta_out) > 1e-9 else np.nan,
                                    'sign_agree_embedding': int(np.sign(repr_signed_emb) == np.sign(delta_out)) if np.isfinite(repr_signed_emb) and np.isfinite(delta_out) and abs(delta_out) > 1e-9 else np.nan,
                                }
                            )
                    if rows:
                        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(rows).sort_values(['baseline', 'condition', 'language', 'probe_axis']).to_csv(args.out_csv, index=False)
                finally:
                    del bi_model
                    if device.type == 'cuda':
                        import torch
                        torch.cuda.empty_cache()
        finally:
            del en_model
            if device.type == 'cuda':
                import torch
                torch.cuda.empty_cache()

    out = pd.DataFrame(rows).sort_values(['baseline', 'condition', 'language', 'probe_axis']).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary_rows = []
    grouped = [((baseline, condition), sub) for (baseline, condition), sub in out.groupby(['baseline', 'condition'])]
    grouped.append((('all', 'theory'), out[out['condition'] == 'theory']))
    for (baseline, condition), sub in grouped:
        valid = sub.dropna(subset=['output_shift_left_minus_right', 'repr_signed_contextual', 'repr_signed_embedding']).copy()
        if valid.empty:
            continue
        ctx_rho, ctx_p = spearmanr(np.abs(valid['repr_signed_contextual']), np.abs(valid['output_shift_left_minus_right']))
        emb_rho, emb_p = spearmanr(np.abs(valid['repr_signed_embedding']), np.abs(valid['output_shift_left_minus_right']))
        summary_rows.append(
            {
                'baseline': baseline,
                'condition': condition,
                'n_cases': int(len(valid)),
                'contextual_sign_agreement': float(valid['sign_agree_contextual'].mean()),
                'embedding_sign_agreement': float(valid['sign_agree_embedding'].mean()),
                'contextual_abs_rho': float(ctx_rho),
                'contextual_abs_p': float(ctx_p),
                'embedding_abs_rho': float(emb_rho),
                'embedding_abs_p': float(emb_p),
                'mean_abs_output_shift': float(np.mean(np.abs(valid['output_shift_left_minus_right']))),
            }
        )
    pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)
    print(f'Wrote: {args.out_csv}')
    print(f'Wrote: {args.summary_csv}')


if __name__ == '__main__':
    main()
