#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def safe_wilcoxon(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.allclose(x, 0.0):
        return float('nan'), float('nan')
    stat, p = wilcoxon(x, alternative='greater', zero_method='wilcox', correction=False, mode='auto')
    return float(stat), float(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Aggregate statistical tests for scope-expansion analyses.')
    p.add_argument('--summary-csv', type=Path, default=Path('outputs/revision/en_ablation/bli_summary_metrics.csv'))
    p.add_argument('--same-language-csv', type=Path, default=None)
    p.add_argument('--behavioral-csv', type=Path, required=True)
    p.add_argument('--specificity-csv', type=Path, required=True)
    p.add_argument('--anchor-csv', type=Path, required=True)
    p.add_argument('--out-csv', type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    summary = pd.read_csv(args.summary_csv)
    pivot = summary.pivot_table(
        index=['model_a', 'model_b'],
        columns='repr_type',
        values='axis_abs_projection_diff_mean',
        aggfunc='first',
    ).reset_index()
    pivot['ctx_minus_emb'] = pivot['pre_lmhead_contextual'] - pivot['embedding_matrix']
    for baseline in ['en_50m', 'en_100m', 'all']:
        sub = pivot if baseline == 'all' else pivot[pivot['model_a'] == baseline]
        stat, p = safe_wilcoxon(sub['ctx_minus_emb'].to_numpy())
        rows.append({
            'test': 'contextual_gt_embedding_axis',
            'slice': baseline,
            'n': int(len(sub)),
            'mean_difference': float(np.mean(sub['ctx_minus_emb'])),
            'statistic': stat,
            'p_value_greater': p,
        })

    if args.same_language_csv and args.same_language_csv.exists():
        same = pd.read_csv(args.same_language_csv)
        same = same.rename(
            columns={
                'language': 'baseline_tag',
                'eval_repr': 'repr_type',
                'axis_abs_projection_diff_mean': 'axis_abs_projection_diff_mean',
            }
        )
        same['model_a'] = same['baseline_tag'].str.lower().str.replace('-', '_', regex=False)
        for baseline in ['en_50m', 'en_100m']:
            for repr_type in ['embedding_matrix', 'pre_lmhead_contextual']:
                bilingual = summary[(summary['model_a'] == baseline) & (summary['repr_type'] == repr_type)]
                null = same[(same['model_a'] == baseline) & (same['repr_type'] == repr_type)]
                if bilingual.empty or null.empty:
                    continue
                diff = bilingual['axis_abs_projection_diff_mean'].to_numpy() - float(null['axis_abs_projection_diff_mean'].mean())
                stat, p = safe_wilcoxon(diff)
                rows.append({
                    'test': 'bilingual_gt_same_language_null',
                    'slice': f'{baseline}:{repr_type}',
                    'n': int(len(diff)),
                    'mean_difference': float(np.mean(diff)),
                    'statistic': stat,
                    'p_value_greater': p,
                })

    # Overlap effect within language families.
    aug = summary.copy()
    aug['family'] = aug['model_b'].astype(str).str.replace(r'_[ab]$', '', regex=True)
    aug['overlap'] = aug['model_b'].astype(str).str[-1]
    for repr_type in ['embedding_matrix', 'pre_lmhead_contextual']:
        for baseline in ['en_50m', 'en_100m']:
            sub = aug[(aug['repr_type'] == repr_type) & (aug['model_a'] == baseline)].copy()
            a = sub[sub['overlap'] == 'a'][['family', 'axis_abs_projection_diff_mean']].rename(columns={'axis_abs_projection_diff_mean': 'a'})
            b = sub[sub['overlap'] == 'b'][['family', 'axis_abs_projection_diff_mean']].rename(columns={'axis_abs_projection_diff_mean': 'b'})
            m = a.merge(b, on='family', how='inner')
            if m.empty:
                continue
            diff = m['b'].to_numpy() - m['a'].to_numpy()
            stat, p = safe_wilcoxon(diff)
            rows.append({
                'test': 'disjoint_gt_overlap_axis',
                'slice': f'{baseline}:{repr_type}',
                'n': int(len(diff)),
                'mean_difference': float(np.mean(diff)),
                'statistic': stat,
                'p_value_greater': p,
            })

    behavioral = pd.read_csv(args.behavioral_csv)
    for baseline in ['en_50m', 'en_100m', 'all']:
        sub = behavioral if baseline == 'all' else behavioral[behavioral['baseline'] == baseline]
        valid = sub[sub['condition'] == 'theory'].dropna(subset=['sign_agree_contextual', 'sign_agree_embedding']).copy()
        if valid.empty:
            continue
        ctx_correct = valid['sign_agree_contextual'].astype(int)
        emb_correct = valid['sign_agree_embedding'].astype(int)
        n10 = int(((ctx_correct == 1) & (emb_correct == 0)).sum())
        n01 = int(((ctx_correct == 0) & (emb_correct == 1)).sum())
        exact = binomtest(n10, n=n10 + n01, p=0.5, alternative='greater') if (n10 + n01) > 0 else None
        rows.append({
            'test': 'behavioral_sign_agreement_ctx_gt_emb',
            'slice': baseline,
            'n': int(len(valid)),
            'mean_difference': float(ctx_correct.mean() - emb_correct.mean()),
            'statistic': float(n10),
            'p_value_greater': float(exact.pvalue) if exact is not None else float('nan'),
        })
        # Theory vs random specificity in the behavioral readout.
        theory = sub[sub['condition'] == 'theory'][['language', 'probe', 'baseline', 'sign_agree_contextual']].rename(columns={'sign_agree_contextual': 'theory_ctx'})
        rand = sub[sub['condition'] == 'random'][['language', 'probe', 'baseline', 'sign_agree_contextual']].rename(columns={'sign_agree_contextual': 'random_ctx'})
        merged = theory.merge(rand, on=['language', 'probe', 'baseline'], how='inner')
        if not merged.empty:
            diff = merged['theory_ctx'].astype(float) - merged['random_ctx'].astype(float)
            stat, p = safe_wilcoxon(diff.to_numpy())
            rows.append({
                'test': 'behavioral_specificity_theory_gt_random',
                'slice': baseline,
                'n': int(len(merged)),
                'mean_difference': float(np.mean(diff)),
                'statistic': stat,
                'p_value_greater': p,
            })

    specificity = pd.read_csv(args.specificity_csv)
    spec_pivot = specificity.pivot_table(
        index=['baseline', 'language', 'repr_type', 'draw'],
        columns='condition',
        values=['axis_abs_projection_diff_mean', 'axis_sign_coherence'],
        aggfunc='first',
    )
    spec_pivot.columns = ['__'.join(col).strip() for col in spec_pivot.columns.to_flat_index()]
    spec_pivot = spec_pivot.reset_index()
    for baseline in ['en_50m', 'en_100m', 'all']:
        sub = spec_pivot if baseline == 'all' else spec_pivot[spec_pivot['baseline'] == baseline]
        for repr_type in ['embedding_matrix', 'pre_lmhead_contextual']:
            sub_rt = sub[sub['repr_type'] == repr_type].copy()
            if sub_rt.empty:
                continue
            if 'axis_sign_coherence__cultural_theory' in sub_rt and 'axis_sign_coherence__negative_theory' in sub_rt:
                diff = sub_rt['axis_sign_coherence__cultural_theory'] - sub_rt['axis_sign_coherence__negative_theory']
                stat, p = safe_wilcoxon(diff.to_numpy())
                rows.append({
                    'test': 'specificity_coherence_cultural_gt_negative',
                    'slice': f'{baseline}:{repr_type}',
                    'n': int(len(diff)),
                    'mean_difference': float(np.mean(diff)),
                    'statistic': stat,
                    'p_value_greater': p,
                })
            rand = sub_rt.groupby(['baseline', 'language', 'repr_type'])['axis_sign_coherence__cultural_random'].mean().reset_index()
            theory = sub_rt.groupby(['baseline', 'language', 'repr_type'])['axis_sign_coherence__cultural_theory'].first().reset_index()
            merged = theory.merge(rand, on=['baseline', 'language', 'repr_type'], how='inner')
            if not merged.empty:
                diff = merged['axis_sign_coherence__cultural_theory'] - merged['axis_sign_coherence__cultural_random']
                stat, p = safe_wilcoxon(diff.to_numpy())
                rows.append({
                    'test': 'specificity_coherence_cultural_gt_random',
                    'slice': f'{baseline}:{repr_type}',
                    'n': int(len(diff)),
                    'mean_difference': float(np.mean(diff)),
                    'statistic': stat,
                    'p_value_greater': p,
                })

    anchor = pd.read_csv(args.anchor_csv)
    for baseline in ['en_50m', 'en_100m', 'all']:
        sub = anchor if baseline == 'all' else anchor[anchor['baseline'] == baseline]
        pivot = sub.pivot_table(
            index=['baseline', 'language', 'subset_size', 'draw'],
            columns='repr_type',
            values='axis_abs_projection_diff_mean',
            aggfunc='first',
        ).reset_index()
        if {'embedding_matrix', 'pre_lmhead_contextual'}.issubset(set(pivot.columns)):
            diff = pivot['pre_lmhead_contextual'] - pivot['embedding_matrix']
            stat, p = safe_wilcoxon(diff.to_numpy())
            rows.append({
                'test': 'anchor_sensitivity_ctx_gt_emb',
                'slice': baseline,
                'n': int(len(diff)),
                'mean_difference': float(np.mean(diff)),
                'statistic': stat,
                'p_value_greater': p,
            })

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f'Wrote: {args.out_csv}')


if __name__ == '__main__':
    main()
