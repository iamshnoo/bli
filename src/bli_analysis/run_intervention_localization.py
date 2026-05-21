#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from shared_utils import (
    DEFAULT_PROMPTS,
    align_source_to_target,
    extract_layer_word_reprs,
    load_probe_set,
    load_tokenizer_and_model,
    pick_word_span,
    preferred_torch_dtype,
    resolve_device,
)

CORE_LANGS = ['zh', 'fr', 'fas', 'nld', 'ukr', 'bul', 'ind', 'deu']
TEMPLATES = [
    'If one had to choose between "{left}" and "{right}" for "{probe}", the closer association is',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run aligned intervention and causal localization analyses.')
    p.add_argument('--probe-set', type=Path, default=Path('data/probes/probe_sets.json'))
    p.add_argument('--cases-csv', type=Path, default=Path('latex/tables/fig5_signflip_hotspots.csv'))
    p.add_argument('--layerwise-csv', type=Path, default=Path('outputs/revision/en_ablation/bli_layerwise_divergence.csv'))
    p.add_argument('--cache-dir', type=Path, default=Path('outputs/revision/scope_expansion/layerwise_repr'))
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--localization-out', type=Path, required=True)
    p.add_argument('--localization-summary-out', type=Path, required=True)
    p.add_argument('--intervention-out', type=Path, required=True)
    p.add_argument('--intervention-summary-out', type=Path, required=True)
    return p.parse_args()


def parse_cases(path: Path) -> list[tuple[str, str, str]]:
    df = pd.read_csv(path)
    out = []
    for raw in df['probe_axis'].astype(str):
        probe, axis = [x.strip() for x in raw.split('|', 1)]
        left, right = [x.strip() for x in axis.split('->', 1)]
        out.append((probe, left, right))
    return out


def pick_intervention_layer(layerwise_csv: Path) -> int:
    df = pd.read_csv(layerwise_csv)
    sub = df[df['model_a'] == 'en_50m'].copy()
    peaks = sub.groupby('model_b')['axis_abs_projection_diff_mean'].idxmax()
    peak_layers = sub.loc[peaks, 'layer'].astype(int).tolist()
    # Stored layers include embedding output at 0; use decoder layer index in [0, 11].
    hidden_idx = int(np.median(peak_layers))
    return max(0, min(hidden_idx - 1, 11))


def load_or_build_layerwise_cache(model_name: str, model_path: Path, words: list[str], cache_dir: Path, batch_size: int, device: torch.device) -> dict[int, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{model_name}.npz'
    if cache_path.exists():
        z = np.load(cache_path)
        return {int(k.replace('layer_', '')): z[k].astype(np.float32) for k in z.files}

    tok, model = load_tokenizer_and_model(model_path, device)
    try:
        mats = extract_layer_word_reprs(model, tok, words, DEFAULT_PROMPTS, batch_size, device)
    finally:
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    save_payload = {f'layer_{k}': v.astype(np.float16) for k, v in mats.items()}
    np.savez_compressed(cache_path, **save_payload)
    return mats


def score_completion_with_hook(model, tokenizer, prompt: str, completion: str, device: torch.device, hook_layer: int, probe_word: str, edit_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)['input_ids']
    if len(full_ids) <= len(prompt_ids):
        return float('nan')
    span = pick_word_span(tokenizer, probe_word, full_ids)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        s, e = span
        hidden[:, s:e, :] = edit_fn(hidden[:, s:e, :])
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    handle = model.model.layers[hook_layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        count = 0
        for pos in range(len(prompt_ids), len(full_ids)):
            total += float(log_probs[pos - 1, full_ids[pos]].item())
            count += 1
        return total / max(1, count)
    finally:
        handle.remove()


def collect_hidden_states(model, tokenizer, prompt: str, completion: str, device: torch.device) -> tuple[list[np.ndarray], list[int]]:
    full_text = prompt + completion
    enc = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
    ids = enc['input_ids'].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = [h.detach().float().cpu().numpy()[0] for h in out.hidden_states]
    return hs, enc['input_ids'][0].cpu().tolist()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    probe = load_probe_set(args.probe_set)
    words = probe['_words']
    w2i = probe['_w2i']
    cases = parse_cases(args.cases_csv)
    chosen_layer = pick_intervention_layer(args.layerwise_csv)

    model_names = ['en_50m'] + [f'en_{lang}_a' for lang in CORE_LANGS]
    layer_cache = {
        name: load_or_build_layerwise_cache(name, Path('models/hf') / name, words, args.cache_dir, args.batch_size, device)
        for name in model_names
    }

    maps_by_lang: dict[str, dict[int, np.ndarray]] = {}
    residue_by_lang: dict[str, dict[str, np.ndarray]] = {}
    for lang in CORE_LANGS:
        bi_name = f'en_{lang}_a'
        maps_by_lang[lang] = {}
        residue_by_lang[lang] = {}
        for layer_idx in range(1, 13):
            aligned_en, W, _ = align_source_to_target(layer_cache['en_50m'][layer_idx], layer_cache[bi_name][layer_idx], probe['_neutral_idx'])
            maps_by_lang[lang][layer_idx - 1] = W.astype(np.float32)
            if layer_idx - 1 == chosen_layer:
                for probe_word, _left, _right in cases:
                    if probe_word not in w2i:
                        continue
                    pidx = w2i[probe_word]
                    residue_by_lang[lang][probe_word] = (layer_cache[bi_name][layer_idx][pidx] - aligned_en[pidx]).astype(np.float32)

    localization_rows = []
    intervention_rows = []

    en_tok, en_model = load_tokenizer_and_model(Path('models/hf/en_50m'), device)
    try:
        for lang in CORE_LANGS:
            bi_name = f'en_{lang}_a'
            bi_tok, bi_model = load_tokenizer_and_model(Path(f'models/hf/{bi_name}'), device)
            try:
                for probe_word, left, right in cases:
                    if probe_word not in w2i:
                        continue
                    loo_residues = [v for k, v in residue_by_lang[lang].items() if k != probe_word]
                    if loo_residues:
                        residue_dir = np.mean(np.stack(loo_residues, axis=0), axis=0).astype(np.float32)
                    else:
                        residue_dir = np.zeros((layer_cache[bi_name][chosen_layer + 1].shape[1],), dtype=np.float32)
                    residue_t = torch.tensor(residue_dir, dtype=preferred_torch_dtype(device), device=device).view(1, 1, -1)

                    per_template_base = []
                    per_template_intervene = []
                    per_layer_rescue: dict[int, list[float]] = {l: [] for l in range(12)}
                    for tmpl in TEMPLATES:
                        prompt = tmpl.format(probe=probe_word, left=left, right=right)
                        pref_en = score_completion_with_hook(en_model, en_tok, prompt, f' {left}', device, 0, probe_word, lambda x: x) - score_completion_with_hook(en_model, en_tok, prompt, f' {right}', device, 0, probe_word, lambda x: x)
                        # Baseline bilingual preference without edits.
                        from shared_utils import score_completion
                        pref_bi = score_completion(bi_model, bi_tok, prompt, f' {left}', device) - score_completion(bi_model, bi_tok, prompt, f' {right}', device)
                        base_delta = pref_bi - pref_en
                        per_template_base.append(base_delta)

                        def intervention_edit(hidden_slice: torch.Tensor) -> torch.Tensor:
                            return hidden_slice - (args.alpha * residue_t)

                        pref_bi_int = score_completion_with_hook(bi_model, bi_tok, prompt, f' {left}', device, chosen_layer, probe_word, intervention_edit) - score_completion_with_hook(bi_model, bi_tok, prompt, f' {right}', device, chosen_layer, probe_word, intervention_edit)
                        int_delta = pref_bi_int - pref_en
                        per_template_intervene.append(int_delta)

                        # Layer-wise aligned patching.
                        hs_left, full_ids_left = collect_hidden_states(en_model, en_tok, prompt, f' {left}', device)
                        hs_right, full_ids_right = collect_hidden_states(en_model, en_tok, prompt, f' {right}', device)
                        span_left = pick_word_span(en_tok, probe_word, full_ids_left)
                        span_right = pick_word_span(en_tok, probe_word, full_ids_right)
                        for hook_layer in range(12):
                            W = maps_by_lang[lang][hook_layer]
                            W_t = torch.tensor(W, dtype=preferred_torch_dtype(device), device=device)
                            en_patch_left = torch.tensor(hs_left[hook_layer + 1][span_left[0]:span_left[1], :], dtype=preferred_torch_dtype(device), device=device) @ W_t
                            en_patch_right = torch.tensor(hs_right[hook_layer + 1][span_right[0]:span_right[1], :], dtype=preferred_torch_dtype(device), device=device) @ W_t

                            def patch_left(_hidden_slice: torch.Tensor, patch=en_patch_left) -> torch.Tensor:
                                out = _hidden_slice.clone()
                                n = min(out.shape[1], patch.shape[0])
                                out[:, :n, :] = patch[:n, :].unsqueeze(0)
                                return out

                            def patch_right(_hidden_slice: torch.Tensor, patch=en_patch_right) -> torch.Tensor:
                                out = _hidden_slice.clone()
                                n = min(out.shape[1], patch.shape[0])
                                out[:, :n, :] = patch[:n, :].unsqueeze(0)
                                return out

                            pref_bi_patch = score_completion_with_hook(bi_model, bi_tok, prompt, f' {left}', device, hook_layer, probe_word, patch_left) - score_completion_with_hook(bi_model, bi_tok, prompt, f' {right}', device, hook_layer, probe_word, patch_right)
                            patched_delta = pref_bi_patch - pref_en
                            denom = abs(base_delta) if abs(base_delta) > 1e-9 else np.nan
                            rescue = (abs(base_delta) - abs(patched_delta)) / denom if np.isfinite(denom) else np.nan
                            per_layer_rescue[hook_layer].append(float(rescue))

                    baseline_delta = float(np.mean(per_template_base))
                    intervened_delta = float(np.mean(per_template_intervene))
                    denom = abs(baseline_delta) if abs(baseline_delta) > 1e-9 else np.nan
                    rescue = (abs(baseline_delta) - abs(intervened_delta)) / denom if np.isfinite(denom) else np.nan
                    intervention_rows.append({
                        'language': lang.upper(),
                        'probe': probe_word,
                        'left_endpoint': left,
                        'right_endpoint': right,
                        'layer': int(chosen_layer + 1),
                        'alpha': float(args.alpha),
                        'baseline_output_shift': baseline_delta,
                        'intervened_output_shift': intervened_delta,
                        'rescue_fraction': rescue,
                    })
                    for hook_layer, vals in per_layer_rescue.items():
                        localization_rows.append({
                            'language': lang.upper(),
                            'probe': probe_word,
                            'left_endpoint': left,
                            'right_endpoint': right,
                            'layer': int(hook_layer + 1),
                            'baseline_output_shift': baseline_delta,
                            'mean_patched_rescue_fraction': float(np.mean(vals)),
                        })
                if localization_rows:
                    args.localization_out.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(localization_rows).sort_values(['layer', 'language', 'probe']).to_csv(args.localization_out, index=False)
                if intervention_rows:
                    args.intervention_out.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(intervention_rows).sort_values(['language', 'probe']).to_csv(args.intervention_out, index=False)
            finally:
                del bi_model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    finally:
        del en_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    loc_df = pd.DataFrame(localization_rows).sort_values(['layer', 'language', 'probe']).reset_index(drop=True)
    int_df = pd.DataFrame(intervention_rows).sort_values(['language', 'probe']).reset_index(drop=True)
    args.localization_out.parent.mkdir(parents=True, exist_ok=True)
    loc_df.to_csv(args.localization_out, index=False)
    int_df.to_csv(args.intervention_out, index=False)

    loc_summary = loc_df.groupby('layer', as_index=False)['mean_patched_rescue_fraction'].agg(['mean', 'median', 'std']).reset_index()
    loc_summary.to_csv(args.localization_summary_out, index=False)
    int_summary = pd.DataFrame([
        {
            'layer': int(chosen_layer + 1),
            'alpha': float(args.alpha),
            'mean_rescue_fraction': float(int_df['rescue_fraction'].mean()),
            'median_rescue_fraction': float(int_df['rescue_fraction'].median()),
            'std_rescue_fraction': float(int_df['rescue_fraction'].std()),
            'n_cases': int(len(int_df)),
        }
    ])
    int_summary.to_csv(args.intervention_summary_out, index=False)
    print(f'Wrote: {args.localization_out}')
    print(f'Wrote: {args.localization_summary_out}')
    print(f'Wrote: {args.intervention_out}')
    print(f'Wrote: {args.intervention_summary_out}')


if __name__ == '__main__':
    main()
