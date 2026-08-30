#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "The word {word} appears in a neutral sentence.",
    "In general usage, {word} is a term.",
    "People may mention {word} in writing.",
]

DEFAULT_NORM_METHODS = ("row_l2", "neutral_z")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run norm-controlled and z-scored D_axis checks on cached and layerwise representations."
    )
    p.add_argument("--probe-set", type=Path, default=Path("data/probes/probe_sets.json"))
    p.add_argument("--rep-dir", type=Path, default=Path("outputs/revision/en_ablation/representations"))
    p.add_argument("--models-json", type=Path, default=Path("configs/models/models_en_ablation.json"))
    p.add_argument("--layer-models-json", type=Path, default=Path("configs/models/models_layerwise.json"))
    p.add_argument("--out-csv", type=Path, default=Path("outputs/revision/en_ablation/bli_norm_controlled_axis.csv"))
    p.add_argument(
        "--layer-out-csv",
        type=Path,
        default=Path("outputs/revision/en_ablation/bli_norm_controlled_layerwise.csv"),
    )
    p.add_argument(
        "--layer-cache-dir",
        type=Path,
        default=Path("outputs/revision/en_ablation/layerwise_representations"),
    )
    p.add_argument(
        "--pair",
        action="append",
        default=[],
        help='Final-representation pair spec in "model_a,model_b" format. Defaults to EN baselines vs EN+L2 variants.',
    )
    p.add_argument(
        "--layer-pair",
        action="append",
        default=[],
        help='Layerwise pair spec in "model_a,model_b" format. Defaults to EN baselines vs shared-doc EN+L2 variants.',
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--normalization",
        action="append",
        choices=["row_l2", "neutral_z", "raw"],
        default=[],
        help="Normalization/control to run. Defaults to row_l2 and neutral_z.",
    )
    p.add_argument("--skip-layerwise", action="store_true")
    p.add_argument("--force-layer-extract", action="store_true")
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), eps)


def fit_orthogonal(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _, vt = svd(source.T @ target, full_matrices=False, check_finite=False)
    return u @ vt


def find_subsequence(seq: Sequence[int], sub: Sequence[int]) -> tuple[int, int] | None:
    m = len(sub)
    if m == 0 or m > len(seq):
        return None
    for i in range(len(seq) - m + 1):
        if list(seq[i : i + m]) == list(sub):
            return i, i + m
    return None


def word_token_candidates(tokenizer, word: str) -> list[list[int]]:
    cands: list[list[int]] = []
    for text in (" " + word, word):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            cands.append(ids)
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for cand in cands:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def pick_word_span(tokenizer, word: str, seq_ids: Sequence[int]) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for cand in word_token_candidates(tokenizer, word):
        found = find_subsequence(seq_ids, cand)
        if found is not None:
            matches.append(found)
    if matches:
        return max(matches, key=lambda x: x[1] - x[0])
    if len(seq_ids) >= 2:
        return len(seq_ids) - 2, len(seq_ids) - 1
    return 0, 1


def load_probe_indices(probe_set: Path) -> tuple[list[str], np.ndarray, np.ndarray, list[tuple[int, int]]]:
    probes = json.loads(probe_set.read_text(encoding="utf-8"))
    words: list[str] = probes["all_probe_words"]
    w2i = {w: i for i, w in enumerate(words)}
    neutral_idx = np.array([w2i[w] for w in probes["neutral_anchor_words"] if w in w2i], dtype=np.int64)
    cultural_idx = np.array([w2i[w] for w in probes["cultural_probe_words"] if w in w2i], dtype=np.int64)
    axis_idx = [(w2i[a], w2i[b]) for a, b in probes["semantic_axes"] if a in w2i and b in w2i]
    return words, neutral_idx, cultural_idx, axis_idx


def parse_pair_specs(pair_args: list[str], model_keys: Sequence[str]) -> list[tuple[str, str]]:
    known = set(model_keys)
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for spec in pair_args:
        if "," not in spec:
            raise ValueError(f"Invalid pair spec '{spec}'. Expected 'model_a,model_b'.")
        left, right = [x.strip() for x in spec.split(",", 1)]
        if left not in known or right not in known:
            raise ValueError(f"Unknown model in pair '{spec}'. Available: {sorted(known)}")
        if left == right:
            raise ValueError(f"Invalid self-pair '{spec}'.")
        pair = (left, right)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def default_final_pairs(model_keys: Sequence[str]) -> list[tuple[str, str]]:
    baselines = [m for m in ("en_50m", "en_100m") if m in model_keys]
    targets = [m for m in model_keys if m.startswith("en_") and m not in set(baselines)]
    return [(base, target) for base in baselines for target in targets]


def default_layer_pairs(model_keys: Sequence[str]) -> list[tuple[str, str]]:
    baselines = [m for m in ("en_50m", "en_100m") if m in model_keys]
    targets = [m for m in model_keys if m.startswith("en_") and m.endswith("_a") and m not in set(baselines)]
    return [(base, target) for base in baselines for target in targets]


def axis_metric_rows(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    neutral_idx: np.ndarray,
    cultural_idx: np.ndarray,
    axis_idx: list[tuple[int, int]],
    normalization: str,
) -> dict[str, float | int]:
    if normalization == "row_l2":
        source = l2_normalize(mat_a)
        target = l2_normalize(mat_b)
    else:
        source = mat_a
        target = mat_b

    w = fit_orthogonal(source[neutral_idx], target[neutral_idx])
    aligned = source @ w

    abs_vals: list[float] = []
    signed_vals: list[float] = []
    max_vals: list[float] = []
    neutral_sd_a: list[float] = []
    neutral_sd_b: list[float] = []
    skipped = 0

    for i, j in axis_idx:
        axis_a = aligned[j] - aligned[i]
        axis_b = target[j] - target[i]
        norm_a = float(np.linalg.norm(axis_a))
        norm_b = float(np.linalg.norm(axis_b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            skipped += 1
            continue
        axis_a = axis_a / norm_a
        axis_b = axis_b / norm_b

        proj_a = aligned[cultural_idx] @ axis_a
        proj_b = target[cultural_idx] @ axis_b

        if normalization == "neutral_z":
            neutral_a = aligned[neutral_idx] @ axis_a
            neutral_b = target[neutral_idx] @ axis_b
            mean_a = float(np.mean(neutral_a))
            mean_b = float(np.mean(neutral_b))
            sd_a = float(np.std(neutral_a))
            sd_b = float(np.std(neutral_b))
            if sd_a < 1e-12 or sd_b < 1e-12:
                skipped += 1
                continue
            proj_a = (proj_a - mean_a) / sd_a
            proj_b = (proj_b - mean_b) / sd_b
            neutral_sd_a.append(sd_a)
            neutral_sd_b.append(sd_b)

        diff = proj_a - proj_b
        abs_diff = np.abs(diff)
        abs_vals.append(float(np.mean(abs_diff)))
        signed_vals.append(float(np.mean(diff)))
        max_vals.append(float(np.max(abs_diff)))

    return {
        "axis_abs_projection_diff_mean": float(np.mean(abs_vals)) if abs_vals else float("nan"),
        "axis_abs_projection_diff_max": float(np.max(max_vals)) if max_vals else float("nan"),
        "axis_signed_projection_diff_mean": float(np.mean(signed_vals)) if signed_vals else float("nan"),
        "axes_used": len(abs_vals),
        "axes_skipped": skipped,
        "neutral_projection_sd_a_mean": float(np.mean(neutral_sd_a)) if neutral_sd_a else float("nan"),
        "neutral_projection_sd_b_mean": float(np.mean(neutral_sd_b)) if neutral_sd_b else float("nan"),
        "cultural_norm_a_mean": float(np.mean(np.linalg.norm(aligned[cultural_idx], axis=1))),
        "cultural_norm_b_mean": float(np.mean(np.linalg.norm(target[cultural_idx], axis=1))),
    }


def load_cached_repr(rep_dir: Path, model: str, repr_type: str) -> np.ndarray:
    path = rep_dir / f"{model}__{repr_type}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached representation: {path}")
    return np.load(path)


def run_final_metrics(
    rep_dir: Path,
    pairs: list[tuple[str, str]],
    neutral_idx: np.ndarray,
    cultural_idx: np.ndarray,
    axis_idx: list[tuple[int, int]],
    norm_methods: Sequence[str],
) -> pd.DataFrame:
    cache: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for repr_type in ("embedding_matrix", "pre_lmhead_contextual"):
        for model_a, model_b in pairs:
            print(f"final {repr_type}: {model_a} vs {model_b}", flush=True)
            for model in (model_a, model_b):
                key = (model, repr_type)
                if key not in cache:
                    cache[key] = load_cached_repr(rep_dir, model, repr_type)
            mat_a = cache[(model_a, repr_type)]
            mat_b = cache[(model_b, repr_type)]
            for normalization in norm_methods:
                metrics = axis_metric_rows(mat_a, mat_b, neutral_idx, cultural_idx, axis_idx, normalization)
                rows.append(
                    {
                        "scope": "final",
                        "repr_type": repr_type,
                        "model_a": model_a,
                        "model_b": model_b,
                        "normalization": normalization,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def load_tokenizer_and_model(model_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=preferred_dtype(device)).to(device)
    model.eval()
    return tok, model


def extract_layer_word_reprs(
    model,
    tokenizer,
    words: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    n_layers = model.config.num_hidden_layers + 1
    dim = model.config.hidden_size
    per_layer_word: dict[int, dict[str, list[np.ndarray]]] = {layer: {w: [] for w in words} for layer in range(n_layers)}

    for prompt in PROMPTS:
        print(f"  extracting prompt: {prompt}", flush=True)
        for start in range(0, len(words), batch_size):
            batch_words = words[start : start + batch_size]
            enc = tokenizer(
                [prompt.format(word=w) for w in batch_words],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
                hidden_states = [h.detach().float().cpu().numpy() for h in out.hidden_states]

            ids_cpu = enc["input_ids"].cpu().numpy()
            mask_cpu = enc["attention_mask"].cpu().numpy()
            for batch_idx, word in enumerate(batch_words):
                valid_len = int(mask_cpu[batch_idx].sum())
                seq_ids = ids_cpu[batch_idx][:valid_len].tolist()
                left, right = pick_word_span(tokenizer, word, seq_ids)
                left = max(0, min(left, valid_len - 1))
                right = max(left + 1, min(right, valid_len))
                for layer, hidden in enumerate(hidden_states):
                    per_layer_word[layer][word].append(hidden[batch_idx, left:right].mean(axis=0))

    out_mats: dict[int, np.ndarray] = {}
    for layer in range(n_layers):
        mat = np.zeros((len(words), dim), dtype=np.float32)
        for word_idx, word in enumerate(words):
            values = per_layer_word[layer][word]
            mat[word_idx] = np.mean(values, axis=0) if values else np.zeros((dim,), dtype=np.float32)
        out_mats[layer] = mat
    return out_mats


def save_layer_cache(cache_path: Path, layer_reprs: dict[int, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{f"layer_{layer}": mat for layer, mat in layer_reprs.items()})


def load_layer_cache(cache_path: Path) -> dict[int, np.ndarray]:
    data = np.load(cache_path)
    out: dict[int, np.ndarray] = {}
    for key in data.files:
        if key.startswith("layer_"):
            out[int(key.split("_", 1)[1])] = data[key]
    return out


def load_or_extract_layer_reprs(
    name: str,
    model_path: str,
    words: list[str],
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
    force: bool,
) -> dict[int, np.ndarray]:
    cache_path = cache_dir / f"{name}__layerwise.npz"
    if cache_path.exists() and not force:
        print(f"loading layer cache: {cache_path}", flush=True)
        return load_layer_cache(cache_path)
    print(f"extracting layer cache: {name} ({model_path})", flush=True)
    tok, model = load_tokenizer_and_model(model_path, device)
    layer_reprs = extract_layer_word_reprs(model, tok, words, batch_size, device)
    save_layer_cache(cache_path, layer_reprs)
    print(f"wrote layer cache: {cache_path}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return layer_reprs


def run_layerwise_metrics(
    models: dict[str, str],
    pairs: list[tuple[str, str]],
    words: list[str],
    neutral_idx: np.ndarray,
    cultural_idx: np.ndarray,
    axis_idx: list[tuple[int, int]],
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
    force: bool,
    norm_methods: Sequence[str],
) -> pd.DataFrame:
    needed_models = sorted(set(itertools.chain.from_iterable(pairs)))
    layer_reprs = {
        name: load_or_extract_layer_reprs(name, models[name], words, cache_dir, batch_size, device, force)
        for name in needed_models
    }

    rows: list[dict[str, object]] = []
    for model_a, model_b in pairs:
        layers = sorted(set(layer_reprs[model_a]) & set(layer_reprs[model_b]))
        for layer in layers:
            print(f"layer {layer}: {model_a} vs {model_b}", flush=True)
            mat_a = layer_reprs[model_a][layer]
            mat_b = layer_reprs[model_b][layer]
            for normalization in norm_methods:
                metrics = axis_metric_rows(mat_a, mat_b, neutral_idx, cultural_idx, axis_idx, normalization)
                rows.append(
                    {
                        "scope": "layerwise",
                        "repr_type": "hidden_state",
                        "model_a": model_a,
                        "model_b": model_b,
                        "layer": layer,
                        "normalization": normalization,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    words, neutral_idx, cultural_idx, axis_idx = load_probe_indices(args.probe_set)
    norm_methods = args.normalization if args.normalization else list(DEFAULT_NORM_METHODS)

    final_models = json.loads(args.models_json.read_text(encoding="utf-8"))
    final_pairs = (
        parse_pair_specs(args.pair, list(final_models.keys()))
        if args.pair
        else default_final_pairs(list(final_models.keys()))
    )
    final_df = run_final_metrics(args.rep_dir, final_pairs, neutral_idx, cultural_idx, axis_idx, norm_methods)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")

    if args.skip_layerwise:
        return

    layer_models = json.loads(args.layer_models_json.read_text(encoding="utf-8"))
    layer_pairs = (
        parse_pair_specs(args.layer_pair, list(layer_models.keys()))
        if args.layer_pair
        else default_layer_pairs(list(layer_models.keys()))
    )
    device = resolve_device(args.device)
    layer_df = run_layerwise_metrics(
        layer_models,
        layer_pairs,
        words,
        neutral_idx,
        cultural_idx,
        axis_idx,
        args.layer_cache_dir,
        args.batch_size,
        device,
        args.force_layer_extract,
        norm_methods,
    )
    args.layer_out_csv.parent.mkdir(parents=True, exist_ok=True)
    layer_df.to_csv(args.layer_out_csv, index=False)
    print(f"Wrote: {args.layer_out_csv}")


if __name__ == "__main__":
    main()
