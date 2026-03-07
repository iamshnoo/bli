#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-head proxy analysis: compute axis divergence on head-width channel slices for selected layers."
    )
    p.add_argument("--models-json", type=Path, required=True, help="JSON mapping model name -> HF path")
    p.add_argument(
        "--probe-set",
        type=Path,
        default=Path("data/probes/probe_sets.json"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/revision/en_ablation/bli_perhead_analysis.csv"),
    )
    p.add_argument("--layers", type=str, default="4,5,6", help="Comma-separated layer indices")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_subseq(seq: list[int], sub: list[int]) -> tuple[int, int] | None:
    m = len(sub)
    if m == 0 or m > len(seq):
        return None
    for i in range(len(seq) - m + 1):
        if seq[i : i + m] == sub:
            return i, i + m
    return None


def word_candidates(tok, w: str) -> list[list[int]]:
    out: list[list[int]] = []
    for s in [" " + w, w]:
        ids = tok.encode(s, add_special_tokens=False)
        if ids:
            out.append(ids)
    dedup: list[list[int]] = []
    seen = set()
    for c in out:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            dedup.append(c)
    return dedup


def pick_span(tok, w: str, seq: list[int]) -> tuple[int, int]:
    cands = word_candidates(tok, w)
    hits = []
    for c in cands:
        h = find_subseq(seq, c)
        if h is not None:
            hits.append(h)
    if hits:
        return max(hits, key=lambda z: z[1] - z[0])
    if len(seq) >= 2:
        return len(seq) - 2, len(seq) - 1
    return 0, 1


def extract_selected_layers(
    model,
    tok,
    words: list[str],
    layers: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    prompts = ["The word {word} appears in a neutral sentence."]
    dim = model.config.hidden_size
    collector: dict[int, dict[str, list[np.ndarray]]] = {l: {w: [] for w in words} for l in layers}

    model.eval()
    for tmpl in prompts:
        for st in tqdm(range(0, len(words), batch_size), desc=f"extract[{tmpl}]"):
            batch_words = words[st : st + batch_size]
            texts = [tmpl.format(word=w) for w in batch_words]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hs = out.hidden_states

            ids_cpu = enc["input_ids"].cpu().numpy()
            mask_cpu = enc["attention_mask"].cpu().numpy()
            for i, w in enumerate(batch_words):
                valid = int(mask_cpu[i].sum())
                seq = ids_cpu[i][:valid].tolist()
                s, e = pick_span(tok, w, seq)
                s = max(0, min(s, valid - 1))
                e = max(s + 1, min(e, valid))
                for li in layers:
                    vec = hs[li][i, s:e].mean(dim=0).detach().float().cpu().numpy()
                    collector[li][w].append(vec)

    out_mats: dict[int, np.ndarray] = {}
    for li in layers:
        mat = np.zeros((len(words), dim), dtype=np.float32)
        for wi, w in enumerate(words):
            vals = collector[li][w]
            mat[wi] = np.mean(vals, axis=0) if vals else np.zeros((dim,), dtype=np.float32)
        out_mats[li] = mat
    return out_mats


def axis_metric(
    Xa: np.ndarray,
    Xb: np.ndarray,
    neutral_idx: np.ndarray,
    cultural_idx: np.ndarray,
    axis_idx: list[tuple[int, int]],
) -> float:
    w, _ = orthogonal_procrustes(Xa[neutral_idx], Xb[neutral_idx])
    A = Xa @ w
    vals = []
    for i, j in axis_idx:
        va = A[j] - A[i]
        vb = Xb[j] - Xb[i]
        nva = np.linalg.norm(va)
        nvb = np.linalg.norm(vb)
        if nva < 1e-12 or nvb < 1e-12:
            continue
        va = va / nva
        vb = vb / nvb
        pa = A[cultural_idx] @ va
        pb = Xb[cultural_idx] @ vb
        vals.append(float(np.mean(np.abs(pa - pb))))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if not layers:
        raise ValueError("No layers provided")

    models = json.loads(args.models_json.read_text(encoding="utf-8"))
    probes = json.loads(args.probe_set.read_text(encoding="utf-8"))
    words: list[str] = probes["all_probe_words"]
    w2i = {w: i for i, w in enumerate(words)}
    neutral_idx = np.array([w2i[w] for w in probes["neutral_anchor_words"] if w in w2i], dtype=np.int64)
    cultural_idx = np.array([w2i[w] for w in probes["cultural_probe_words"] if w in w2i], dtype=np.int64)
    axis_idx = [(w2i[a], w2i[b]) for a, b in probes["semantic_axes"] if a in w2i and b in w2i]

    baselines = [m for m in ["en_50m", "en_100m"] if m in models]
    targets = sorted(
        m for m in models if m.startswith("en_") and m.endswith("_a") and m not in {"en_50m", "en_100m"}
    )
    pairs = [(base, tgt) for base in baselines for tgt in targets]
    if not pairs:
        raise ValueError("No EN-centered A-setup pairs found in --models-json")

    selected_names = sorted(set([m for pair in pairs for m in pair]))
    model_paths = {k: v for k, v in models.items() if k in selected_names}

    per_model_layers: dict[str, dict[int, np.ndarray]] = {}
    n_heads = None
    head_dim = None

    for name, path in model_paths.items():
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)

        if n_heads is None:
            n_heads = int(model.config.num_attention_heads)
            head_dim = int(model.config.hidden_size // model.config.num_attention_heads)

        per_model_layers[name] = extract_selected_layers(
            model=model,
            tok=tok,
            words=words,
            layers=layers,
            batch_size=args.batch_size,
            device=device,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert n_heads is not None and head_dim is not None

    rows = []
    for ma, mb in pairs:
        for li in layers:
            A = per_model_layers[ma][li]
            B = per_model_layers[mb][li]
            for h in range(n_heads):
                s = h * head_dim
                e = (h + 1) * head_dim
                da = axis_metric(
                    Xa=A[:, s:e],
                    Xb=B[:, s:e],
                    neutral_idx=neutral_idx,
                    cultural_idx=cultural_idx,
                    axis_idx=axis_idx,
                )
                rows.append(
                    {
                        "model_a": ma,
                        "model_b": mb,
                        "layer": li,
                        "head": h,
                        "axis_abs_projection_diff_mean": da,
                    }
                )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
