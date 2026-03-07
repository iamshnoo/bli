#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-wise axis divergence analysis")
    p.add_argument("--models-json", type=Path, required=True, help="JSON mapping model_name -> HF path")
    p.add_argument("--probe-set", type=Path, default=Path("data/probes/probe_sets.json"))
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument(
        "--pair",
        action="append",
        default=[],
        help='Optional explicit pair spec in "model_a,model_b" format. Repeat for multiple pairs.',
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), eps)


def find_subseq(seq: list[int], sub: list[int]) -> tuple[int, int] | None:
    m = len(sub)
    if m == 0 or m > len(seq):
        return None
    for i in range(len(seq) - m + 1):
        if seq[i : i + m] == sub:
            return i, i + m
    return None


def word_candidates(tok, w: str) -> list[list[int]]:
    out = []
    for s in [" " + w, w]:
        ids = tok.encode(s, add_special_tokens=False)
        if ids:
            out.append(ids)
    # dedup
    dedup: list[list[int]] = []
    seen = set()
    for x in out:
        t = tuple(x)
        if t not in seen:
            seen.add(t)
            dedup.append(x)
    return dedup


def pick_span(tok, w: str, seq: list[int]) -> tuple[int, int]:
    cands = word_candidates(tok, w)
    m = []
    for c in cands:
        f = find_subseq(seq, c)
        if f is not None:
            m.append(f)
    if m:
        return max(m, key=lambda z: z[1] - z[0])
    if len(seq) >= 2:
        return len(seq) - 2, len(seq) - 1
    return 0, 1


def extract_layer_word_reprs(model, tok, words: list[str], prompts: list[str], batch_size: int, device: torch.device) -> dict[int, np.ndarray]:
    n_layers = model.config.num_hidden_layers + 1  # include embedding output layer 0
    dim = model.config.hidden_size
    per_layer_word: dict[int, dict[str, list[np.ndarray]]] = {
        l: {w: [] for w in words} for l in range(n_layers)
    }

    model.eval()
    for tmpl in prompts:
        for st in tqdm(range(0, len(words), batch_size), desc=f"extract[{tmpl}]"):
            bwords = words[st : st + batch_size]
            texts = [tmpl.format(word=w) for w in bwords]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
                hs = [h.detach().float().cpu().numpy() for h in out.hidden_states]

            ids_cpu = enc["input_ids"].cpu().numpy()
            m_cpu = enc["attention_mask"].cpu().numpy()
            for i, w in enumerate(bwords):
                valid = int(m_cpu[i].sum())
                seq = ids_cpu[i][:valid].tolist()
                s, e = pick_span(tok, w, seq)
                s = max(0, min(s, valid - 1))
                e = max(s + 1, min(e, valid))
                for li, h in enumerate(hs):
                    per_layer_word[li][w].append(h[i, s:e].mean(axis=0))

    out_mats: dict[int, np.ndarray] = {}
    for li in range(n_layers):
        mat = np.zeros((len(words), dim), dtype=np.float32)
        for wi, w in enumerate(words):
            vals = per_layer_word[li][w]
            mat[wi] = np.mean(vals, axis=0) if vals else np.zeros((dim,), dtype=np.float32)
        out_mats[li] = mat
    return out_mats


def axis_metric(Xa: np.ndarray, Xb: np.ndarray, nidx: np.ndarray, cidx: np.ndarray, aidx: list[tuple[int, int]]) -> float:
    W, _ = orthogonal_procrustes(Xa[nidx], Xb[nidx])
    A = Xa @ W
    vals = []
    for i, j in aidx:
        va = A[j] - A[i]
        vb = Xb[j] - Xb[i]
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            continue
        va = va / na
        vb = vb / nb
        pa = A[cidx] @ va
        pb = Xb[cidx] @ vb
        vals.append(float(np.mean(np.abs(pa - pb))))
    return float(np.mean(vals)) if vals else float("nan")


def resolve_pairs(pair_args: list[str], model_keys: list[str]) -> list[tuple[str, str]]:
    if not pair_args:
        return list(itertools.combinations(model_keys, 2))

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    known = set(model_keys)
    for spec in pair_args:
        if "," not in spec:
            raise ValueError(f"Invalid --pair '{spec}'. Expected 'model_a,model_b'")
        left, right = [x.strip() for x in spec.split(",", 1)]
        if left == right:
            raise ValueError(f"Invalid --pair '{spec}': model names must differ")
        if left not in known or right not in known:
            raise ValueError(f"Invalid --pair '{spec}': unknown model name (available: {model_keys})")
        pair = (left, right)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    models = json.loads(args.models_json.read_text(encoding="utf-8"))
    probes = json.loads(args.probe_set.read_text(encoding="utf-8"))
    words: list[str] = probes["all_probe_words"]
    w2i = {w: i for i, w in enumerate(words)}
    nidx = np.array([w2i[w] for w in probes["neutral_anchor_words"] if w in w2i], dtype=np.int64)
    cidx = np.array([w2i[w] for w in probes["cultural_probe_words"] if w in w2i], dtype=np.int64)
    aidx = [(w2i[a], w2i[b]) for a, b in probes["semantic_axes"] if a in w2i and b in w2i]

    prompts = [
        "The word {word} appears in a neutral sentence.",
        "In general usage, {word} is a term.",
        "People may mention {word} in writing.",
    ]

    reprs: dict[str, dict[int, np.ndarray]] = {}
    for name, path in models.items():
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)
        reprs[name] = extract_layer_word_reprs(model, tok, words, prompts, args.batch_size, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pairs = resolve_pairs(args.pair, list(models.keys()))

    rows = []
    for a, b in pairs:
        n_layers = min(len(reprs[a]), len(reprs[b]))
        for li in range(n_layers):
            d = axis_metric(reprs[a][li], reprs[b][li], nidx, cidx, aidx)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "layer": li,
                    "axis_abs_projection_diff_mean": d,
                }
            )

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
