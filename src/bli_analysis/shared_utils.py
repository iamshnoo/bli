#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "The word {word} appears in a neutral sentence.",
    "In general usage, {word} is a term.",
    "People may mention {word} in writing.",
]


def resolve_device(arg: str = "auto") -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preferred_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    # T4-class GPUs are safer on float16 than bfloat16.
    return torch.float16


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), eps)


def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    norm = l2_normalize(mat)
    return norm @ norm.T


def topk_neighbor_indices(norm_mat: np.ndarray, idx: int, k: int) -> np.ndarray:
    sims = norm_mat @ norm_mat[idx]
    sims[idx] = -1e9
    candidate = np.argpartition(-sims, k)[:k]
    candidate = candidate[np.argsort(-sims[candidate])]
    return candidate


def jaccard_divergence(a: set[int], b: set[int]) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - (len(a & b) / union)


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
    for s in [" " + word, word]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            cands.append(ids)
    dedup: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for c in cands:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            dedup.append(c)
    return dedup


def pick_word_span(tokenizer, word: str, seq_ids: Sequence[int]) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for c in word_token_candidates(tokenizer, word):
        found = find_subsequence(seq_ids, c)
        if found is not None:
            matches.append(found)
    if matches:
        return max(matches, key=lambda z: z[1] - z[0])
    if len(seq_ids) >= 2:
        return len(seq_ids) - 2, len(seq_ids) - 1
    return 0, 1


def load_probe_set(probe_set_path: Path) -> dict:
    probe = json.loads(probe_set_path.read_text(encoding="utf-8"))
    words = probe["all_probe_words"]
    w2i = {w: i for i, w in enumerate(words)}
    probe["_words"] = words
    probe["_w2i"] = w2i
    probe["_neutral_idx"] = [w2i[w] for w in probe["neutral_anchor_words"] if w in w2i]
    probe["_cultural_idx"] = [w2i[w] for w in probe["cultural_probe_words"] if w in w2i]
    probe["_negative_idx"] = [w2i[w] for w in probe.get("negative_control_words", []) if w in w2i]
    probe["_axis_idx"] = [(w2i[a], w2i[b]) for a, b in probe["semantic_axes"] if a in w2i and b in w2i]
    return probe


def load_tokenizer_and_model(model_path: str | Path, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=preferred_torch_dtype(device)).to(device)
    model.eval()
    return tok, model


def score_completion(model, tokenizer, prompt: str, completion: str, device: torch.device) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return float("nan")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    count = 0
    for pos in range(len(prompt_ids), len(full_ids)):
        token_id = full_ids[pos]
        total += float(log_probs[pos - 1, token_id].item())
        count += 1
    return total / max(1, count)


def align_source_to_target(mat_source: np.ndarray, mat_target: np.ndarray, neutral_idx: Sequence[int]) -> tuple[np.ndarray, np.ndarray, float]:
    w, _ = orthogonal_procrustes(mat_source[list(neutral_idx)], mat_target[list(neutral_idx)])
    aligned = mat_source @ w
    resid = aligned[list(neutral_idx)] - mat_target[list(neutral_idx)]
    resid_per = float(np.linalg.norm(resid, ord="fro") / max(1, len(neutral_idx)))
    return aligned, w, resid_per


def case_signed_shift(aligned_source: np.ndarray, target: np.ndarray, probe_idx: int, axis_left_idx: int, axis_right_idx: int) -> float:
    axis_source = aligned_source[axis_right_idx] - aligned_source[axis_left_idx]
    axis_target = target[axis_right_idx] - target[axis_left_idx]
    na = np.linalg.norm(axis_source)
    nb = np.linalg.norm(axis_target)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    axis_source = axis_source / na
    axis_target = axis_target / nb
    source_score = float(aligned_source[probe_idx] @ axis_source)
    target_score = float(target[probe_idx] @ axis_target)
    return source_score - target_score


def discover_representation_file(model_name: str, repr_type: str, rep_roots: Sequence[Path]) -> Path:
    for root in rep_roots:
        candidate = root / f"{model_name}__{repr_type}.npy"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find cached representation for {model_name} [{repr_type}] in {rep_roots}")


def load_repr(model_name: str, repr_type: str, rep_roots: Sequence[Path]) -> np.ndarray:
    return np.load(discover_representation_file(model_name, repr_type, rep_roots))


def infer_language_from_model(model_name: str) -> str:
    parts = str(model_name).split("_")
    if len(parts) >= 2 and parts[0] == "en":
        return parts[1].upper()
    return parts[0].upper()


def safe_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_std(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.std(vals)) if vals else float("nan")


def extract_layer_word_reprs(model, tok, words: list[str], prompts: list[str], batch_size: int, device: torch.device) -> dict[int, np.ndarray]:
    n_layers = model.config.num_hidden_layers + 1
    dim = model.config.hidden_size
    per_layer_word: dict[int, dict[str, list[np.ndarray]]] = {l: {w: [] for w in words} for l in range(n_layers)}

    model.eval()
    for tmpl in prompts:
        for st in range(0, len(words), batch_size):
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
                s, e = pick_word_span(tok, w, seq)
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


def extract_embedding_matrix_reprs(model, tokenizer, words: list[str]) -> np.ndarray:
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    vecs = np.zeros((len(words), emb.shape[1]), dtype=np.float32)
    for i, w in enumerate(words):
        cands = word_token_candidates(tokenizer, w)
        ids = cands[0] if cands else tokenizer.encode(w, add_special_tokens=False)
        if not ids:
            ids = [tokenizer.eos_token_id]
        vecs[i] = emb[np.array(ids)].mean(axis=0)
    return vecs


def extract_contextual_pre_lmhead_reprs(
    model,
    tokenizer,
    words: list[str],
    prompts: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dim = model.config.hidden_size
    per_word_vecs: dict[str, list[np.ndarray]] = {w: [] for w in words}
    model.eval()
    for prompt_tmpl in prompts:
        for start in range(0, len(words), batch_size):
            batch_words = words[start : start + batch_size]
            texts = [prompt_tmpl.format(word=w) for w in batch_words]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = out.hidden_states[-1].detach().float().cpu().numpy()
            ids_cpu = enc["input_ids"].cpu().numpy()
            mask_cpu = enc["attention_mask"].cpu().numpy()
            for i, w in enumerate(batch_words):
                valid_len = int(mask_cpu[i].sum())
                seq_ids = ids_cpu[i][:valid_len].tolist()
                s, e = pick_word_span(tokenizer, w, seq_ids)
                s = max(0, min(s, valid_len - 1))
                e = max(s + 1, min(e, valid_len))
                vec = hidden[i, s:e].mean(axis=0)
                per_word_vecs[w].append(vec)
    mat = np.zeros((len(words), dim), dtype=np.float32)
    for i, w in enumerate(words):
        vals = per_word_vecs[w]
        mat[i] = np.mean(vals, axis=0) if vals else np.zeros((dim,), dtype=np.float32)
    return mat
