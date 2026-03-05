#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = {
    "eng_only": "/scratch/amukher6/bli/models/hf/babylm_160m_eng_only_3g40_hf",
    "eng_zho": "/scratch/amukher6/bli/models/hf/babylm_160m_eng_zho_3g40_hf",
    "eng_fra": "/scratch/amukher6/bli/models/hf/babylm_160m_eng_fra_3g40_hf",
}

DEFAULT_PROMPTS = [
    "The word {word} appears in a neutral sentence.",
    "In general usage, {word} is a term.",
    "People may mention {word} in writing.",
]


@dataclass
class PairMetrics:
    repr_type: str
    model_a: str
    model_b: str
    jaccard_at_k_mean: float
    jaccard_at_k_std: float
    frobenius_cultural_similarity: float
    procrustes_anchor_residual_fro: float
    procrustes_anchor_residual_per_anchor: float
    axis_abs_projection_diff_mean: float
    axis_abs_projection_diff_rel_mean: float
    axis_abs_projection_diff_max: float
    axis_signed_projection_diff_mean: float


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return mat / denom


def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    norm = l2_normalize(mat)
    return norm @ norm.T


def find_subsequence(seq: list[int], sub: list[int]) -> tuple[int, int] | None:
    if not sub or len(sub) > len(seq):
        return None
    m = len(sub)
    for i in range(len(seq) - m + 1):
        if seq[i : i + m] == sub:
            return i, i + m
    return None


def word_token_candidates(tokenizer, word: str) -> list[list[int]]:
    cands: list[list[int]] = []
    for s in [" " + word, word]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            cands.append(ids)
    # dedup
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for c in cands:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            out.append(c)
    return out


def pick_word_span(tokenizer, word: str, seq_ids: list[int]) -> tuple[int, int]:
    cands = word_token_candidates(tokenizer, word)
    # prefer longest exact match
    matches: list[tuple[int, int]] = []
    for c in cands:
        found = find_subsequence(seq_ids, c)
        if found is not None:
            matches.append(found)
    if matches:
        return max(matches, key=lambda x: x[1] - x[0])

    # fallback to first non-special token span near end
    if len(seq_ids) >= 2:
        return len(seq_ids) - 2, len(seq_ids) - 1
    return 0, 1


def extract_embedding_matrix_reprs(model, tokenizer, words: list[str]) -> np.ndarray:
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    vecs = np.zeros((len(words), emb.shape[1]), dtype=np.float32)

    for i, w in enumerate(tqdm(words, desc="Embedding-matrix extraction")):
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
        for start in tqdm(range(0, len(words), batch_size), desc=f"Contextual extraction [{prompt_tmpl}]"):
            batch_words = words[start : start + batch_size]
            texts = [prompt_tmpl.format(word=w) for w in batch_words]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = out.hidden_states[-1].detach().float().cpu().numpy()  # pre-LM-head

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
        if per_word_vecs[w]:
            mat[i] = np.mean(per_word_vecs[w], axis=0)
        else:
            mat[i] = np.zeros((dim,), dtype=np.float32)
    return mat


def topk_neighbor_indices(norm_mat: np.ndarray, idx: int, k: int) -> np.ndarray:
    sims = norm_mat @ norm_mat[idx]
    sims[idx] = -1e9
    # argpartition then sort exact top-k
    candidate = np.argpartition(-sims, k)[:k]
    candidate = candidate[np.argsort(-sims[candidate])]
    return candidate


def jaccard(a: set[int], b: set[int]) -> float:
    u = len(a | b)
    if u == 0:
        return 0.0
    return 1.0 - (len(a & b) / u)


def run_pair_metrics(
    repr_type: str,
    model_a: str,
    model_b: str,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    words: list[str],
    neutral_idx: list[int],
    cultural_idx: list[int],
    axis_idx: list[tuple[int, int]],
    topk: int,
) -> tuple[PairMetrics, pd.DataFrame, pd.DataFrame]:
    anchors_a = mat_a[neutral_idx]
    anchors_b = mat_b[neutral_idx]
    W, _ = orthogonal_procrustes(anchors_a, anchors_b)
    a_aligned = mat_a @ W
    anchor_residual = anchors_a @ W - anchors_b
    anchor_resid_fro = float(np.linalg.norm(anchor_residual, ord="fro"))
    anchor_resid_per = float(anchor_resid_fro / max(1, len(neutral_idx)))

    norm_a = l2_normalize(a_aligned)
    norm_b = l2_normalize(mat_b)

    # Neighbor divergence per cultural word
    word_rows = []
    jaccards = []
    for idx in cultural_idx:
        na = topk_neighbor_indices(norm_a, idx, topk)
        nb = topk_neighbor_indices(norm_b, idx, topk)
        ja = jaccard(set(na.tolist()), set(nb.tolist()))
        jaccards.append(ja)
        word_rows.append(
            {
                "repr_type": repr_type,
                "pair": f"{model_a}__vs__{model_b}",
                "word": words[idx],
                "jaccard_divergence": ja,
                "neighbors_a": ", ".join(words[i] for i in na[:10]),
                "neighbors_b": ", ".join(words[i] for i in nb[:10]),
            }
        )
    word_div_df = pd.DataFrame(word_rows).sort_values("jaccard_divergence", ascending=False)

    # Axis disagreement
    axis_rows = []
    axis_diffs_all = []
    axis_diffs_rel_all = []
    axis_signed_all = []
    for i, j in axis_idx:
        va = a_aligned[j] - a_aligned[i]
        vb = mat_b[j] - mat_b[i]
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            continue
        va = va / na
        vb = vb / nb
        proj_a = a_aligned[cultural_idx] @ va
        proj_b = mat_b[cultural_idx] @ vb
        d = proj_a - proj_b
        mean_abs = float(np.mean(np.abs(d)))
        denom = np.maximum(np.abs(proj_a) + np.abs(proj_b), 1e-12)
        mean_rel = float(np.mean(np.abs(d) / denom))
        mean_signed = float(np.mean(d))
        axis_diffs_all.append(mean_abs)
        axis_diffs_rel_all.append(mean_rel)
        axis_signed_all.append(mean_signed)
        axis_rows.append(
            {
                "repr_type": repr_type,
                "pair": f"{model_a}__vs__{model_b}",
                "axis": f"{words[i]}->{words[j]}",
                "mean_abs_projection_diff": mean_abs,
                "mean_rel_projection_diff": mean_rel,
                "mean_signed_projection_diff": mean_signed,
            }
        )
    axis_df = pd.DataFrame(axis_rows).sort_values("mean_abs_projection_diff", ascending=False)

    # Structural disagreement on cultural set
    c_a = cosine_similarity_matrix(a_aligned[cultural_idx])
    c_b = cosine_similarity_matrix(mat_b[cultural_idx])
    frob = float(np.linalg.norm(c_a - c_b, ord="fro") / max(1, len(cultural_idx)))

    metrics = PairMetrics(
        repr_type=repr_type,
        model_a=model_a,
        model_b=model_b,
        jaccard_at_k_mean=float(np.mean(jaccards)) if jaccards else float("nan"),
        jaccard_at_k_std=float(np.std(jaccards)) if jaccards else float("nan"),
        frobenius_cultural_similarity=frob,
        procrustes_anchor_residual_fro=anchor_resid_fro,
        procrustes_anchor_residual_per_anchor=anchor_resid_per,
        axis_abs_projection_diff_mean=float(np.mean(axis_diffs_all)) if axis_diffs_all else float("nan"),
        axis_abs_projection_diff_rel_mean=float(np.mean(axis_diffs_rel_all)) if axis_diffs_rel_all else float("nan"),
        axis_abs_projection_diff_max=float(np.max(axis_diffs_all)) if axis_diffs_all else float("nan"),
        axis_signed_projection_diff_mean=float(np.mean(axis_signed_all)) if axis_signed_all else float("nan"),
    )

    return metrics, word_div_df, axis_df


def dataframe_page(pdf: PdfPages, df: pd.DataFrame, title: str, max_rows: int = 25):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    show = df.head(max_rows).copy()
    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full BLI analysis for models")
    parser.add_argument(
        "--probe-set",
        type=Path,
        default=Path("/scratch/amukher6/bli/data/probes/probe_sets.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/scratch/amukher6/bli/outputs/bli"),
    )
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--models-json",
        type=Path,
        default=None,
        help="Optional model registry JSON mapping model_name -> HF path",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help='Optional explicit pair spec in "model_a,model_b" format. Repeat for multiple pairs.',
    )
    parser.add_argument("--force-reextract", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_models(models_json: Path | None) -> dict[str, str]:
    if models_json is None:
        return dict(DEFAULT_MODELS)
    payload = json.loads(models_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("--models-json must be a non-empty JSON object")
    return {str(k): str(v) for k, v in payload.items()}


def resolve_pairs(pair_args: list[str], model_keys: list[str]) -> list[tuple[str, str]]:
    if not pair_args:
        return list(itertools.combinations(model_keys, 2))

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    model_key_set = set(model_keys)
    for spec in pair_args:
        if "," not in spec:
            raise ValueError(f"Invalid --pair '{spec}'. Expected 'model_a,model_b'")
        left, right = [x.strip() for x in spec.split(",", 1)]
        if left == right:
            raise ValueError(f"Invalid --pair '{spec}': model names must differ")
        if left not in model_key_set or right not in model_key_set:
            raise ValueError(f"Invalid --pair '{spec}': unknown model name (available: {model_keys})")
        pair = (left, right)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = resolve_models(args.models_json)

    with open(args.probe_set) as f:
        probes = json.load(f)

    words: list[str] = probes["all_probe_words"]
    neutral_words: list[str] = probes["neutral_anchor_words"]
    cultural_words: list[str] = probes["cultural_probe_words"]
    semantic_axes: list[list[str]] = probes["semantic_axes"]

    word_to_idx = {w: i for i, w in enumerate(words)}

    neutral_idx = [word_to_idx[w] for w in neutral_words if w in word_to_idx]
    cultural_idx = [word_to_idx[w] for w in cultural_words if w in word_to_idx]
    axis_idx = [(word_to_idx[a], word_to_idx[b]) for a, b in semantic_axes if a in word_to_idx and b in word_to_idx]

    if len(neutral_idx) < 50:
        raise RuntimeError("Not enough neutral anchors after filtering; need at least 50.")
    if len(cultural_idx) < 50:
        raise RuntimeError("Not enough cultural probes after filtering; need at least 50.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    rep_cache_dir = args.output_dir / "representations"
    rep_cache_dir.mkdir(parents=True, exist_ok=True)

    repr_data: dict[str, dict[str, np.ndarray]] = {"embedding_matrix": {}, "pre_lmhead_contextual": {}}

    for model_name, model_path in models.items():
        print(f"\n=== Loading model: {model_name} ({model_path})")

        emb_cache = rep_cache_dir / f"{model_name}__embedding_matrix.npy"
        ctx_cache = rep_cache_dir / f"{model_name}__pre_lmhead_contextual.npy"

        need_load_model = args.force_reextract or (not emb_cache.exists()) or (not ctx_cache.exists())
        reextract_emb = args.force_reextract or (not emb_cache.exists())
        reextract_ctx = args.force_reextract or (not ctx_cache.exists())
        if not need_load_model:
            try:
                emb_shape = np.load(emb_cache, mmap_mode="r").shape
                ctx_shape = np.load(ctx_cache, mmap_mode="r").shape
                if emb_shape[0] != len(words) or ctx_shape[0] != len(words):
                    print(
                        f"Cache length mismatch for {model_name}: "
                        f"emb={emb_shape[0]}, ctx={ctx_shape[0]}, words={len(words)}. Re-extracting."
                    )
                    need_load_model = True
                    reextract_emb = True
                    reextract_ctx = True
            except Exception:
                need_load_model = True
                reextract_emb = True
                reextract_ctx = True

        if need_load_model:
            torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
            model.to(device)
            model.eval()

            if reextract_emb:
                emb_mat = extract_embedding_matrix_reprs(model, tokenizer, words)
                np.save(emb_cache, emb_mat)
            if reextract_ctx:
                ctx_mat = extract_contextual_pre_lmhead_reprs(
                    model=model,
                    tokenizer=tokenizer,
                    words=words,
                    prompts=DEFAULT_PROMPTS,
                    batch_size=args.batch_size,
                    device=device,
                )
                np.save(ctx_cache, ctx_mat)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        repr_data["embedding_matrix"][model_name] = np.load(emb_cache)
        repr_data["pre_lmhead_contextual"][model_name] = np.load(ctx_cache)

    metrics_rows: list[dict] = []
    word_div_tables: list[pd.DataFrame] = []
    axis_tables: list[pd.DataFrame] = []

    model_pairs = resolve_pairs(args.pair, list(models.keys()))
    for repr_type, model_mats in repr_data.items():
        for m_a, m_b in model_pairs:
            print(f"\nRunning metrics: {repr_type} | {m_a} vs {m_b}")
            metrics, word_div_df, axis_df = run_pair_metrics(
                repr_type=repr_type,
                model_a=m_a,
                model_b=m_b,
                mat_a=model_mats[m_a],
                mat_b=model_mats[m_b],
                words=words,
                neutral_idx=neutral_idx,
                cultural_idx=cultural_idx,
                axis_idx=axis_idx,
                topk=args.topk,
            )
            metrics_rows.append(metrics.__dict__)
            word_div_tables.append(word_div_df)
            axis_tables.append(axis_df)

    summary_df = pd.DataFrame(metrics_rows)
    word_div_df = pd.concat(word_div_tables, ignore_index=True)
    axis_df = pd.concat(axis_tables, ignore_index=True)

    summary_csv = args.output_dir / "bli_summary_metrics.csv"
    word_csv = args.output_dir / "bli_word_neighbor_divergence.csv"
    axis_csv = args.output_dir / "bli_axis_divergence.csv"

    summary_df.to_csv(summary_csv, index=False)
    word_div_df.to_csv(word_csv, index=False)
    axis_df.to_csv(axis_csv, index=False)

    # Build PDF report
    pdf_path = args.output_dir / "bli_report.pdf"
    with PdfPages(pdf_path) as pdf:
        dataframe_page(pdf, summary_df, "BLI Summary Metrics", max_rows=20)

        for repr_type in summary_df["repr_type"].unique():
            sub = summary_df[summary_df["repr_type"] == repr_type]
            dataframe_page(pdf, sub, f"Summary ({repr_type})", max_rows=20)

            for _, row in sub.iterrows():
                pair = f"{row['model_a']}__vs__{row['model_b']}"
                wd = word_div_df[(word_div_df["repr_type"] == repr_type) & (word_div_df["pair"] == pair)]
                axd = axis_df[(axis_df["repr_type"] == repr_type) & (axis_df["pair"] == pair)]
                dataframe_page(pdf, wd, f"Top Neighbor Divergence ({repr_type} | {pair})", max_rows=25)
                dataframe_page(pdf, axd, f"Top Axis Divergence ({repr_type} | {pair})", max_rows=25)

    meta_out = args.output_dir / "bli_run_metadata.json"
    meta = {
        "models": models,
        "model_pairs": [{"model_a": a, "model_b": b} for a, b in model_pairs],
        "probe_set": str(args.probe_set),
        "word_count": len(words),
        "neutral_anchor_count": len(neutral_idx),
        "cultural_probe_count": len(cultural_idx),
        "axis_count": len(axis_idx),
        "topk": args.topk,
        "device": str(device),
        "prompts": DEFAULT_PROMPTS,
        "outputs": {
            "summary_csv": str(summary_csv),
            "word_csv": str(word_csv),
            "axis_csv": str(axis_csv),
            "pdf": str(pdf_path),
        },
    }
    meta_out.write_text(json.dumps(meta, indent=2))

    print("\n=== Completed BLI pipeline ===")
    print(f"Summary: {summary_csv}")
    print(f"Word divergence: {word_csv}")
    print(f"Axis divergence: {axis_csv}")
    print(f"Report PDF: {pdf_path}")


if __name__ == "__main__":
    main()
