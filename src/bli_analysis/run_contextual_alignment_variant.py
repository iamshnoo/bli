#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare contextual-space metrics under embedding-anchor vs contextual-anchor alignment."
    )
    p.add_argument(
        "--probe-set",
        type=Path,
        default=Path("/scratch/amukher6/bli/data/probes/probe_sets.json"),
    )
    p.add_argument(
        "--rep-dir",
        type=Path,
        default=Path("/scratch/amukher6/bli/outputs/revision/en_ablation/representations"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/scratch/amukher6/bli/outputs/revision/en_ablation/bli_contextual_alignment_variant.csv"),
    )
    p.add_argument("--topk", type=int, default=25)
    return p.parse_args()


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(denom, eps)


def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    x = l2_normalize(mat)
    return x @ x.T


def topk_neighbor_indices(norm_mat: np.ndarray, idx: int, k: int) -> np.ndarray:
    sims = norm_mat @ norm_mat[idx]
    sims[idx] = -1e9
    cand = np.argpartition(-sims, k)[:k]
    return cand[np.argsort(-sims[cand])]


def jaccard_divergence(a: set[int], b: set[int]) -> float:
    u = len(a | b)
    if u == 0:
        return 0.0
    return 1.0 - (len(a & b) / u)


def compute_metrics(
    eval_a: np.ndarray,
    eval_b: np.ndarray,
    align_a: np.ndarray,
    align_b: np.ndarray,
    neutral_idx: list[int],
    cultural_idx: list[int],
    axis_idx: list[tuple[int, int]],
    topk: int,
) -> dict[str, float]:
    wa = align_a[neutral_idx]
    wb = align_b[neutral_idx]
    w, _ = orthogonal_procrustes(wa, wb)

    a_aligned = eval_a @ w
    resid = wa @ w - wb
    resid_fro = float(np.linalg.norm(resid, ord="fro"))
    resid_per = float(resid_fro / max(1, len(neutral_idx)))

    na = l2_normalize(a_aligned)
    nb = l2_normalize(eval_b)
    dnn = []
    for idx in cultural_idx:
        n1 = topk_neighbor_indices(na, idx, topk)
        n2 = topk_neighbor_indices(nb, idx, topk)
        dnn.append(jaccard_divergence(set(n1.tolist()), set(n2.tolist())))
    dnn = float(np.mean(dnn)) if dnn else float("nan")

    sa = cosine_similarity_matrix(a_aligned[cultural_idx])
    sb = cosine_similarity_matrix(eval_b[cultural_idx])
    dstruct = float(np.linalg.norm(sa - sb, ord="fro") / max(1, len(cultural_idx)))

    axis_abs = []
    axis_signed = []
    for i, j in axis_idx:
        va = a_aligned[j] - a_aligned[i]
        vb = eval_b[j] - eval_b[i]
        nva = np.linalg.norm(va)
        nvb = np.linalg.norm(vb)
        if nva < 1e-12 or nvb < 1e-12:
            continue
        va = va / nva
        vb = vb / nvb
        pa = a_aligned[cultural_idx] @ va
        pb = eval_b[cultural_idx] @ vb
        d = pa - pb
        axis_abs.append(float(np.mean(np.abs(d))))
        axis_signed.append(float(np.mean(d)))

    return {
        "jaccard_at_k_mean": dnn,
        "frobenius_cultural_similarity": dstruct,
        "axis_abs_projection_diff_mean": float(np.mean(axis_abs)) if axis_abs else float("nan"),
        "axis_signed_projection_diff_mean": float(np.mean(axis_signed)) if axis_signed else float("nan"),
        "procrustes_anchor_residual_fro": resid_fro,
        "procrustes_anchor_residual_per_anchor": resid_per,
    }


def main() -> None:
    args = parse_args()

    probes = json.loads(args.probe_set.read_text(encoding="utf-8"))
    words = probes["all_probe_words"]
    w2i = {w: i for i, w in enumerate(words)}

    neutral_idx = [w2i[w] for w in probes["neutral_anchor_words"] if w in w2i]
    cultural_idx = [w2i[w] for w in probes["cultural_probe_words"] if w in w2i]
    axis_idx = [(w2i[a], w2i[b]) for a, b in probes["semantic_axes"] if a in w2i and b in w2i]

    pairs = [
        ("en_50m", "en_zh_a"),
        ("en_50m", "en_fr_a"),
        ("en_100m", "en_zh_a"),
        ("en_100m", "en_fr_a"),
    ]
    rows: list[dict] = []
    for ma, mb in pairs:
        emb_a = np.load(args.rep_dir / f"{ma}__embedding_matrix.npy")
        emb_b = np.load(args.rep_dir / f"{mb}__embedding_matrix.npy")
        ctx_a = np.load(args.rep_dir / f"{ma}__pre_lmhead_contextual.npy")
        ctx_b = np.load(args.rep_dir / f"{mb}__pre_lmhead_contextual.npy")

        # Variant 1: align contextual evaluation using embedding anchors.
        m1 = compute_metrics(
            eval_a=ctx_a,
            eval_b=ctx_b,
            align_a=emb_a,
            align_b=emb_b,
            neutral_idx=neutral_idx,
            cultural_idx=cultural_idx,
            axis_idx=axis_idx,
            topk=args.topk,
        )
        rows.append(
            {
                "model_a": ma,
                "model_b": mb,
                "eval_repr": "pre_lmhead_contextual",
                "alignment_source": "embedding_matrix",
                **m1,
            }
        )

        # Variant 2: align contextual evaluation using contextual anchors.
        m2 = compute_metrics(
            eval_a=ctx_a,
            eval_b=ctx_b,
            align_a=ctx_a,
            align_b=ctx_b,
            neutral_idx=neutral_idx,
            cultural_idx=cultural_idx,
            axis_idx=axis_idx,
            topk=args.topk,
        )
        rows.append(
            {
                "model_a": ma,
                "model_b": mb,
                "eval_repr": "pre_lmhead_contextual",
                "alignment_source": "pre_lmhead_contextual",
                **m2,
            }
        )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
