#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate cultural probes vs negative controls from cached representations")
    p.add_argument("--probe-set", type=Path, required=True)
    p.add_argument("--rep-dir", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--topk", type=int, default=25)
    return p.parse_args()


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return mat / denom


def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    norm = l2_normalize(mat)
    return norm @ norm.T


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


def eval_subset(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    neutral_idx: list[int],
    eval_idx: list[int],
    axis_idx: list[tuple[int, int]],
    topk: int,
) -> tuple[float, float, float]:
    wa, wb = mat_a[neutral_idx], mat_b[neutral_idx]
    w, _ = orthogonal_procrustes(wa, wb)
    a_aligned = mat_a @ w

    norm_a = l2_normalize(a_aligned)
    norm_b = l2_normalize(mat_b)

    jacc = []
    for idx in eval_idx:
        na = topk_neighbor_indices(norm_a, idx, topk)
        nb = topk_neighbor_indices(norm_b, idx, topk)
        jacc.append(jaccard_divergence(set(na.tolist()), set(nb.tolist())))

    sa = cosine_similarity_matrix(a_aligned[eval_idx])
    sb = cosine_similarity_matrix(mat_b[eval_idx])
    d_struct = float(np.linalg.norm(sa - sb, ord="fro") / max(1, len(eval_idx)))

    d_axis = []
    for i, j in axis_idx:
        va = a_aligned[j] - a_aligned[i]
        vb = mat_b[j] - mat_b[i]
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            continue
        va /= na
        vb /= nb
        proj_a = a_aligned[eval_idx] @ va
        proj_b = mat_b[eval_idx] @ vb
        d_axis.append(float(np.mean(np.abs(proj_a - proj_b))))

    return float(np.mean(jacc)), d_struct, float(np.mean(d_axis)) if d_axis else float("nan")


def main() -> None:
    args = parse_args()
    probes = json.loads(args.probe_set.read_text(encoding="utf-8"))
    words: list[str] = probes["all_probe_words"]
    neutral_words: list[str] = probes["neutral_anchor_words"]
    cultural_words: list[str] = probes["cultural_probe_words"]
    negative_words: list[str] = probes.get("negative_control_words", [])
    semantic_axes: list[list[str]] = probes["semantic_axes"]

    w2i = {w: i for i, w in enumerate(words)}
    neutral_idx = [w2i[w] for w in neutral_words if w in w2i]
    cultural_idx = [w2i[w] for w in cultural_words if w in w2i]
    negative_idx = [w2i[w] for w in negative_words if w in w2i]
    axis_idx = [(w2i[a], w2i[b]) for a, b in semantic_axes if a in w2i and b in w2i]

    available = set()
    for p in args.rep_dir.glob("*__embedding_matrix.npy"):
        name = p.name.replace("__embedding_matrix.npy", "")
        if (args.rep_dir / f"{name}__pre_lmhead_contextual.npy").exists():
            available.add(name)

    baselines = [m for m in ["en_50m", "en_100m"] if m in available]
    targets = sorted(
        m for m in available if m.startswith("en_") and m.endswith("_a") and m not in {"en_50m", "en_100m"}
    )
    core_pairs = [(base, tgt) for base in baselines for tgt in targets]
    if not core_pairs:
        raise ValueError(f"No EN-centered A-setup pairs found in {args.rep_dir}")

    repr_types = ["embedding_matrix", "pre_lmhead_contextual"]

    rows: list[dict] = []
    for repr_type in repr_types:
        for ma, mb in core_pairs:
            pa = np.load(args.rep_dir / f"{ma}__{repr_type}.npy")
            pb = np.load(args.rep_dir / f"{mb}__{repr_type}.npy")
            for group, eval_idx in [("cultural_probes", cultural_idx), ("negative_controls", negative_idx)]:
                dnn, dstruct, daxis = eval_subset(pa, pb, neutral_idx, eval_idx, axis_idx, args.topk)
                rows.append(
                    {
                        "repr_type": repr_type,
                        "pair": f"{ma}__vs__{mb}",
                        "group": group,
                        "n_eval_words": int(len(eval_idx)),
                        "jaccard_at_k_mean": dnn,
                        "frobenius_similarity": dstruct,
                        "axis_abs_projection_diff_mean": daxis,
                    }
                )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
