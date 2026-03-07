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
        description="Run EN-centered alignment-method comparison (orthogonal vs affine)."
    )
    p.add_argument(
        "--probe-set",
        type=Path,
        default=Path("data/probes/probe_sets.json"),
    )
    p.add_argument(
        "--rep-dir",
        type=Path,
        default=Path("outputs/revision/en_ablation/representations"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/revision/en_ablation/bli_alignment_method_comparison.csv"),
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


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    # Solve [src, 1] @ M ~= dst where M has shape (d+1, d).
    src_aug = np.hstack([src, np.ones((src.shape[0], 1), dtype=src.dtype)])
    M, *_ = np.linalg.lstsq(src_aug, dst, rcond=None)
    return M


def apply_affine(mat: np.ndarray, M: np.ndarray) -> np.ndarray:
    mat_aug = np.hstack([mat, np.ones((mat.shape[0], 1), dtype=mat.dtype)])
    return mat_aug @ M


def compute_metrics(
    aligned_a: np.ndarray,
    eval_b: np.ndarray,
    cultural_idx: list[int],
    axis_idx: list[tuple[int, int]],
    topk: int,
) -> dict[str, float]:
    na = l2_normalize(aligned_a)
    nb = l2_normalize(eval_b)
    dnn = []
    for idx in cultural_idx:
        n1 = topk_neighbor_indices(na, idx, topk)
        n2 = topk_neighbor_indices(nb, idx, topk)
        dnn.append(jaccard_divergence(set(n1.tolist()), set(n2.tolist())))
    dnn = float(np.mean(dnn)) if dnn else float("nan")

    sa = cosine_similarity_matrix(aligned_a[cultural_idx])
    sb = cosine_similarity_matrix(eval_b[cultural_idx])
    dstruct = float(np.linalg.norm(sa - sb, ord="fro") / max(1, len(cultural_idx)))

    axis_abs = []
    axis_signed = []
    for i, j in axis_idx:
        va = aligned_a[j] - aligned_a[i]
        vb = eval_b[j] - eval_b[i]
        nva = np.linalg.norm(va)
        nvb = np.linalg.norm(vb)
        if nva < 1e-12 or nvb < 1e-12:
            continue
        va = va / nva
        vb = vb / nvb
        pa = aligned_a[cultural_idx] @ va
        pb = eval_b[cultural_idx] @ vb
        d = pa - pb
        axis_abs.append(float(np.mean(np.abs(d))))
        axis_signed.append(float(np.mean(d)))

    return {
        "jaccard_at_k_mean": dnn,
        "frobenius_cultural_similarity": dstruct,
        "axis_abs_projection_diff_mean": float(np.mean(axis_abs)) if axis_abs else float("nan"),
        "axis_signed_projection_diff_mean": float(np.mean(axis_signed)) if axis_signed else float("nan"),
    }


def main() -> None:
    args = parse_args()

    probes = json.loads(args.probe_set.read_text(encoding="utf-8"))
    words = probes["all_probe_words"]
    w2i = {w: i for i, w in enumerate(words)}

    neutral_idx = [w2i[w] for w in probes["neutral_anchor_words"] if w in w2i]
    cultural_idx = [w2i[w] for w in probes["cultural_probe_words"] if w in w2i]
    axis_idx = [(w2i[a], w2i[b]) for a, b in probes["semantic_axes"] if a in w2i and b in w2i]

    available = set()
    for p in args.rep_dir.glob("*__embedding_matrix.npy"):
        name = p.name.replace("__embedding_matrix.npy", "")
        if (args.rep_dir / f"{name}__pre_lmhead_contextual.npy").exists():
            available.add(name)

    baselines = [m for m in ["en_50m", "en_100m"] if m in available]
    targets = sorted(
        m for m in available if m.startswith("en_") and m.endswith("_a") and m not in {"en_50m", "en_100m"}
    )
    pairs = [(base, tgt) for base in baselines for tgt in targets]
    if not pairs:
        raise ValueError(f"No EN-centered C3 pairs found in {args.rep_dir}")

    rows: list[dict[str, float | str]] = []
    for ma, mb in pairs:
        emb_a = np.load(args.rep_dir / f"{ma}__embedding_matrix.npy")
        emb_b = np.load(args.rep_dir / f"{mb}__embedding_matrix.npy")
        ctx_a = np.load(args.rep_dir / f"{ma}__pre_lmhead_contextual.npy")
        ctx_b = np.load(args.rep_dir / f"{mb}__pre_lmhead_contextual.npy")

        align_variants: list[dict[str, str | np.ndarray | float]] = []

        # Variant 1: orthogonal + embedding anchors.
        w_emb, _ = orthogonal_procrustes(emb_a[neutral_idx], emb_b[neutral_idx])
        emb_resid = emb_a[neutral_idx] @ w_emb - emb_b[neutral_idx]
        align_variants.append(
            {
                "alignment_method": "orthogonal",
                "alignment_source": "embedding_matrix",
                "transform": w_emb,
                "affine": False,
                "anchor_residual_fro": float(np.linalg.norm(emb_resid, ord="fro")),
                "anchor_residual_per_anchor": float(np.linalg.norm(emb_resid, ord="fro") / max(1, len(neutral_idx))),
            }
        )

        # Variant 2: orthogonal + contextual anchors.
        w_ctx, _ = orthogonal_procrustes(ctx_a[neutral_idx], ctx_b[neutral_idx])
        ctx_resid = ctx_a[neutral_idx] @ w_ctx - ctx_b[neutral_idx]
        align_variants.append(
            {
                "alignment_method": "orthogonal",
                "alignment_source": "pre_lmhead_contextual",
                "transform": w_ctx,
                "affine": False,
                "anchor_residual_fro": float(np.linalg.norm(ctx_resid, ord="fro")),
                "anchor_residual_per_anchor": float(np.linalg.norm(ctx_resid, ord="fro") / max(1, len(neutral_idx))),
            }
        )

        # Variant 3: affine + embedding anchors.
        m_aff = fit_affine(emb_a[neutral_idx], emb_b[neutral_idx])
        emb_aff_resid = apply_affine(emb_a[neutral_idx], m_aff) - emb_b[neutral_idx]
        align_variants.append(
            {
                "alignment_method": "affine",
                "alignment_source": "embedding_matrix",
                "transform": m_aff,
                "affine": True,
                "anchor_residual_fro": float(np.linalg.norm(emb_aff_resid, ord="fro")),
                "anchor_residual_per_anchor": float(np.linalg.norm(emb_aff_resid, ord="fro") / max(1, len(neutral_idx))),
            }
        )

        eval_reprs = [
            ("embedding_matrix", emb_a, emb_b),
            ("pre_lmhead_contextual", ctx_a, ctx_b),
        ]
        for eval_repr, eval_a, eval_b in eval_reprs:
            for av in align_variants:
                if bool(av["affine"]):
                    aligned = apply_affine(eval_a, av["transform"])  # type: ignore[arg-type]
                else:
                    aligned = eval_a @ av["transform"]  # type: ignore[operator]
                metrics = compute_metrics(
                    aligned_a=aligned,
                    eval_b=eval_b,
                    cultural_idx=cultural_idx,
                    axis_idx=axis_idx,
                    topk=args.topk,
                )
                rows.append(
                    {
                        "model_a": ma,
                        "model_b": mb,
                        "eval_repr": eval_repr,
                        "alignment_method": str(av["alignment_method"]),
                        "alignment_source": str(av["alignment_source"]),
                        "anchor_residual_fro": float(av["anchor_residual_fro"]),
                        "anchor_residual_per_anchor": float(av["anchor_residual_per_anchor"]),
                        **metrics,
                    }
                )

    out = pd.DataFrame(rows).sort_values(
        ["eval_repr", "alignment_method", "alignment_source", "model_a", "model_b"]
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
