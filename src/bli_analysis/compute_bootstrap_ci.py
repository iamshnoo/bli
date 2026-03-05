#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap CIs for BLI metrics")
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--word-csv", type=Path, required=True)
    p.add_argument("--axis-csv", type=Path, required=True)
    p.add_argument("--repr-dir", type=Path, required=True)
    p.add_argument("--probe-set", type=Path, default=Path("data/probes/probe_sets.json"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, required=True)
    return p.parse_args()


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(denom, eps)


def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    nm = l2_normalize(mat)
    return nm @ nm.T


def ci_bounds(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    summary = pd.read_csv(args.summary_csv)
    word_df = pd.read_csv(args.word_csv)
    axis_df = pd.read_csv(args.axis_csv)

    probes = json.loads(args.probe_set.read_text(encoding="utf-8"))
    words: list[str] = probes["all_probe_words"]
    neutral = probes["neutral_anchor_words"]
    cultural = probes["cultural_probe_words"]

    w2i = {w: i for i, w in enumerate(words)}
    nidx = np.array([w2i[w] for w in neutral if w in w2i], dtype=np.int64)
    cidx = np.array([w2i[w] for w in cultural if w in w2i], dtype=np.int64)

    if len(nidx) == 0 or len(cidx) == 0:
        raise RuntimeError("Probe indices are empty; check probe set.")

    # Load all required representation arrays once.
    needed_models = sorted(set(summary["model_a"]).union(set(summary["model_b"])))
    repr_types = sorted(set(summary["repr_type"]))
    repr_cache: dict[tuple[str, str], np.ndarray] = {}
    for rt in repr_types:
        for m in needed_models:
            p = args.repr_dir / f"{m}__{rt}.npy"
            if not p.exists():
                raise FileNotFoundError(f"Missing representation file: {p}")
            repr_cache[(m, rt)] = np.load(p)

    rows: list[dict] = []

    # Per-pair metric bootstrap.
    for _, r in summary.iterrows():
        rt = str(r["repr_type"])
        ma = str(r["model_a"])
        mb = str(r["model_b"])
        pair = f"{ma}__vs__{mb}"

        wd = word_df[(word_df["repr_type"] == rt) & (word_df["pair"] == pair)]["jaccard_divergence"].to_numpy()
        ad = axis_df[(axis_df["repr_type"] == rt) & (axis_df["pair"] == pair)]["mean_abs_projection_diff"].to_numpy()
        if len(wd) == 0 or len(ad) == 0:
            continue

        # NN and axis CIs from item-level bootstrap.
        boot_nn = np.empty(args.n_bootstrap, dtype=np.float64)
        boot_axis = np.empty(args.n_bootstrap, dtype=np.float64)
        for b in range(args.n_bootstrap):
            i_nn = rng.integers(0, len(wd), len(wd))
            i_ax = rng.integers(0, len(ad), len(ad))
            boot_nn[b] = wd[i_nn].mean()
            boot_axis[b] = ad[i_ax].mean()

        nn_mean, nn_low, nn_high = ci_bounds(boot_nn)
        ax_mean, ax_low, ax_high = ci_bounds(boot_axis)

        rows.append(
            {
                "repr_type": rt,
                "model_a": ma,
                "model_b": mb,
                "metric": "jaccard_at_k_mean",
                "mean": nn_mean,
                "ci_low": nn_low,
                "ci_high": nn_high,
            }
        )
        rows.append(
            {
                "repr_type": rt,
                "model_a": ma,
                "model_b": mb,
                "metric": "axis_abs_projection_diff_mean",
                "mean": ax_mean,
                "ci_low": ax_low,
                "ci_high": ax_high,
            }
        )

        # Structural CI from cultural-word bootstrap over similarity matrices.
        Xa = repr_cache[(ma, rt)]
        Xb = repr_cache[(mb, rt)]
        W, _ = orthogonal_procrustes(Xa[nidx], Xb[nidx])
        Xa_aligned = Xa @ W
        Sa = cosine_similarity_matrix(Xa_aligned[cidx])
        Sb = cosine_similarity_matrix(Xb[cidx])

        n = len(cidx)
        boot_struct = np.empty(args.n_bootstrap, dtype=np.float64)
        for b in range(args.n_bootstrap):
            idx = rng.integers(0, n, n)
            d = Sa[np.ix_(idx, idx)] - Sb[np.ix_(idx, idx)]
            boot_struct[b] = np.linalg.norm(d, ord="fro") / n

        st_mean, st_low, st_high = ci_bounds(boot_struct)
        rows.append(
            {
                "repr_type": rt,
                "model_a": ma,
                "model_b": mb,
                "metric": "frobenius_cultural_similarity",
                "mean": st_mean,
                "ci_low": st_low,
                "ci_high": st_high,
            }
        )

    out = pd.DataFrame(rows)

    # Ratio CIs: contextual / embedding for the axis metric per identical pair.
    ratio_rows: list[dict] = []
    axis_only = out[out["metric"] == "axis_abs_projection_diff_mean"]
    pairs = axis_only[["model_a", "model_b"]].drop_duplicates()
    for _, p in pairs.iterrows():
        ma, mb = str(p["model_a"]), str(p["model_b"])
        e = axis_df[
            (axis_df["repr_type"] == "embedding_matrix")
            & (axis_df["pair"] == f"{ma}__vs__{mb}")
        ]["mean_abs_projection_diff"].to_numpy()
        c = axis_df[
            (axis_df["repr_type"] == "pre_lmhead_contextual")
            & (axis_df["pair"] == f"{ma}__vs__{mb}")
        ]["mean_abs_projection_diff"].to_numpy()
        if len(e) == 0 or len(c) == 0:
            continue

        m = min(len(e), len(c))
        boot_ratio = np.empty(args.n_bootstrap, dtype=np.float64)
        for b in range(args.n_bootstrap):
            ie = rng.integers(0, len(e), m)
            ic = rng.integers(0, len(c), m)
            denom = max(1e-12, float(e[ie].mean()))
            boot_ratio[b] = float(c[ic].mean()) / denom

        ratio_mean, ratio_low, ratio_high = ci_bounds(boot_ratio)
        ratio_rows.append(
            {
                "repr_type": "contextual_over_embedding",
                "model_a": ma,
                "model_b": mb,
                "metric": "axis_ratio_contextual_over_embedding",
                "mean": ratio_mean,
                "ci_low": ratio_low,
                "ci_high": ratio_high,
            }
        )

    if ratio_rows:
        out = pd.concat([out, pd.DataFrame(ratio_rows)], ignore_index=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
