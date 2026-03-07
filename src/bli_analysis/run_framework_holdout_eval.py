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
        description="Framework/category holdout robustness: evaluate category probes with matched vs held-out axis sets."
    )
    p.add_argument("--probe-set", type=Path, default=Path("data/probes/probe_sets.json"))
    p.add_argument("--rep-dir", type=Path, default=Path("outputs/revision/en_ablation/representations"))
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/revision/en_ablation/bli_framework_holdout_eval.csv"),
    )
    return p.parse_args()


def compute_daxis(
    abs_diff_by_axis_probe: np.ndarray,
    probe_idx: list[int],
    axis_pos: list[int],
) -> float:
    if not probe_idx or not axis_pos:
        return float("nan")
    sub = abs_diff_by_axis_probe[np.array(axis_pos)][:, np.array(probe_idx)]
    return float(np.mean(sub))


def main() -> None:
    args = parse_args()
    probe = json.loads(args.probe_set.read_text(encoding="utf-8"))

    words = probe["all_probe_words"]
    w2i = {w: i for i, w in enumerate(words)}
    neutral_idx = [w2i[w] for w in probe["neutral_anchor_words"] if w in w2i]

    # Word -> category for cultural probes.
    probe_cat = probe.get("cultural_probe_categories", {})
    categories = sorted({c for c in probe_cat.values() if isinstance(c, str)})

    probes_by_category: dict[str, list[int]] = {c: [] for c in categories}
    for w in probe["cultural_probe_words"]:
        c = probe_cat.get(w)
        if c in probes_by_category and w in w2i:
            probes_by_category[c].append(w2i[w])

    axis_meta = probe.get("semantic_axis_metadata", [])
    semantic_axes = probe.get("semantic_axes", [])
    if not axis_meta or len(axis_meta) != len(semantic_axes):
        raise ValueError("semantic_axis_metadata must exist and align with semantic_axes.")

    axis_by_category: dict[str, list[tuple[int, int]]] = {c: [] for c in categories}
    all_axes: list[tuple[int, int, str]] = []
    for meta, (a, b) in zip(axis_meta, semantic_axes):
        if a not in w2i or b not in w2i:
            continue
        c = str(meta.get("category", ""))
        ai = (w2i[a], w2i[b])
        all_axes.append((ai[0], ai[1], c))
        if c in axis_by_category:
            axis_by_category[c].append(ai)

    pairs: list[tuple[str, str]] = []
    for base in ["en_50m", "en_100m"]:
        for lang in ["zh", "fr", "fas", "nld", "ukr", "bul", "ind", "deu"]:
            cand = f"en_{lang}_a"
            if (args.rep_dir / f"{base}__embedding_matrix.npy").exists() and (args.rep_dir / f"{cand}__embedding_matrix.npy").exists():
                pairs.append((base, cand))

    rows: list[dict[str, float | str | int]] = []
    for model_a, model_b in pairs:
        for eval_repr in ["embedding_matrix", "pre_lmhead_contextual"]:
            pa = args.rep_dir / f"{model_a}__{eval_repr}.npy"
            pb = args.rep_dir / f"{model_b}__{eval_repr}.npy"
            if not pa.exists() or not pb.exists():
                continue
            mat_a = np.load(pa)
            mat_b = np.load(pb)
            w, _ = orthogonal_procrustes(mat_a[neutral_idx], mat_b[neutral_idx])
            aligned = mat_a @ w

            # Build axis-wise |projection diff| matrix once per pair+repr.
            cultural_idx = [w2i[w] for w in probe["cultural_probe_words"] if w in w2i]
            axis_pairs_only = [(i, j) for i, j, _ in all_axes]
            abs_by_axis_probe = []
            for i, j in axis_pairs_only:
                va = aligned[i] - aligned[j]
                vb = mat_b[i] - mat_b[j]
                nva = np.linalg.norm(va)
                nvb = np.linalg.norm(vb)
                if nva < 1e-12 or nvb < 1e-12:
                    abs_by_axis_probe.append(np.zeros((len(cultural_idx),), dtype=np.float32))
                    continue
                va = va / nva
                vb = vb / nvb
                pa_proj = aligned[cultural_idx] @ va
                pb_proj = mat_b[cultural_idx] @ vb
                abs_by_axis_probe.append(np.abs(pa_proj - pb_proj).astype(np.float32))
            abs_by_axis_probe_arr = np.stack(abs_by_axis_probe, axis=0)

            # Global->cultural positions once.
            cpos = {idx: k for k, idx in enumerate(cultural_idx)}
            axis_pos_by_cat = {c: [k for k, (_, _, ac) in enumerate(all_axes) if ac == c] for c in categories}
            axis_pos_holdout = {c: [k for k, (_, _, ac) in enumerate(all_axes) if ac != c] for c in categories}

            cat_rows = []
            for c in categories:
                probe_idx = probes_by_category.get(c, [])
                if not probe_idx:
                    continue
                matched_axes = axis_by_category.get(c, [])
                heldout_axes = [(i, j) for i, j, ac in all_axes if ac != c]
                if not matched_axes or not heldout_axes:
                    continue
                probe_pos = [cpos[idx] for idx in probe_idx if idx in cpos]
                matched_pos = axis_pos_by_cat[c]
                heldout_pos = axis_pos_holdout[c]

                d_matched = compute_daxis(abs_by_axis_probe_arr, probe_pos, matched_pos)
                d_heldout = compute_daxis(abs_by_axis_probe_arr, probe_pos, heldout_pos)
                row = {
                    "scope": "category",
                    "model_a": model_a,
                    "model_b": model_b,
                    "eval_repr": eval_repr,
                    "category": c,
                    "n_probes": len(probe_idx),
                    "n_axes_matched": len(matched_axes),
                    "n_axes_heldout": len(heldout_axes),
                    "daxis_matched": d_matched,
                    "daxis_heldout": d_heldout,
                    "delta_heldout_minus_matched": d_heldout - d_matched,
                }
                cat_rows.append(row)
                rows.append(row)

            if cat_rows:
                cat_df = pd.DataFrame(cat_rows)
                weighted_m = float(np.average(cat_df["daxis_matched"], weights=cat_df["n_probes"]))
                weighted_h = float(np.average(cat_df["daxis_heldout"], weights=cat_df["n_probes"]))
                rows.append(
                    {
                        "scope": "aggregate",
                        "model_a": model_a,
                        "model_b": model_b,
                        "eval_repr": eval_repr,
                        "category": "ALL",
                        "n_probes": int(cat_df["n_probes"].sum()),
                        "n_axes_matched": int(cat_df["n_axes_matched"].mean()),
                        "n_axes_heldout": int(cat_df["n_axes_heldout"].mean()),
                        "daxis_matched": weighted_m,
                        "daxis_heldout": weighted_h,
                        "delta_heldout_minus_matched": weighted_h - weighted_m,
                    }
                )

    out = pd.DataFrame(rows).sort_values(["scope", "eval_repr", "model_a", "model_b", "category"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
