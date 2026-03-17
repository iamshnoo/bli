#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable

import pandas as pd


CORE_METRICS = (
    "jaccard_at_k_mean",
    "frobenius_cultural_similarity",
    "axis_abs_projection_diff_mean",
)

EN_SUBSPACE_MODEL_B_RE = re.compile(r"^en_[a-z]+_[ab]$")
EN_SUBSPACE_PAIR_RE = re.compile(r"^en_(50m|100m)__vs__en_[a-z]+_[ab]$")
SEED_MODEL_RE = re.compile(r"^en_(50m|100m)_s\d+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Center EN-subspace metrics in en_ablation artifacts using EN seed-null means: "
            "delta = value - EN_null_mean."
        )
    )
    p.add_argument(
        "--seed-summary-csv",
        type=Path,
        default=Path("outputs/revision/en_seed_null/bli_summary_metrics.csv"),
        help="Seed-null summary CSV with pairwise seed metrics.",
    )
    p.add_argument(
        "--en-ablation-dir",
        type=Path,
        default=Path("outputs/revision/en_ablation"),
        help="Directory containing en_ablation CSV artifacts to center in-place.",
    )
    return p.parse_args()


def checkpoint_from_model(model_name: str) -> str | None:
    if model_name in {"en_50m", "en_100m"}:
        return "50m" if model_name == "en_50m" else "100m"
    m = SEED_MODEL_RE.match(model_name)
    if m:
        return m.group(1)
    return None


def checkpoint_from_pair(pair: str) -> str | None:
    m = EN_SUBSPACE_PAIR_RE.match(pair)
    if not m:
        return None
    return m.group(1)


def is_en_subspace_model_row(model_a: str, model_b: str) -> bool:
    return model_a in {"en_50m", "en_100m"} and EN_SUBSPACE_MODEL_B_RE.match(model_b) is not None


def is_en_subspace_pair(pair: str) -> bool:
    return EN_SUBSPACE_PAIR_RE.match(pair) is not None


def repr_key_from_row(row: pd.Series) -> str:
    if "repr_type" in row and isinstance(row["repr_type"], str):
        return row["repr_type"]
    if "eval_repr" in row and isinstance(row["eval_repr"], str):
        return row["eval_repr"]
    raise ValueError("Row missing representation column (`repr_type` or `eval_repr`).")


def build_baseline_means(seed_summary: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    rows: list[dict[str, object]] = []
    for _, r in seed_summary.iterrows():
        model_a = str(r["model_a"])
        model_b = str(r["model_b"])
        ck_a = checkpoint_from_model(model_a)
        ck_b = checkpoint_from_model(model_b)
        if ck_a is None or ck_b is None:
            continue
        if ck_a != ck_b:
            raise ValueError(f"Mismatched checkpoints in seed-null row: {model_a} vs {model_b}")
        repr_type = str(r["repr_type"])
        for metric in CORE_METRICS:
            rows.append(
                {
                    "checkpoint": ck_a,
                    "repr_type": repr_type,
                    "metric": metric,
                    "value": float(r[metric]),
                }
            )

    bdf = pd.DataFrame(rows)
    if bdf.empty:
        raise ValueError("No seed-null baseline rows found.")

    means = (
        bdf.groupby(["checkpoint", "repr_type", "metric"], as_index=False)["value"]
        .mean()
        .set_index(["checkpoint", "repr_type", "metric"])["value"]
        .to_dict()
    )
    if len(means) != 12:
        raise ValueError(f"Expected 12 baseline cells, found {len(means)}.")
    return means


def center_metric_columns(
    df: pd.DataFrame,
    means: dict[tuple[str, str, str], float],
    row_selector: Callable[[pd.Series], bool],
    metric_to_baseline_metric: dict[str, str],
    checkpoint_getter: Callable[[pd.Series], str | None],
) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        if not row_selector(row):
            continue
        ck = checkpoint_getter(row)
        if ck is None:
            raise ValueError(f"Could not infer checkpoint for row index {idx}.")
        repr_type = repr_key_from_row(row)
        for metric_col, baseline_metric in metric_to_baseline_metric.items():
            if metric_col not in out.columns:
                continue
            key = (ck, repr_type, baseline_metric)
            if key not in means:
                raise ValueError(f"Missing baseline mean for key={key} at row index {idx}.")
            out.at[idx, metric_col] = float(row[metric_col]) - float(means[key])
    return out


def center_summary_csv(path: Path, means: dict[tuple[str, str, str], float]) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    out = center_metric_columns(
        df=df,
        means=means,
        row_selector=lambda r: is_en_subspace_model_row(str(r["model_a"]), str(r["model_b"])),
        metric_to_baseline_metric={m: m for m in CORE_METRICS},
        checkpoint_getter=lambda r: checkpoint_from_model(str(r["model_a"])),
    )
    out.to_csv(path, index=False)
    print(f"Centered: {path}")


def center_bootstrap_ci_csv(path: Path, means: dict[tuple[str, str, str], float]) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    out = df.copy()
    for idx, row in out.iterrows():
        metric = str(row["metric"])
        if metric not in CORE_METRICS:
            continue
        if not is_en_subspace_model_row(str(row["model_a"]), str(row["model_b"])):
            continue
        ck = checkpoint_from_model(str(row["model_a"]))
        if ck is None:
            raise ValueError(f"Could not infer checkpoint for bootstrap row index {idx}.")
        repr_type = str(row["repr_type"])
        key = (ck, repr_type, metric)
        if key not in means:
            raise ValueError(f"Missing baseline mean for key={key} at bootstrap row index {idx}.")
        baseline = float(means[key])
        for col in ("mean", "ci_low", "ci_high"):
            out.at[idx, col] = float(row[col]) - baseline

    out.to_csv(path, index=False)
    print(f"Centered: {path}")


def center_framework_holdout_csv(path: Path, means: dict[tuple[str, str, str], float]) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    out = center_metric_columns(
        df=df,
        means=means,
        row_selector=lambda r: is_en_subspace_model_row(str(r["model_a"]), str(r["model_b"])),
        metric_to_baseline_metric={
            "daxis_matched": "axis_abs_projection_diff_mean",
            "daxis_heldout": "axis_abs_projection_diff_mean",
        },
        checkpoint_getter=lambda r: checkpoint_from_model(str(r["model_a"])),
    )
    if {"daxis_matched", "daxis_heldout", "delta_heldout_minus_matched"}.issubset(set(out.columns)):
        out["delta_heldout_minus_matched"] = out["daxis_heldout"] - out["daxis_matched"]
    out.to_csv(path, index=False)
    print(f"Centered: {path}")


def center_pair_csv(path: Path, means: dict[tuple[str, str, str], float], metric_map: dict[str, str]) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    out = center_metric_columns(
        df=df,
        means=means,
        row_selector=lambda r: is_en_subspace_pair(str(r["pair"])),
        metric_to_baseline_metric=metric_map,
        checkpoint_getter=lambda r: checkpoint_from_pair(str(r["pair"])),
    )
    out.to_csv(path, index=False)
    print(f"Centered: {path}")


def main() -> None:
    args = parse_args()

    seed_df = pd.read_csv(args.seed_summary_csv)
    baseline_means = build_baseline_means(seed_df)
    print(f"Baseline cells built: {len(baseline_means)}")

    en_dir = args.en_ablation_dir

    center_summary_csv(en_dir / "bli_summary_metrics.csv", baseline_means)
    center_bootstrap_ci_csv(en_dir / "bli_bootstrap_ci.csv", baseline_means)

    # Files with model_a/model_b rows (repr_type/eval_repr available).
    for name in ("bli_alignment_method_comparison.csv", "bli_contextual_alignment_variant.csv"):
        path = en_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        out = center_metric_columns(
            df=df,
            means=baseline_means,
            row_selector=lambda r: is_en_subspace_model_row(str(r["model_a"]), str(r["model_b"])),
            metric_to_baseline_metric={m: m for m in CORE_METRICS},
            checkpoint_getter=lambda r: checkpoint_from_model(str(r["model_a"])),
        )
        out.to_csv(path, index=False)
        print(f"Centered: {path}")

    # Pair-style artifacts.
    center_pair_csv(
        en_dir / "bli_negative_control_eval.csv",
        baseline_means,
        metric_map={
            "jaccard_at_k_mean": "jaccard_at_k_mean",
            "frobenius_similarity": "frobenius_cultural_similarity",
            "axis_abs_projection_diff_mean": "axis_abs_projection_diff_mean",
        },
    )

    center_framework_holdout_csv(en_dir / "bli_framework_holdout_eval.csv", baseline_means)


if __name__ == "__main__":
    main()
