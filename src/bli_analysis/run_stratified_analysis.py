#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Category-stratified BLI analysis")
    p.add_argument("--word-csv", type=Path, required=True)
    p.add_argument("--axis-csv", type=Path, required=True)
    p.add_argument("--probe-set", type=Path, default=Path("/scratch/amukher6/bli/data/probes/probe_sets.json"))
    p.add_argument("--out-csv", type=Path, required=True)
    return p.parse_args()


def infer_axis_group(axis: str) -> str:
    a = axis.lower()
    mapping = {
        "values_norms": ["honor", "shame", "duty", "freedom", "authority", "equality", "obedience", "autonomy", "hierarchy", "merit", "collective", "personal", "community", "self", "individual", "collectivism"],
        "food_cuisine": ["rice", "bread", "tea", "coffee", "spicy", "mild", "vegetarian", "meat"],
        "gender_roles": ["male", "female", "monogamy", "polygamy", "arranged", "chosen"],
        "religion_ritual": ["sacred", "secular", "religion", "science", "ceremony", "routine", "festive", "solemn"],
        "space_society": ["rural", "urban", "private", "public", "local", "global", "east", "west", "north", "south", "tradition", "modernity"],
    }
    for g, keys in mapping.items():
        if any(k in a for k in keys):
            return g
    return "other_axis"


def main() -> None:
    args = parse_args()
    probes = json.loads(args.probe_set.read_text(encoding="utf-8"))
    cat_map = probes.get("cultural_probe_categories", {})

    word_df = pd.read_csv(args.word_csv)
    axis_df = pd.read_csv(args.axis_csv)

    if not cat_map:
        # Backward-compatible fallback if probe file predates category annotations.
        cat_map = {w: "other_cultural" for w in word_df["word"].unique()}

    word_df["category"] = word_df["word"].map(cat_map).fillna("other_cultural")
    axis_df["category"] = axis_df["axis"].map(infer_axis_group)

    word_g = (
        word_df.groupby(["repr_type", "pair", "category"], as_index=False)
        .agg(
            n_words=("word", "count"),
            jaccard_divergence_mean=("jaccard_divergence", "mean"),
            jaccard_divergence_std=("jaccard_divergence", "std"),
        )
    )
    word_g["metric"] = "jaccard_divergence"

    axis_g = (
        axis_df.groupby(["repr_type", "pair", "category"], as_index=False)
        .agg(
            n_axes=("axis", "count"),
            mean_abs_projection_diff=("mean_abs_projection_diff", "mean"),
            std_abs_projection_diff=("mean_abs_projection_diff", "std"),
        )
    )
    axis_g["metric"] = "axis_abs_projection_diff"

    # Long-format output for downstream plotting.
    word_out = word_g.rename(columns={
        "n_words": "n_items",
        "jaccard_divergence_mean": "mean",
        "jaccard_divergence_std": "std",
    })[["repr_type", "pair", "metric", "category", "n_items", "mean", "std"]]

    axis_out = axis_g.rename(columns={
        "n_axes": "n_items",
        "mean_abs_projection_diff": "mean",
        "std_abs_projection_diff": "std",
    })[["repr_type", "pair", "metric", "category", "n_items", "mean", "std"]]

    out = pd.concat([word_out, axis_out], ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
