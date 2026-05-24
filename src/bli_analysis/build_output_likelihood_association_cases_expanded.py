#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expanded output-likelihood association theory cases.")
    parser.add_argument("--probe-set", type=Path, default=Path("data/probes/probe_sets.json"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/probes/output_likelihood_association_cases_expanded.csv"),
    )
    parser.add_argument("--max-probes-per-axis", type=int, default=4)
    return parser.parse_args()


def build_cases(probe_set: dict, max_probes_per_axis: int) -> list[dict[str, object]]:
    endpoint_words = {word for axis in probe_set["semantic_axes"] for word in axis}
    categories = probe_set["cultural_probe_categories"]
    cultural_words = probe_set["cultural_probe_words"]

    rows: list[dict[str, object]] = []
    for meta in probe_set["semantic_axis_metadata"]:
        category = meta["category"]
        left = meta["endpoint_1"]
        right = meta["endpoint_2"]
        eligible = [
            word
            for word in cultural_words
            if categories.get(word) == category and word not in endpoint_words
        ]
        for probe_word in eligible[:max_probes_per_axis]:
            rows.append(
                {
                    "probe": probe_word,
                    "left_endpoint": left,
                    "right_endpoint": right,
                    "axis_index": meta["index"],
                    "category": category,
                    "case_source": "probe_sets_category_match_v1",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    probe_set = json.loads(args.probe_set.read_text(encoding="utf-8"))
    rows = build_cases(probe_set, args.max_probes_per_axis)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "probe",
                "left_endpoint",
                "right_endpoint",
                "axis_index",
                "category",
                "case_source",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cases to {args.out_csv}")


if __name__ == "__main__":
    main()
