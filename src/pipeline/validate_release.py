#!/usr/bin/env python3
"""Validate the tracked manifests, lineage table, artifacts, and public models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the BLI repository release")
    parser.add_argument(
        "--check-hub",
        action="store_true",
        help="Also verify that all model IDs resolve publicly on the Hugging Face Hub.",
    )
    return parser.parse_args()


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_manifests(errors: list[str]) -> set[str]:
    train = load_json("configs/train_manifest.json")
    data = load_json("configs/data_manifest.json")
    test = load_json("configs/test_manifest.json")
    hub = load_json("configs/models/models_hub.json")

    runs = train["runs"]
    run_ids = [run["run_id"] for run in runs]
    check(train["run_count"] == len(runs) == 40, "train manifest must contain 40 runs", errors)
    check(len(run_ids) == len(set(run_ids)), "training run IDs must be unique", errors)
    check(train["published_model_count"] == 40, "all 40 runs must be marked as published", errors)
    check(set(run_ids) == set(hub), "Hub registry keys must exactly match the training runs", errors)
    check(
        all(run["published_model_hf_id"] == hub[run["run_id"]] for run in runs),
        "per-run Hugging Face IDs must match the Hub registry",
        errors,
    )
    check(all(value == f"iamshnoo/{key}" for key, value in hub.items()), "unexpected Hub model ID", errors)

    training_types = Counter(run["training_type"] for run in runs)
    check(
        training_types == {"monolingual": 18, "same_language_seed_control": 6, "bilingual_alternating": 16},
        f"unexpected training-run breakdown: {dict(training_types)}",
        errors,
    )

    for run in runs:
        if "stage_config" in run:
            check((ROOT / run["stage_config"]).is_file(), f"missing {run['stage_config']}", errors)

    check(len(data["pretraining_datasets"]) == 11, "data manifest must list 9 corpora and 2 English partitions", errors)
    check(data["model_assets"]["published_model_count"] == 40, "data manifest model count must be 40", errors)
    check(test["evaluation_suite_count"] == len(test["evaluation_suites"]) == 18, "test manifest must contain 18 suites", errors)
    suite_ids = [suite["id"] for suite in test["evaluation_suites"]]
    check(len(suite_ids) == len(set(suite_ids)), "evaluation suite IDs must be unique", errors)

    return set(hub.values())


def validate_stages(errors: list[str]) -> None:
    paths = sorted((ROOT / "configs/stages").glob("*.json"))
    check(len(paths) == 16, "expected 16 bilingual stage files", errors)
    expected_starts = list(range(1, 3001, 50))
    for path in paths:
        stages = json.loads(path.read_text(encoding="utf-8"))
        starts = [stage["start_training_step"] for stage in stages]
        check(len(stages) == 60, f"{path.relative_to(ROOT)} must contain 60 stages", errors)
        check(starts == expected_starts, f"{path.relative_to(ROOT)} has incorrect stage boundaries", errors)
        names = [stage["name"].split()[0] for stage in stages]
        check(names == ["EN" if index % 2 == 0 else "L2" for index in range(60)], f"{path.relative_to(ROOT)} does not alternate EN/L2", errors)


def validate_lineage(errors: list[str]) -> None:
    path = ROOT / "data/probes/evaluation_data_lineage.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["record_type"] for row in rows)
    expected = {"concept": 9000, "axis_endpoint": 900, "alignment_anchor": 3000, "negative_control": 100}
    check(len(rows) == 13000, "lineage CSV must contain 13,000 rows", errors)
    check(counts == expected, f"unexpected lineage counts: {dict(counts)}", errors)
    ids = [row["record_id"] for row in rows]
    check(len(ids) == len(set(ids)), "lineage record IDs must be unique", errors)
    axis_rows = [row for row in rows if row["record_type"] == "axis_endpoint"]
    check(all(row["source_urls"] for row in axis_rows), "every axis endpoint row must include its sources", errors)
    check(
        Counter((row["axis_endpoint"], row["axis_polarity"]) for row in axis_rows)
        == {("endpoint_1", "-1"): 450, ("endpoint_2", "1"): 450},
        "axis endpoint directions are incomplete",
        errors,
    )


def validate_artifacts(errors: list[str]) -> None:
    manifest = load_json("artifacts/manifest.json")
    files = manifest["files"]
    check(manifest["file_count"] == len(files) == 100, "artifact manifest must contain 100 files", errors)
    for item in files:
        path = ROOT / item["path"]
        check(path.is_file(), f"missing artifact {item['path']}", errors)
        if not path.is_file():
            continue
        check(path.stat().st_size == item["bytes"], f"size mismatch for {item['path']}", errors)
        check(digest(path) == item["sha256"], f"SHA-256 mismatch for {item['path']}", errors)
        if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            check("/scratch/" not in text and "/home/" not in text, f"local absolute path in {item['path']}", errors)


def validate_hub(expected_ids: set[str], errors: list[str]) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        errors.append("huggingface_hub is required for --check-hub")
        return
    visible = {model.id: model for model in HfApi().list_models(author="iamshnoo", limit=1000)}
    missing = sorted(expected_ids - set(visible))
    private = sorted(model_id for model_id in expected_ids if model_id in visible and visible[model_id].private)
    check(not missing, f"models missing from the Hub: {missing}", errors)
    check(not private, f"models are not public: {private}", errors)


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    expected_hub_ids = validate_manifests(errors)
    validate_stages(errors)
    validate_lineage(errors)
    validate_artifacts(errors)
    if args.check_hub:
        validate_hub(expected_hub_ids, errors)
    if errors:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(errors))
    suffix = " including 40 public Hub models" if args.check_hub else ""
    print(f"Release validation passed{suffix}.")


if __name__ == "__main__":
    main()
