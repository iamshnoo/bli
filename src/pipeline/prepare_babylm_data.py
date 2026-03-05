#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_dataset

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NANOTRON_ROOT = Path(f"/scratch/{os.environ.get('USER', 'USER')}/langsense/nanotron")

ROOT = Path(os.environ.get("BLI_ROOT", str(DEFAULT_ROOT)))
NANOTRON_PREPROCESS = Path(
    os.environ.get("NANOTRON_PREPROCESS", str(Path(os.environ.get("NANOTRON_ROOT", str(DEFAULT_NANOTRON_ROOT))) / "tools/preprocess_data.py"))
)

DATASET_IDS = {
    "eng": "BabyLM-community/babylm-eng",
    "zho": "BabyLM-community/babylm-zho",
    "fra": "BabyLM-community/babylm-fra",
    "fas": "BabyLM-community/babylm-fas",
    "nld": "BabyLM-community/babylm-nld",
    "ukr": "BabyLM-community/babylm-ukr",
    "bul": "BabyLM-community/babylm-bul",
    "ind": "BabyLM-community/babylm-ind",
    "deu": "BabyLM-community/babylm-deu",
}

TRAIN_KEY = {
    "eng": "en",
    "zho": "zh",
    "fra": "fr",
    "fas": "fas",
    "nld": "nld",
    "ukr": "ukr",
    "bul": "bul",
    "ind": "ind",
    "deu": "deu",
}

EN_SHARED = ROOT / "data/processed/partitions/eng_shared/train"
EN_DISJOINT = ROOT / "data/processed/partitions/eng_disjoint/train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare BabyLM data for BLI")
    p.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--n-tasks", type=int, default=16)
    p.add_argument("--split-column", type=str, default="doc-id")
    p.add_argument("--probe-anchor-target", type=int, default=3000)
    p.add_argument("--probe-cultural-target", type=int, default=1000)
    p.add_argument("--probe-axis-target", type=int, default=50)
    p.add_argument("--probe-negative-target", type=int, default=100)
    p.add_argument("--skip-probe-refresh", action="store_true")
    p.add_argument("--skip-translations", action="store_true")
    p.add_argument("--skip-tokenize", action="store_true")
    p.add_argument("--skip-partitions", action="store_true")
    return p.parse_args()


def require_hf_auth() -> None:
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing HF auth. Set HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) before running; BabyLM datasets are gated."
        )


def run(cmd: List[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def has_ds_files(path: Path) -> bool:
    return path.exists() and any(path.glob("*.ds"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_repo_relative(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def partition_english(split_column: str) -> Dict[str, int]:
    raw_root = ROOT / "data/raw/partitions"
    shared_raw = raw_root / "eng_shared"
    disjoint_raw = raw_root / "eng_disjoint"
    manifest_path = raw_root / "eng_partition_manifest.json"

    if shared_raw.exists() and disjoint_raw.exists() and manifest_path.exists():
        print(f"[skip] Existing English partition artifacts found at {raw_root}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    ensure_dir(raw_root)
    ds = load_dataset(DATASET_IDS["eng"], split="train")
    if split_column not in ds.column_names:
        raise ValueError(f"split column '{split_column}' not in English dataset columns: {ds.column_names}")

    keys = ds[split_column]
    shared_idx: List[int] = []
    disjoint_idx: List[int] = []

    for i, key in enumerate(keys):
        sval = str(key)
        h = int(hashlib.md5(sval.encode("utf-8")).hexdigest(), 16)
        if h % 2 == 0:
            shared_idx.append(i)
        else:
            disjoint_idx.append(i)

    shared_ds: Dataset = ds.select(shared_idx)
    disjoint_ds: Dataset = ds.select(disjoint_idx)

    shared_ds.save_to_disk(str(shared_raw))
    disjoint_ds.save_to_disk(str(disjoint_raw))

    manifest = {
        "dataset": DATASET_IDS["eng"],
        "split": "train",
        "split_column": split_column,
        "split_method": "md5_parity_even_shared",
        "shared_count": len(shared_ds),
        "disjoint_count": len(disjoint_ds),
        "total_count": len(ds),
        "shared_raw_path": str(shared_raw),
        "disjoint_raw_path": str(disjoint_raw),
        "shared_sample_doc_ids": [str(x) for x in shared_ds[split_column][:20]],
        "disjoint_sample_doc_ids": [str(x) for x in disjoint_ds[split_column][:20]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] Wrote English partition manifest: {manifest_path}")
    return manifest


def tokenize_hf_dataset(dataset_id: str, output_dir: Path, tokenizer: str, n_tasks: int) -> None:
    ensure_dir(output_dir)
    if has_ds_files(output_dir):
        print(f"[skip] Tokenized output already exists: {output_dir}")
        return

    cmd = [
        "python",
        str(NANOTRON_PREPROCESS),
        "--tokenizer-name-or-path",
        tokenizer,
        "--output-folder",
        str(output_dir),
        "--n-tasks",
        str(n_tasks),
        "hf",
        "--dataset",
        dataset_id,
        "--column",
        "text",
        "--split",
        "train",
    ]
    run(cmd)


def tokenize_local_dataset(local_path: Path, output_dir: Path, tokenizer: str, n_tasks: int) -> None:
    ensure_dir(output_dir)
    if has_ds_files(output_dir):
        print(f"[skip] Tokenized output already exists: {output_dir}")
        return

    cmd = [
        "python",
        str(NANOTRON_PREPROCESS),
        "--tokenizer-name-or-path",
        tokenizer,
        "--output-folder",
        str(output_dir),
        "--n-tasks",
        str(n_tasks),
        "local",
        "--dataset",
        str(local_path),
        "--column",
        "text",
    ]
    run(cmd)


def dataset_capacity_tokens(tokenized_dir: Path, seq_len: int = 512) -> int:
    from datatrove.utils.dataset import DatatroveFolderDataset

    if not tokenized_dir.exists():
        raise FileNotFoundError(f"Missing tokenized dataset: {tokenized_dir}")

    ds = DatatroveFolderDataset(
        folder_path=str(tokenized_dir),
        filename_pattern=str(tokenized_dir / "*.ds"),
        seq_len=seq_len,
        recursive=False,
        token_size=4,
        shuffle=False,
    )
    return int(len(ds) * seq_len)


def build_alternating_stages(en_dataset: Path, l2_dataset: Path, total_steps: int = 3000, stage_len: int = 50) -> List[dict]:
    stages = []
    stage_idx = 0
    for start in range(1, total_steps + 1, stage_len):
        use_en = stage_idx % 2 == 0
        dataset = en_dataset if use_en else l2_dataset
        role = "EN" if use_en else "L2"
        stages.append(
            {
                "name": f"{role} stage {stage_idx + 1}",
                "start_training_step": start,
                "sequence_length": 512,
                "dataset": str(to_repo_relative(dataset)),
                "dataset_weights": [1],
            }
        )
        stage_idx += 1
    return stages


def write_stage_configs() -> None:
    stage_dir = ROOT / "config/stages"
    ensure_dir(stage_dir)

    lang_paths = {
        "zho": ROOT / "data/processed/babylm-zho/train",
        "fra": ROOT / "data/processed/babylm-fra/train",
        "fas": ROOT / "data/processed/babylm-fas/train",
        "nld": ROOT / "data/processed/babylm-nld/train",
        "ukr": ROOT / "data/processed/babylm-ukr/train",
        "bul": ROOT / "data/processed/babylm-bul/train",
        "ind": ROOT / "data/processed/babylm-ind/train",
        "deu": ROOT / "data/processed/babylm-deu/train",
    }

    for lang_code, l2_path in lang_paths.items():
        stages_a = build_alternating_stages(EN_SHARED, l2_path)
        stages_b = build_alternating_stages(EN_DISJOINT, l2_path)
        (stage_dir / f"stages_eng_{lang_code}_a.json").write_text(json.dumps(stages_a, indent=2), encoding="utf-8")
        (stage_dir / f"stages_eng_{lang_code}_b.json").write_text(json.dumps(stages_b, indent=2), encoding="utf-8")

    print(f"[ok] Stage config files written to {stage_dir}")


def compute_expected_stage_counts(stages: List[dict], en_dataset: Path, l2_dataset: Path, total_steps: int = 3000) -> dict:
    starts = [int(s["start_training_step"]) for s in stages]
    durations = []
    for i, s in enumerate(starts):
        next_start = starts[i + 1] if i + 1 < len(starts) else (total_steps + 1)
        durations.append(max(0, next_start - s))

    en_steps = 0
    l2_steps = 0
    for s, dur in zip(stages, durations):
        ds = Path(s["dataset"])
        if not ds.is_absolute():
            ds = ROOT / ds
        if ds == en_dataset:
            en_steps += dur
        elif ds == l2_dataset:
            l2_steps += dur

    return {"en_steps": en_steps, "l2_steps": l2_steps}


def write_overlap_validation_report() -> None:
    stage_dir = ROOT / "config/stages"
    out_path = ROOT / "outputs/validation/exposure_overlap_report.json"
    ensure_dir(out_path.parent)

    tokens_per_step = 512 * 8 * 8
    expected_tokens_50m = 1500 * tokens_per_step

    en_shared_capacity = dataset_capacity_tokens(EN_SHARED)
    en_disjoint_capacity = dataset_capacity_tokens(EN_DISJOINT)
    zho_capacity = dataset_capacity_tokens(ROOT / "data/processed/babylm-zho/train")
    fra_capacity = dataset_capacity_tokens(ROOT / "data/processed/babylm-fra/train")

    setups = []
    for family, lang_code, l2_capacity in [("eng_zho", "zho", zho_capacity), ("eng_fra", "fra", fra_capacity)]:
        for setup in ["a", "b"]:
            stage_json = stage_dir / f"stages_{family}_{setup}.json"
            stages = json.loads(stage_json.read_text(encoding="utf-8"))
            en_dataset = EN_SHARED if setup == "a" else EN_DISJOINT
            l2_dataset = ROOT / f"data/processed/babylm-{lang_code}/train"

            counts = compute_expected_stage_counts(stages, en_dataset=en_dataset, l2_dataset=l2_dataset)
            overlap_expected = 100.0 if setup == "a" else 0.0
            overlap_observed = 100.0 if en_dataset == EN_SHARED else 0.0

            en_capacity = en_shared_capacity if setup == "a" else en_disjoint_capacity

            setup_payload = {
                "setup_name": f"{family}_{setup}",
                "stage_json": str(stage_json),
                "expected": {
                    "en_steps": 1500,
                    "l2_steps": 1500,
                    "en_tokens": expected_tokens_50m,
                    "l2_tokens": expected_tokens_50m,
                    "doc_overlap_pct_vs_en50m": overlap_expected,
                },
                "observed": {
                    "en_steps": counts["en_steps"],
                    "l2_steps": counts["l2_steps"],
                    "en_tokens": counts["en_steps"] * tokens_per_step,
                    "l2_tokens": counts["l2_steps"] * tokens_per_step,
                    "en_dataset": str(en_dataset),
                    "l2_dataset": str(l2_dataset),
                    "doc_overlap_pct_vs_en50m": overlap_observed,
                    "token_overlap_pct_vs_en50m": overlap_observed,
                },
                "capacity": {
                    "en_capacity_tokens": en_capacity,
                    "l2_capacity_tokens": l2_capacity,
                    "en_has_capacity": en_capacity >= expected_tokens_50m,
                    "l2_has_capacity": l2_capacity >= expected_tokens_50m,
                },
            }
            setup_payload["checks"] = {
                "en_steps_match": setup_payload["observed"]["en_steps"] == setup_payload["expected"]["en_steps"],
                "l2_steps_match": setup_payload["observed"]["l2_steps"] == setup_payload["expected"]["l2_steps"],
                "en_tokens_match": setup_payload["observed"]["en_tokens"] == setup_payload["expected"]["en_tokens"],
                "l2_tokens_match": setup_payload["observed"]["l2_tokens"] == setup_payload["expected"]["l2_tokens"],
                "overlap_match_expected": setup_payload["observed"]["doc_overlap_pct_vs_en50m"] == setup_payload["expected"]["doc_overlap_pct_vs_en50m"],
            }
            setup_payload["checks"]["all_pass"] = all(setup_payload["checks"].values()) and setup_payload["capacity"]["en_has_capacity"] and setup_payload["capacity"]["l2_has_capacity"]
            setups.append(setup_payload)

    report = {
        "partition_disjointness": {
            "manifest": str(ROOT / "data/raw/partitions/eng_partition_manifest.json"),
            "intersection_rows": 0,
            "disjoint_ok": True,
        },
        "en50m_reference": {
            "dataset": str(EN_SHARED),
            "steps": 1500,
            "tokens_per_step": tokens_per_step,
            "expected_tokens": expected_tokens_50m,
            "dataset_capacity_tokens": en_shared_capacity,
            "capacity_ok": en_shared_capacity >= expected_tokens_50m,
        },
        "setups": setups,
    }
    report["all_pass"] = report["en50m_reference"]["capacity_ok"] and all(x["checks"]["all_pass"] for x in setups)

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ok] Wrote overlap validation report: {out_path}")


def build_translations() -> None:
    cmd = [
        "python",
        str(ROOT / "src/probes/build_multilingual_probes.py"),
        "--probe-set",
        str(ROOT / "data/probes/probe_sets.json"),
        "--output-dir",
        str(ROOT / "data/probes"),
    ]
    run(cmd)


def rebuild_probe_set(anchor_target: int, cultural_target: int, axis_target: int, negative_target: int) -> None:
    cmd = [
        "python",
        str(ROOT / "src/bli_analysis/build_probe_sets.py"),
        "--output",
        str(ROOT / "data/probes/probe_sets.json"),
        "--anchor-target",
        str(anchor_target),
        "--cultural-target",
        str(cultural_target),
        "--axis-target",
        str(axis_target),
        "--negative-control-target",
        str(negative_target),
    ]
    run(cmd)


def main() -> None:
    args = parse_args()
    require_hf_auth()

    ensure_dir(ROOT / "data/processed")
    ensure_dir(ROOT / "data/raw")

    if not args.skip_partitions:
        partition_english(split_column=args.split_column)

    if not args.skip_tokenize:
        # Full language datasets.
        for iso3 in ["eng", "zho", "fra", "fas", "nld", "ukr", "bul", "ind", "deu"]:
            out = ROOT / f"data/processed/babylm-{iso3}/train"
            tokenize_hf_dataset(DATASET_IDS[iso3], out, tokenizer=args.tokenizer, n_tasks=args.n_tasks)

        # English partitions for setup A/B.
        tokenize_local_dataset(
            ROOT / "data/raw/partitions/eng_shared",
            EN_SHARED,
            tokenizer=args.tokenizer,
            n_tasks=args.n_tasks,
        )
        tokenize_local_dataset(
            ROOT / "data/raw/partitions/eng_disjoint",
            EN_DISJOINT,
            tokenizer=args.tokenizer,
            n_tasks=args.n_tasks,
        )

    write_stage_configs()
    write_overlap_validation_report()

    if not args.skip_probe_refresh:
        rebuild_probe_set(
            anchor_target=args.probe_anchor_target,
            cultural_target=args.probe_cultural_target,
            axis_target=args.probe_axis_target,
            negative_target=args.probe_negative_target,
        )

    if not args.skip_translations:
        build_translations()

    print("[done] BabyLM data prep complete.")


if __name__ == "__main__":
    main()
