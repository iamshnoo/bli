#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from shared_utils import load_tokenizer_and_model, resolve_device, score_completion

DEFAULT_MODELS = [
    "en_50m",
    "en_100m",
    "en_zh_a",
    "en_fr_a",
    "en_fas_a",
    "en_nld_a",
    "en_ukr_a",
    "en_bul_a",
    "en_ind_a",
    "en_deu_a",
]
TARGET_COUNTRIES = {
    "zh": "China",
    "fr": "France",
    "fas": "Iran",
    "nld": "Netherlands",
    "ukr": "Ukraine",
    "bul": "Bulgaria",
    "ind": "Indonesia",
    "deu": "Germany",
}
COUNTRY_ALIASES = {
    "persia": "Iran",
    "iran/persia": "Iran",
    "islamic republic of iran": "Iran",
    "iran islamic republic of": "Iran",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "deutschland": "Germany",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "hong kong sar": "Hong Kong",
}
COUNTRY_TO_CONTINENT = {
    "China": "Asia",
    "France": "Europe",
    "Iran": "Asia",
    "Netherlands": "Europe",
    "Ukraine": "Europe",
    "Bulgaria": "Europe",
    "Indonesia": "Asia",
    "Germany": "Europe",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BLI checkpoints on the WorldValuesBench target-country slice.")
    p.add_argument(
        "--data-path",
        "--worldvaluesbench-root",
        dest="worldvaluesbench_root",
        type=Path,
        default=Path(os.environ.get("WORLDVALUESBENCH_ROOT", "external/WorldValuesBench")),
        help="Prepared WorldValuesBench root containing WorldValuesBench/{probe,full} and dataset_construction.",
    )
    p.add_argument("--model-root", type=Path, default=Path("models/hf"))
    p.add_argument("--models", nargs="+", default=[",".join(DEFAULT_MODELS)], help="Model names under --model-root; accepts spaces and/or commas.")
    p.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Comma-separated countries to keep. Unavailable countries are reported and skipped.",
    )
    p.add_argument(
        "--country-language-map",
        type=Path,
        default=None,
        help="Optional JSON mapping language/model keys to country names; overrides default BLI target-country map.",
    )
    p.add_argument("--max-items-per-country", type=int, default=0, help="0 means the full available target-country slice.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None, help="Per-item prediction CSV path.")
    p.add_argument("--summary-json", type=Path, default=None, help="Summary JSON path.")
    p.add_argument("--emd-by-group-csv", type=Path, default=None, help="Per-model/country/question EMD CSV path.")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dry-run", action="store_true", help="Only load/filter the benchmark and write selected items.")
    return p.parse_args()


def split_values(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)
    out: list[str] = []
    for value in raw_values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def normalize_key(x: object) -> str:
    return re.sub(r"[^a-z0-9/]+", " ", str(x).strip().lower()).strip()


def normalize_country(x: object) -> str:
    text = str(x).strip()
    return COUNTRY_ALIASES.get(normalize_key(text), text)


def load_target_country_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(TARGET_COUNTRIES)
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            if not value:
                continue
            value = value[0]
        out[str(key)] = normalize_country(value)
    return out


def softmax(xs: list[float]) -> list[float]:
    arr = np.array(xs, dtype=float)
    if not np.isfinite(arr).any():
        return [float("nan")] * len(xs)
    arr = arr - np.nanmax(arr)
    exp = np.exp(arr)
    denom = float(np.nansum(exp))
    if denom <= 0 or not np.isfinite(denom):
        return [float("nan")] * len(xs)
    return [float(x) for x in exp / denom]


def wasserstein_equal_weight(values_a: list[float], values_b: list[float]) -> float:
    if not values_a or not values_b:
        return float("nan")
    a = np.sort(np.array(values_a, dtype=float))
    b = np.sort(np.array(values_b, dtype=float))
    if len(a) == len(b):
        return float(np.mean(np.abs(a - b)))
    n = max(len(a), len(b))
    q = (np.arange(n) + 0.5) / n
    qa = np.quantile(a, q, method="linear")
    qb = np.quantile(b, q, method="linear")
    return float(np.mean(np.abs(qa - qb)))


def load_worldvaluebench_items(
    root: Path,
    countries: list[str],
    max_items_per_country: int,
    seed: int,
) -> pd.DataFrame:
    wvb_root = root / "WorldValuesBench"
    probe_path = wvb_root / "probe" / "samples.tsv"
    full_value_path = wvb_root / "full" / "full_value_qa.tsv"
    full_demo_path = wvb_root / "full" / "full_demographic_qa.tsv"
    value_q_path = root / "dataset_construction" / "probe_set_construction" / "value_questions.json"
    qmeta_path = root / "dataset_construction" / "question_metadata.json"
    required = [probe_path, full_value_path, full_demo_path, value_q_path, qmeta_path]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing WorldValuesBench files:\n" + "\n".join(missing))

    probe = pd.read_csv(probe_path, sep="\t")
    full_demo = (
        pd.read_csv(full_demo_path, sep="\t", usecols=["D_INTERVIEW", "B_COUNTRY"], low_memory=False)
        .drop_duplicates("D_INTERVIEW")
    )
    full_value = pd.read_csv(full_value_path, sep="\t", low_memory=False).set_index("D_INTERVIEW")
    with value_q_path.open("r", encoding="utf-8") as f:
        value_questions = json.load(f)
    with qmeta_path.open("r", encoding="utf-8") as f:
        question_meta = json.load(f)

    wanted = {normalize_country(c) for c in countries}
    joined = probe.merge(full_demo, on="D_INTERVIEW", how="left")
    joined["country"] = joined["B_COUNTRY"].map(normalize_country)
    available = set(joined["country"].dropna().unique())
    unavailable = sorted(wanted - available)
    if unavailable:
        print(f"[!] WorldValuesBench has no probe rows for requested countries: {', '.join(unavailable)}")
    joined = joined[joined["country"].isin(wanted)].copy()

    rows = []
    skipped = 0
    for source_index, row in joined.iterrows():
        qid = str(row["Question"]).strip()
        participant = row["D_INTERVIEW"]
        if qid not in full_value.columns or participant not in full_value.index:
            skipped += 1
            continue
        gold = full_value.at[participant, qid]
        if gold is None or (isinstance(gold, float) and math.isnan(gold)):
            skipped += 1
            continue
        try:
            gold_i = int(float(gold))
        except Exception:
            skipped += 1
            continue
        meta = question_meta.get(qid, {})
        try:
            min_i = int(meta["answer_scale_min"])
            max_i = int(meta["answer_scale_max"])
        except Exception:
            skipped += 1
            continue
        if gold_i < min_i or gold_i > max_i or max_i <= min_i:
            skipped += 1
            continue
        options = [str(x) for x in range(min_i, max_i + 1)]
        q_text = str(value_questions.get(qid, meta.get("question", qid))).strip()
        country = str(row["country"])
        continent = COUNTRY_TO_CONTINENT.get(country, str(row.get("Continent", "")).strip())
        prompt = (
            f"{q_text}\n\n"
            f"Respondent profile: Country={country}; Continent={continent}; "
            f"Settlement={row.get('Urban / Rural')}; Education={row.get('Education')}.\n\n"
            f"Answer with one number from {min_i} to {max_i}:"
        )
        rows.append(
            {
                "item_id": f"{qid}:{int(participant)}",
                "question_key": qid,
                "question_category": row.get("Question Category"),
                "country": country,
                "continent": continent,
                "participant_id": int(participant),
                "source_index": int(source_index),
                "question": q_text,
                "prompt": prompt,
                "options": json.dumps(options),
                "min_option": min_i,
                "max_option": max_i,
                "gold_answer": str(gold_i),
                "gold_norm": (gold_i - min_i) / (max_i - min_i),
            }
        )
    out = pd.DataFrame(rows).sort_values(["country", "question_key", "participant_id"]).reset_index(drop=True)
    if max_items_per_country > 0 and not out.empty:
        sampled = []
        for _, sub in out.groupby("country", sort=False):
            sampled.append(
                sub.sample(n=min(max_items_per_country, len(sub)), random_state=seed).sort_values(
                    ["question_key", "participant_id"]
                )
            )
        out = pd.concat(sampled, ignore_index=True) if sampled else out.iloc[0:0].copy()
    print(
        f"[WorldValuesBench] selected_rows={len(out)} countries={out['country'].nunique() if not out.empty else 0} skipped={skipped}"
    )
    return out


def score_model(model_name: str, items: pd.DataFrame, model_root: Path, device) -> pd.DataFrame:
    tok, model = load_tokenizer_and_model(model_root / model_name, device)
    rows = []
    try:
        for _, item in items.iterrows():
            options = json.loads(item["options"])
            scores = [score_completion(model, tok, item["prompt"], f" {opt}", device) for opt in options]
            probs = softmax(scores)
            best_idx = int(np.nanargmax(np.array(scores, dtype=float)))
            pred = str(options[best_idx])
            min_i = int(item["min_option"])
            max_i = int(item["max_option"])
            pred_norm = (int(pred) - min_i) / (max_i - min_i)
            expected_norm = float(
                np.nansum([(int(opt) - min_i) / (max_i - min_i) * prob for opt, prob in zip(options, probs)])
            )
            out = item.to_dict()
            out.update(
                {
                    "model": model_name,
                    "scores_json": json.dumps({opt: float(score) for opt, score in zip(options, scores)}),
                    "probs_json": json.dumps({opt: float(prob) for opt, prob in zip(options, probs)}),
                    "pred_answer": pred,
                    "pred_norm": pred_norm,
                    "expected_norm": expected_norm,
                    "correct": int(pred == str(item["gold_answer"])),
                    "abs_error_norm": abs(pred_norm - float(item["gold_norm"])),
                    "expected_abs_error_norm": abs(expected_norm - float(item["gold_norm"])),
                }
            )
            rows.append(out)
    finally:
        del model
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def build_model_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, country), sub in pred.groupby(["model", "country"]):
        grouped = []
        for (question_key, _country), g in sub.groupby(["question_key", "country"]):
            grouped.append(
                {
                    "question_key": question_key,
                    "emd_hard": wasserstein_equal_weight(g["gold_norm"].tolist(), g["pred_norm"].tolist()),
                    "emd_expected": wasserstein_equal_weight(g["gold_norm"].tolist(), g["expected_norm"].tolist()),
                    "n": len(g),
                }
            )
        gdf = pd.DataFrame(grouped)
        rows.append(
            {
                "model": model,
                "country": country,
                "n_items": int(len(sub)),
                "n_questions": int(sub["question_key"].nunique()),
                "accuracy": float(sub["correct"].mean()),
                "mean_abs_error_norm": float(sub["abs_error_norm"].mean()),
                "mean_expected_abs_error_norm": float(sub["expected_abs_error_norm"].mean()),
                "overall_emd_hard": float(np.average(gdf["emd_hard"], weights=gdf["n"])) if not gdf.empty else float("nan"),
                "overall_emd_expected": float(np.average(gdf["emd_expected"], weights=gdf["n"])) if not gdf.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "country"])


def build_emd_by_group(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, country, continent, question_key), g in pred.groupby(["model", "country", "continent", "question_key"]):
        rows.append(
            {
                "model": model,
                "country": country,
                "continent": continent,
                "question_key": question_key,
                "n": int(len(g)),
                "emd_hard": wasserstein_equal_weight(g["gold_norm"].tolist(), g["pred_norm"].tolist()),
                "emd_expected": wasserstein_equal_weight(g["gold_norm"].tolist(), g["expected_norm"].tolist()),
                "correct_answers_scaled": json.dumps([float(x) for x in g["gold_norm"].tolist()]),
                "predicted_answers_scaled": json.dumps([float(x) for x in g["pred_norm"].tolist()]),
                "expected_answers_scaled": json.dumps([float(x) for x in g["expected_norm"].tolist()]),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "country", "question_key"]) if rows else pd.DataFrame(rows)


def model_language(model_name: str) -> str | None:
    parts = model_name.split("_")
    if len(parts) >= 2 and parts[0] == "en" and parts[1] not in {"50m", "100m"}:
        return parts[1]
    return None


def build_pair_summary(pred: pd.DataFrame, model_summary: pd.DataFrame, target_country_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    bilingual_models = sorted(m for m in pred["model"].unique() if model_language(str(m)) in target_country_map)
    for baseline in ["en_50m", "en_100m"]:
        if baseline not in set(pred["model"]):
            continue
        for model_b in bilingual_models:
            lang = model_language(str(model_b))
            country = target_country_map.get(str(lang))
            if not country:
                continue
            a = pred[(pred["model"] == baseline) & (pred["country"] == country)]
            b = pred[(pred["model"] == model_b) & (pred["country"] == country)]
            merged = a.merge(
                b,
                on=["item_id", "question_key", "country", "participant_id"],
                suffixes=("_baseline", "_bilingual"),
            )
            if merged.empty:
                rows.append(
                    {
                        "baseline": baseline,
                        "model_b": model_b,
                        "language": lang,
                        "country": country,
                        "n_items": 0,
                        "available": 0,
                    }
                )
                continue
            a_sum = model_summary[(model_summary["model"] == baseline) & (model_summary["country"] == country)]
            b_sum = model_summary[(model_summary["model"] == model_b) & (model_summary["country"] == country)]
            rows.append(
                {
                    "baseline": baseline,
                    "model_b": model_b,
                    "language": lang,
                    "country": country,
                    "n_items": int(len(merged)),
                    "available": 1,
                    "mean_abs_hard_prediction_shift": float(
                        np.mean(np.abs(merged["pred_norm_bilingual"] - merged["pred_norm_baseline"]))
                    ),
                    "mean_abs_expected_prediction_shift": float(
                        np.mean(np.abs(merged["expected_norm_bilingual"] - merged["expected_norm_baseline"]))
                    ),
                    "baseline_emd_expected": float(a_sum["overall_emd_expected"].iloc[0]) if not a_sum.empty else float("nan"),
                    "bilingual_emd_expected": float(b_sum["overall_emd_expected"].iloc[0]) if not b_sum.empty else float("nan"),
                    "delta_emd_expected_bilingual_minus_baseline": (
                        float(b_sum["overall_emd_expected"].iloc[0] - a_sum["overall_emd_expected"].iloc[0])
                        if not a_sum.empty and not b_sum.empty
                        else float("nan")
                    ),
                }
            )
    columns = [
        "baseline",
        "model_b",
        "language",
        "country",
        "n_items",
        "available",
        "mean_abs_hard_prediction_shift",
        "mean_abs_expected_prediction_shift",
        "baseline_emd_expected",
        "bilingual_emd_expected",
        "delta_emd_expected_bilingual_minus_baseline",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns).sort_values(["baseline", "country", "model_b"])


def main() -> None:
    args = parse_args()
    models = split_values(args.models)
    target_country_map = load_target_country_map(args.country_language_map)
    countries = [normalize_country(x) for x in split_values(args.countries)] if args.countries else list(target_country_map.values())
    if args.out_dir is None and (args.out_csv is None or args.summary_json is None):
        raise ValueError("Provide --out-dir, or both --out-csv and --summary-json.")
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_items_path = (
        args.out_dir / "worldvaluebench_selected_items.csv"
        if args.out_dir is not None
        else args.out_csv.parent / "worldvaluebench_selected_items.csv"
    )
    prediction_path = args.out_csv or (args.out_dir / "worldvaluebench_predictions.csv")
    model_summary_path = args.out_dir / "worldvaluebench_model_summary.csv" if args.out_dir is not None else None
    pair_summary_path = args.out_dir / "worldvaluebench_pair_summary.csv" if args.out_dir is not None else None
    emd_by_group_path = args.emd_by_group_csv or (args.out_dir / "worldvaluebench_emd_by_group.csv" if args.out_dir is not None else None)
    items = load_worldvaluebench_items(args.worldvaluesbench_root, countries, args.max_items_per_country, args.seed)
    selected_items_path.parent.mkdir(parents=True, exist_ok=True)
    items.to_csv(selected_items_path, index=False)
    if args.dry_run:
        print(f"Wrote: {selected_items_path}")
        return

    device = resolve_device(args.device)
    pred_parts = []
    for model_name in models:
        print(f"[WorldValuesBench] scoring {model_name} on {len(items)} items")
        part = score_model(model_name, items, args.model_root, device)
        pred_parts.append(part)
        partial = pd.concat(pred_parts, ignore_index=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        partial.to_csv(prediction_path, index=False)

    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(prediction_path, index=False)
    model_summary = build_model_summary(pred)
    if model_summary_path is not None:
        model_summary.to_csv(model_summary_path, index=False)
    emd_by_group = build_emd_by_group(pred)
    if emd_by_group_path is not None:
        emd_by_group_path.parent.mkdir(parents=True, exist_ok=True)
        emd_by_group.to_csv(emd_by_group_path, index=False)
    pair_summary = build_pair_summary(pred, model_summary, target_country_map)
    if pair_summary_path is not None:
        pair_summary.to_csv(pair_summary_path, index=False)
    if args.summary_json is not None:
        summary_payload = {
            "data_path": str(args.worldvaluesbench_root),
            "models": models,
            "requested_countries": countries,
            "selected_items": int(len(items)),
            "selected_countries": sorted(items["country"].unique().tolist()) if not items.empty else [],
            "prediction_csv": str(prediction_path),
            "emd_by_group_csv": str(emd_by_group_path) if emd_by_group_path is not None else None,
            "summary": model_summary.to_dict(orient="records"),
            "pair_summary": pair_summary.to_dict(orient="records"),
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Wrote: {prediction_path}")
    if model_summary_path is not None:
        print(f"Wrote: {model_summary_path}")
    if emd_by_group_path is not None:
        print(f"Wrote: {emd_by_group_path}")
    if pair_summary_path is not None:
        print(f"Wrote: {pair_summary_path}")
    if args.summary_json is not None:
        print(f"Wrote: {args.summary_json}")


if __name__ == "__main__":
    main()
