#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import fasttext
from datasets import load_dataset


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

TARGET_LABEL = {
    "eng": "en",
    "zho": "zh",
    "fra": "fr",
    "fas": "fa",
    "nld": "nl",
    "ukr": "uk",
    "bul": "bg",
    "ind": "id",
    "deu": "de",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk-level mixed-language audit for BabyBabelLM documents.")
    p.add_argument("--lid-model", type=Path, default=Path("outputs/validation/lid.176.ftz"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/validation/mixed_language_audit"))
    p.add_argument("--max-segments", type=int, default=12)
    p.add_argument("--chunk-chars", type=int, default=500)
    p.add_argument("--min-segment-chars", type=int, default=30)
    p.add_argument("--min-alpha", type=int, default=12)
    p.add_argument("--min-confidence", type=float, default=0.50)
    p.add_argument("--mixed-min-frac", type=float, default=0.20)
    p.add_argument("--mixed-min-segments", type=int, default=3)
    p.add_argument("--source-min-docs", type=int, default=100)
    p.add_argument("--dataset-batch-size", type=int, default=2048)
    return p.parse_args()


def clean_segment(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def alpha_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def fixed_chunks(text: str, chunk_chars: int) -> list[str]:
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def take_evenly(items: list[str], max_items: int) -> list[str]:
    if len(items) <= max_items:
        return items
    if max_items <= 1:
        return [items[0]]
    idxs = [round(i * (len(items) - 1) / (max_items - 1)) for i in range(max_items)]
    return [items[i] for i in idxs]


def candidate_segments(text: str, args: argparse.Namespace) -> list[str]:
    raw_parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > args.chunk_chars * 2:
            raw_parts.extend(fixed_chunks(line, args.chunk_chars))
        else:
            raw_parts.append(line)

    if len(raw_parts) < 3 and len(text) > args.chunk_chars:
        raw_parts = fixed_chunks(text, args.chunk_chars)

    parts = [
        clean_segment(part)
        for part in raw_parts
        if len(clean_segment(part)) >= args.min_segment_chars and alpha_count(part) >= args.min_alpha
    ]
    return take_evenly(parts, args.max_segments)


def predict_labels_batch(model, segments: list[str], min_confidence: float) -> list[str | None]:
    if not segments:
        return []
    pred_labels, probs = model.predict([s.replace("\n", " ") for s in segments], k=1)
    out: list[str | None] = []
    for labels_i, probs_i in zip(pred_labels, probs):
        if not labels_i:
            out.append(None)
            continue
        prob = float(probs_i[0])
        if prob < min_confidence:
            out.append(None)
            continue
        out.append(str(labels_i[0]).replace("__label__", ""))
    return out


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = fasttext.load_model(str(args.lid_model))
    summary_rows: list[dict] = []
    source_rows: list[dict] = []
    category_rows: list[dict] = []

    for lang, dataset_id in DATASET_IDS.items():
        target = TARGET_LABEL[lang]
        ds = load_dataset(dataset_id, split="train")
        total_docs = 0
        total_tokens = 0
        auditable_docs = 0
        mixed_docs = 0
        english_mixed_docs = 0
        mostly_other_docs = 0
        target_dominant_docs = 0
        non_target_segment_counts: Counter[str] = Counter()
        source_stats = defaultdict(lambda: Counter(docs=0, auditable=0, mixed=0, english_mixed=0, mostly_other=0))
        category_stats = defaultdict(lambda: Counter(docs=0, auditable=0, mixed=0, english_mixed=0, mostly_other=0))

        for batch in ds.iter(batch_size=args.dataset_batch_size):
            texts = batch.get("text", [])
            sources = batch.get("data-source", ["unknown"] * len(texts))
            categories = batch.get("category", ["unknown"] * len(texts))
            token_counts = batch.get("num-tokens", [0] * len(texts))

            doc_segments: list[list[str]] = []
            flat_segments: list[str] = []
            for text in texts:
                segments = candidate_segments(str(text or ""), args)
                doc_segments.append(segments)
                flat_segments.extend(segments)

            flat_labels = predict_labels_batch(model, flat_segments, args.min_confidence)
            cursor = 0
            for segments, source_raw, category_raw, ntok_raw in zip(doc_segments, sources, categories, token_counts):
                total_docs += 1
                ntok = int(ntok_raw or 0)
                total_tokens += ntok
                source = str(source_raw or "unknown")
                category = str(category_raw or "unknown")
                source_stats[source]["docs"] += 1
                category_stats[category]["docs"] += 1

                labels_slice = flat_labels[cursor : cursor + len(segments)]
                cursor += len(segments)
                labels = [label for label in labels_slice if label is not None]
                if len(labels) < args.mixed_min_segments:
                    continue

                auditable_docs += 1
                source_stats[source]["auditable"] += 1
                category_stats[category]["auditable"] += 1
                counts = Counter(labels)
                target_count = counts.get(target, 0)
                non_target_count = len(labels) - target_count
                dominant, dominant_count = counts.most_common(1)[0]
                non_target_frac = non_target_count / len(labels)
                target_frac = target_count / len(labels)

                for label, count in counts.items():
                    if label != target:
                        non_target_segment_counts[label] += count

                is_mixed = target_count > 0 and non_target_count > 0 and target_frac >= args.mixed_min_frac and non_target_frac >= args.mixed_min_frac
                is_english_mixed = target != "en" and counts.get("en", 0) > 0 and target_count > 0 and counts["en"] / len(labels) >= args.mixed_min_frac
                is_mostly_other = dominant != target and dominant_count / len(labels) >= 0.60
                is_target_dominant = dominant == target

                if is_mixed:
                    mixed_docs += 1
                    source_stats[source]["mixed"] += 1
                    category_stats[category]["mixed"] += 1
                if is_english_mixed:
                    english_mixed_docs += 1
                    source_stats[source]["english_mixed"] += 1
                    category_stats[category]["english_mixed"] += 1
                if is_mostly_other:
                    mostly_other_docs += 1
                    source_stats[source]["mostly_other"] += 1
                    category_stats[category]["mostly_other"] += 1
                if is_target_dominant:
                    target_dominant_docs += 1

        top_non_target = ";".join(f"{label}:{count}" for label, count in non_target_segment_counts.most_common(8))
        summary_rows.append(
            {
                "lang": lang,
                "dataset": dataset_id,
                "target_lid_label": target,
                "docs": total_docs,
                "num_tokens_metadata": total_tokens,
                "auditable_docs": auditable_docs,
                "auditable_pct_docs": pct(auditable_docs, total_docs),
                "mixed_candidate_docs": mixed_docs,
                "mixed_candidate_pct_all_docs": pct(mixed_docs, total_docs),
                "mixed_candidate_pct_auditable_docs": pct(mixed_docs, auditable_docs),
                "english_mixed_candidate_docs": english_mixed_docs,
                "english_mixed_candidate_pct_all_docs": pct(english_mixed_docs, total_docs),
                "mostly_other_docs": mostly_other_docs,
                "mostly_other_pct_auditable_docs": pct(mostly_other_docs, auditable_docs),
                "target_dominant_docs": target_dominant_docs,
                "target_dominant_pct_auditable_docs": pct(target_dominant_docs, auditable_docs),
                "top_non_target_segment_labels": top_non_target,
            }
        )

        for source, stats in source_stats.items():
            if stats["docs"] < args.source_min_docs:
                continue
            source_rows.append(
                {
                    "lang": lang,
                    "group_type": "data-source",
                    "group": source,
                    "docs": stats["docs"],
                    "auditable_docs": stats["auditable"],
                    "mixed_candidate_docs": stats["mixed"],
                    "mixed_candidate_pct_all_docs": pct(stats["mixed"], stats["docs"]),
                    "mixed_candidate_pct_auditable_docs": pct(stats["mixed"], stats["auditable"]),
                    "english_mixed_candidate_docs": stats["english_mixed"],
                    "mostly_other_docs": stats["mostly_other"],
                }
            )
        for category, stats in category_stats.items():
            category_rows.append(
                {
                    "lang": lang,
                    "group_type": "category",
                    "group": category,
                    "docs": stats["docs"],
                    "auditable_docs": stats["auditable"],
                    "mixed_candidate_docs": stats["mixed"],
                    "mixed_candidate_pct_all_docs": pct(stats["mixed"], stats["docs"]),
                    "mixed_candidate_pct_auditable_docs": pct(stats["mixed"], stats["auditable"]),
                    "english_mixed_candidate_docs": stats["english_mixed"],
                    "mostly_other_docs": stats["mostly_other"],
                }
            )
        print(f"{lang}: docs={total_docs} auditable={auditable_docs} mixed={mixed_docs} en_mixed={english_mixed_docs}", flush=True)

    summary_path = args.output_dir / "mixed_language_audit_summary.csv"
    source_path = args.output_dir / "mixed_language_audit_by_source.csv"
    category_path = args.output_dir / "mixed_language_audit_by_category.csv"
    meta_path = args.output_dir / "mixed_language_audit_config.json"

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(summary_path, summary_rows)
    write_csv(source_path, source_rows)
    write_csv(category_path, category_rows)
    meta_path.write_text(
        json.dumps(
            {
                "lid_model": str(args.lid_model),
                "max_segments": args.max_segments,
                "chunk_chars": args.chunk_chars,
                "min_segment_chars": args.min_segment_chars,
                "min_alpha": args.min_alpha,
                "min_confidence": args.min_confidence,
                "mixed_min_frac": args.mixed_min_frac,
                "mixed_min_segments": args.mixed_min_segments,
                "note": "Heuristic chunk-level audit. Mixed-candidate docs require both target and non-target high-confidence segment labels.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {source_path}")
    print(f"Wrote {category_path}")


if __name__ == "__main__":
    main()
