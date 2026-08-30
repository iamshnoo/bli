#!/usr/bin/env python3
"""Build one normalized CSV describing every probe term used by the study."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = ROOT / "data/probes/probe_sets.json"
LANGUAGE_PATH = ROOT / "data/language_metadata.csv"
DATA_MANIFEST_PATH = ROOT / "configs/data_manifest.json"
OUTPUT_PATH = ROOT / "data/probes/evaluation_data_lineage.csv"

LANGUAGE_CODES = {
    "EN": ("en", "eng"),
    "ZH": ("zh", "zho"),
    "FR": ("fr", "fra"),
    "FAS": ("fas", "fas"),
    "NLD": ("nld", "nld"),
    "UKR": ("ukr", "ukr"),
    "BUL": ("bul", "bul"),
    "IND": ("ind", "ind"),
    "DEU": ("deu", "deu"),
}

FIELDS = [
    "record_id",
    "record_type",
    "used_for",
    "actual_model_input",
    "model_language_conditions",
    "availability_status",
    "language_code",
    "dataset_code",
    "language_name",
    "language_family",
    "script",
    "source_term_en",
    "localized_term",
    "concept_name_en",
    "concept_category",
    "concept_category_name",
    "axis_id",
    "axis_category",
    "axis_endpoint",
    "axis_polarity",
    "axis_endpoint_1_en",
    "axis_endpoint_1_localized",
    "axis_endpoint_2_en",
    "axis_endpoint_2_localized",
    "source_citation_keys",
    "source_titles",
    "source_years",
    "source_urls",
    "translation_nllb_code",
    "back_translated_en",
    "back_similarity",
    "comet_kiwi_score",
    "qe_tier",
    "needs_manual_review",
    "duplicate_translation",
    "origin_files",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def joined(values: list[object]) -> str:
    return " | ".join(str(value) for value in values)


def citation_fields(keys: list[str], sources: dict[str, dict]) -> dict[str, str]:
    missing = [key for key in keys if key not in sources]
    if missing:
        raise KeyError(f"Missing citation metadata for: {missing}")
    return {
        "source_citation_keys": joined(keys),
        "source_titles": joined([sources[key]["title"] for key in keys]),
        "source_years": joined([sources[key]["year"] for key in keys]),
        "source_urls": joined([sources[key]["url"] for key in keys]),
    }


def translation_fields(row: dict[str, str] | None) -> dict[str, str]:
    if row is None:
        return {
            "translation_nllb_code": "",
            "back_translated_en": "",
            "back_similarity": "",
            "comet_kiwi_score": "",
            "qe_tier": "",
            "needs_manual_review": "",
            "duplicate_translation": "",
        }
    return {
        "translation_nllb_code": row["target_lang_nllb"],
        "back_translated_en": row["back_translated_en"],
        "back_similarity": row["back_similarity"],
        "comet_kiwi_score": row["comet_kiwi_score"],
        "qe_tier": row["qe_tier"],
        "needs_manual_review": row["needs_manual_review"],
        "duplicate_translation": row["duplicate_translation"],
    }


def empty_row() -> dict[str, str]:
    return {field: "" for field in FIELDS}


def main() -> None:
    probes = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    data_manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    sources = data_manifest["axis_and_category_sources"]

    language_rows = read_csv(LANGUAGE_PATH)
    languages: list[dict[str, str]] = []
    for metadata in language_rows:
        analysis_code, dataset_code = LANGUAGE_CODES[metadata["code"]]
        languages.append(
            {
                **metadata,
                "analysis_code": analysis_code,
                "dataset_code": dataset_code,
            }
        )

    cultural_words = probes["cultural_probe_words"]
    cultural_set = set(cultural_words)
    categories = probes["cultural_probe_categories"]
    frameworks = probes["category_frameworks"]
    axes = probes["semantic_axis_metadata"]

    if set(categories) != cultural_set:
        raise ValueError("Every cultural concept must have exactly one category")
    if len(axes) != 50:
        raise ValueError(f"Expected 50 semantic axes, found {len(axes)}")

    translations: dict[str, dict[str, dict[str, str]]] = {"en": {}}
    for language in languages:
        code = language["analysis_code"]
        if code == "en":
            continue
        path = ROOT / f"data/probes/translations_{code}.csv"
        rows = read_csv(path)
        by_term = {row["source_term_en"]: row for row in rows}
        if len(rows) != 1000 or len(by_term) != 1000:
            raise ValueError(f"{path}: expected 1,000 unique translations, found {len(rows)} rows and {len(by_term)} terms")
        if set(by_term) != cultural_set:
            missing = sorted(cultural_set - set(by_term))[:5]
            extra = sorted(set(by_term) - cultural_set)[:5]
            raise ValueError(f"{path}: translation inventory mismatch; missing={missing}, extra={extra}")
        translations[code] = by_term

    output_rows: list[dict[str, str]] = []

    all_model_conditions = joined([language["analysis_code"] for language in languages])

    # English concepts are model inputs. Their translations support the paper's
    # translation-quality strata but are not substituted into the model prompts.
    for language in languages:
        code = language["analysis_code"]
        for concept_index, term in enumerate(cultural_words, start=1):
            translation = translations[code].get(term)
            localized = term if code == "en" else translation["translated_term"]
            category = categories[term]
            framework = frameworks[category]
            row = empty_row()
            row.update(
                {
                    "record_id": f"concept:{code}:{concept_index:04d}",
                    "record_type": "concept",
                    "used_for": (
                        "concept_neighborhood_and_similarity_evaluation"
                        if code == "en"
                        else "translation_quality_stratification"
                    ),
                    "actual_model_input": "1" if code == "en" else "0",
                    "model_language_conditions": all_model_conditions if code == "en" else code,
                    "availability_status": "available",
                    "language_code": code,
                    "dataset_code": language["dataset_code"],
                    "language_name": language["language"],
                    "language_family": language["family"],
                    "script": language["script"],
                    "source_term_en": term,
                    "localized_term": localized,
                    "concept_name_en": term,
                    "concept_category": category,
                    "concept_category_name": framework["display_name"],
                    "origin_files": (
                        "data/probes/probe_sets.json"
                        if code == "en"
                        else f"data/probes/probe_sets.json | data/probes/translations_{code}.csv"
                    ),
                }
            )
            row.update(citation_fields(framework["citations"], sources))
            row.update(translation_fields(translation))
            output_rows.append(row)

    # Axis endpoints are repeated for every language so availability is explicit.
    # The study evaluates the English axis terms in every model; translated axis
    # terms are documentation/QC assets and are blank where no translation exists.
    for language in languages:
        code = language["analysis_code"]
        for axis in axes:
            endpoint_1 = axis["endpoint_1"]
            endpoint_2 = axis["endpoint_2"]
            translation_1 = translations[code].get(endpoint_1)
            translation_2 = translations[code].get(endpoint_2)
            localized_1 = endpoint_1 if code == "en" else (translation_1 or {}).get("translated_term", "")
            localized_2 = endpoint_2 if code == "en" else (translation_2 or {}).get("translated_term", "")
            for endpoint_number, polarity, term, localized, translation in (
                (1, -1, endpoint_1, localized_1, translation_1),
                (2, 1, endpoint_2, localized_2, translation_2),
            ):
                row = empty_row()
                row.update(
                    {
                        "record_id": f"axis_endpoint:{code}:{int(axis['index']):02d}:{endpoint_number}",
                        "record_type": "axis_endpoint",
                        "used_for": (
                            "signed_semantic_axis_evaluation"
                            if code == "en"
                            else "translated_axis_documentation"
                        ),
                        "actual_model_input": "1" if code == "en" else "0",
                        "model_language_conditions": all_model_conditions if code == "en" else code,
                        "availability_status": (
                            "available" if code == "en" or translation is not None else "not_translated"
                        ),
                        "language_code": code,
                        "dataset_code": language["dataset_code"],
                        "language_name": language["language"],
                        "language_family": language["family"],
                        "script": language["script"],
                        "source_term_en": term,
                        "localized_term": localized,
                        "concept_name_en": term,
                        "concept_category": categories.get(term, axis["category"]),
                        "concept_category_name": frameworks[axis["category"]]["display_name"],
                        "axis_id": f"axis_{int(axis['index']):02d}",
                        "axis_category": axis["category"],
                        "axis_endpoint": f"endpoint_{endpoint_number}",
                        "axis_polarity": str(polarity),
                        "axis_endpoint_1_en": endpoint_1,
                        "axis_endpoint_1_localized": localized_1,
                        "axis_endpoint_2_en": endpoint_2,
                        "axis_endpoint_2_localized": localized_2,
                        "origin_files": (
                            "data/probes/probe_sets.json"
                            if code == "en" or translation is None
                            else f"data/probes/probe_sets.json | data/probes/translations_{code}.csv"
                        ),
                    }
                )
                row.update(citation_fields(axis["citations"], sources))
                row.update(translation_fields(translation))
                output_rows.append(row)

    english = next(language for language in languages if language["analysis_code"] == "en")
    for record_type, used_for, words in (
        ("alignment_anchor", "fit_representation_alignment", probes["neutral_anchor_words"]),
        ("negative_control", "concrete_word_negative_control", probes["negative_control_words"]),
    ):
        for index, term in enumerate(words, start=1):
            row = empty_row()
            row.update(
                {
                    "record_id": f"{record_type}:en:{index:04d}",
                    "record_type": record_type,
                    "used_for": used_for,
                    "actual_model_input": "1",
                    "model_language_conditions": (
                        all_model_conditions if record_type == "alignment_anchor" else "en"
                    ),
                    "availability_status": "available",
                    "language_code": "en",
                    "dataset_code": english["dataset_code"],
                    "language_name": english["language"],
                    "language_family": english["family"],
                    "script": english["script"],
                    "source_term_en": term,
                    "localized_term": term,
                    "concept_name_en": term,
                    "origin_files": "data/probes/probe_sets.json",
                }
            )
            row.update(translation_fields(None))
            output_rows.append(row)

    expected_counts = {
        "concept": 9 * 1000,
        "axis_endpoint": 9 * 50 * 2,
        "alignment_anchor": 3000,
        "negative_control": 100,
    }
    actual_counts = {
        record_type: sum(row["record_type"] == record_type for row in output_rows)
        for record_type in expected_counts
    }
    if actual_counts != expected_counts:
        raise AssertionError(f"Unexpected lineage counts: {actual_counts} != {expected_counts}")
    record_ids = [row["record_id"] for row in output_rows]
    if len(record_ids) != len(set(record_ids)):
        raise AssertionError("Lineage record IDs are not unique")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} ({len(output_rows):,} rows)")
    print(json.dumps(actual_counts, indent=2))


if __name__ == "__main__":
    main()
