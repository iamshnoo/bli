#!/usr/bin/env python3
"""Build the repository's human- and machine-readable research manifests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"
TOKENS_PER_STEP = 512 * 8 * 8

LANGUAGES = [
    ("en", "eng", "English", "BabyLM-community/babylm-eng"),
    ("zh", "zho", "Chinese (Mandarin)", "BabyLM-community/babylm-zho"),
    ("fr", "fra", "French", "BabyLM-community/babylm-fra"),
    ("fas", "fas", "Persian (Farsi)", "BabyLM-community/babylm-fas"),
    ("nld", "nld", "Dutch", "BabyLM-community/babylm-nld"),
    ("ukr", "ukr", "Ukrainian", "BabyLM-community/babylm-ukr"),
    ("bul", "bul", "Bulgarian", "BabyLM-community/babylm-bul"),
    ("ind", "ind", "Indonesian", "BabyLM-community/babylm-ind"),
    ("deu", "deu", "German", "BabyLM-community/babylm-deu"),
]
SECOND_LANGUAGES = LANGUAGES[1:]

AXIS_SOURCES = {
    "hofstede2001culture": {
        "title": "Culture's Consequences: Comparing Values, Behaviors, Institutions and Organizations Across Nations",
        "year": 2001,
        "url": "https://digitalcommons.usu.edu/unf_research/53/",
    },
    "schwartz2006theory": {
        "title": "A theory of cultural value orientations: Explication and applications",
        "year": 2006,
        "url": "https://doi.org/10.1163/156913306778667357",
    },
    "inglehart2005modernization": {
        "title": "Modernization, Cultural Change, and Democracy: The Human Development Sequence",
        "year": 2005,
        "url": "https://doi.org/10.1017/CBO9780511790881",
    },
    "house2004culture": {
        "title": "Culture, Leadership, and Organizations: The GLOBE Study of 62 Societies",
        "year": 2004,
        "url": "https://www.sagepub.com/shop/buy-a-book/culture-leadership-and-organizations-1-226013",
    },
    "hershcovich2022challenges": {
        "title": "Challenges and Strategies in Cross-Cultural NLP",
        "year": 2022,
        "url": "https://aclanthology.org/2022.acl-long.482/",
    },
    "arora2023probing": {
        "title": "Probing Pre-Trained Language Models for Cross-Cultural Differences in Values",
        "year": 2023,
        "url": "https://aclanthology.org/2023.c3nlp-1.12/",
    },
    "berlin1969basic": {
        "title": "Basic Color Terms: Their Universality and Evolution",
        "year": 1969,
        "url": "https://philpapers.org/rec/BERBCT",
    },
}


def write_json(name: str, payload: dict) -> None:
    path = CONFIG_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


def model_run(
    run_id: str,
    training_type: str,
    steps: int,
    dataset_refs: list[str],
    dataset_paths: list[str],
    seed: int = 42,
    stage_config: str | None = None,
) -> dict:
    payload = {
        "run_id": run_id,
        "training_type": training_type,
        "dataset_refs": dataset_refs,
        "dataset_paths": dataset_paths,
        "steps": steps,
        "warmup_steps": 150 if steps == 1500 else 300,
        "tokens_per_step": TOKENS_PER_STEP,
        "approximate_training_tokens": steps * TOKENS_PER_STEP,
        "seed": seed,
        "initialization": "from_scratch",
        "tokenizer_hf_id": "meta-llama/Llama-3.2-1B",
        "pretrained_weight_hf_id": None,
        "huggingface_export_path": f"models/hf/{run_id}",
        "published_model_hf_id": f"iamshnoo/{run_id}",
    }
    if stage_config is not None:
        payload["stage_config"] = stage_config
        payload["schedule"] = "50-step English/L2 blocks, alternating for 3,000 steps"
        payload["steps_by_language"] = {"English": 1500, "second_language": 1500}
    return payload


def build_train_manifest() -> dict:
    runs = [
        model_run(
            "en_50m",
            "monolingual",
            1500,
            ["babylm_eng_shared"],
            ["data/processed/partitions/eng_shared/train"],
        ),
        model_run(
            "en_100m",
            "monolingual",
            3000,
            ["babylm_eng_full"],
            ["data/processed/babylm-eng/train"],
        ),
    ]

    for index, seed in enumerate((101, 202, 303), start=1):
        runs.extend(
            [
                model_run(
                    f"en_50m_s{index}",
                    "same_language_seed_control",
                    1500,
                    ["babylm_eng_shared"],
                    ["data/processed/partitions/eng_shared/train"],
                    seed=seed,
                ),
                model_run(
                    f"en_100m_s{index}",
                    "same_language_seed_control",
                    3000,
                    ["babylm_eng_full"],
                    ["data/processed/babylm-eng/train"],
                    seed=seed,
                ),
            ]
        )

    for short, dataset_code, _name, _dataset_id in SECOND_LANGUAGES:
        dataset_ref = f"babylm_{dataset_code}"
        dataset_path = f"data/processed/babylm-{dataset_code}/train"
        runs.extend(
            [
                model_run(f"{short}_50m", "monolingual", 1500, [dataset_ref], [dataset_path]),
                model_run(f"{short}_100m", "monolingual", 3000, [dataset_ref], [dataset_path]),
            ]
        )
        for setup, english_ref, english_path in (
            ("a", "babylm_eng_shared", "data/processed/partitions/eng_shared/train"),
            ("b", "babylm_eng_disjoint", "data/processed/partitions/eng_disjoint/train"),
        ):
            runs.append(
                model_run(
                    f"en_{short}_{setup}",
                    "bilingual_alternating",
                    3000,
                    [english_ref, dataset_ref],
                    [english_path, dataset_path],
                    stage_config=f"configs/stages/stages_eng_{dataset_code}_{setup}.json",
                )
            )

    return {
        "schema_version": "1.0",
        "description": "Complete manifest for the 40 models used in the controlled bilingual-pretraining study.",
        "data_manifest": "configs/data_manifest.json",
        "model_architecture": {
            "family": "LlamaForCausalLM",
            "nanotron_model_preset": "160m",
            "reported_parameter_count": "approximately 310M parameters with the 128,256-token vocabulary",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "max_position_embeddings": 512,
            "vocab_size": 128256,
            "activation": "SiLU",
            "normalization": "RMSNorm",
            "tie_word_embeddings": False,
            "dtype": "bfloat16",
        },
        "optimization": {
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "sequence_length": 512,
            "tokens_per_step": TOKENS_PER_STEP,
            "learning_rate": 0.003,
            "minimum_learning_rate": 0.0003,
            "weight_decay": 0.033,
            "gradient_clip": 0.1,
            "checkpoint_interval_steps": 500,
            "validation": "disabled during pretraining",
            "parallelism": {"data": 1, "tensor": 1, "pipeline": 1, "context": 1, "expert": 1},
            "hardware_per_run": "1 x NVIDIA A100 80GB",
        },
        "notes": [
            "The tokenizer vocabulary comes from meta-llama/Llama-3.2-1B; no pretrained Llama weights are loaded.",
            "All 40 models are public under iamshnoo on the Hugging Face Hub.",
            "The 50M and 100M labels denote the intended token budgets; exact step-derived budgets are recorded per run.",
        ],
        "run_count": len(runs),
        "published_model_count": sum(run["published_model_hf_id"] is not None for run in runs),
        "runs": runs,
    }


def build_hub_model_registry(train_manifest: dict) -> dict[str, str]:
    return {
        run["run_id"]: run["published_model_hf_id"]
        for run in train_manifest["runs"]
        if run["published_model_hf_id"] is not None
    }


def build_hub_suite_registries(hub_registry: dict[str, str]) -> dict[str, dict[str, str]]:
    suite_names = [
        "models_en_ablation.json",
        "models_layerwise.json",
        "models_seed_null.json",
        *[f"models_{short}_shared.json" for short, _dataset, _name, _hf_id in SECOND_LANGUAGES],
    ]
    registries = {}
    for name in suite_names:
        local_registry = json.loads((CONFIG_ROOT / "models" / name).read_text(encoding="utf-8"))
        missing = [model_name for model_name in local_registry if model_name not in hub_registry]
        if missing:
            raise KeyError(f"No public Hugging Face model for {name}: {missing}")
        registries[name.replace(".json", "_hub.json")] = {
            model_name: hub_registry[model_name] for model_name in local_registry
        }
    return registries


def build_data_manifest() -> dict:
    datasets = []
    for short, dataset_code, name, dataset_id in LANGUAGES:
        datasets.append(
            {
                "id": f"babylm_{dataset_code}",
                "language": name,
                "analysis_code": short,
                "dataset_code": dataset_code,
                "hf_dataset_id": dataset_id,
                "split": "train",
                "text_column": "text",
                "tokenized_path": f"data/processed/babylm-{dataset_code}/train",
            }
        )
    datasets.extend(
        [
            {
                "id": "babylm_eng_shared",
                "language": "English",
                "parent_dataset": "babylm_eng",
                "split_rule": "even MD5 parity of the string-valued doc-id field",
                "raw_path": "data/raw/partitions/eng_shared",
                "tokenized_path": "data/processed/partitions/eng_shared/train",
                "role": "English data shared with the en_50m reference",
            },
            {
                "id": "babylm_eng_disjoint",
                "language": "English",
                "parent_dataset": "babylm_eng",
                "split_rule": "odd MD5 parity of the string-valued doc-id field",
                "raw_path": "data/raw/partitions/eng_disjoint",
                "tokenized_path": "data/processed/partitions/eng_disjoint/train",
                "role": "English data disjoint from the en_50m reference",
            },
        ]
    )
    return {
        "schema_version": "1.0",
        "description": "Lineage for pretraining corpora, probe inventories, translations, and evaluation inputs.",
        "pretraining_datasets": datasets,
        "probe_assets": {
            "language_metadata": "data/language_metadata.csv",
            "canonical_probe_set": "data/probes/probe_sets.json",
            "evaluation_data_lineage": "data/probes/evaluation_data_lineage.csv",
            "lineage_builder": "src/probes/build_eval_data_lineage.py",
            "translation_files": "data/probes/translations_{zh,fr,fas,nld,ukr,bul,ind,deu}.csv",
            "translation_summary": "data/probes/translation_summary.json",
            "output_likelihood_cases": "data/probes/output_likelihood_association_cases_expanded.csv",
            "inventory": {
                "alignment_anchors_english": 3000,
                "cultural_concepts_per_language": 1000,
                "semantic_axes": 50,
                "axis_endpoints_per_language": 100,
                "negative_controls_english": 100,
                "languages_with_concept_terms": 9,
            },
            "axis_direction_note": "Endpoint 1 is coded -1 and endpoint 2 +1 only to orient signed differences. The signs are not value judgments.",
            "term_usage_note": "All reported representation comparisons use the English concept and axis terms in every model. The eight translation files support translation-quality stratification; 35 axis endpoint terms outside the 1,000-concept inventory were not translated and are marked not_translated in the lineage CSV.",
        },
        "translation_resources": {
            "model_hf_id": "facebook/nllb-200-distilled-600M",
            "quality_estimation_model_hf_id": "Unbabel/wmt22-cometkiwi-da",
            "quality_fields": [
                "back_translated_en",
                "back_similarity",
                "comet_kiwi_score",
                "qe_tier",
                "needs_manual_review",
                "duplicate_translation",
            ],
        },
        "tokenizer": {
            "hf_id": "meta-llama/Llama-3.2-1B",
            "role": "tokenizer and vocabulary only; model weights are initialized from scratch",
        },
        "model_assets": {
            "public_huggingface_registry": "configs/models/models_hub.json",
            "public_namespace": "iamshnoo",
            "published_model_count": 40,
            "local_export_root": "models/hf",
            "local_only_models": [],
        },
        "axis_and_category_sources": AXIS_SOURCES,
        "validation_outputs": {
            "exposure_overlap": "artifacts/results/validation/exposure_overlap_report.json",
            "mixed_language_audit": "artifacts/results/validation/mixed_language_audit_summary.csv",
        },
    }


def suite(
    suite_id: str,
    purpose: str,
    entrypoint: str,
    inputs: list[str],
    model_registry: str | None,
    outputs: list[str],
    paper_role: str,
) -> dict:
    payload = {
        "id": suite_id,
        "purpose": purpose,
        "entrypoint": entrypoint,
        "inputs": inputs,
        "outputs": outputs,
        "paper_role": paper_role,
    }
    if model_registry is not None:
        payload["model_registry"] = model_registry
        public_candidate = model_registry.replace(".json", "_hub.json")
        if public_candidate != model_registry and (ROOT / public_candidate).is_file():
            payload["public_model_registry"] = public_candidate
    return payload


def build_test_manifest() -> dict:
    main_root = "outputs/revision/en_ablation"
    suites = [
        suite(
            "english_control_matrix",
            "Compare English representations after English-only or English-plus-second-language pretraining under matched-data and matched-compute controls.",
            "src/bli_analysis/run_bli_pipeline.py",
            ["data/probes/probe_sets.json"],
            "configs/models/models_en_ablation.json",
            [f"{main_root}/bli_summary_metrics.csv", f"{main_root}/bli_word_neighbor_divergence.csv", f"{main_root}/bli_axis_divergence.csv"],
            "core",
        ),
        suite(
            "target_language_validation",
            "Repeat the comparison in each second-language model space using the same English concept inventory.",
            "src/bli_analysis/run_bli_pipeline.py",
            ["data/probes/probe_sets.json"],
            "configs/models/models_{language}_shared.json",
            ["outputs/{revision_or_multilingual}/{language}_shared_language/bli_summary_metrics.csv", "outputs/{revision_or_multilingual}/{language}_shared_language/bli_axis_divergence.csv"],
            "core",
        ),
        suite(
            "bootstrap_confidence_intervals",
            "Bootstrap word-, axis-, and similarity-based estimates.",
            "src/bli_analysis/compute_bootstrap_ci.py",
            [f"{main_root}/bli_summary_metrics.csv", f"{main_root}/bli_word_neighbor_divergence.csv", f"{main_root}/bli_axis_divergence.csv", f"{main_root}/representations"],
            None,
            [f"{main_root}/bli_bootstrap_ci.csv"],
            "statistical_validation",
        ),
        suite(
            "document_overlap_tests",
            "Test shared-versus-disjoint English comparisons and translation-quality strata.",
            "src/bli_analysis/run_statistical_tests.py",
            [f"{main_root}/bli_word_neighbor_divergence.csv", "data/probes/translations_*.csv"],
            None,
            [f"{main_root}/bli_wilcoxon_overlap.csv"],
            "statistical_validation",
        ),
        suite(
            "same_language_seed_controls",
            "Estimate variation between English-only runs that differ only in random seed.",
            "src/bli_analysis/run_same_language_controls.py",
            ["data/probes/probe_sets.json", "outputs/revision/en_seed_null/representations", f"{main_root}/representations"],
            "configs/models/models_seed_null.json",
            [f"{main_root}/bli_same_language_controls.csv", f"{main_root}/bli_same_language_controls_summary.csv"],
            "robustness",
        ),
        suite(
            "layerwise_analysis",
            "Trace semantic-axis differences through every transformer layer.",
            "src/bli_analysis/run_layerwise_analysis.py",
            ["data/probes/probe_sets.json"],
            "configs/models/models_layerwise.json",
            [f"{main_root}/bli_layerwise_divergence.csv"],
            "core",
        ),
        suite(
            "norm_controlled_axes",
            "Repeat signed-axis analysis after unit-normalizing representations and report layerwise controls.",
            "src/bli_analysis/run_norm_controlled_axis.py",
            ["data/probes/probe_sets.json", f"{main_root}/representations"],
            "configs/models/models_en_ablation.json",
            [f"{main_root}/bli_norm_controlled_axis.csv", f"{main_root}/bli_norm_controlled_layerwise.csv"],
            "robustness",
        ),
        suite(
            "alignment_method_comparison",
            "Compare orthogonal and affine maps fit on embeddings or contextual states.",
            "src/bli_analysis/run_alignment_method_comparison.py",
            ["data/probes/probe_sets.json", f"{main_root}/representations"],
            None,
            [f"{main_root}/bli_alignment_method_comparison.csv"],
            "robustness",
        ),
        suite(
            "contextual_alignment_variant",
            "Fit the orthogonal map directly on contextual states.",
            "src/bli_analysis/run_contextual_alignment_variant.py",
            ["data/probes/probe_sets.json", f"{main_root}/representations"],
            None,
            [f"{main_root}/bli_contextual_alignment_variant.csv"],
            "robustness",
        ),
        suite(
            "anchor_sensitivity",
            "Refit alignment maps with repeated anchor subsets of 500, 1,000, 2,000, and 3,000 words.",
            "src/bli_analysis/run_anchor_sensitivity.py",
            ["data/probes/probe_sets.json", f"{main_root}/representations"],
            None,
            [f"{main_root}/bli_anchor_sensitivity.csv", f"{main_root}/bli_anchor_sensitivity_summary.csv"],
            "robustness",
        ),
        suite(
            "knn_sensitivity",
            "Recompute nearest-neighbor disagreement for k in {5, 10, 25, 50, 100}.",
            "src/bli_analysis/run_k_sensitivity.py",
            ["data/probes/probe_sets.json", f"{main_root}/representations", "outputs/revision/en_seed_null/representations"],
            None,
            ["outputs/revision/scope_expansion/k_sensitivity.csv"],
            "robustness",
        ),
        suite(
            "framework_holdout",
            "Refit alignment while holding out each cultural framework category.",
            "src/bli_analysis/run_framework_holdout_eval.py",
            ["data/probes/probe_sets.json", f"{main_root}/representations"],
            None,
            [f"{main_root}/bli_framework_holdout_eval.csv"],
            "robustness",
        ),
        suite(
            "negative_controls",
            "Compare cultural concepts with concrete negative-control words.",
            "src/bli_analysis/run_negative_control_eval.py",
            ["data/probes/probe_sets.json", f"{main_root}/representations"],
            None,
            [f"{main_root}/bli_negative_control_eval.csv"],
            "robustness",
        ),
        suite(
            "training_progress",
            "Measure differences over training checkpoints and test the final-step trend.",
            "src/pipeline/run_dense_progress_trajectory.sh",
            ["data/probes/probe_sets.json"],
            "configs/models/models_dense_progress_all.json",
            [f"{main_root}/bli_dense_progress_trajectory.csv", f"{main_root}/bli_progress_sensitivity.csv"],
            "robustness",
        ),
        suite(
            "tokenizer_audit",
            "Verify tokenizer identity and vocabulary consistency across model exports.",
            "src/bli_analysis/run_tokenizer_audit.py",
            ["models/hf"],
            None,
            [f"{main_root}/bli_tokenizer_audit.csv", f"{main_root}/bli_tokenizer_audit.json"],
            "validation",
        ),
        suite(
            "mixed_language_corpus_audit",
            "Use chunk-level language identification to flag mixed-language or wrong-language documents in each training corpus.",
            "src/bli_analysis/run_mixed_language_audit.py",
            ["BabyLM-community/babylm-* train splits", "fastText lid.176.ftz"],
            None,
            ["outputs/validation/mixed_language_audit/mixed_language_audit_summary.csv", "outputs/validation/mixed_language_audit/mixed_language_audit_by_source.csv", "outputs/validation/mixed_language_audit/mixed_language_audit_by_category.csv"],
            "validation",
        ),
        suite(
            "output_likelihood_association",
            "Test whether signed representation differences predict controlled output-likelihood preferences.",
            "slurm/run_output_likelihood_association.sbatch",
            ["data/probes/output_likelihood_association_cases_expanded.csv"],
            "configs/models/models_en_ablation.json",
            [f"{main_root}/bli_output_likelihood_association.csv", f"{main_root}/bli_output_likelihood_association_summary.csv"],
            "scope_extension",
        ),
        suite(
            "worldvaluesbench",
            "Evaluate target-country WorldValuesBench items and compare behavioral residuals with representation differences.",
            "slurm/run_worldvaluebench_bli.sbatch",
            ["external/WorldValuesBench", "data/probes/probe_sets.json"],
            "configs/models/models_en_ablation.json",
            [f"{main_root}/worldvaluebench_bli/worldvaluebench_model_summary.csv", f"{main_root}/worldvaluebench_bli/worldvaluebench_residual_association.csv"],
            "scope_extension",
        ),
    ]
    return {
        "schema_version": "1.0",
        "description": "Evaluation, validation, and robustness suites used by the paper and tracked extensions.",
        "data_manifest": "configs/data_manifest.json",
        "train_manifest": "configs/train_manifest.json",
        "public_model_registry": "configs/models/models_hub.json",
        "comparison_setups": [
            {"id": "C1", "baseline": "en_50m", "bilingual": "en_{language}_a", "english_data": "shared", "total_compute": "bilingual model has twice the training steps", "isolates": "adding a second language while holding English documents fixed"},
            {"id": "C2", "baseline": "en_50m", "bilingual": "en_{language}_b", "english_data": "disjoint", "total_compute": "bilingual model has twice the training steps", "isolates": "adding a second language without shared English documents"},
            {"id": "C3", "baseline": "en_100m", "bilingual": "en_{language}_a", "english_data": "shared with the 50M English subset", "total_compute": "matched", "isolates": "language mixture at matched total compute"},
            {"id": "C4", "baseline": "en_100m", "bilingual": "en_{language}_b", "english_data": "disjoint from the 50M English subset", "total_compute": "matched", "isolates": "language mixture at matched total compute without shared English documents"},
        ],
        "representations": {
            "embedding_matrix": "input token embedding for each single-token English probe",
            "pre_lmhead_contextual": "final-token hidden state before the language-model head for a fixed prompt",
            "layerwise_contextual": "the same prompted state extracted after each transformer layer",
        },
        "alignment_strategies": [
            {"id": "orthogonal_embedding", "map": "orthogonal Procrustes", "fit_on": "3,000 neutral English embedding anchors", "default": True},
            {"id": "orthogonal_contextual", "map": "orthogonal Procrustes", "fit_on": "3,000 neutral English contextual anchors", "default": False},
            {"id": "affine_embedding", "map": "affine least squares", "fit_on": "3,000 neutral English embedding anchors", "default": False},
            {"id": "layerwise_orthogonal", "map": "separate orthogonal Procrustes map per layer", "fit_on": "3,000 neutral English contextual anchors per layer", "default": False},
        ],
        "evaluation_suite_count": len(suites),
        "evaluation_suites": suites,
        "released_result_root": "artifacts/results",
        "note": "Paths under outputs/ are generated locally; curated summary artifacts used by the manuscript are mirrored under artifacts/results/.",
    }


def main() -> None:
    train_manifest = build_train_manifest()
    hub_registry = build_hub_model_registry(train_manifest)
    write_json("train_manifest.json", train_manifest)
    write_json("models/models_hub.json", hub_registry)
    for name, registry in build_hub_suite_registries(hub_registry).items():
        write_json(f"models/{name}", registry)
    write_json("data_manifest.json", build_data_manifest())
    write_json("test_manifest.json", build_test_manifest())


if __name__ == "__main__":
    main()
