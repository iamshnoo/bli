#!/usr/bin/env python3
"""Export the compact, tracked result snapshot used by the manuscript."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts"

PAPER_FIGURES = [
    "fig1.pdf",
    "fig2-exp1-divergence.pdf",
    "fig3-exp2-overlap.pdf",
    "fig4-multilingual-validation.pdf",
    "fig5-contextual-vs-embedding-ratio.pdf",
    "fig6-signed-axis-shifts.pdf",
    "fig7-layerwise-axis-divergence.pdf",
    "fig8-alignment-method-comparison.pdf",
    "fig8-dense-progress-summary.pdf",
    "fig9-appendix-perhead-heatmap.pdf",
    "fig10-appendix-category-heatmap.pdf",
    "fig11-appendix-multilingual-overview.pdf",
    "fig12-typology-regression-scatter.pdf",
    "fig13-appendix-l2-signed-hotspots-panel1.pdf",
    "fig14-appendix-l2-signed-hotspots-panel2.pdf",
    "fig15-appendix-exp1-divergence-100m.pdf",
    "fig16-appendix-exp2-overlap-100m.pdf",
    "fig17-appendix-multilingual-validation-100m.pdf",
    "fig18-appendix-contextual-vs-embedding-100m.pdf",
    "fig19-appendix-layerwise-axis-divergence-100m.pdf",
    "fig20-appendix-alignment-method-comparison-100m.pdf",
    "fig21-appendix-signed-axis-shifts-100m.pdf",
    "fig22-appendix-dense-progress-trajectory.pdf",
    "fig23-appendix-norm-controlled-layerwise.pdf",
]

MAIN_RESULTS = [
    "bli_alignment_method_comparison.csv",
    "bli_anchor_sensitivity.csv",
    "bli_anchor_sensitivity_summary.csv",
    "bli_axis_divergence.csv",
    "bli_behavioral_bridge_hotspots_summary.csv",
    "bli_behavioral_bridge_summary.csv",
    "bli_bootstrap_ci.csv",
    "bli_causal_localization_summary.csv",
    "bli_contextual_alignment_variant.csv",
    "bli_dense_progress_trajectory.csv",
    "bli_framework_holdout_eval.csv",
    "bli_layerwise_divergence.csv",
    "bli_minimal_intervention_summary.csv",
    "bli_negative_control_eval.csv",
    "bli_norm_controlled_axis.csv",
    "bli_norm_controlled_layerwise.csv",
    "bli_output_likelihood_association_summary.csv",
    "bli_perhead_analysis.csv",
    "bli_progress_sensitivity.csv",
    "bli_run_metadata.json",
    "bli_same_language_controls.csv",
    "bli_same_language_controls_summary.csv",
    "bli_scope_tests.csv",
    "bli_specificity_bridge_summary.csv",
    "bli_stratified_metrics.csv",
    "bli_summary_metrics.csv",
    "bli_tokenizer_audit.csv",
    "bli_tokenizer_audit.json",
    "bli_tokenizer_audit_summary.csv",
    "bli_wilcoxon_overlap.csv",
]

TARGET_RESULT_FILES = [
    "bli_axis_divergence.csv",
    "bli_bootstrap_ci.csv",
    "bli_run_metadata.json",
    "bli_summary_metrics.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_text_artifact(path: Path) -> None:
    """Remove machine-specific repository prefixes from released text files."""
    if path.suffix.lower() not in {".csv", ".json", ".md", ".txt"}:
        return
    text = path.read_text(encoding="utf-8")
    # Historical outputs came from both .../bli/ and .../bli/v2/ worktrees.
    text = re.sub(r"/(?:scratch|home)/[^/\s\"']+/bli(?:/v2)?/", "", text)
    text = re.sub(r"/(?:scratch|home)/[^/\s\"']+/bli(?:/v2)?(?=[\"'])", ".", text)
    text = text.replace("config/stages/", "configs/stages/")
    path.write_text(text, encoding="utf-8")


def build_copy_map() -> list[tuple[Path, Path]]:
    copies: list[tuple[Path, Path]] = []
    figure_root = ROOT / "latex/emnlp/anon/figures"
    for name in PAPER_FIGURES:
        copies.append((figure_root / name, ARTIFACT_ROOT / "figures" / name))

    main_root = ROOT / "outputs/revision/en_ablation"
    for name in MAIN_RESULTS:
        copies.append((main_root / name, ARTIFACT_ROOT / "results/en_ablation" / name))

    for code in ("zh", "fr"):
        source_root = ROOT / f"outputs/revision/{code}_shared_language"
        for name in TARGET_RESULT_FILES:
            copies.append((source_root / name, ARTIFACT_ROOT / f"results/target_language/{code}" / name))

    for code in ("fas", "nld", "ukr", "bul", "ind", "deu"):
        source_root = ROOT / f"outputs/multilingual_expansion/{code}_shared_language"
        for name in TARGET_RESULT_FILES:
            copies.append((source_root / name, ARTIFACT_ROOT / f"results/target_language/{code}" / name))

    for name in ("language_ratio_summary.csv", "multilingual_summary.csv"):
        copies.append(
            (
                ROOT / "outputs/multilingual_expansion" / name,
                ARTIFACT_ROOT / "results/multilingual" / name,
            )
        )

    for name in ("bli_axis_divergence.csv", "bli_run_metadata.json", "bli_summary_metrics.csv"):
        copies.append(
            (
                ROOT / "outputs/revision/en_seed_null" / name,
                ARTIFACT_ROOT / "results/seed_controls" / name,
            )
        )

    copies.extend(
        [
            (
                ROOT / "outputs/revision/scope_expansion/k_sensitivity.csv",
                ARTIFACT_ROOT / "results/en_ablation/k_sensitivity.csv",
            ),
            (
                ROOT / "outputs/revision/en_ablation/worldvaluebench_bli/worldvaluebench_model_summary.csv",
                ARTIFACT_ROOT / "results/worldvaluesbench/worldvaluebench_model_summary.csv",
            ),
            (
                ROOT / "outputs/revision/en_ablation/worldvaluebench_bli/worldvaluebench_pair_summary.csv",
                ARTIFACT_ROOT / "results/worldvaluesbench/worldvaluebench_pair_summary.csv",
            ),
            (
                ROOT / "outputs/revision/en_ablation/worldvaluebench_bli/worldvaluebench_residual_association.csv",
                ARTIFACT_ROOT / "results/worldvaluesbench/worldvaluebench_residual_association.csv",
            ),
            (
                ROOT / "outputs/validation/exposure_overlap_report.json",
                ARTIFACT_ROOT / "results/validation/exposure_overlap_report.json",
            ),
        ]
    )
    for name in (
        "mixed_language_audit_summary.csv",
        "mixed_language_audit_by_source.csv",
        "mixed_language_audit_by_category.csv",
        "mixed_language_audit_config.json",
    ):
        copies.append(
            (
                ROOT / "outputs/validation/mixed_language_audit" / name,
                ARTIFACT_ROOT / "results/validation" / name,
            )
        )
    return copies


def main() -> None:
    copies = build_copy_map()
    missing = [source.relative_to(ROOT) for source, _destination in copies if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing release inputs:\n" + "\n".join(str(path) for path in missing))

    manifest_files = []
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sanitize_text_artifact(destination)
        manifest_files.append(
            {
                "path": str(destination.relative_to(ROOT)),
                "source": str(source.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    payload = {
        "schema_version": "1.0",
        "description": "Compact result and figure snapshot used by the manuscript.",
        "file_count": len(manifest_files),
        "files": sorted(manifest_files, key=lambda row: row["path"]),
    }
    manifest_path = ARTIFACT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(manifest_files)} files to {ARTIFACT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
