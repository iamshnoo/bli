#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate revision artifacts for the BLI paper")
    p.add_argument("--output-root", type=Path, default=Path("outputs/revision"))
    p.add_argument(
        "--multilingual-output-root",
        type=Path,
        default=Path("outputs/multilingual_expansion"),
    )
    p.add_argument("--latex-root", type=Path, default=Path("latex"))
    p.add_argument("--probe-set", type=Path, default=Path("data/probes/probe_sets.json"))
    p.add_argument("--translations-dir", type=Path, default=Path("data/probes"))
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Suppress table/CSV side effects while regenerating figures.",
    )
    p.add_argument(
        "--main-figures-only",
        action="store_true",
        help="Regenerate only the main-paper figure pipeline for Figures 2 to 8.",
    )
    return p.parse_args()


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12})
    # Keep vector figure text searchable and avoid legacy Type 3 font outlines
    # in the compiled proceedings PDF.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"]
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["hatch.linewidth"] = 0.8


COLORS = {
    "embedding_matrix": "#cdcdcd",
    "pre_lmhead_contextual": "#9fc4e6",
}
HATCH = {
    "embedding_matrix": "..",
    "pre_lmhead_contextual": "//",
}
REPR_LABEL = {
    "embedding_matrix": "Embedding",
    "pre_lmhead_contextual": "Contextual",
}

PAIR_LABEL: dict[tuple[str, str], str] = {}
CORE_LANGS = ["zh", "fr", "fas", "nld", "ukr", "bul", "ind", "deu"]

# Language colors matching the LaTeX tcolorbox scheme used throughout the paper.
LANG_COLORS: dict[str, str] = {
    "ZH":  "#E67E22",  # orange
    "FR":  "#00ACC1",  # cyan
    "FAS": "#8E24AA",  # violet
    "NLD": "#00897B",  # teal
    "UKR": "#F9A825",  # yellow/amber
    "BUL": "#43A047",  # green
    "IND": "#E53935",  # red
    "DEU": "#6D4C41",  # brown
}

LANG_HATCHES: dict[str, str] = {
    "ZH": "///",
    "FR": "\\\\\\",
    "FAS": "xx",
    "NLD": "..",
    "UKR": "++",
    "BUL": "oo",
    "IND": "**",
    "DEU": "||",
}

PAPER_BG = "#faf9f4"
PANEL_BG = "#ffffff"
INK = "#253142"
MUTED = "#687386"
GRID = "#d7d9d4"
NAVY = "#2f6f9f"
TEAL = "#2a9d8f"
ORANGE = "#e6954a"
ROSE = "#d86565"
SOFT_BLUE = "#e8eff5"
SOFT_TEAL = "#e4eed8"
SOFT_ORANGE = "#f2e1c8"
SOFT_GRAY = "#ececea"
CTRL_HEADER_BG = "#f0f1ee"
CTRL_HEADER_BORDER = "#cfd4d8"
CTRL_ROW_BORDER = "#c7ccd0"
CTRL_DIVIDER = "#cfd3d6"
REF_RANGE_BG = "#c6ccd2"
REF_RANGE_ALPHA = 0.42
OVERLAP_REF_BG = "#c9ced5"
OVERLAP_REF_ALPHA = 0.42
ZERO_RANGE_BG = "#e3e4e1"
ZERO_RANGE_ALPHA = 0.52
ZERO_RANGE_LINE = "#a9afb4"
RAIL_BG = "#d2d6da"
RAIL_LINE = "#eceff1"
COL_FIG_W = 3.42
COL_FIG_H = 2.25


def style_paper_axis(ax, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(PAPER_BG)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="-", linewidth=0.55, color=GRID, alpha=0.82)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#b3bac1")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelcolor=INK)


CONTROL_SETTINGS = [
    {
        "code": "C1+C3",
        "short": "Matched EN\nshared docs",
        "title": "Matched English",
        "doc": "shared English docs",
        "model_a": "en_50m",
        "setup": "a",
        "chips": ["same EN steps", "same docs"],
        "color": "#557d9f",
        "fill": SOFT_BLUE,
    },
    {
        "code": "C1+C4",
        "short": "Matched EN\ndisjoint docs",
        "title": "Matched English",
        "doc": "disjoint English docs",
        "model_a": "en_50m",
        "setup": "b",
        "chips": ["same EN steps", "disjoint docs"],
        "color": "#1f5f85",
        "fill": "#edf1f3",
    },
    {
        "code": "C2+C3",
        "short": "Matched compute\nshared docs",
        "title": "Matched compute",
        "doc": "shared English docs",
        "model_a": "en_100m",
        "setup": "a",
        "chips": ["same updates", "same docs"],
        "color": "#6f7f92",
        "fill": SOFT_GRAY,
    },
    {
        "code": "C2+C4",
        "short": "Matched compute\ndisjoint docs",
        "title": "Matched compute",
        "doc": "disjoint English docs",
        "model_a": "en_100m",
        "setup": "b",
        "chips": ["same updates", "disjoint docs"],
        "color": "#566579",
        "fill": SOFT_GRAY,
    },
]


def blend_hex(hex_color: str, target_rgb: tuple[float, float, float], frac: float) -> tuple[float, float, float]:
    c = np.array(colors.to_rgb(hex_color), dtype=float)
    t = np.array(target_rgb, dtype=float)
    return tuple(((1.0 - frac) * c + frac * t).tolist())


def pastel_lang_color(lang_code: str, lighten: float = 0.35) -> tuple[float, float, float]:
    base = LANG_COLORS.get(str(lang_code).upper(), "#4C78A8")
    return blend_hex(base, (1.0, 1.0, 1.0), lighten)


def color_language_ticklabels(ax, axis: str = "y") -> None:
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for tick in labels:
        code = tick.get_text().strip().upper()
        if code in LANG_COLORS:
            tick.set_color(LANG_COLORS[code])
            tick.set_fontweight("bold")

# Domain colors for the cultural category strip.
DOMAIN_COLORS: dict[str, str] = {
    "values_norms":        "#4e79a7",
    "family_kinship":      "#59a14f",
    "religion_ritual":     "#9467bd",
    "daily_customs":       "#17becf",
    "food_cuisine":        "#ff7f0e",
    "festivals_holidays":  "#e377c2",
    "clothing_appearance": "#00bcd4",
    "governance_law":      "#7f7f7f",
    "social_identity":     "#8c564b",
    "symbols_colors":      "#d62728",
}


def build_control_matrix_figure(en: pd.DataFrame, latex_root: Path) -> None:
    med_rows = []
    for spec in CONTROL_SETTINGS:
        lang_vals = []
        for lang in CORE_LANGS:
            sub = en[
                (en["repr_type"] == "pre_lmhead_contextual")
                & (en["model_a"] == spec["model_a"])
                & (en["model_b"] == f"en_{lang}_{spec['setup']}")
            ]
            if not sub.empty:
                lang_vals.append(float(sub["axis_abs_projection_diff_mean"].iloc[0]))
        med_rows.append(float(np.median(lang_vals)) if lang_vals else np.nan)

    vals = np.array(med_rows, dtype=float)
    fig, ax = plt.subplots(figsize=(7.45, 3.10), facecolor=PAPER_BG)
    ax.set_facecolor(PAPER_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.025,
        0.965,
        "Control comparisons and what they rule out",
        ha="left",
        va="top",
        fontsize=10.4,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.025,
        0.902,
        r"All values are median contextual $\Delta D_{Axis}$ across eight EN vs EN+L2 families after English-anchor alignment.",
        ha="left",
        va="top",
        fontsize=10.0,
        color=MUTED,
    )

    rows = [
        {
            "name": "EN-matched\nshared-doc",
            "english": "matched\n1500 vs 1500\nEN steps",
            "updates": "not matched\n1500 vs 3000\nupdates",
            "docs": "shared",
            "test": "rules out\nless English\nevidence",
        },
        {
            "name": "EN-matched\ndisjoint-doc",
            "english": "matched\n1500 vs 1500\nEN steps",
            "updates": "not matched\n1500 vs 3000\nupdates",
            "docs": "disjoint",
            "test": "rules out\nless English +\nshared-doc overlap",
        },
        {
            "name": "Compute-matched\nshared-doc",
            "english": "not matched\n3000 vs 1500\nEN steps",
            "updates": "matched\n3000 vs 3000\nupdates",
            "docs": "shared",
            "test": "tests equal\noptimization\nbudget",
        },
        {
            "name": "Compute-matched\ndisjoint-doc",
            "english": "not matched\n3000 vs 1500\nEN steps",
            "updates": "matched\n3000 vs 3000\nupdates",
            "docs": "disjoint",
            "test": "tests equal\nbudget without\nshared documents",
        },
    ]

    left, right = 0.025, 0.985
    top = 0.805
    header_h = 0.085
    row_h = 0.145
    gap = 0.018
    col_edges = np.array([left, 0.175, 0.345, 0.515, 0.640, 0.865, right])
    headers = [
        "Comparison",
        "English evidence",
        "Total updates",
        "English docs",
        "Interpretation",
        r"Median $\Delta D$",
    ]

    header = FancyBboxPatch(
        (left, top - header_h),
        right - left,
        header_h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=CTRL_HEADER_BG,
        edgecolor=CTRL_HEADER_BORDER,
        linewidth=0.65,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(header)
    for x0 in col_edges[1:-1]:
        ax.plot([x0, x0], [top - header_h + 0.012, top - 0.012], color="#cbd4de", lw=0.45, transform=ax.transAxes)
    for i, hdr in enumerate(headers):
        x0, x1 = col_edges[i], col_edges[i + 1]
        ax.text(
            (x0 + x1) / 2,
            top - header_h / 2,
            hdr,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=INK,
        )

    y0 = top - header_h - gap
    for idx, (spec, row) in enumerate(zip(CONTROL_SETTINGS, rows)):
        y = y0 - idx * (row_h + gap)
        is_clean = spec["setup"] == "b" and spec["model_a"] == "en_50m"
        face = spec["fill"] if is_clean else blend_hex(str(spec["fill"]), (1, 1, 1), 0.20)
        edge = spec["color"] if is_clean else CTRL_ROW_BORDER
        row_box = FancyBboxPatch(
            (left, y - row_h),
            right - left,
            row_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2 if is_clean else 0.55,
            transform=ax.transAxes,
            zorder=0,
        )
        ax.add_patch(row_box)
        for xline in col_edges[1:-1]:
            ax.plot([xline, xline], [y - row_h + 0.014, y - 0.014], color=CTRL_DIVIDER, lw=0.45, transform=ax.transAxes, zorder=1)

        cell_text = [
            row["name"],
            row["english"],
            row["updates"],
            row["docs"],
            row["test"],
        ]
        aligns = ["left", "center", "center", "center", "left"]
        weights = ["bold", "normal", "normal", "bold", "normal"]
        colors_txt = [INK, INK, INK, spec["color"], INK]
        for cidx, txt in enumerate(cell_text):
            x0, x1 = col_edges[cidx], col_edges[cidx + 1]
            x_text = x0 + 0.012 if aligns[cidx] == "left" else (x0 + x1) / 2
            ax.text(
                x_text,
                y - row_h / 2,
                txt,
                transform=ax.transAxes,
                ha=aligns[cidx],
                va="center",
                fontsize=10.0,
                color=colors_txt[cidx],
                fontweight=weights[cidx],
                linespacing=1.03,
            )

        x0, x1 = col_edges[-2], col_edges[-1]
        chip_w = (x1 - x0) * 0.68
        chip_h = row_h * 0.49
        chip = FancyBboxPatch(
            ((x0 + x1) / 2 - chip_w / 2, y - row_h / 2 - chip_h / 2),
            chip_w,
            chip_h,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor="white",
            edgecolor=spec["color"],
            linewidth=0.9,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(chip)
        ax.text(
            (x0 + x1) / 2,
            y - row_h / 2,
            f"{vals[idx]:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.2,
            color=spec["color"],
            fontweight="bold",
            zorder=3,
        )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.03)
    fig.savefig(latex_root / "figures" / "fig2-control-matrix.pdf", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

FIGURE_NAME_MAP = {
    "exp1_controls.pdf": "fig2-exp1-divergence.pdf",
    "exp1_controls_100m.pdf": "fig15-appendix-exp1-divergence-100m.pdf",
    "exp2_overlap.pdf": "fig3-exp2-overlap.pdf",
    "exp2_overlap_100m.pdf": "fig16-appendix-exp2-overlap-100m.pdf",
    "combined_multilingual_fig5_fig11.pdf": "fig4-multilingual-validation.pdf",
    "combined_multilingual_fig5_fig11_100m.pdf": "fig17-appendix-multilingual-validation-100m.pdf",
    "exp4_signed_axes.pdf": "fig6-signed-axis-shifts.pdf",
    "exp4_signed_axes_100m.pdf": "fig21-appendix-signed-axis-shifts-100m.pdf",
    "exp4_ratio.pdf": "fig5-contextual-vs-embedding-ratio.pdf",
    "exp4_ratio_100m.pdf": "fig18-appendix-contextual-vs-embedding-100m.pdf",
    "exp4_layerwise.pdf": "fig7-layerwise-axis-divergence.pdf",
    "exp4_layerwise_100m.pdf": "fig19-appendix-layerwise-axis-divergence-100m.pdf",
    "exp5_alignment_methods.pdf": "fig8-alignment-method-comparison.pdf",
    "exp5_alignment_methods_100m.pdf": "fig20-appendix-alignment-method-comparison-100m.pdf",
    "main_dense_progress_summary.pdf": "fig8-dense-progress-summary.pdf",
    "appendix_perhead_heatmap.pdf": "fig9-appendix-perhead-heatmap.pdf",
    "category_heatmap.pdf": "fig10-appendix-category-heatmap.pdf",
    "appendix_multilingual_overview.pdf": "fig11-appendix-multilingual-overview.pdf",
    "appendix_multilingual_regression_scatter.pdf": "fig12-typology-regression-scatter.pdf",
    "appendix_l2_signed_hotspots_panel1.pdf": "fig13-appendix-l2-signed-hotspots-panel1.pdf",
    "appendix_l2_signed_hotspots_panel2.pdf": "fig14-appendix-l2-signed-hotspots-panel2.pdf",
    "appendix_dense_progress_trajectory.pdf": "fig22-appendix-dense-progress-trajectory.pdf",
    "norm_controlled_layerwise.pdf": "fig23-appendix-norm-controlled-layerwise.pdf",
}

TABLE_NAME_MAP = {
    "appendix_axes.tex": "tab1-appendix-axes.tex",
    "appendix_axis_grounding_part1.tex": "tab2-appendix-axis-grounding-part1.tex",
    "appendix_axis_grounding_part2.tex": "tab3-appendix-axis-grounding-part2.tex",
    "appendix_probe_qc.tex": "tab4-appendix-probe-qc.tex",
    "appendix_exp2_quality_tiers.tex": "tab5-appendix-exp2-quality-tiers.tex",
    "appendix_daxis_interpretation.tex": "tab6-appendix-daxis-interpretation.tex",
    "exp1_ci_summary_emb.tex": "tab7-exp1-ci-summary-embedding.tex",
    "exp1_ci_summary_ctx.tex": "tab8-exp1-ci-summary-contextual.tex",
    "exp1_negative_controls.tex": "tab9-exp1-negative-controls.tex",
    "exp1_procrustes.tex": "tab10-exp1-procrustes.tex",
    "exp2_overlap.tex": "tab11-exp2-overlap.tex",
    "exp2_wilcoxon.tex": "tab12-exp2-wilcoxon.tex",
    "exp4_ratio.tex": "tab13-exp3-ratio.tex",
    "main_multilingual_ratios.tex": "tab15-main-multilingual-ratios.tex",
    "exp4_layerwise.tex": "tab16-exp3-layerwise.tex",
    "exp5_alignment_methods.tex": "tab17-exp4-alignment-methods.tex",
    "appendix_contextual_alignment_variant.tex": "tab18-appendix-contextual-alignment-variant.tex",
    "appendix_perhead_top.tex": "tab19-appendix-perhead-top.tex",
    "exp1_hotspots.tex": "tab20-exp1-hotspots.tex",
    "appendix_multilingual_summary.tex": "tab21-appendix-multilingual-summary.tex",
    "appendix_multilingual_ratio_ci.tex": "tab22-appendix-multilingual-ratio-ci.tex",
    "appendix_multilingual_regression.tex": "tab23-appendix-multilingual-regression.tex",
    "exp4_signed_quadrants.tex": "tab24-exp3-signed-quadrants.tex",
    "same_language_controls.tex": "tab25-same-language-controls.tex",
    "framework_holdout.tex": "tab26-framework-holdout.tex",
    "progress_sensitivity.tex": "tab27-progress-sensitivity.tex",
    "anchor_sensitivity.tex": "tab28-anchor-sensitivity.tex",
    "tokenizer_check.tex": "tab29-tokenizer-check.tex",
    "scope_tests.tex": "tab30-aggregate-scope-tests.tex",
    "norm_controlled_axis.tex": "tab31-norm-controlled-axis.tex",
}

# No legacy figure sources are retained in publishing.
LEGACY_FIGURE_SOURCES_TO_KEEP: set[str] = set()

# Legacy artifacts produced by older paths that are not part of numbered publishing.
LEGACY_EXTRA_FIGURES = {
    "exp3_shared.png",
    "main_multilingual_regression.png",
}
LEGACY_EXTRA_TABLES = {
    "exp1_results.tex",
    "exp1_ci_summary.tex",
    "exp2_quality_tiers.tex",
    "exp3_shared.tex",
    "exp4_signed_top_axes.tex",
    "appendix_axis_grounding.tex",
}

EMIT_TABLES = True

MAIN_FIGURE_SOURCE_NAMES = {
    "exp1_controls.pdf",
    "exp1_controls_100m.pdf",
    "exp2_overlap.pdf",
    "exp2_overlap_100m.pdf",
    "combined_multilingual_fig5_fig11.pdf",
    "combined_multilingual_fig5_fig11_100m.pdf",
    "exp4_signed_axes.pdf",
    "exp4_signed_axes_100m.pdf",
    "exp4_ratio.pdf",
    "exp4_ratio_100m.pdf",
    "exp4_layerwise.pdf",
    "exp4_layerwise_100m.pdf",
    "exp5_alignment_methods.pdf",
    "exp5_alignment_methods_100m.pdf",
    "main_dense_progress_summary.pdf",
}


def parse_en_target(model_name: str) -> tuple[str, str] | None:
    if not model_name.startswith("en_") or not model_name.endswith(("_a", "_b")):
        return None
    parts = model_name.split("_")
    if len(parts) < 3:
        return None
    lang = parts[1]
    setup = parts[2]
    return lang, setup


def en_pair_label(model_a: str, model_b: str, include_setup: bool = False) -> str:
    parsed = parse_en_target(model_b)
    if parsed and model_a in {"en_50m", "en_100m"}:
        lang, setup = parsed
        base = "EN-50M" if model_a == "en_50m" else "EN-100M"
        label = f"{base} vs EN+{lang.upper()}"
        if include_setup:
            setup_label = {"a": "C3", "b": "C4"}.get(setup, setup.upper())
            label += f" ({setup_label})"
        return label
    return f"{model_a} vs {model_b}"


def add_agreement(df: pd.DataFrame, mode: str = "agreement") -> pd.DataFrame:
    x = df.copy()
    x["pair"] = x.apply(lambda r: f"{r['model_a']}__vs__{r['model_b']}", axis=1)
    if mode == "direct":
        # Use divergence-scale values directly (for EN-null-centered EN-subspace outputs).
        x["nn_agree"] = x["jaccard_at_k_mean"]
        x["struct_agree"] = x["frobenius_cultural_similarity"]
        x["axis_agree"] = x["axis_abs_projection_diff_mean"]
    else:
        x["nn_agree"] = 1.0 - x["jaccard_at_k_mean"]
        x["struct_agree"] = 1.0 / (1.0 + x["frobenius_cultural_similarity"])
        x["axis_agree"] = 1.0 / (1.0 + x["axis_abs_projection_diff_mean"])
    return x


def seed_null_metric_stats(
    ctrl_df: pd.DataFrame | None,
    baseline_label: str,
    eval_repr: str = "pre_lmhead_contextual",
    metric_col: str = "axis_abs_projection_diff_mean",
) -> dict[str, float] | None:
    if ctrl_df is None or ctrl_df.empty:
        return None
    need = {"language", "eval_repr", metric_col}
    if not need.issubset(set(ctrl_df.columns)):
        return None
    sub = ctrl_df[
        (ctrl_df["language"].astype(str) == baseline_label)
        & (ctrl_df["eval_repr"].astype(str) == eval_repr)
    ]
    if sub.empty:
        return None
    vals = sub[metric_col].astype(float).to_numpy()
    return {
        "mean": float(np.mean(vals)),
        "low": float(np.min(vals)),
        "high": float(np.max(vals)),
    }


def ensure_dirs(latex_root: Path) -> None:
    (latex_root / "tables").mkdir(parents=True, exist_ok=True)
    (latex_root / "figures").mkdir(parents=True, exist_ok=True)


def _convert_image_to_pdf(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".pdf":
        shutil.copy2(src, dst)
        return
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        rgb.save(dst, "PDF", resolution=300.0)


def publish_numbered_assets(
    latex_root: Path,
    include_tables: bool = True,
    figure_names: set[str] | None = None,
) -> None:
    figures_dir = latex_root / "figures"
    tables_dir = latex_root / "tables"

    for src_name, dst_name in FIGURE_NAME_MAP.items():
        if figure_names is not None and src_name not in figure_names:
            continue
        src = figures_dir / src_name
        dst = figures_dir / dst_name
        if src.exists():
            _convert_image_to_pdf(src, dst)

    if include_tables:
        for src_name, dst_name in TABLE_NAME_MAP.items():
            src = tables_dir / src_name
            dst = tables_dir / dst_name
            if src.exists():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Remove legacy-named intermediate outputs so the repo keeps a single naming scheme.
    for src_name in FIGURE_NAME_MAP:
        if figure_names is not None and src_name not in figure_names:
            continue
        if src_name in LEGACY_FIGURE_SOURCES_TO_KEEP:
            continue
        src = figures_dir / src_name
        if src.exists():
            src.unlink()
    if include_tables:
        for src_name in TABLE_NAME_MAP:
            src = tables_dir / src_name
            if src.exists():
                src.unlink()
    for name in LEGACY_EXTRA_FIGURES:
        p = figures_dir / name
        if p.exists():
            p.unlink()
    for name in LEGACY_EXTRA_TABLES:
        p = tables_dir / name
        if p.exists():
            p.unlink()


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def write_table(lines: list[str], out: Path) -> None:
    if not EMIT_TABLES:
        return
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _annotate_heatmap_adaptive(
    ax: plt.Axes, data: np.ndarray, cmap_name: str, norm: colors.Normalize, fontsize: int = 8
) -> None:
    cmap = plt.get_cmap(cmap_name)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            r, g, b, _ = cmap(norm(val))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def pretty_pair(pair: str) -> str:
    if "__vs__" not in pair:
        return pair
    left, right = pair.split("__vs__", 1)
    return PAIR_LABEL.get((left, right), en_pair_label(left, right, include_setup=True))


def en_families_in_df(en: pd.DataFrame, setup: str = "a") -> list[str]:
    fams = set()
    for mb in en["model_b"].astype(str).unique():
        parsed = parse_en_target(mb)
        if parsed is None:
            continue
        lang, found_setup = parsed
        if found_setup == setup:
            fams.add(lang)
    return sorted(fams)


def build_exp1_tables_fig(
    en: pd.DataFrame,
    ci: pd.DataFrame | None,
    latex_root: Path,
    same_lang_df: pd.DataFrame | None = None,
) -> None:
    families = en_families_in_df(en, setup="a")
    specs = []
    for base in ["en_50m", "en_100m"]:
        for fam in families:
            specs.append((base, fam, en_pair_label(base, f"en_{fam}_a")))

    rows = []
    for repr_type in ["embedding_matrix", "pre_lmhead_contextual"]:
        for left, fam, label in specs:
            sub = en[(en["repr_type"] == repr_type) & (en["model_a"] == left) & (en["model_b"] == f"en_{fam}_a")]
            if sub.empty:
                continue
            rows.append(
                {
                    "repr": REPR_LABEL[repr_type],
                    "repr_type": repr_type,
                    "comparison": label,
                    "nn": sub["nn_agree"].mean(),
                    "struct": sub["struct_agree"].mean(),
                    "axis": sub["axis_agree"].mean(),
                }
            )
    tdf = pd.DataFrame(rows)
    if tdf.empty:
        return

    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Representation & Comparison & $\Delta D_{NN}$ & $\Delta D_{Struct}$ & $\Delta D_{Axis}$ \\",
        r"\midrule",
    ]
    for repr_name in ["Embedding", "Contextual"]:
        sub = tdf[tdf["repr"] == repr_name].reset_index(drop=True)
        for i, r in sub.iterrows():
            left = repr_name if i == 0 else ""
            lines.append(f"{left} & {r['comparison']} & {_fmt(r['nn'])} & {_fmt(r['struct'])} & {_fmt(r['axis'])} \\\\")
        if repr_name == "Embedding":
            lines.append(r"\cmidrule(lr){1-5}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp1_results.tex")

    # Simplified main figure: contextual-only EN-50M/C3 D_Axis by language.
    ci_lookup: dict[str, tuple[float, float]] = {}
    if ci is not None and not ci.empty:
        csub_axis = ci[
            (ci["repr_type"] == "pre_lmhead_contextual")
            & (ci["model_a"] == "en_50m")
            & (ci["metric"] == "axis_abs_projection_diff_mean")
        ].copy()
        for _, rr in csub_axis.iterrows():
            parsed = parse_en_target(str(rr["model_b"]))
            if parsed is None or parsed[1] != "a":
                continue
            lang = parsed[0].upper()
            ci_lookup[lang] = (float(rr["ci_low"]), float(rr["ci_high"]))

    def _plot_ctx_slice(model_a: str, seed_lbl: str, out_name: str) -> None:
        rows_local = []
        ci_slice = {}
        if ci is not None and not ci.empty:
            csub_axis_local = ci[
                (ci["repr_type"] == "pre_lmhead_contextual")
                & (ci["model_a"] == model_a)
                & (ci["metric"] == "axis_abs_projection_diff_mean")
            ].copy()
            for _, rr in csub_axis_local.iterrows():
                parsed = parse_en_target(str(rr["model_b"]))
                if parsed is None or parsed[1] != "a":
                    continue
                ci_slice[parsed[0].upper()] = (float(rr["ci_low"]), float(rr["ci_high"]))
        for fam in families:
            sub = en[
                (en["repr_type"] == "pre_lmhead_contextual")
                & (en["model_a"] == model_a)
                & (en["model_b"] == f"en_{fam}_a")
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows_local.append(
                {
                    "lang": fam.upper(),
                    "daxis": float(r["axis_abs_projection_diff_mean"]),
                    "low": ci_slice.get(fam.upper(), (np.nan, np.nan))[0],
                    "high": ci_slice.get(fam.upper(), (np.nan, np.nan))[1],
                }
            )
        pdf = pd.DataFrame(rows_local)
        if pdf.empty:
            return
        ref = seed_null_metric_stats(
            same_lang_df,
            seed_lbl,
            eval_repr="pre_lmhead_contextual",
            metric_col="axis_abs_projection_diff_mean",
        )
        pdf = pdf.assign(is_ref=False)
        lang_order = [x.upper() for x in CORE_LANGS]
        pdf["lang_order"] = pdf["lang"].map(lambda x: lang_order.index(x) if x in lang_order else 999)
        pdf = pdf.sort_values("lang_order").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(7.15, 3.75))
        if ref is not None:
            ax.axhspan(
                ref["low"] - ref["mean"],
                ref["high"] - ref["mean"],
                color=REF_RANGE_BG,
                alpha=REF_RANGE_ALPHA,
                zorder=0,
            )
            ax.axhline(0.0, color="#555555", linestyle=(0, (4, 2)), linewidth=1.1, zorder=1)
        x = np.arange(len(pdf))
        bars = ax.bar(
            x,
            pdf["daxis"].to_numpy(),
            width=0.62,
            edgecolor="#222222",
            linewidth=0.85,
            alpha=0.96,
            zorder=2,
        )
        for b, (_, r) in zip(bars, pdf.iterrows()):
            lg = str(r["lang"]).upper()
            b.set_facecolor(pastel_lang_color(lg, lighten=0.30))
            b.set_hatch(LANG_HATCHES.get(lg, "//"))

        for i, rr in pdf.iterrows():
            if not np.isfinite(float(rr["low"])) or not np.isfinite(float(rr["high"])):
                continue
            lo = float(rr["low"])
            hi = float(rr["high"])
            center = float(rr["daxis"])
            lower = max(0.0, center - lo)
            upper = max(0.0, hi - center)
            ax.errorbar(
                [i],
                [center],
                yerr=[[lower], [upper]],
                fmt="none",
                ecolor="#2f3b4a",
                elinewidth=1.0,
                capsize=3.5,
                capthick=1.0,
                zorder=4,
            )
        ymax = float(np.nanmax(pdf["daxis"].to_numpy(dtype=float)))
        for rect, value in zip(bars, pdf["daxis"].to_numpy(dtype=float)):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                value + max(0.015, 0.025 * ymax),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.0,
                color="#222222",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(pdf["lang"].tolist(), fontsize=10.0, fontweight="bold")
        color_language_ticklabels(ax, axis="x")
        ax.set_xlabel("Language")
        ax.set_ylabel(r"EN-null-centered contextual $D_{Axis}$")
        ax.text(
            0.01,
            0.98,
            "zero = EN seed variation",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.0,
            color="#4f5965",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#c5ccd5", linewidth=0.6),
        )
        ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.52, alpha=0.40)
        ax.set_axisbelow(True)
        ax.set_ylim(top=max(ymax * 1.22, ymax + 0.08))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ref is not None:
            ax.legend(
                handles=[
                    plt.Line2D([0], [0], color="#555555", linestyle=(0, (4, 2)), lw=1.1, label="EN seed-null mean"),
                    plt.Rectangle((0, 0), 1, 1, facecolor=REF_RANGE_BG, alpha=REF_RANGE_ALPHA, edgecolor="none", label="EN seed-null range"),
                ],
                loc="upper left",
                frameon=True,
                edgecolor="#8a8a8a",
                framealpha=0.96,
                fontsize=10.0,
            )
        fig.tight_layout(pad=0.35)
        fig.subplots_adjust(left=0.14)
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    def _plot_control_trajectory(out_name: str) -> None:
        rows_h = []
        lang_labels = []
        for fam in CORE_LANGS:
            row = []
            for spec in CONTROL_SETTINGS:
                sub = en[
                    (en["repr_type"] == "pre_lmhead_contextual")
                    & (en["model_a"] == spec["model_a"])
                    & (en["model_b"] == f"en_{fam}_{spec['setup']}")
                ]
                row.append(float(sub["axis_abs_projection_diff_mean"].iloc[0]) if not sub.empty else np.nan)
            rows_h.append(row)
            lang_labels.append(fam.upper())
        mat = np.array(rows_h, dtype=float)
        if np.all(np.isnan(mat)):
            return
        fig, ax = plt.subplots(figsize=(COL_FIG_W, COL_FIG_H), facecolor=PAPER_BG)
        style_paper_axis(ax, grid_axis="y")
        x = np.arange(len(CONTROL_SETTINGS), dtype=float)
        clean_idx = 1
        ax.axvspan(clean_idx - 0.36, clean_idx + 0.36, color=SOFT_BLUE, alpha=0.94, zorder=0)
        ax.axhline(0.0, color="#6c7280", linestyle=(0, (3, 2)), linewidth=0.95, zorder=1)

        offsets = np.linspace(-0.13, 0.13, len(lang_labels))
        for i, (lang, row) in enumerate(zip(lang_labels, mat)):
            if np.all(np.isnan(row)):
                continue
            c = LANG_COLORS.get(lang, NAVY)
            ax.scatter(
                x + offsets[i],
                row,
                s=22,
                color=blend_hex(c, (1, 1, 1), 0.22),
                edgecolors=blend_hex(c, (0, 0, 0), 0.18),
                linewidths=0.6,
                alpha=0.92,
                zorder=3,
            )

        med = np.nanmedian(mat, axis=0)
        q20 = np.nanpercentile(mat, 20, axis=0)
        q80 = np.nanpercentile(mat, 80, axis=0)
        for xi, lo, hi in zip(x, q20, q80):
            ax.plot([xi, xi], [lo, hi], color=INK, linewidth=4.5, alpha=0.22, solid_capstyle="round", zorder=2)
        ax.plot(
            x,
            med,
            color=INK,
            linewidth=2.2,
            marker="D",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=1.0,
            zorder=5,
            label="Median across languages",
        )
        for xi, yi in zip(x, med):
            ax.text(
                xi,
                yi + 0.075,
                f"{yi:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.0,
                color=INK,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.08", facecolor=PAPER_BG, edgecolor="none", alpha=0.88),
                zorder=6,
            )
        ymax = float(np.nanmax(mat))
        ymin = min(0.0, float(np.nanmin(mat)))
        ax.set_ylim(ymin - 0.06, max(0.62, ymax * 1.22))
        ax.set_xlim(-0.34, len(CONTROL_SETTINGS) - 1 + 0.34)
        ax.set_xticks(x)
        ax.set_xticklabels(["EN\nshared", "EN\ndisjoint", "Compute\nshared", "Compute\ndisjoint"], fontsize=10.0, color=INK)
        ax.set_ylabel(r"Contextual $\Delta D_{Axis}$", fontsize=10.0, color=INK)
        ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=10.0)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            frameon=True,
            facecolor="white",
            edgecolor="#c5ccd7",
            framealpha=0.96,
            fontsize=10.0,
            handlelength=1.8,
        )
        fig.tight_layout(pad=0.45)
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_control_trajectory("exp1_controls.pdf")
    _plot_ctx_slice("en_100m", "EN-100M", "exp1_controls_100m.pdf")

    if ci is not None and not ci.empty:
        # compact CI table for exp1 pairs only
        keep_pairs = []
        for base in ["en_50m", "en_100m"]:
            for fam in families:
                keep_pairs.append((base, f"en_{fam}_a"))
        csub = ci[ci["metric"].isin(["jaccard_at_k_mean", "frobenius_cultural_similarity", "axis_abs_projection_diff_mean"])].copy()
        lines = [
            r"\begin{tabular}{@{}llll@{}}",
            r"\toprule",
            r"Representation & Pair & Metric & Mean [95\% CI] \\",
            r"\midrule",
        ]
        for rt in ["embedding_matrix", "pre_lmhead_contextual"]:
            c1 = csub[csub["repr_type"] == rt]
            pretty_rt = REPR_LABEL.get(rt, rt)
            wrote_any = False
            for ma, mb in keep_pairs:
                pair_df = c1[(c1["model_a"] == ma) & (c1["model_b"] == mb)]
                if pair_df.empty:
                    continue
                wrote_any = True
                pretty_pair = en_pair_label(ma, mb)
                for _, rr in pair_df.iterrows():
                    metric_short = {
                        "jaccard_at_k_mean": r"$\Delta D_{NN}$",
                        "frobenius_cultural_similarity": r"$\Delta D_{Struct}$",
                        "axis_abs_projection_diff_mean": r"$\Delta D_{Axis}$",
                    }[rr["metric"]]
                    lines.append(
                        f"{pretty_rt} & {pretty_pair} & {metric_short} & {_fmt(rr['mean'])} [{_fmt(rr['ci_low'])}, {_fmt(rr['ci_high'])}] \\\\")
                    pretty_rt = ""
                    pretty_pair = ""
            if wrote_any:
                lines.append(r"\cmidrule(lr){1-4}")
        if lines[-1] == r"\cmidrule(lr){1-4}":
            lines.pop()
        lines += [r"\bottomrule", r"\end{tabular}"]
        write_table(lines, latex_root / "tables" / "exp1_ci_summary.tex")

        # Per-representation appendix tables used by main.tex.
        for rt, out_name in [
            ("embedding_matrix", "exp1_ci_summary_emb.tex"),
            ("pre_lmhead_contextual", "exp1_ci_summary_ctx.tex"),
        ]:
            c1 = csub[csub["repr_type"] == rt].copy()
            if c1.empty:
                continue
            ls = [
                r"\begin{tabular}{@{}lll@{}}",
                r"\toprule",
                r"Pair & Metric & Mean [95\% CI] \\",
                r"\midrule",
            ]
            for ma, mb in keep_pairs:
                pair_df = c1[(c1["model_a"] == ma) & (c1["model_b"] == mb)]
                if pair_df.empty:
                    continue
                pair_lbl = en_pair_label(ma, mb)
                first_metric = True
                for _, rr in pair_df.iterrows():
                    metric_short = {
                        "jaccard_at_k_mean": r"$\Delta D_{NN}$",
                        "frobenius_cultural_similarity": r"$\Delta D_{Struct}$",
                        "axis_abs_projection_diff_mean": r"$\Delta D_{Axis}$",
                    }[rr["metric"]]
                    left_pair = pair_lbl if first_metric else ""
                    ls.append(
                        f"{left_pair} & {metric_short} & {_fmt(rr['mean'])} [{_fmt(rr['ci_low'])}, {_fmt(rr['ci_high'])}] \\\\"
                    )
                    first_metric = False
                ls.append(r"\cmidrule(lr){1-3}")
            if ls[-1] == r"\cmidrule(lr){1-3}":
                ls.pop()
            ls += [r"\bottomrule", r"\end{tabular}"]
            write_table(ls, latex_root / "tables" / out_name)


def build_exp1_hotspots_table(word_df: pd.DataFrame, latex_root: Path) -> None:
    core_pairs = []
    for p in sorted(word_df["pair"].astype(str).unique()):
        if "__vs__" not in p:
            continue
        ma, mb = p.split("__vs__", 1)
        if ma in {"en_50m", "en_100m"}:
            parsed = parse_en_target(mb)
            if parsed is not None and parsed[1] == "a":
                core_pairs.append(p)
    sub = word_df[
        (word_df["repr_type"] == "pre_lmhead_contextual")
        & (word_df["pair"].isin(core_pairs))
    ].copy()
    if sub.empty:
        return
    agg = sub.groupby("word", as_index=False)["jaccard_divergence"].mean()
    top = agg.sort_values("jaccard_divergence", ascending=False).head(10).reset_index(drop=True)
    bot = agg.sort_values("jaccard_divergence", ascending=True).head(10).reset_index(drop=True)

    lines = [
        r"\begin{tabular}{@{}r l r r l r@{}}",
        r"\toprule",
        r"\multicolumn{3}{c}{Highest contextual divergence} & \multicolumn{3}{c}{Lowest contextual divergence} \\",
        r"\cmidrule(lr){1-3}\cmidrule(lr){4-6}",
        r"Rank & Word & Mean $D_{NN}$ & Rank & Word & Mean $D_{NN}$ \\",
        r"\midrule",
    ]
    for i in range(10):
        lines.append(
            f"{i+1} & {top.loc[i, 'word']} & {_fmt(top.loc[i, 'jaccard_divergence'])} & "
            f"{i+1} & {bot.loc[i, 'word']} & {_fmt(bot.loc[i, 'jaccard_divergence'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp1_hotspots.tex")


def build_exp1_negative_controls_table(neg_df: pd.DataFrame | None, latex_root: Path) -> None:
    if neg_df is None or neg_df.empty:
        return
    c = neg_df.copy()
    c["pair_pretty"] = c["pair"].map(pretty_pair)
    c["group_pretty"] = c["group"].map({"cultural_probes": "Cultural probes", "negative_controls": "Negative controls"})

    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Representation & Group & $\Delta D_{NN}$ & $\Delta D_{Struct}$ & $\Delta D_{Axis}$ \\",
        r"\midrule",
    ]
    for rt in ["embedding_matrix", "pre_lmhead_contextual"]:
        srt = c[c["repr_type"] == rt]
        if srt.empty:
            continue
        # average across the four core pairs to keep compact
        g = srt.groupby("group_pretty", as_index=False)[["jaccard_at_k_mean", "frobenius_similarity", "axis_abs_projection_diff_mean"]].mean()
        tag = REPR_LABEL.get(rt, rt)
        for i, (_, r) in enumerate(g.iterrows()):
            left = tag if i == 0 else ""
            lines.append(
                f"{left} & {r['group_pretty']} & {_fmt(r['jaccard_at_k_mean'])} & {_fmt(r['frobenius_similarity'])} & {_fmt(r['axis_abs_projection_diff_mean'])} \\\\"
            )
        lines.append(r"\cmidrule(lr){1-5}")
    if lines[-1] == r"\cmidrule(lr){1-5}":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp1_negative_controls.tex")


def build_exp1_procrustes_table(en: pd.DataFrame, latex_root: Path) -> None:
    if "procrustes_anchor_residual_per_anchor" not in en.columns:
        return
    core = en[en["model_a"].isin(["en_50m", "en_100m"])].copy()
    core = core[
        core["model_b"].astype(str).map(
            lambda mb: parse_en_target(mb) is not None and parse_en_target(mb)[1] == "a"
        )
    ].copy()
    if core.empty:
        return
    lines = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Representation & Pair & Anchor residual / anchor & Anchor residual (Frobenius) \\",
        r"\midrule",
    ]
    for rt in ["embedding_matrix", "pre_lmhead_contextual"]:
        sub = core[core["repr_type"] == rt].copy()
        sub["pair_pretty"] = sub.apply(lambda r: en_pair_label(r["model_a"], r["model_b"]), axis=1)
        sub = sub.sort_values("pair_pretty")
        tag = REPR_LABEL.get(rt, rt)
        for i, (_, r) in enumerate(sub.iterrows()):
            left = tag if i == 0 else ""
            lines.append(
                f"{left} & {r['pair_pretty']} & {_fmt(r['procrustes_anchor_residual_per_anchor'])} & {_fmt(r['procrustes_anchor_residual_fro'])} \\\\"
            )
        lines.append(r"\cmidrule(lr){1-4}")
    if lines[-1] == r"\cmidrule(lr){1-4}":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp1_procrustes.tex")


def build_exp2_table_fig(
    en: pd.DataFrame,
    stats_df: pd.DataFrame | None,
    latex_root: Path,
    same_lang_df: pd.DataFrame | None = None,
) -> None:
    rows = []
    families = [(f"en_{lang}", f"EN+{lang.upper()}") for lang in en_families_in_df(en, setup="a")]
    for repr_type in ["embedding_matrix", "pre_lmhead_contextual"]:
        for base in ["en_50m", "en_100m"]:
            for fam_key, fam_name in families:
                a = en[(en["repr_type"] == repr_type) & (en["model_a"] == base) & (en["model_b"] == f"{fam_key}_a")]
                b = en[(en["repr_type"] == repr_type) & (en["model_a"] == base) & (en["model_b"] == f"{fam_key}_b")]
                if a.empty or b.empty:
                    continue
                ra, rb = a.iloc[0], b.iloc[0]
                rows.append(
                    {
                        "repr": REPR_LABEL[repr_type],
                        "family": fam_name,
                        "baseline": "EN-50M" if base == "en_50m" else "EN-100M",
                        "delta_nn": ra["nn_agree"] - rb["nn_agree"],
                        "delta_struct": ra["struct_agree"] - rb["struct_agree"],
                        "delta_axis": ra["axis_agree"] - rb["axis_agree"],
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return

    lines = [
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Representation & Family & Baseline & $\Delta D_{NN}$ & $\Delta D_{Struct}$ & $\Delta D_{Axis}$ \\",
        r"\midrule",
    ]
    for repr_name in ["Embedding", "Contextual"]:
        sub = df[df["repr"] == repr_name]
        for i, r in sub.reset_index(drop=True).iterrows():
            left = repr_name if i == 0 else ""
            lines.append(f"{left} & {r['family']} & {r['baseline']} & {_fmt(r['delta_nn'])} & {_fmt(r['delta_struct'])} & {_fmt(r['delta_axis'])} \\\\")
        if repr_name == "Embedding":
            lines.append(r"\cmidrule(lr){1-6}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp2_overlap.tex")

    # Simplified main figure: contextual-only EN-50M, shared (C3) vs disjoint (C4),
    # centered against EN-null so language offsets are immediately visible.
    def _plot_overlap_slice(model_a: str, seed_lbl: str, out_name: str) -> None:
        overlap_rows = []
        for fam_key, fam_name in families:
            a = en[
                (en["repr_type"] == "pre_lmhead_contextual")
                & (en["model_a"] == model_a)
                & (en["model_b"] == f"{fam_key}_a")
            ]
            b = en[
                (en["repr_type"] == "pre_lmhead_contextual")
                & (en["model_a"] == model_a)
                & (en["model_b"] == f"{fam_key}_b")
            ]
            if a.empty or b.empty:
                continue
            ra, rb = a.iloc[0], b.iloc[0]
            overlap_rows.append(
                {
                    "lang": fam_name.replace("EN+", ""),
                    "daxis_c3": float(ra["axis_abs_projection_diff_mean"]),
                    "daxis_c4": float(rb["axis_abs_projection_diff_mean"]),
                }
            )
        odf = pd.DataFrame(overlap_rows)
        if odf.empty:
            return
        lang_order = [f"EN+{x.upper()}".replace("EN+", "") for x in CORE_LANGS]
        odf["lang_order"] = odf["lang"].map(lambda x: lang_order.index(x) if x in lang_order else 999)
        odf = odf.sort_values("lang_order").reset_index(drop=True)

        ref = seed_null_metric_stats(
            same_lang_df,
            seed_lbl,
            eval_repr="pre_lmhead_contextual",
            metric_col="axis_abs_projection_diff_mean",
        )
        ref_low, ref_high = (np.nan, np.nan)
        if ref is not None:
            ref_low = ref["low"] - ref["mean"]
            ref_high = ref["high"] - ref["mean"]

        fig, ax = plt.subplots(figsize=(6.95, 4.10))
        y = np.arange(len(odf))
        all_vals = np.r_[odf["daxis_c3"].to_numpy(dtype=float), odf["daxis_c4"].to_numpy(dtype=float)]
        span = max(0.10, float(np.nanmax(all_vals) - np.nanmin(all_vals)))
        for i, rr in odf.iterrows():
            ax.plot(
                [rr["daxis_c3"], rr["daxis_c4"]],
                [i, i],
                color="#9aa4b0",
                linewidth=1.45,
                alpha=0.95,
                zorder=1,
            )
            lang = str(rr["lang"]).upper()
            c = pastel_lang_color(lang)
            ec = blend_hex(LANG_COLORS.get(lang, "#4C78A8"), (0, 0, 0), 0.30)
            ax.scatter([rr["daxis_c3"]], [i], s=44, marker="o", color=c, edgecolors=ec, linewidths=0.8, zorder=3)
            ax.scatter([rr["daxis_c4"]], [i], s=46, marker="s", facecolors="white", edgecolors=ec, linewidths=1.0, zorder=4)
            delta = float(rr["daxis_c4"] - rr["daxis_c3"])
            label_x = max(float(rr["daxis_c3"]), float(rr["daxis_c4"])) + 0.035 * span
            ax.text(
                label_x,
                i,
                f"{delta:+.2f}",
                va="center",
                ha="left",
                fontsize=10.0,
                color="#333333",
            )
        if ref is not None:
            ax.axvspan(ref_low, ref_high, color=OVERLAP_REF_BG, alpha=0.52, zorder=0)
            ax.axvline(0.0, color="#666666", linestyle=(0, (4, 2)), linewidth=1.1, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(odf["lang"].tolist(), fontsize=10.0, fontweight="bold")
        color_language_ticklabels(ax, axis="y")
        ax.set_ylabel("Language")
        ax.invert_yaxis()
        xmin = min(float(np.nanmin(all_vals)), 0.0 if ref is not None else float(np.nanmin(all_vals)))
        xmax = max(float(np.nanmax(all_vals)), 0.0 if ref is not None else float(np.nanmax(all_vals)))
        ax.set_xlim(xmin - 0.10 * span, xmax + 0.26 * span)
        ax.set_xlabel(r"EN-null-centered contextual $D_{Axis}$  (right label: disjoint minus shared)")
        ax.grid(axis="x", linestyle=(0, (3, 2)), linewidth=0.5, alpha=0.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#c7d8ea", markeredgecolor="#394a5a", label="Shared English docs", markersize=6),
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="#394a5a", label="Disjoint English docs", markersize=6),
        ]
        if ref is not None:
            handles.append(
                plt.Line2D([0], [0], color="#6b6b6b", linestyle=(0, (3, 2)), lw=1.2, label="EN-null mean/range")
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3 if ref is not None else 2,
            frameon=True,
            edgecolor="gray",
            fontsize=10.0,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    def _plot_overlap_delta_swarm(out_name: str) -> None:
        cols = [
            ("EN-50M", "Embedding", "en_50m", "embedding_matrix", 0.0),
            ("EN-50M", "Contextual", "en_50m", "pre_lmhead_contextual", 1.0),
            ("EN-100M", "Embedding", "en_100m", "embedding_matrix", 2.55),
            ("EN-100M", "Contextual", "en_100m", "pre_lmhead_contextual", 3.55),
        ]
        rows_delta = []
        for lang in CORE_LANGS:
            for base_lbl, repr_lbl, model_a, repr_type, xpos in cols:
                shared = en[
                    (en["repr_type"] == repr_type)
                    & (en["model_a"] == model_a)
                    & (en["model_b"] == f"en_{lang}_a")
                ]
                disjoint = en[
                    (en["repr_type"] == repr_type)
                    & (en["model_a"] == model_a)
                    & (en["model_b"] == f"en_{lang}_b")
                ]
                if shared.empty or disjoint.empty:
                    continue
                rows_delta.append(
                    {
                        "lang": lang.upper(),
                        "baseline": base_lbl,
                        "repr": repr_lbl,
                        "x": xpos,
                        "delta": float(disjoint["axis_abs_projection_diff_mean"].iloc[0] - shared["axis_abs_projection_diff_mean"].iloc[0]),
                    }
                )
        ddf = pd.DataFrame(rows_delta)
        if ddf.empty:
            return

        fig, ax = plt.subplots(figsize=(COL_FIG_W, COL_FIG_H), facecolor=PAPER_BG)
        style_paper_axis(ax, grid_axis="x")
        row_order = [
            ("EN-50M", "Embedding", "50M Emb."),
            ("EN-50M", "Contextual", "50M Ctx."),
            ("EN-100M", "Embedding", "100M Emb."),
            ("EN-100M", "Contextual", "100M Ctx."),
        ]
        ypos = {key: i for i, key in enumerate((b, r) for b, r, _ in row_order)}
        ax.axvline(0.0, color="#5f6672", linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        ax.axhspan(-0.45, 1.45, color=SOFT_BLUE, alpha=0.66, zorder=0)
        ax.axhspan(1.55, 3.45, color=SOFT_GRAY, alpha=0.82, zorder=0)
        offsets = np.linspace(-0.34, 0.34, len(CORE_LANGS))
        offset_map = {lang.upper(): offsets[i] for i, lang in enumerate(CORE_LANGS)}
        for _, rr in ddf.iterrows():
            lang = str(rr["lang"]).upper()
            yrow = ypos[(str(rr["baseline"]), str(rr["repr"]))] + offset_map.get(lang, 0.0)
            c = LANG_COLORS.get(lang, NAVY)
            marker = "o" if rr["repr"] == "Embedding" else "s"
            face = blend_hex(c, (1, 1, 1), 0.25) if rr["repr"] == "Contextual" else "white"
            ax.scatter(
                [float(rr["delta"])],
                [yrow],
                s=18 if marker == "s" else 15,
                marker=marker,
                facecolors=face,
                edgecolors=blend_hex(c, (0, 0, 0), 0.18),
                linewidths=0.72,
                alpha=0.96,
                zorder=4,
            )
        med_table = ddf.groupby(["baseline", "repr"], as_index=False).agg(median_delta=("delta", "median"))
        for _, rr in med_table.iterrows():
            yrow = ypos[(str(rr["baseline"]), str(rr["repr"]))]
            medv = float(rr["median_delta"])
            ax.plot([medv, medv], [yrow + 0.39, yrow + 0.51], color=INK, linewidth=1.8, solid_capstyle="round", zorder=5)
        xabs = max(0.18, float(np.nanpercentile(np.abs(ddf["delta"].to_numpy(dtype=float)), 99)) * 1.18)
        ax.set_xlim(-xabs, xabs)
        ax.set_ylim(-0.55, 3.92)
        ax.invert_yaxis()
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels([lbl for _, _, lbl in row_order], fontsize=10.0, color=INK)
        ax.set_xlabel(r"Disjoint minus shared $\Delta D_{Axis}$", fontsize=10.0, color=INK)
        ax.tick_params(axis="x", labelsize=10.0)
        ax.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="white",
                    markeredgecolor="#526273",
                    markeredgewidth=0.8,
                    label="Languages",
                    markersize=4.6,
                ),
                plt.Line2D([0], [0], color=INK, lw=1.8, label="Median tick"),
            ],
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="#c5ccd7",
            framealpha=0.96,
            fontsize=10.0,
            borderpad=0.24,
            handlelength=1.25,
            labelspacing=0.28,
        )
        fig.tight_layout(pad=0.35)
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_overlap_delta_swarm("exp2_overlap.pdf")
    _plot_overlap_slice("en_100m", "EN-100M", "exp2_overlap_100m.pdf")

    if stats_df is not None and not stats_df.empty:
        base_label_map = {"en_50m": "EN-50M", "en_100m": "EN-100M"}
        fam_label_map = {f"en_{lang}": f"EN+{lang.upper()}" for lang in en_families_in_df(en, setup="a")}
        if "quality_tier" in stats_df.columns:
            stats_df = stats_df[stats_df["quality_tier"] == "all"].copy()
        ls = [
            r"\begin{tabular}{@{}llllrr@{}}",
            r"\toprule",
            r"Representation & Baseline & Family & $n$ & Med.\ $\Delta D_{NN}$ & $p$ \\",
            r"\midrule",
        ]
        for rt in ["embedding_matrix", "pre_lmhead_contextual"]:
            sub = stats_df[stats_df["repr_type"] == rt].reset_index(drop=True)
            p_rt = REPR_LABEL.get(rt, rt)
            for i, rr in sub.iterrows():
                left = p_rt if i == 0 else ""
                base = base_label_map.get(str(rr.get("baseline", "")), str(rr.get("baseline", "")))
                fam = fam_label_map.get(str(rr.get("family", "")), str(rr.get("family", "")))
                ls.append(
                    f"{left} & {base} & {fam} & {int(rr['n_words'])} & {_fmt(rr['median_delta_jaccard'])} & {rr['p_value']:.2e} \\\\")
            if rt == "embedding_matrix":
                ls.append(r"\cmidrule(lr){1-6}")
        ls += [r"\bottomrule", r"\end{tabular}"]
        write_table(ls, latex_root / "tables" / "exp2_wilcoxon.tex")


def build_exp2_quality_table(stats_df: pd.DataFrame | None, latex_root: Path) -> None:
    if stats_df is None or stats_df.empty or "quality_tier" not in stats_df.columns:
        return
    s = stats_df[
        stats_df["quality_tier"].isin(["high", "medium", "low"])
        & (stats_df["repr_type"] == "pre_lmhead_contextual")
    ].copy()
    if s.empty:
        return
    s["baseline"] = s["baseline"].map({"en_50m": "EN-50M", "en_100m": "EN-100M"}).fillna(s["baseline"])
    fam_map = {f"en_{lang}": f"EN+{lang.upper()}" for lang in CORE_LANGS}
    s["family"] = s["family"].map(fam_map).fillna(s["family"])
    s["quality_tier"] = s["quality_tier"].map(
        {"high": "High QE", "medium": "Medium QE", "low": "Low QE"}
    ).fillna(s["quality_tier"])
    lines = [
        r"\begin{tabular}{@{}llllr@{}}",
        r"\toprule",
        r"Baseline & Family & Quality tier & $n$ & Wilcoxon $p$ \\",
        r"\midrule",
    ]
    for _, r in s.sort_values(["baseline", "family", "quality_tier"]).iterrows():
        lines.append(
            f"{r['baseline']} & {r['family']} & {r['quality_tier']} & {int(r['n_words'])} & {float(r['p_value']):.2e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp2_quality_tiers.tex")
    write_table(lines, latex_root / "tables" / "appendix_exp2_quality_tiers.tex")


def build_appendix_daxis_interpretation(en: pd.DataFrame, latex_root: Path) -> None:
    core = en[
        en["model_a"].isin(["en_50m", "en_100m"])
        & en["repr_type"].isin(["embedding_matrix", "pre_lmhead_contextual"])
    ].copy()
    core = core[
        core["model_b"].astype(str).map(
            lambda mb: parse_en_target(mb) is not None and parse_en_target(mb)[1] == "a"
        )
    ].copy()
    if core.empty:
        return

    piv = core.pivot_table(
        index=["model_a", "model_b"],
        columns="repr_type",
        values="axis_abs_projection_diff_mean",
        aggfunc="mean",
    ).reset_index()
    if "embedding_matrix" not in piv.columns or "pre_lmhead_contextual" not in piv.columns:
        return

    piv["pair"] = piv.apply(lambda r: en_pair_label(r["model_a"], r["model_b"]), axis=1)
    piv["gap"] = piv["pre_lmhead_contextual"] - piv["embedding_matrix"]
    piv = piv.sort_values("pair")

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Pair & Emb. & Ctx. & Difference \\",
        r"\midrule",
    ]
    for _, r in piv.iterrows():
        lines.append(
            f"{r['pair']} & {_fmt(r['embedding_matrix'])} & {_fmt(r['pre_lmhead_contextual'])} & {_fmt(r['gap'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "appendix_daxis_interpretation.tex")


def build_exp3_table_fig(zh: pd.DataFrame, fr: pd.DataFrame, latex_root: Path) -> None:
    rows = []
    for target, d in [("ZH", zh), ("FR", fr)]:
        for repr_type in ["embedding_matrix", "pre_lmhead_contextual"]:
            for mono in [f"{target.lower()}_50m", f"{target.lower()}_100m"]:
                for setup in ["a", "b"]:
                    bi = f"en_{target.lower()}_{setup}"
                    sub = d[(d["repr_type"] == repr_type) & (d["model_a"] == mono) & (d["model_b"] == bi)]
                    if sub.empty:
                        continue
                    r = sub.iloc[0]
                    rows.append(
                        {
                            "target": target,
                            "repr": REPR_LABEL[repr_type],
                            "mono": "50M" if mono.endswith("50m") else "100M",
                            "setup": setup.upper(),
                            "nn": r["nn_agree"],
                            "struct": r["struct_agree"],
                            "axis": r["axis_agree"],
                        }
                    )
    df = pd.DataFrame(rows)
    lines = [
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Target & Representation & Setup & $A_{NN}$ & $A_{Struct}$ & $A_{Axis}$ \\",
        r"\midrule",
    ]
    for target in ["ZH", "FR"]:
        tsub = df[df["target"] == target]
        for mono in ["50M", "100M"]:
            mono_sub = tsub[tsub["mono"] == mono].reset_index(drop=True)
            for i, r in mono_sub.iterrows():
                left = f"{target} ({mono})" if i == 0 else ""
                lines.append(f"{left} & {r['repr']} & {r['setup']} & {_fmt(r['nn'])} & {_fmt(r['struct'])} & {_fmt(r['axis'])} \\\\")
        if target == "ZH":
            lines.append(r"\cmidrule(lr){1-6}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp3_shared.tex")

    agg = df.groupby(["target", "repr"], as_index=False)[["nn", "struct", "axis"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), gridspec_kw={"wspace": 0.38})
    metrics = [("nn", r"$A_{NN}$"), ("struct", r"$A_{Struct}$"), ("axis", r"$A_{Axis}$")]
    for ax, target in zip(axes, ["ZH", "FR"]):
        sub = agg[agg["target"] == target]
        x = np.arange(len(metrics))
        width = 0.34
        for j, repr_name in enumerate(["Embedding", "Contextual"]):
            short_label = repr_name
            vals = [sub[sub["repr"] == repr_name][m].iloc[0] for m, _ in metrics]
            bars = ax.bar(
                x + (j - 0.5) * width,
                vals,
                width,
                label=short_label,
                color=COLORS["embedding_matrix" if j == 0 else "pre_lmhead_contextual"],
                edgecolor="black",
                linewidth=0.7,
            )
            for b in bars:
                b.set_hatch(HATCH["embedding_matrix" if j == 0 else "pre_lmhead_contextual"])
        ax.set_title(f"{target} target language", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Agreement (higher is better)")
        ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.5, alpha=0.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="lower right", frameon=True, edgecolor="gray", fontsize=10)
    fig.savefig(latex_root / "figures" / "exp3_shared.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def build_exp4_ratio(en: pd.DataFrame, ci: pd.DataFrame | None, latex_root: Path) -> None:
    # Use setup A for the headline representation-family gap. We intentionally
    # report absolute centered values plus a difference rather than ratios,
    # because centered embedding denominators can be close to zero.
    pairs = []
    families = en_families_in_df(en, setup="a")
    for ma in ["en_50m", "en_100m"]:
        for fam in families:
            mb = f"en_{fam}_a"
            pairs.append((ma, mb, en_pair_label(ma, mb)))
    rows = []
    for ma, mb, label in pairs:
        e = en[(en["repr_type"] == "embedding_matrix") & (en["model_a"] == ma) & (en["model_b"] == mb)]
        c = en[(en["repr_type"] == "pre_lmhead_contextual") & (en["model_a"] == ma) & (en["model_b"] == mb)]
        if e.empty or c.empty:
            continue
        e_axis = float(e["axis_abs_projection_diff_mean"].iloc[0])
        c_axis = float(c["axis_abs_projection_diff_mean"].iloc[0])
        rows.append({
            "pair": label,
            "embedding_axis": e_axis,
            "contextual_axis": c_axis,
            "gap": c_axis - e_axis,
        })
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return

    def _plot_repr_gap_slice(base_lbl: str, out_name: str) -> None:
        plot_df = rdf[rdf["pair"].str.startswith(base_lbl)].copy()
        if plot_df.empty:
            return
        plot_df["lang"] = plot_df["pair"].map(lambda x: x.split(" vs ")[1].replace("EN+", ""))
        lang_rank = {lang.upper(): i for i, lang in enumerate([l.upper() for l in CORE_LANGS])}
        plot_df["lang_rank"] = plot_df["lang"].map(lambda x: lang_rank.get(str(x).upper(), 999))
        plot_df = plot_df.sort_values("lang_rank").reset_index(drop=True)
        plot_df["delta_ctx_minus_emb"] = plot_df["contextual_axis"] - plot_df["embedding_axis"]
        y = np.arange(len(plot_df))

        fig, ax = plt.subplots(figsize=(COL_FIG_W, COL_FIG_H), facecolor=PAPER_BG)
        style_paper_axis(ax, grid_axis="x")
        ax.axvline(0.0, color="#666666", linestyle=(0, (4, 2)), linewidth=1.0, zorder=0)
        for i, rr in plot_df.iterrows():
            lang = str(rr["lang"]).upper()
            c = pastel_lang_color(lang)
            ec = blend_hex(LANG_COLORS.get(lang, "#4C78A8"), (0, 0, 0), 0.35)
            ax.plot(
                [rr["embedding_axis"], rr["contextual_axis"]],
                [i, i],
                color="#9aa6b4",
                linewidth=2.4,
                alpha=0.95,
                zorder=1,
            )
            ax.scatter(
                [rr["embedding_axis"]],
                [i],
                s=42,
                marker="o",
                facecolors="white",
                edgecolors=ec,
                linewidths=1.0,
                zorder=3,
            )
            ax.scatter(
                [rr["contextual_axis"]],
                [i],
                s=52,
                marker="s",
                facecolors=c,
                edgecolors=ec,
                linewidths=0.9,
                zorder=4,
            )
            ax.text(
                rr["contextual_axis"] + 0.025 * max(0.3, float(plot_df["contextual_axis"].max())),
                i,
                f"+{rr['delta_ctx_minus_emb']:.2f}",
                va="center",
                ha="left",
                fontsize=10.0,
                color="#3a3a3a",
            )
        xmax = float(max(plot_df["contextual_axis"].max(), plot_df["embedding_axis"].max()))
        xmin = float(min(0.0, plot_df["embedding_axis"].min()))
        ax.set_xlim(xmin - 0.05 * max(0.5, xmax), xmax * 1.20)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["lang"].str.upper().tolist(), fontsize=10.0, fontweight="bold")
        color_language_ticklabels(ax, axis="y")
        ax.invert_yaxis()
        ax.set_xlabel(
            f"EN-null-centered $D_{{Axis}}$ ({base_lbl}, shared docs)",
            labelpad=4,
            fontsize=10.0,
        )
        ax.set_ylabel("Language", fontsize=10.0)
        ax.tick_params(axis="x", labelsize=10.0)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#4c5a67", label="Token embedding", markersize=6),
                plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#c9dff2", markeredgecolor="#4c5a67", label="Contextual state", markersize=6),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#c5ccd7",
            framealpha=0.96,
            fontsize=10.0,
        )
        fig.tight_layout(pad=0.35)
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_repr_gap_slice("EN-50M", "exp4_ratio.pdf")
    _plot_repr_gap_slice("EN-100M", "exp4_ratio_100m.pdf")

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Pair & $\Delta D_A$ (Embedding) & $\Delta D_A$ (Contextual) & Ctx.-Emb. \\",
        r"\midrule",
    ]
    for _, r in rdf.iterrows():
        lines.append(
            f"{r['pair']} & {_fmt(r['embedding_axis'])} & {_fmt(r['contextual_axis'])} & {_fmt(r['gap'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp4_ratio.tex")


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return r"$<10^{-4}$"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def build_exp5_alignment_method_artifacts(aln_df: pd.DataFrame | None, latex_root: Path) -> None:
    if aln_df is None or aln_df.empty:
        return
    d = aln_df.copy()
    d = d[d["model_a"].isin(["en_50m", "en_100m"])].copy()
    if d.empty:
        return

    d["pair"] = d.apply(lambda r: en_pair_label(str(r["model_a"]), str(r["model_b"])), axis=1)
    d["repr"] = d["eval_repr"].map(
        {
            "embedding_matrix": "Embedding",
            "pre_lmhead_contextual": "Contextual",
        }
    ).fillna(d["eval_repr"])

    def _method_label(row: pd.Series) -> str:
        method = str(row["alignment_method"]).lower()
        source = str(row["alignment_source"])
        if method == "orthogonal" and source == "embedding_matrix":
            return "Orthogonal\n(Embedding anchors)"
        if method == "orthogonal" and source == "pre_lmhead_contextual":
            return "Orthogonal\n(Contextual anchors)"
        if method == "affine" and source == "embedding_matrix":
            return "Affine\n(Embedding anchors)"
        return f"{row['alignment_method']}\n({source})"

    d["method"] = d.apply(_method_label, axis=1)
    method_order = [
        "Orthogonal\n(Embedding anchors)",
        "Orthogonal\n(Contextual anchors)",
        "Affine\n(Embedding anchors)",
    ]

    def _plot_alignment_slice(model_a: str, out_name: str) -> None:
        sub = d[(d["repr"] == "Contextual") & (d["model_a"] == model_a)].copy()
        if sub.empty:
            fig, ax = plt.subplots(figsize=(6.8, 4.4))
            ax.text(0.5, 0.5, "Alignment comparison data unavailable", ha="center", va="center")
            ax.axis("off")
            fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
            plt.close(fig)
            return
        sub["lang"] = sub["model_b"].astype(str).map(lambda x: parse_en_target(x)[0].upper() if parse_en_target(x) else x)
        sub = sub[
            (
                (sub["alignment_method"].astype(str).str.lower() == "orthogonal")
                & (sub["alignment_source"].astype(str) == "embedding_matrix")
            )
            | (
                (sub["alignment_method"].astype(str).str.lower() == "affine")
                & (sub["alignment_source"].astype(str) == "embedding_matrix")
            )
        ].copy()
        if sub.empty:
            fig, ax = plt.subplots(figsize=(6.8, 4.4))
            ax.text(0.5, 0.5, "Orthogonal/Affine contextual rows unavailable", ha="center", va="center")
            ax.axis("off")
            fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
            plt.close(fig)
            return
        sub["method_short"] = sub["alignment_method"].astype(str).str.lower().map(
            {"orthogonal": "Orthogonal", "affine": "Affine"}
        )
        piv = sub.pivot_table(
            index="lang",
            columns="method_short",
            values="axis_abs_projection_diff_mean",
            aggfunc="mean",
        ).reset_index()
        if not {"Orthogonal", "Affine"}.issubset(set(piv.columns)):
            fig, ax = plt.subplots(figsize=(6.8, 4.4))
            ax.text(0.5, 0.5, "Orthogonal/Affine contextual rows unavailable", ha="center", va="center")
            ax.axis("off")
            fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
            plt.close(fig)
            return
        lang_rank = {lang.upper(): i for i, lang in enumerate([l.upper() for l in CORE_LANGS])}
        piv["lang_rank"] = piv["lang"].map(lambda x: lang_rank.get(str(x).upper(), 999))
        piv = piv.sort_values("lang_rank").reset_index(drop=True)
        piv["delta_affine_minus_orth"] = piv["Affine"] - piv["Orthogonal"]
        y = np.arange(len(piv))

        fig, ax = plt.subplots(figsize=(COL_FIG_W, COL_FIG_H), facecolor=PAPER_BG)
        style_paper_axis(ax, grid_axis="x")
        ax.axvline(0.0, color="#666666", linestyle=(0, (4, 2)), linewidth=1.0, zorder=0)
        for i, rr in piv.iterrows():
            lang = str(rr["lang"]).upper()
            c = pastel_lang_color(lang)
            ec = blend_hex(LANG_COLORS.get(lang, "#4C78A8"), (0, 0, 0), 0.35)
            ax.plot(
                [rr["Orthogonal"], rr["Affine"]],
                [i, i],
                color="#a7b2be",
                linewidth=2.2,
                alpha=0.95,
                zorder=1,
            )
            ax.scatter([rr["Orthogonal"]], [i], s=42, marker="o", facecolors="white", edgecolors=ec, linewidths=1.0, zorder=3)
            ax.scatter([rr["Affine"]], [i], s=48, marker="s", facecolors=c, edgecolors=ec, linewidths=0.9, zorder=4)
            label_x = max(float(rr["Orthogonal"]), float(rr["Affine"])) + 0.02 * max(
                0.2,
                float(np.max(np.abs(np.r_[piv["Orthogonal"].to_numpy(), piv["Affine"].to_numpy()]))),
            )
            ax.text(
                label_x,
                i,
                f"{rr['delta_affine_minus_orth']:+.2f}",
                va="center",
                ha="left",
                fontsize=10.0,
                color="#3a3a3a",
            )
        xmax = float(np.max(np.r_[piv["Orthogonal"].to_numpy(), piv["Affine"].to_numpy()]))
        xmin = float(min(0.0, np.min(np.r_[piv["Orthogonal"].to_numpy(), piv["Affine"].to_numpy()])))
        ax.set_xlim(xmin - 0.05 * max(0.3, xmax), xmax * 1.18)
        ax.set_yticks(y)
        ax.set_yticklabels(piv["lang"].astype(str).str.upper().tolist(), fontsize=10.0, fontweight="bold")
        color_language_ticklabels(ax, axis="y")
        ax.invert_yaxis()
        ax.set_xlabel(
            f"Contextual $\Delta D_{{Axis}}$ by alignment ({model_a.replace('en_', 'EN-').upper()}, shared docs)",
            fontsize=10.0,
        )
        ax.set_ylabel("Language", fontsize=10.0)
        ax.tick_params(axis="x", labelsize=10.0)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#4c5a67", label="Orthogonal", markersize=6),
                plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#c9dff2", markeredgecolor="#4c5a67", label="Affine", markersize=6),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#c5ccd7",
            framealpha=0.96,
            fontsize=10.0,
        )
        fig.tight_layout(pad=0.35)
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_alignment_slice("en_50m", "exp5_alignment_methods.pdf")
    _plot_alignment_slice("en_100m", "exp5_alignment_methods_100m.pdf")

    # Appendix table with compact numeric summary.
    metrics = [
        ("jaccard_at_k_mean", r"$\Delta D_{NN}$"),
        ("frobenius_cultural_similarity", r"$\Delta D_{Struct}$"),
        ("axis_abs_projection_diff_mean", r"$\Delta D_{Axis}$"),
    ]
    lines = [
        r"\begin{tabular}{@{}llllrr@{}}",
        r"\toprule",
        r"Representation & Alignment method & Metric & Mean & Std. dev. & Anchor residual/anchor \\",
        r"\midrule",
    ]
    for repr_name in ["Embedding", "Contextual"]:
        srepr = d[d["repr"] == repr_name]
        for method in method_order:
            sm = srepr[srepr["method"] == method]
            if sm.empty:
                continue
            residual = float(sm["anchor_residual_per_anchor"].mean())
            first = True
            for metric_col, metric_lbl in metrics:
                vals = sm[metric_col].astype(float)
                left_repr = repr_name if first else ""
                left_method = method.replace("\n", " ") if first else ""
                lines.append(
                    f"{left_repr} & {left_method} & {metric_lbl} & {_fmt(vals.mean())} & {_fmt(vals.std(ddof=0))} & {_fmt(residual)} \\\\"
                )
                first = False
            lines.append(r"\cmidrule(lr){1-6}")
    if lines[-1] == r"\cmidrule(lr){1-6}":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp5_alignment_methods.tex")


def build_exp4_signed_axis_scatter(
    axis_df: pd.DataFrame | None,
    latex_root: Path,
    output_root: Path | None = None,
    probe_set: dict | None = None,
) -> None:
    if axis_df is None or axis_df.empty or "mean_signed_projection_diff" not in axis_df.columns:
        return

    def _row_norm(x: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(x, axis=1, keepdims=True)
        d = np.maximum(d, 1e-12)
        return x / d

    # Main Figure 5: sign-flip hotspots over (probe, axis), with language columns.
    # This explicitly surfaces cases like probe direction reversal across languages.
    built_main = False
    if output_root is not None and probe_set is not None:
        repr_dir = output_root / "en_ablation" / "representations"
        req = {
            "all_probe_words",
            "neutral_anchor_words",
            "cultural_probe_words",
            "semantic_axes",
        }
        if repr_dir.exists() and req.issubset(set(probe_set.keys())):
            words = list(probe_set["all_probe_words"])
            w2i = {w: i for i, w in enumerate(words)}
            neutral_idx = np.array([w2i[w] for w in probe_set["neutral_anchor_words"] if w in w2i], dtype=np.int64)
            cultural_words = [w for w in probe_set["cultural_probe_words"] if w in w2i]
            cultural_idx = np.array([w2i[w] for w in cultural_words], dtype=np.int64)
            probe_to_cat = probe_set.get("cultural_probe_categories", {})
            axis_cat_lookup: dict[tuple[str, str], str] = {}
            for m in probe_set.get("semantic_axis_metadata", []):
                a = m.get("endpoint_1")
                b = m.get("endpoint_2")
                c = m.get("category")
                if isinstance(a, str) and isinstance(b, str) and isinstance(c, str):
                    axis_cat_lookup[(a, b)] = c
            axis_meta = []
            for a, b in probe_set["semantic_axes"]:
                if a in w2i and b in w2i:
                    axis_cat = axis_cat_lookup.get((a, b), axis_cat_lookup.get((b, a), ""))
                    axis_meta.append((a, b, axis_cat))

            # Compatibility map to keep probe-axis rows semantically relevant.
            compat = {
                "values_norms": {"values_norms"},
                "family_kinship": {"family_kinship", "clothing_appearance", "symbols_colors"},
                "religion_ritual": {"religion_ritual", "festivals_holidays", "symbols_colors"},
                "food_cuisine": {"food_cuisine"},
                "festivals_holidays": {"festivals_holidays", "religion_ritual", "symbols_colors"},
                "clothing_appearance": {"clothing_appearance", "symbols_colors"},
                "symbols_colors": {"symbols_colors", "clothing_appearance", "religion_ritual", "festivals_holidays"},
                "governance_law": {"governance_law", "social_identity"},
                "social_identity": {"social_identity", "governance_law", "values_norms"},
                "daily_customs": {"daily_customs", "values_norms", "family_kinship", "religion_ritual", "festivals_holidays"},
            }

            if len(neutral_idx) >= 10 and len(cultural_idx) >= 50 and len(axis_meta) > 0:
                a_path = repr_dir / "en_50m__pre_lmhead_contextual.npy"
                if a_path.exists():
                    Xa = _row_norm(np.load(a_path))
                    records: list[dict[str, object]] = []
                    for lang in CORE_LANGS:
                        b_path = repr_dir / f"en_{lang}_a__pre_lmhead_contextual.npy"
                        if not b_path.exists():
                            continue
                        Xb = _row_norm(np.load(b_path))

                        # Orthogonal alignment on neutral anchors.
                        M = Xa[neutral_idx].T @ Xb[neutral_idx]
                        U, _, Vt = np.linalg.svd(M, full_matrices=False)
                        W = U @ Vt
                        Xa_aligned = Xa @ W

                        for a_word, b_word, axis_cat in axis_meta:
                            i, j = w2i[a_word], w2i[b_word]
                            va = Xa_aligned[j] - Xa_aligned[i]
                            vb = Xb[j] - Xb[i]
                            na = np.linalg.norm(va)
                            nb = np.linalg.norm(vb)
                            if na < 1e-12 or nb < 1e-12:
                                continue
                            va = va / na
                            vb = vb / nb

                            pa = Xa_aligned[cultural_idx] @ va
                            pb = Xb[cultural_idx] @ vb
                            signed = pa - pb
                            for pw, sv in zip(cultural_words, signed):
                                probe_cat = probe_to_cat.get(pw, "")
                                if axis_cat and probe_cat:
                                    allowed = compat.get(probe_cat, {probe_cat})
                                    if axis_cat not in allowed:
                                        continue
                                records.append(
                                    {
                                        "lang": lang.upper(),
                                        "axis": f"{a_word}->{b_word}",
                                        "probe": pw,
                                        "signed": float(sv),
                                        "probe_axis": f"{pw} | {a_word}->{b_word}",
                                    }
                                )

                    if records:
                        long_df = pd.DataFrame(records)
                        mat = long_df.pivot_table(
                            index="probe_axis",
                            columns="lang",
                            values="signed",
                            aggfunc="mean",
                        )
                        cols = [c.upper() for c in CORE_LANGS if c.upper() in mat.columns]
                        mat = mat[cols]
                        if not mat.empty:
                            eps = 0.05
                            agg = pd.DataFrame(index=mat.index)
                            agg["n_pos"] = (mat > eps).sum(axis=1)
                            agg["n_neg"] = (mat < -eps).sum(axis=1)
                            agg["mean_abs"] = mat.abs().mean(axis=1)
                            agg["range"] = (mat.max(axis=1) - mat.min(axis=1)).abs()
                            agg["flip"] = (agg["n_pos"] > 0) & (agg["n_neg"] > 0)
                            agg["score"] = agg["mean_abs"] * agg["range"] * (agg["n_pos"] * agg["n_neg"])
                            cand = agg[agg["flip"]].sort_values("score", ascending=False)
                            if not cand.empty:
                                top_n = 4
                                # Prefer a compact, human-readable main-paper panel over
                                # an exhaustive hotspot list. Fill from the ranked list only
                                # if some preferred rows are unavailable.
                                per_axis_cap = 1
                                keep: list[str] = []
                                axis_counts: dict[str, int] = {}
                                cand_index = cand.index.tolist()

                                preferred_rows = [
                                    "earring | ornate->plain",
                                    "ceremonial mask | uniform->personalized",
                                    "hosting | tradition->modernity",
                                    "meal | clan->individual",
                                    "handshake | formal->informal",
                                ]

                                for pinned in preferred_rows:
                                    if pinned in cand_index and len(keep) < top_n:
                                        keep.append(pinned)
                                        p_axis = pinned.split(" | ", 1)[1]
                                        axis_counts[p_axis] = axis_counts.get(p_axis, 0) + 1

                                for idx in cand_index:
                                    if len(keep) >= top_n:
                                        break
                                    if idx in keep:
                                        continue
                                    parts = idx.split(" | ", 1)
                                    axis_name = parts[1] if len(parts) == 2 else ""
                                    if axis_name:
                                        count = axis_counts.get(axis_name, 0)
                                        if count >= per_axis_cap:
                                            continue
                                        axis_counts[axis_name] = count + 1
                                    keep.append(idx)
                                    if len(keep) >= top_n:
                                        break
                                h = mat.loc[keep].copy()

                                # Shorten row labels for visual scan in one figure.
                                def _short_label(s: str) -> str:
                                    probe, axis = s.split(" | ", 1)
                                    probe_s = probe if len(probe) <= 22 else probe[:21] + "..."
                                    axis_s = axis.replace("->", " -> ")
                                    axis_s = axis_s if len(axis_s) <= 26 else axis_s[:25] + "..."
                                    return f"{probe_s}\n{axis_s}"

                                # Save original labels before index reassignment (used for
                                # domain lookup and for the supporting CSV below).
                                orig_labels = keep[:]
                                h.index = [_short_label(x) for x in h.index]
                                q = np.nanpercentile(np.abs(h.to_numpy()), 95)
                                max_abs = float(np.nanmax(np.abs(h.to_numpy())))
                                vmax = float(max(0.25, q, max_abs * 1.02))

                                langs_ordered = list(h.columns)  # ZH, FR, FAS, ...
                                sign_counts = pd.DataFrame(
                                    {
                                        "en_only": (h > eps).sum(axis=1).astype(int),
                                        "en_l2": (h < -eps).sum(axis=1).astype(int),
                                        "near_zero": ((h >= -eps) & (h <= eps)).sum(axis=1).astype(int),
                                    },
                                    index=h.index,
                                )

                                n_rows = len(h.index)
                                fig, ax = plt.subplots(figsize=(7.20, 2.72), facecolor=PAPER_BG)
                                style_paper_axis(ax, grid_axis="x")
                                ax.axvspan(-eps, eps, color=SOFT_GRAY, alpha=0.96, zorder=0)
                                ax.axvline(0.0, color="#5f6672", lw=1.0, ls=(0, (3, 2)), zorder=2)
                                ax.text(
                                    -vmax,
                                    1.02,
                                    "EN+L2 side",
                                    transform=ax.get_xaxis_transform(),
                                    ha="left",
                                    va="bottom",
                                    fontsize=10.0,
                                    color=MUTED,
                                )
                                ax.text(
                                    vmax,
                                    1.02,
                                    "EN-only side",
                                    transform=ax.get_xaxis_transform(),
                                    ha="right",
                                    va="bottom",
                                    fontsize=10.0,
                                    color=MUTED,
                                )
                                ax.text(
                                    vmax * 1.23,
                                    1.02,
                                    "side counts\nL2 / EN / 0",
                                    transform=ax.get_xaxis_transform(),
                                    ha="center",
                                    va="bottom",
                                    fontsize=10.0,
                                    color=MUTED,
                                )
                                lang_offsets = {
                                    str(lg).upper(): off
                                    for lg, off in zip(langs_ordered, np.linspace(-0.32, 0.32, len(langs_ordered)))
                                }

                                for i_row in range(n_rows):
                                    row_vals = h.iloc[i_row].dropna().astype(float)
                                    if row_vals.empty:
                                        continue
                                    yrow = float(i_row)
                                    rmin = float(row_vals.min())
                                    rmax = float(row_vals.max())
                                    rail = FancyBboxPatch(
                                        (rmin, yrow - 0.055),
                                        max(rmax - rmin, 0.002),
                                        0.11,
                                        boxstyle="round,pad=0.0,rounding_size=0.045",
                                        facecolor=RAIL_BG,
                                        edgecolor="none",
                                        alpha=0.88,
                                        transform=ax.transData,
                                        zorder=1,
                                    )
                                    ax.add_patch(rail)
                                    ax.axhline(yrow, color=RAIL_LINE, lw=0.62, zorder=0)
                                    for lg, val in h.iloc[i_row].items():
                                        if pd.isna(val):
                                            continue
                                        c = LANG_COLORS.get(str(lg).upper(), NAVY)
                                        ax.scatter(
                                            [float(val)],
                                            [yrow + lang_offsets.get(str(lg).upper(), 0.0)],
                                            s=23,
                                            marker="o",
                                            color=blend_hex(c, (1, 1, 1), 0.18),
                                            edgecolors="white",
                                            linewidths=0.60,
                                            zorder=4,
                                        )
                                    pos = int(sign_counts.iloc[i_row]["en_only"])
                                    neg = int(sign_counts.iloc[i_row]["en_l2"])
                                    zero = int(sign_counts.iloc[i_row]["near_zero"])
                                    ax.text(
                                        vmax * 1.23,
                                        yrow,
                                        f"{neg} / {pos} / {zero}",
                                        ha="center",
                                        va="center",
                                        fontsize=10.0,
                                        color=INK,
                                        bbox=dict(
                                            boxstyle="round,pad=0.18,rounding_size=0.06",
                                            facecolor="white",
                                            edgecolor="#c7cfd9",
                                            linewidth=0.65,
                                        ),
                                        zorder=5,
                                    )

                                ax.set_xlim(-vmax * 1.08, vmax * 1.40)
                                ax.set_ylim(-0.58, n_rows - 0.35)
                                ax.invert_yaxis()
                                ax.set_yticks(range(n_rows))
                                ax.set_yticklabels(h.index, fontsize=10.0, color=INK)
                                for tick in ax.get_yticklabels():
                                    tick.set_linespacing(0.88)
                                ax.tick_params(axis="y", length=0, pad=3)
                                ax.set_xticks([-vmax, -vmax / 2.0, 0.0, vmax / 2.0, vmax])
                                ax.set_xticklabels(
                                    [f"{-vmax:.2f}", f"{-vmax/2:.2f}", "0", f"{vmax/2:.2f}", f"{vmax:.2f}"],
                                    fontsize=10.0,
                                    color=INK,
                                )
                                ax.set_xlabel(r"Signed contextual shift $\Delta s$ (EN-only minus EN+L2)", fontsize=10.0, color=INK, labelpad=4)
                                leg_handles = [
                                    plt.Line2D(
                                        [0],
                                        [0],
                                        marker="o",
                                        color="none",
                                        markerfacecolor=blend_hex(LANG_COLORS.get(lg, NAVY), (1, 1, 1), 0.18),
                                        markeredgecolor="white",
                                        label=lg,
                                        markersize=5.4,
                                    )
                                    for lg in langs_ordered
                                ]
                                ax.legend(
                                    handles=leg_handles,
                                    loc="lower center",
                                    bbox_to_anchor=(0.5, 1.08),
                                    ncol=min(8, len(langs_ordered)),
                                    frameon=True,
                                    facecolor="white",
                                    edgecolor="#c5ccd7",
                                    framealpha=0.96,
                                    fontsize=10.0,
                                    handletextpad=0.20,
                                    columnspacing=0.58,
                                )
                                fig.subplots_adjust(left=0.21, right=0.95, bottom=0.20, top=0.78)
                                fig.savefig(
                                    latex_root / "figures" / "exp4_signed_axes.pdf",
                                    dpi=450,
                                    bbox_inches="tight",
                                    pad_inches=0.05,
                                )
                                plt.close(fig)

                                # Supporting data for direct lookup of a row from the figure.
                                support = mat.loc[keep].copy()
                                support = support.reset_index().rename(columns={"index": "probe_axis"})
                                if EMIT_TABLES:
                                    support.to_csv(
                                        latex_root / "tables" / "fig5_signflip_hotspots.csv",
                                        index=False,
                                    )
                                # Full probe-axis matrix (not only top rows) for direct querying
                                # of arbitrary probes/axes from the same computation.
                                full_support = mat.copy().reset_index().rename(columns={"index": "probe_axis"})
                                if EMIT_TABLES:
                                    full_support.to_csv(
                                        latex_root / "tables" / "fig5_probe_axis_signed_full.csv",
                                        index=False,
                                    )

                                # 100M companion for Figure 5: recompute signed-shift values
                                # using EN-100M, then plot the same selected probe-axis rows.
                                eps_icon = eps
                                a100_path = repr_dir / "en_100m__pre_lmhead_contextual.npy"
                                if a100_path.exists():
                                    Xa100 = _row_norm(np.load(a100_path))
                                    records100: list[dict[str, object]] = []
                                    for lang in CORE_LANGS:
                                        b_path = repr_dir / f"en_{lang}_a__pre_lmhead_contextual.npy"
                                        if not b_path.exists():
                                            continue
                                        Xb = _row_norm(np.load(b_path))
                                        M = Xa100[neutral_idx].T @ Xb[neutral_idx]
                                        U, _, Vt = np.linalg.svd(M, full_matrices=False)
                                        W = U @ Vt
                                        Xa100_aligned = Xa100 @ W
                                        for a_word, b_word, axis_cat in axis_meta:
                                            i, j = w2i[a_word], w2i[b_word]
                                            va = Xa100_aligned[j] - Xa100_aligned[i]
                                            vb = Xb[j] - Xb[i]
                                            na = np.linalg.norm(va)
                                            nb = np.linalg.norm(vb)
                                            if na < 1e-12 or nb < 1e-12:
                                                continue
                                            va = va / na
                                            vb = vb / nb
                                            pa = Xa100_aligned[cultural_idx] @ va
                                            pb = Xb[cultural_idx] @ vb
                                            signed = pa - pb
                                            for pw, sv in zip(cultural_words, signed):
                                                probe_cat = probe_to_cat.get(pw, "")
                                                if axis_cat and probe_cat:
                                                    allowed = compat.get(probe_cat, {probe_cat})
                                                    if axis_cat not in allowed:
                                                        continue
                                                records100.append(
                                                    {
                                                        "lang": lang.upper(),
                                                        "axis": f"{a_word}->{b_word}",
                                                        "probe": pw,
                                                        "signed": float(sv),
                                                        "probe_axis": f"{pw} | {a_word}->{b_word}",
                                                    }
                                                )
                                    if records100:
                                        mat100 = (
                                            pd.DataFrame(records100)
                                            .pivot_table(index="probe_axis", columns="lang", values="signed", aggfunc="mean")
                                        )
                                        cols100 = [c.upper() for c in CORE_LANGS if c.upper() in mat100.columns]
                                        mat100 = mat100[cols100]
                                        keep100 = [k for k in orig_labels if k in mat100.index]
                                        if keep100:
                                            h100 = mat100.loc[keep100].copy()
                                            h100.index = [_short_label(x) for x in keep100]
                                            q100 = np.nanpercentile(np.abs(h100.to_numpy()), 95)
                                            max_abs100 = float(np.nanmax(np.abs(h100.to_numpy())))
                                            vmax100 = float(max(0.25, q100, max_abs100 * 1.02))
                                            langs100 = list(h100.columns)
                                            n_rows100 = len(h100)
                                            fig_h100 = max(4.80, n_rows100 * 0.58 + 2.05)
                                            fig100, ax100 = plt.subplots(figsize=(7.15, fig_h100), facecolor=PAPER_BG)
                                            fig100.subplots_adjust(left=0.36, right=0.97, bottom=0.18, top=0.82)
                                            style_paper_axis(ax100, grid_axis="x")
                                            lang_offsets100 = {
                                                str(lg).upper(): off
                                                for lg, off in zip(langs100, np.linspace(-0.32, 0.32, len(langs100)))
                                            }
                                            ax100.axvspan(-eps_icon, eps_icon, color=SOFT_GRAY, alpha=0.96, zorder=0)
                                            ax100.axvline(0.0, color="#5f6672", lw=1.0, ls=(0, (3, 2)), zorder=2)
                                            ax100.text(
                                                -vmax100,
                                                1.02,
                                                "EN+L2 side",
                                                transform=ax100.get_xaxis_transform(),
                                                ha="left",
                                                va="bottom",
                                                fontsize=10.0,
                                                color=MUTED,
                                            )
                                            ax100.text(
                                                vmax100,
                                                1.02,
                                                "EN-only side",
                                                transform=ax100.get_xaxis_transform(),
                                                ha="right",
                                                va="bottom",
                                                fontsize=10.0,
                                                color=MUTED,
                                            )
                                            for i_row in range(n_rows100):
                                                row_vals = h100.iloc[i_row]
                                                vals = row_vals.dropna().astype(float)
                                                if vals.empty:
                                                    continue
                                                vmin_i = float(vals.min())
                                                vmax_i = float(vals.max())
                                                rail = FancyBboxPatch(
                                                    (vmin_i, i_row - 0.055),
                                                    max(vmax_i - vmin_i, 0.002),
                                                    0.11,
                                                    boxstyle="round,pad=0.0,rounding_size=0.045",
                                                    facecolor=RAIL_BG,
                                                    edgecolor="none",
                                                    alpha=0.88,
                                                    transform=ax100.transData,
                                                    zorder=1,
                                                )
                                                ax100.add_patch(rail)
                                                for lg, val in zip(langs100, row_vals):
                                                    if pd.isna(val):
                                                        continue
                                                    c = LANG_COLORS.get(str(lg).upper(), NAVY)
                                                    ax100.scatter(
                                                        [float(val)],
                                                        [i_row + lang_offsets100.get(str(lg).upper(), 0.0)],
                                                        color=blend_hex(c, (1, 1, 1), 0.18),
                                                        s=22,
                                                        zorder=4,
                                                        edgecolors="white",
                                                        linewidths=0.55,
                                                    )
                                                ax100.axhline(i_row, color=RAIL_LINE, lw=0.62, zorder=0)
                                            ax100.set_xlim(-vmax100 * 1.08, vmax100 * 1.08)
                                            ax100.set_ylim(-0.6, n_rows100 - 0.35)
                                            ax100.invert_yaxis()
                                            ax100.set_yticks(range(n_rows100))
                                            ax100.set_yticklabels(h100.index, fontsize=10.0, color=INK)
                                            for tick in ax100.get_yticklabels():
                                                tick.set_linespacing(0.90)
                                            ax100.set_xlabel(
                                                "Signed contextual shift $\\Delta s$ (EN-100M minus EN+L2)",
                                                fontsize=10.0,
                                                color=INK,
                                                labelpad=4,
                                            )
                                            ax100.set_xticks([-vmax100, -vmax100 / 2, 0.0, vmax100 / 2, vmax100])
                                            ax100.set_xticklabels(
                                                [f"{-vmax100:.2f}", f"{-vmax100/2:.2f}", "0", f"{vmax100/2:.2f}", f"{vmax100:.2f}"],
                                                fontsize=10.0,
                                                color=INK,
                                            )
                                            ax100.tick_params(axis="y", length=0, pad=3)
                                            ax100.tick_params(axis="x", labelsize=10.0)
                                            leg_handles100 = [
                                                plt.Line2D(
                                                    [0],
                                                    [0],
                                                    marker="o",
                                                    color="none",
                                                    markerfacecolor=blend_hex(LANG_COLORS.get(lg, NAVY), (1, 1, 1), 0.18),
                                                    markeredgecolor="white",
                                                    label=lg,
                                                    markersize=5.0,
                                                )
                                                for lg in langs100
                                            ]
                                            ax100.legend(
                                                handles=leg_handles100,
                                                loc="lower center",
                                                bbox_to_anchor=(0.5, 1.07),
                                                ncol=min(8, max(1, len(langs100))),
                                                fontsize=10.0,
                                                frameon=True,
                                                facecolor="white",
                                                edgecolor="#c5ccd7",
                                                framealpha=0.96,
                                                handletextpad=0.20,
                                                columnspacing=0.58,
                                            )
                                            fig100.savefig(
                                                latex_root / "figures" / "exp4_signed_axes_100m.pdf",
                                                dpi=450,
                                                bbox_inches="tight",
                                                pad_inches=0.08,
                                            )
                                            plt.close(fig100)
                                built_main = True

    if not built_main:
        # Fallback: language-wise top signed axes when probe-level recomputation is unavailable.
        sub = axis_df[(axis_df["repr_type"] == "pre_lmhead_contextual")].copy()
        sub = sub[sub["pair"].astype(str).str.match(r"en_50m__vs__en_[a-z]+_a$")].copy()
        if sub.empty:
            return
        sub["lang"] = sub["pair"].astype(str).str.extract(r"__vs__en_([a-z]+)_a$", expand=False)
        sub = sub[sub["lang"].isin(CORE_LANGS)].copy()
        if sub.empty:
            return
        fig, ax = plt.subplots(figsize=(6.9, 4.2))
        g = (
            sub.groupby("lang", as_index=False)
            .agg(mean_abs_signed=("mean_signed_projection_diff", lambda s: float(np.mean(np.abs(s)))))
        )
        g["lang"] = g["lang"].str.upper()
        ax.bar(g["lang"], g["mean_abs_signed"], color="#9fc4e6", edgecolor="black", linewidth=0.8)
        ax.set_xlabel("Language")
        ax.set_ylabel("Mean |signed axis shift|")
        ax.grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.5, alpha=0.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(latex_root / "figures" / "exp4_signed_axes.pdf", dpi=450, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # Keep the previous quadrant summary table in appendix using ZH/FR illustration.
    a = axis_df[(axis_df["repr_type"] == "pre_lmhead_contextual") & (axis_df["pair"] == "en_100m__vs__en_zh_a")][
        ["axis", "mean_signed_projection_diff"]
    ].rename(columns={"mean_signed_projection_diff": "signed_zh"})
    b = axis_df[(axis_df["repr_type"] == "pre_lmhead_contextual") & (axis_df["pair"] == "en_100m__vs__en_fr_a")][
        ["axis", "mean_signed_projection_diff"]
    ].rename(columns={"mean_signed_projection_diff": "signed_fr"})
    m = a.merge(b, on="axis", how="inner")
    if m.empty:
        return

    q1 = ((m["signed_zh"] > 0) & (m["signed_fr"] > 0)).sum()
    q2 = ((m["signed_zh"] < 0) & (m["signed_fr"] > 0)).sum()
    q3 = ((m["signed_zh"] < 0) & (m["signed_fr"] < 0)).sum()
    q4 = ((m["signed_zh"] > 0) & (m["signed_fr"] < 0)).sum()

    # Quadrant summary table.
    lines = [
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"Quadrant & Count of axes \\",
        r"\midrule",
        f"ZH$>$0, FR$>$0 (shared positive tilt) & {int(q1)} \\\\",
        f"ZH$<$0, FR$>$0 (opposite tilt) & {int(q2)} \\\\",
        f"ZH$<$0, FR$<$0 (shared negative tilt) & {int(q3)} \\\\",
        f"ZH$>$0, FR$<$0 (opposite tilt) & {int(q4)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_table(lines, latex_root / "tables" / "exp4_signed_quadrants.tex")

    # top signed axes table for interpretability
    top_rows = []
    for pair in ["en_100m__vs__en_zh_a", "en_100m__vs__en_fr_a"]:
        ps = axis_df[(axis_df["repr_type"] == "pre_lmhead_contextual") & (axis_df["pair"] == pair)].copy()
        if ps.empty:
            continue
        ps["abs_signed"] = ps["mean_signed_projection_diff"].abs()
        ps = ps.sort_values("abs_signed", ascending=False).head(10).copy()
        pair_lbl = "EN+ZH vs EN" if pair == "en_100m__vs__en_zh_a" else "EN+FR vs EN"
        ps["pair_lbl"] = pair_lbl
        top_rows.append(ps[["pair_lbl", "axis", "mean_signed_projection_diff", "mean_abs_projection_diff"]])
    if top_rows:
        t = pd.concat(top_rows, ignore_index=True)
        lines = [
            r"\begin{tabular}{@{}llrr@{}}",
            r"\toprule",
            r"Pair & Axis & Signed shift & Abs.\ divergence \\",
            r"\midrule",
        ]
        for pair_lbl in ["EN+ZH vs EN", "EN+FR vs EN"]:
            sub = t[t["pair_lbl"] == pair_lbl].reset_index(drop=True)
            for i, (_, r) in enumerate(sub.iterrows()):
                left = pair_lbl if i == 0 else ""
                lines.append(
                    f"{left} & {r['axis']} & {_fmt(r['mean_signed_projection_diff'])} & {_fmt(r['mean_abs_projection_diff'])} \\\\"
                )
            lines.append(r"\cmidrule(lr){1-4}")
        if lines[-1] == r"\cmidrule(lr){1-4}":
            lines.pop()
        lines += [r"\bottomrule", r"\end{tabular}"]
        write_table(lines, latex_root / "tables" / "exp4_signed_top_axes.tex")


def build_appendix_l2_signed_hotspots(
    output_root: Path,
    multilingual_root: Path,
    latex_root: Path,
    probe_set: dict | None = None,
) -> None:
    """Appendix-only signed-shift hotspots in target-language shared subspaces.

    Computes contextual signed shifts for L2 baseline pairs:
    (L2-50M vs EN+L2_a), aligned on neutral anchors in the same L2 space.
    """
    if not probe_set:
        return
    req = {"all_probe_words", "neutral_anchor_words", "cultural_probe_words", "semantic_axes"}
    if not req.issubset(set(probe_set.keys())):
        return

    repr_dirs = {
        "zh": output_root / "zh_shared_language" / "representations",
        "fr": output_root / "fr_shared_language" / "representations",
        "fas": multilingual_root / "fas_shared_language" / "representations",
        "nld": multilingual_root / "nld_shared_language" / "representations",
        "ukr": multilingual_root / "ukr_shared_language" / "representations",
        "bul": multilingual_root / "bul_shared_language" / "representations",
        "ind": multilingual_root / "ind_shared_language" / "representations",
        "deu": multilingual_root / "deu_shared_language" / "representations",
    }

    def _row_norm(x: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(x, axis=1, keepdims=True)
        d = np.maximum(d, 1e-12)
        return x / d

    words = list(probe_set["all_probe_words"])
    w2i = {w: i for i, w in enumerate(words)}
    neutral_idx = np.array([w2i[w] for w in probe_set["neutral_anchor_words"] if w in w2i], dtype=np.int64)
    cultural_words = [w for w in probe_set["cultural_probe_words"] if w in w2i]
    cultural_idx = np.array([w2i[w] for w in cultural_words], dtype=np.int64)
    if len(neutral_idx) < 10 or len(cultural_idx) < 50:
        return

    probe_to_cat = probe_set.get("cultural_probe_categories", {})
    axis_cat_lookup: dict[tuple[str, str], str] = {}
    for m in probe_set.get("semantic_axis_metadata", []):
        a = m.get("endpoint_1")
        b = m.get("endpoint_2")
        c = m.get("category")
        if isinstance(a, str) and isinstance(b, str) and isinstance(c, str):
            axis_cat_lookup[(a, b)] = c

    axis_meta: list[tuple[str, str, str]] = []
    for a, b in probe_set["semantic_axes"]:
        if a in w2i and b in w2i:
            axis_cat = axis_cat_lookup.get((a, b), axis_cat_lookup.get((b, a), ""))
            axis_meta.append((a, b, axis_cat))
    if not axis_meta:
        return

    # Same compatibility mapping used for Figure 5 row filtering.
    compat = {
        "values_norms": {"values_norms"},
        "family_kinship": {"family_kinship", "clothing_appearance", "symbols_colors"},
        "religion_ritual": {"religion_ritual", "festivals_holidays", "symbols_colors"},
        "food_cuisine": {"food_cuisine"},
        "festivals_holidays": {"festivals_holidays", "religion_ritual", "symbols_colors"},
        "clothing_appearance": {"clothing_appearance", "symbols_colors"},
        "symbols_colors": {"symbols_colors", "clothing_appearance", "religion_ritual", "festivals_holidays"},
        "governance_law": {"governance_law", "social_identity"},
        "social_identity": {"social_identity", "governance_law", "values_norms"},
        "daily_customs": {"daily_customs", "values_norms", "family_kinship", "religion_ritual", "festivals_holidays"},
    }

    records_full: list[dict[str, object]] = []
    records_filtered: list[dict[str, object]] = []
    for lang in CORE_LANGS:
        rdir = repr_dirs.get(lang)
        if rdir is None:
            continue
        a_path = rdir / f"{lang}_50m__pre_lmhead_contextual.npy"
        b_path = rdir / f"en_{lang}_a__pre_lmhead_contextual.npy"
        if not (a_path.exists() and b_path.exists()):
            continue
        Xa = _row_norm(np.load(a_path))
        Xb = _row_norm(np.load(b_path))

        # Orthogonal alignment on neutral anchors in this language's space.
        M = Xa[neutral_idx].T @ Xb[neutral_idx]
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        W = U @ Vt
        Xa_aligned = Xa @ W

        for a_word, b_word, axis_cat in axis_meta:
            i, j = w2i[a_word], w2i[b_word]
            va = Xa_aligned[j] - Xa_aligned[i]
            vb = Xb[j] - Xb[i]
            na = np.linalg.norm(va)
            nb = np.linalg.norm(vb)
            if na < 1e-12 or nb < 1e-12:
                continue
            va = va / na
            vb = vb / nb
            signed = (Xa_aligned[cultural_idx] @ va) - (Xb[cultural_idx] @ vb)
            axis_name = f"{a_word}->{b_word}"

            for pw, sv in zip(cultural_words, signed):
                row = {
                    "lang": lang.upper(),
                    "pair": f"{lang}_50m__vs__en_{lang}_a",
                    "probe": pw,
                    "axis": axis_name,
                    "probe_axis": f"{pw} | {axis_name}",
                    "signed": float(sv),
                }
                records_full.append(row)
                probe_cat = probe_to_cat.get(pw, "")
                if axis_cat and probe_cat:
                    allowed = compat.get(probe_cat, {probe_cat})
                    if axis_cat not in allowed:
                        continue
                records_filtered.append(row)

    if not records_full:
        return

    full_df = pd.DataFrame(records_full)
    filt_df = pd.DataFrame(records_filtered)
    full_mat = full_df.pivot_table(index="probe_axis", columns="lang", values="signed", aggfunc="mean")
    filt_mat = filt_df.pivot_table(index="probe_axis", columns="lang", values="signed", aggfunc="mean")
    if filt_mat.empty:
        return

    cols = [c.upper() for c in CORE_LANGS if c.upper() in filt_mat.columns]
    filt_mat = filt_mat[cols]
    full_mat = full_mat[[c for c in cols if c in full_mat.columns]]

    tables_dir = latex_root / "tables"
    (tables_dir / "fig5_probe_axis_signed_all_lang50m_full.csv").write_text(
        full_df.to_csv(index=False), encoding="utf-8"
    )
    (tables_dir / "fig5_probe_axis_signed_all_lang50m_filtered.csv").write_text(
        filt_df.to_csv(index=False), encoding="utf-8"
    )
    (tables_dir / "fig5_probe_axis_signed_all_lang50m_full_matrix.csv").write_text(
        full_mat.reset_index().to_csv(index=False), encoding="utf-8"
    )
    (tables_dir / "fig5_probe_axis_signed_all_lang50m_filtered_matrix.csv").write_text(
        filt_mat.reset_index().to_csv(index=False), encoding="utf-8"
    )

    eps = 0.05
    agg = pd.DataFrame(index=filt_mat.index)
    agg["n_pos"] = (filt_mat > eps).sum(axis=1)
    agg["n_neg"] = (filt_mat < -eps).sum(axis=1)
    agg["mean_abs"] = filt_mat.abs().mean(axis=1)
    agg["range"] = (filt_mat.max(axis=1) - filt_mat.min(axis=1)).abs()
    agg["flip"] = (agg["n_pos"] > 0) & (agg["n_neg"] > 0)
    agg["score"] = agg["mean_abs"] * agg["range"] * (agg["n_pos"] * agg["n_neg"])
    cand = agg[agg["flip"]].sort_values("score", ascending=False)
    if cand.empty:
        return

    # Axis-balanced selection for readability.
    top_n = 22
    per_axis_cap = 2
    keep: list[str] = []
    axis_counts: dict[str, int] = {}
    cand_index = cand.index.tolist()
    for pinned in [
        "wedding dress | white->red",
        "formal wedding dress | white->red",
    ]:
        if pinned in cand_index and len(keep) < top_n:
            keep.append(pinned)
            p_axis = pinned.split(" | ", 1)[1]
            axis_counts[p_axis] = axis_counts.get(p_axis, 0) + 1
    for idx in cand_index:
        if idx in keep:
            continue
        parts = idx.split(" | ", 1)
        axis_name = parts[1] if len(parts) == 2 else ""
        if axis_name:
            count = axis_counts.get(axis_name, 0)
            if count >= per_axis_cap:
                continue
            axis_counts[axis_name] = count + 1
        keep.append(idx)
        if len(keep) >= top_n:
            break

    h = filt_mat.loc[keep].copy()
    (tables_dir / "fig5_signflip_hotspots_all_lang50m_filtered.csv").write_text(
        h.reset_index().to_csv(index=False), encoding="utf-8"
    )
    if not full_mat.empty:
        full_top = full_mat.loc[[k for k in keep if k in full_mat.index]].copy()
        (tables_dir / "fig5_signflip_hotspots_all_lang50m_full.csv").write_text(
            full_top.reset_index().to_csv(index=False), encoding="utf-8"
        )

    # Quick lookup rows used in interpretation checks.
    needles = [
        "wedding dress | white->red",
        "formal wedding dress | white->red",
        "wedding dress | black->gold",
        "formal wedding dress | black->gold",
    ]
    rows = []
    for n in needles:
        if n not in full_mat.index:
            continue
        row = {"probe_axis": n}
        for c in full_mat.columns:
            row[c] = float(full_mat.loc[n, c])
        rows.append(row)
    if rows:
        (tables_dir / "fig5_wedding_color_lookup_all_lang50m.csv").write_text(
            pd.DataFrame(rows).to_csv(index=False), encoding="utf-8"
        )

    # Visuals: per-language signed hotspots (appendix companion), packed as two 2x2 panels.
    def _short_label(s: str) -> str:
        probe, axis = s.split(" | ", 1)
        probe_s = probe if len(probe) <= 20 else probe[:19] + "…"
        axis_s = axis if len(axis) <= 22 else axis[:21] + "…"
        return f"{probe_s} | {axis_s}"

    def _select_lang_rows(series: pd.Series) -> list[str]:
        s = series.dropna().astype(float)
        if s.empty:
            return []
        df = s.to_frame(name="signed")
        df["abs"] = df["signed"].abs()
        rows: list[str] = []
        for pinned in ["wedding dress | white->red", "formal wedding dress | white->red"]:
            if pinned in df.index:
                rows.append(pinned)
        for idx in df.sort_values("abs", ascending=False).index:
            if idx in rows:
                continue
            rows.append(idx)
        return rows

    # Build per-language candidates, then assign globally unique examples so
    # y-axis entries do not repeat across the appendix panels.
    top_n = 12
    cand_per_lang: dict[str, list[str]] = {lg: _select_lang_rows(filt_mat[lg]) for lg in cols if lg in filt_mat.columns}
    used_rows: set[str] = set()
    per_lang: dict[str, pd.DataFrame] = {}
    for lg in cols:
        candidates = cand_per_lang.get(lg, [])
        selected: list[str] = []
        for idx in candidates:
            if idx in used_rows:
                continue
            selected.append(idx)
            used_rows.add(idx)
            if len(selected) >= top_n:
                break
        if selected:
            per_lang[lg] = filt_mat.loc[selected, [lg]].rename(columns={lg: "signed"}).copy()
        else:
            per_lang[lg] = pd.DataFrame(columns=["signed"])
    vmax_vals = []
    for sub in per_lang.values():
        if not sub.empty:
            vmax_vals.append(float(np.max(np.abs(sub["signed"].to_numpy()))))
    if not vmax_vals:
        return
    vmax = max(0.25, float(np.percentile(vmax_vals, 95)))
    xlim = vmax * 1.12

    def _draw_panel(lang_list: list[str], out_png: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6), sharex=True)
        axes = axes.flatten()
        for ax, lg in zip(axes, lang_list):
            sub = per_lang.get(lg, pd.DataFrame(columns=["signed"]))
            if sub.empty:
                ax.text(0.5, 0.5, f"{lg}: no rows", ha="center", va="center", fontsize=10.0)
                ax.axis("off")
                continue
            sub = sub.sort_values("signed").copy()
            y = np.arange(len(sub))
            vals = sub["signed"].to_numpy()
            labels = [_short_label(x) for x in sub.index.tolist()]
            # Prevent duplicates after truncation by suffixing repeated labels.
            seen: dict[str, int] = {}
            labels_unique: list[str] = []
            for lbl in labels:
                c = seen.get(lbl, 0) + 1
                seen[lbl] = c
                labels_unique.append(lbl if c == 1 else f"{lbl} [{c}]")

            ax.axvspan(-eps, eps, color=ZERO_RANGE_BG, alpha=0.82, zorder=0)
            ax.axvline(0.0, color="#444444", lw=0.9, ls=(0, (3, 2)), zorder=1, alpha=0.75)
            ax.hlines(y, 0.0, vals, color="#bcc5d1", lw=1.8, zorder=1)
            ax.scatter(vals, y, s=26, color=LANG_COLORS.get(lg, "#4c78a8"), edgecolors="white", linewidths=0.45, zorder=2)

            ax.set_yticks(y)
            ax.set_yticklabels(labels_unique, fontsize=12.0, fontfamily="monospace")
            ax.set_title(f"{lg} (L2-50M vs EN+L2)", fontsize=13, pad=4)
            ax.set_xlim(-xlim, xlim)
            ax.grid(axis="x", linestyle=(0, (3, 2)), linewidth=0.45, alpha=0.42)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=2)
            ax.tick_params(axis="x", labelsize=11)

        for ax in axes[len(lang_list):]:
            ax.axis("off")
        fig.supxlabel("Signed shift (L2-50M minus EN+L2): left = EN+L2 higher, right = L2-50M higher", fontsize=12)
        fig.subplots_adjust(left=0.24, right=0.99, bottom=0.08, top=0.96, wspace=0.60, hspace=0.34)
        fig.savefig(latex_root / "figures" / out_png, dpi=450, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    _draw_panel(["ZH", "FR", "FAS", "NLD"], "appendix_l2_signed_hotspots_panel1.pdf")
    _draw_panel(["UKR", "BUL", "IND", "DEU"], "appendix_l2_signed_hotspots_panel2.pdf")


def build_axis_inventory_table(probe_set: dict, latex_root: Path) -> None:
    axes = probe_set.get("semantic_axes", [])
    if not axes:
        return
    half = (len(axes) + 1) // 2
    left = list(enumerate(axes[:half], start=1))
    right = list(enumerate(axes[half:], start=half + 1))
    while len(right) < len(left):
        right.append((None, ["", ""]))
    lines = [
        r"\begin{tabular}{@{}rll@{\hspace{1.2em}}rll@{}}",
        r"\toprule",
        r"\multicolumn{3}{c}{Axis set A} & \multicolumn{3}{c}{Axis set B} \\",
        r"\cmidrule(lr){1-3}\cmidrule(lr){4-6}",
        r"\# & Endpoint 1 & Endpoint 2 & \# & Endpoint 1 & Endpoint 2 \\",
        r"\midrule",
    ]
    for (li, lpair), (ri, rpair) in zip(left, right):
        la, lb = lpair
        ra, rb = rpair
        ri_str = "" if ri is None else str(ri)
        lines.append(f"{li} & {la} & {lb} & {ri_str} & {ra} & {rb} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "appendix_axes.tex")


def build_axis_grounding_table(probe_set: dict, latex_root: Path) -> None:
    axis_meta = probe_set.get("semantic_axis_metadata", [])
    if not axis_meta:
        return
    cat_meta = probe_set.get("category_frameworks", {})

    def _rows(rows: list[dict]) -> list[str]:
        lines = [
            r"\begin{tabular}{@{}>{\raggedleft\arraybackslash}p{0.04\textwidth}>{\raggedright\arraybackslash}p{0.15\textwidth}>{\raggedright\arraybackslash}p{0.15\textwidth}>{\raggedright\arraybackslash}p{0.25\textwidth}>{\raggedright\arraybackslash}p{0.28\textwidth}@{}}",
            r"\toprule",
            r"\# & Endpoint 1 & Endpoint 2 & Category & Citation(s) \\",
            r"\midrule",
        ]
        for row in rows:
            idx = int(row.get("index", 0))
            left = str(row.get("endpoint_1", ""))
            right = str(row.get("endpoint_2", ""))
            cat = str(row.get("category", ""))
            cat_info = cat_meta.get(cat, {})
            cat_label = str(cat_info.get("display_name", cat))
            cites = row.get("citations", [])
            cite_txt = f"\\cite{{{','.join(cites)}}}" if cites else "NA"
            lines.append(f"{idx} & {left} & {right} & {cat_label} & {cite_txt} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return lines

    split = (len(axis_meta) + 1) // 2
    part1 = axis_meta[:split]
    part2 = axis_meta[split:]
    write_table(_rows(part1), latex_root / "tables" / "appendix_axis_grounding_part1.tex")
    write_table(_rows(part2), latex_root / "tables" / "appendix_axis_grounding_part2.tex")
    write_table(_rows(axis_meta), latex_root / "tables" / "appendix_axis_grounding.tex")


def build_probe_qc_table(translations: dict[str, pd.DataFrame], latex_root: Path) -> None:
    if not translations:
        return
    rows = []
    for lang, df in sorted(translations.items()):
        if df is None or df.empty:
            continue
        x = df.copy()
        score_col = "comet_kiwi_score" if "comet_kiwi_score" in x.columns else "back_similarity"
        x[score_col] = pd.to_numeric(x.get(score_col), errors="coerce")
        score = x[score_col].dropna()
        if score_col == "comet_kiwi_score":
            high = int((score >= 0.80).sum())
            med = int(((score >= 0.60) & (score < 0.80)).sum())
            low = int((score < 0.60).sum())
        else:
            high = int((score > 0.80).sum())
            med = int(((score >= 0.55) & (score <= 0.80)).sum())
            low = int((score < 0.55).sum())
        rows.append(
            {
                "lang": lang.upper(),
                "rows": int(len(x)),
                "mean_qe": float(score.mean()) if not score.empty else float("nan"),
                "p50_qe": float(score.median()) if not score.empty else float("nan"),
                "high": high,
                "medium": med,
                "low": low,
                "manual_review": int(pd.to_numeric(x.get("needs_manual_review", 0), errors="coerce").fillna(0).astype(int).sum()),
                "duplicates": int(pd.to_numeric(x.get("duplicate_translation", 0), errors="coerce").fillna(0).astype(int).sum()),
                "score_label": "COMETKiwi" if score_col == "comet_kiwi_score" else "BackSim",
            }
        )

    if not rows:
        return

    lines = [
        r"\begin{tabular}{@{}lrrrcccrr@{}}",
        r"\toprule",
        r"Lang & $n$ & Mean QE & Median QE & High & Medium & Low & Manual-review & Duplicate-target \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['lang']} & {r['rows']} & {r['mean_qe']:.3f} & {r['p50_qe']:.3f} & {r['high']} & {r['medium']} & {r['low']} & {r['manual_review']} & {r['duplicates']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "appendix_probe_qc.tex")


def build_category_heatmap(strat: pd.DataFrame | None, latex_root: Path) -> None:
    if strat is None or strat.empty:
        return
    category_labels = {
        "values_norms": "Values/Norms",
        "family_kinship": "Family/Kinship",
        "religion_ritual": "Religion/Ritual",
        "daily_customs": "Daily Customs",
        "food_cuisine": "Food/Cuisine",
        "festivals_holidays": "Festivals/Holidays",
        "clothing_appearance": "Clothing/Appearance",
        "governance_law": "Governance/Law",
        "social_identity": "Social Identity",
        "symbols_colors": "Symbols/Colors",
    }
    sub = strat[(strat["repr_type"] == "pre_lmhead_contextual") & (strat["metric"] == "axis_abs_projection_diff")].copy()
    if sub.empty:
        return
    top_cats = (
        sub.groupby("category", as_index=False)["mean"].mean().sort_values("mean", ascending=False).head(10)["category"].tolist()
    )
    sub = sub[sub["category"].isin(top_cats)]
    pvt = sub.pivot_table(index="category", columns="pair", values="mean", aggfunc="mean")
    # Sort rows by mean divergence descending
    pvt = pvt.loc[pvt.mean(axis=1).sort_values(ascending=False).index]
    # Keep all model-pair cells, but make the axis readable in the compiled appendix.
    # Columns are grouped by baseline, document setting, and target-language color.
    def _cat_col_key(col: object) -> tuple[int, int, int, str]:
        raw = str(col)
        if "__vs__" not in raw:
            return (99, 99, 99, raw)
        left, right = raw.split("__vs__", 1)
        base_order = 0 if left == "en_50m" else 1 if left == "en_100m" else 9
        parsed = parse_en_target(right)
        if parsed is None:
            return (base_order, 99, 99, raw)
        lang, setup = parsed
        setup_order = 0 if setup == "a" else 1 if setup == "b" else 9
        lang_order = CORE_LANGS.index(lang) if lang in CORE_LANGS else 99
        return (base_order, setup_order, lang_order, raw)

    def _cat_col_label(col: object) -> str:
        raw = str(col)
        if "__vs__" not in raw:
            return raw
        left, right = raw.split("__vs__", 1)
        parsed = parse_en_target(right)
        if parsed is None:
            return raw
        lang, setup = parsed
        base = "50" if left == "en_50m" else "100" if left == "en_100m" else left
        setting = "S" if setup == "a" else "D" if setup == "b" else setup.upper()
        return f"{base}{setting}-{lang.upper()}"

    sorted_cols = sorted(list(pvt.columns), key=_cat_col_key)
    pvt = pvt.loc[:, sorted_cols]
    col_rename = {c: _cat_col_label(c) for c in pvt.columns}
    pvt = pvt.rename(columns=col_rename)
    row_color_lookup: dict[str, str] = {}
    display_index: list[str] = []
    for raw_label in pvt.index.astype(str):
        pretty = category_labels.get(raw_label, raw_label.replace("_", " ").title())
        display_index.append(pretty)
        row_color_lookup[pretty] = DOMAIN_COLORS.get(raw_label, INK)
    pvt.index = display_index

    fig, ax = plt.subplots(figsize=(8.4, 5.8), facecolor=PAPER_BG)
    vmin = float(np.nanmin(pvt.values))
    vmax = float(np.nanmax(pvt.values))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sns.heatmap(
        pvt,
        cmap="YlOrBr",
        norm=norm,
        annot=False,
        linewidths=0.8,
        linecolor="black",
        cbar_kws={"label": r"Mean $D_{Axis}$ (higher = worse mismatch)", "shrink": 0.8},
        ax=ax,
    )
    ax.collections[0].colorbar.ax.locator = MaxNLocator(4)
    ax.collections[0].colorbar.update_ticks()
    ax.set_xlabel("Model pair", fontsize=11)
    ax.set_ylabel("Probe category", fontsize=11)
    ax.tick_params(axis="x", labelsize=11.2, pad=2)
    ax.tick_params(axis="y", labelsize=11.2, rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", va="top")
    for boundary in range(8, len(pvt.columns), 8):
        ax.axvline(boundary, color=INK, linewidth=1.2)
    for tick in ax.get_yticklabels():
        tick.set_color(row_color_lookup.get(tick.get_text(), INK))
        tick.set_fontweight("bold")
    for tick in ax.get_xticklabels():
        lang = tick.get_text().split("-")[-1]
        tick.set_color(LANG_COLORS.get(lang, INK))
        tick.set_fontweight("bold")
    ax.collections[0].colorbar.ax.tick_params(labelsize=11)
    ax.text(4, -0.70, "50M shared", ha="center", va="bottom", fontsize=11.2, color=INK)
    ax.text(12, -0.70, "50M disjoint", ha="center", va="bottom", fontsize=11.2, color=INK)
    ax.text(20, -0.70, "100M shared", ha="center", va="bottom", fontsize=11.2, color=INK)
    ax.text(28, -0.70, "100M disjoint", ha="center", va="bottom", fontsize=11.2, color=INK)
    fig.subplots_adjust(left=0.11, right=0.93, bottom=0.34, top=0.92)
    fig.savefig(latex_root / "figures" / "category_heatmap.pdf", dpi=450, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def build_layerwise_artifacts(layerwise_df: pd.DataFrame | None, latex_root: Path) -> None:
    if layerwise_df is None or layerwise_df.empty:
        return

    df = layerwise_df.copy()
    df["pair"] = df.apply(lambda r: en_pair_label(str(r["model_a"]), str(r["model_b"])), axis=1)
    df = df.sort_values(["pair", "layer"]).reset_index(drop=True)
    pairs = sorted(df["pair"].astype(str).unique())
    if not pairs:
        return

    def _plot_layerwise_slice(model_a: str, out_name: str) -> None:
        dbase = df[df["model_a"] == model_a].copy()
        dbase = dbase[
            dbase["model_b"].astype(str).map(
                lambda mb: parse_en_target(mb) is not None and parse_en_target(mb)[1] == "a"
            )
        ].copy()
        if dbase.empty:
            return
        fig, ax = plt.subplots(figsize=(COL_FIG_W, 2.70), facecolor=PAPER_BG)
        style_paper_axis(ax, grid_axis="both")
        vals = []
        layers_ref = None
        for lang in CORE_LANGS:
            sub = dbase[dbase["model_b"] == f"en_{lang}_a"].sort_values("layer")
            if sub.empty:
                continue
            raw_yv = sub["axis_abs_projection_diff_mean"].to_numpy(dtype=float)
            denom = max(float(np.nanmax(raw_yv)), 1e-12)
            yv = raw_yv / denom
            lv = sub["layer"].to_numpy(dtype=float)
            vals.append(yv)
            if layers_ref is None:
                layers_ref = lv
            c = pastel_lang_color(lang.upper(), lighten=0.28)
            ec = blend_hex(LANG_COLORS.get(lang.upper(), "#4C78A8"), (0, 0, 0), 0.25)
            ax.plot(
                lv,
                yv,
                color=c,
                alpha=0.50,
                linewidth=1.25,
                marker="o",
                markersize=2.6,
                markerfacecolor=c,
                markeredgecolor=ec,
                markeredgewidth=0.4,
                zorder=1,
            )
        if vals and layers_ref is not None:
            arr = np.vstack(vals)
            y_mean = arr.mean(axis=0)
            y_low = np.percentile(arr, 20, axis=0)
            y_high = np.percentile(arr, 80, axis=0)
            peak_idx = int(np.argmax(y_mean))
            ax.fill_between(
                layers_ref,
                y_low,
                y_high,
                color=NAVY,
                alpha=0.28,
                linewidth=0,
                label="20 to 80% language band",
                zorder=2,
            )
            ax.plot(
                layers_ref,
                y_mean,
                color=INK,
                linewidth=2.35,
                marker="o",
                markersize=3.6,
                markerfacecolor="white",
                markeredgecolor=INK,
                markeredgewidth=0.9,
                label=f"{model_a.replace('en_', 'EN-').upper()} mean across L2",
                zorder=3,
            )
            ax.annotate(
                "peak",
                xy=(layers_ref[peak_idx], y_mean[peak_idx]),
                xytext=(layers_ref[peak_idx] + 0.7, y_mean[peak_idx] + 0.08),
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.8),
                fontsize=10.0,
                color=INK,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#c7c7c7", linewidth=0.6),
            )
        ax.axhline(0.0, color="#666666", linestyle=(0, (4, 2)), linewidth=0.9, alpha=0.65, zorder=0)
        ax.set_xlabel("Transformer layer", fontsize=10.0)
        ax.set_ylabel(r"Normalized contextual $D_{Axis}$", fontsize=10.0)
        ax.tick_params(labelsize=10.0)
        fig.tight_layout(pad=0.35)
        fig.subplots_adjust(left=0.16)
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_layerwise_slice("en_50m", "exp4_layerwise.pdf")
    _plot_layerwise_slice("en_100m", "exp4_layerwise_100m.pdf")

    # Table: peak and last-layer summaries.
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Pair & Peak & $D_A$(peak) & $D_A$(final) & F/P \\",
        r"\midrule",
    ]
    for pair in pairs:
        sub = df[df["pair"] == pair].sort_values("layer")
        if sub.empty:
            continue
        peak_idx = sub["axis_abs_projection_diff_mean"].idxmax()
        peak_layer = int(sub.loc[peak_idx, "layer"])
        peak_val = float(sub.loc[peak_idx, "axis_abs_projection_diff_mean"])
        final_val = float(sub.iloc[-1]["axis_abs_projection_diff_mean"])
        ratio = final_val / max(1e-12, peak_val)
        lines.append(
            f"{pair} & {peak_layer} & {peak_val:.1f} & {_fmt(final_val)} & {ratio:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp4_layerwise.tex")


def build_norm_control_artifacts(
    final_df: pd.DataFrame | None,
    layer_df: pd.DataFrame | None,
    latex_root: Path,
) -> None:
    if final_df is None or final_df.empty:
        return

    final = final_df.copy()
    final = final[final["normalization"].isin(["row_l2", "neutral_z"])].copy()
    final = final[
        final["model_b"].astype(str).map(
            lambda mb: parse_en_target(mb) is not None and parse_en_target(mb)[1] == "a"
        )
    ].copy()
    if final.empty:
        return

    norm_label = {
        "row_l2": "Row-L2",
        "neutral_z": "Neutral-z",
    }
    base_label = {
        "en_50m": "EN-50M",
        "en_100m": "EN-100M",
    }

    pivot = final.pivot_table(
        index=["model_a", "model_b", "normalization"],
        columns="repr_type",
        values="axis_abs_projection_diff_mean",
    ).reset_index()
    if "embedding_matrix" not in pivot.columns or "pre_lmhead_contextual" not in pivot.columns:
        return
    pivot["gap"] = pivot["pre_lmhead_contextual"] - pivot["embedding_matrix"]

    layer_summary: dict[tuple[str, str], dict[str, float]] = {}
    if layer_df is not None and not layer_df.empty:
        layer = layer_df[layer_df["normalization"].isin(["row_l2", "neutral_z"])].copy()
        for (base, norm), sub in layer.groupby(["model_a", "normalization"]):
            by_layer = sub.groupby("layer")["axis_abs_projection_diff_mean"].mean().sort_index()
            if by_layer.empty:
                continue
            peak_layer = int(by_layer.idxmax())
            layer0 = float(by_layer.iloc[0])
            final_val = float(by_layer.iloc[-1])
            layer_summary[(str(base), str(norm))] = {
                "layer0": layer0,
                "peak_layer": float(peak_layer),
                "final": final_val,
                "final_over_input": final_val / max(layer0, 1e-12),
            }

    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Baseline & Control & Embed. & Context. & Gap & +Pairs & Final/Input \\",
        r"\midrule",
    ]
    for base in ["en_50m", "en_100m"]:
        for norm in ["row_l2", "neutral_z"]:
            sub = pivot[(pivot["model_a"] == base) & (pivot["normalization"] == norm)]
            if sub.empty:
                continue
            emb = float(sub["embedding_matrix"].mean())
            ctx = float(sub["pre_lmhead_contextual"].mean())
            gap = float(sub["gap"].mean())
            pos = int((sub["gap"] > 0).sum())
            total = int(sub["gap"].notna().sum())
            layer_info = layer_summary.get((base, norm), {})
            final_over_input = layer_info.get("final_over_input", float("nan"))
            lines.append(
                f"{base_label.get(base, base)} & {norm_label.get(norm, norm)} & "
                f"{_fmt(emb)} & {_fmt(ctx)} & +{_fmt(gap)} & {pos}/{total} & {_fmt(final_over_input)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "norm_controlled_axis.tex")

    if layer_df is None or layer_df.empty:
        return
    layer = layer_df[layer_df["normalization"].isin(["row_l2", "neutral_z"])].copy()
    if layer.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(COL_FIG_W, 3.72), sharex=True, facecolor=PAPER_BG)
    configs = [
        ("row_l2", "Row-normalized vectors", r"Row-L2 $D_{Axis}$"),
        ("neutral_z", "Neutral-anchor z-scored axes", r"Neutral-z $D_{Axis}$"),
    ]
    base_styles = {
        "en_50m": {"label": "EN-50M", "color": NAVY, "linestyle": "-"},
        "en_100m": {"label": "EN-100M", "color": ORANGE, "linestyle": (0, (3, 2))},
    }
    for ax, (norm, title, ylabel) in zip(axes, configs):
        style_paper_axis(ax, grid_axis="both")
        dnorm = layer[layer["normalization"] == norm]
        for base, style in base_styles.items():
            dbase = dnorm[dnorm["model_a"] == base]
            if dbase.empty:
                continue
            by_pair = dbase.pivot_table(
                index="model_b",
                columns="layer",
                values="axis_abs_projection_diff_mean",
            )
            by_pair = by_pair.dropna(axis=1, how="all")
            if by_pair.empty:
                continue
            layers = by_pair.columns.to_numpy(dtype=float)
            vals = by_pair.to_numpy(dtype=float)
            mean = np.nanmean(vals, axis=0)
            low = np.nanpercentile(vals, 20, axis=0)
            high = np.nanpercentile(vals, 80, axis=0)
            ax.fill_between(layers, low, high, color=style["color"], alpha=0.28, linewidth=0)
            ax.plot(
                layers,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.85,
                marker="o",
                markersize=2.6,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=0.7,
                label=style["label"],
            )
        ax.set_title(title, fontsize=10.0, fontweight="bold", color=INK, pad=3)
        ax.set_ylabel(ylabel, fontsize=10.0)
        ax.tick_params(labelsize=10.0)
        ax.legend(frameon=False, fontsize=10.0, loc="upper left", ncol=2, handlelength=1.8, columnspacing=0.8)
    axes[-1].set_xlabel("Transformer layer", fontsize=10.0)
    fig.tight_layout(pad=0.35)
    fig.subplots_adjust(hspace=0.34, left=0.18)
    fig.savefig(latex_root / "figures" / "norm_controlled_layerwise.pdf", dpi=450, bbox_inches="tight")
    plt.close(fig)


def build_contextual_alignment_variant_artifacts(ctx_align_df: pd.DataFrame | None, latex_root: Path) -> None:
    if ctx_align_df is None or ctx_align_df.empty:
        return
    d = ctx_align_df.copy()
    d["pair"] = d.apply(lambda r: en_pair_label(str(r["model_a"]), str(r["model_b"]), include_setup=True), axis=1)
    d["align_lbl"] = d["alignment_source"].map(
        {
            "embedding_matrix": "Align on Embedding anchors",
            "pre_lmhead_contextual": "Align on Contextual anchors",
        }
    ).fillna(d["alignment_source"])

    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Pair & Alignment source & $\Delta D_{NN}$ & $\Delta D_{Struct}$ & $\Delta D_{Axis}$ & Residual/anchor \\",
        r"\midrule",
    ]
    for pair in d["pair"].drop_duplicates().tolist():
        sub = d[d["pair"] == pair].reset_index(drop=True)
        for i, (_, r) in enumerate(sub.iterrows()):
            left = pair if i == 0 else ""
            lines.append(
                f"{left} & {r['align_lbl']} & {_fmt(r['jaccard_at_k_mean'])} & {_fmt(r['frobenius_cultural_similarity'])} & "
                f"{_fmt(r['axis_abs_projection_diff_mean'])} & {_fmt(r['procrustes_anchor_residual_per_anchor'])} \\\\"
            )
        lines.append(r"\cmidrule(lr){1-6}")
    if lines[-1] == r"\cmidrule(lr){1-6}":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "appendix_contextual_alignment_variant.tex")


def build_perhead_artifacts(perhead_df: pd.DataFrame | None, latex_root: Path) -> None:
    if perhead_df is None or perhead_df.empty:
        return
    d = perhead_df.copy()
    d["pair"] = d.apply(lambda r: en_pair_label(str(r["model_a"]), str(r["model_b"]), include_setup=True), axis=1)
    pivot = d.pivot_table(index=["pair", "layer"], columns="head", values="axis_abs_projection_diff_mean", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    vmin = float(np.nanmin(pivot.values))
    vmax = float(np.nanmax(pivot.values))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sns.heatmap(
        pivot,
        cmap="YlOrBr",
        norm=norm,
        annot=False,
        linewidths=0.35,
        linecolor="black",
        cbar_kws={"label": r"Mean $D_{Axis}$ (higher = worse mismatch)", "shrink": 0.85},
        ax=ax,
    )
    ax.collections[0].colorbar.ax.locator = MaxNLocator(4)
    ax.collections[0].colorbar.update_ticks()
    ax.set_xlabel("Attention head index")
    ax.set_ylabel("Pair / Layer")
    ax.tick_params(axis="x", labelsize=11.5)
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    if len(ytick_labels) > 32:
        show_every = 2
        ax.set_yticks(np.arange(0.5, len(ytick_labels), show_every))
        ax.set_yticklabels(ytick_labels[::show_every], fontsize=11.5)
    else:
        ax.tick_params(axis="y", labelsize=11.5)
    fig.tight_layout()
    fig.savefig(latex_root / "figures" / "appendix_perhead_heatmap.pdf", dpi=450, bbox_inches="tight")
    plt.close(fig)

    top = (
        d.groupby(["pair", "layer", "head"], as_index=False)["axis_abs_projection_diff_mean"]
        .mean()
        .sort_values("axis_abs_projection_diff_mean", ascending=False)
        .head(12)
    )
    lines = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Pair & Layer & Head & Mean $D_{Axis}$ \\",
        r"\midrule",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"{r['pair']} & {int(r['layer'])} & {int(r['head'])} & {_fmt(r['axis_abs_projection_diff_mean'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "appendix_perhead_top.tex")


def build_multilingual_expansion_artifacts(multilingual_root: Path, latex_root: Path) -> None:
    langs = ["fas", "nld", "ukr", "bul", "ind", "deu"]
    lang_label = {
        "fas": "FAS",
        "nld": "NLD",
        "ukr": "UKR",
        "bul": "BUL",
        "ind": "IND",
        "deu": "DEU",
    }
    rows = []
    for lang in langs:
        p = multilingual_root / f"{lang}_shared_language" / "bli_summary_metrics.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)

        def pick(rt: str, ma: str, mb: str) -> pd.Series | None:
            s = d[(d["repr_type"] == rt) & (d["model_a"] == ma) & (d["model_b"] == mb)]
            if s.empty:
                return None
            return s.iloc[0]

        e50a = pick("embedding_matrix", f"{lang}_50m", f"en_{lang}_a")
        c50a = pick("pre_lmhead_contextual", f"{lang}_50m", f"en_{lang}_a")
        e100a = pick("embedding_matrix", f"{lang}_100m", f"en_{lang}_a")
        c100a = pick("pre_lmhead_contextual", f"{lang}_100m", f"en_{lang}_a")
        e50b = pick("embedding_matrix", f"{lang}_50m", f"en_{lang}_b")
        c50b = pick("pre_lmhead_contextual", f"{lang}_50m", f"en_{lang}_b")
        e100b = pick("embedding_matrix", f"{lang}_100m", f"en_{lang}_b")
        c100b = pick("pre_lmhead_contextual", f"{lang}_100m", f"en_{lang}_b")
        if any(x is None for x in [e50a, c50a, e100a, c100a, e50b, c50b, e100b, c100b]):
            continue

        ratio_50 = float(c50a["axis_abs_projection_diff_mean"]) / max(1e-12, float(e50a["axis_abs_projection_diff_mean"]))
        ratio_100 = float(c100a["axis_abs_projection_diff_mean"]) / max(1e-12, float(e100a["axis_abs_projection_diff_mean"]))

        gain_nn_emb = float(np.mean([
            float(e50b["jaccard_at_k_mean"]) - float(e50a["jaccard_at_k_mean"]),
            float(e100b["jaccard_at_k_mean"]) - float(e100a["jaccard_at_k_mean"]),
        ]))
        gain_nn_ctx = float(np.mean([
            float(c50b["jaccard_at_k_mean"]) - float(c50a["jaccard_at_k_mean"]),
            float(c100b["jaccard_at_k_mean"]) - float(c100a["jaccard_at_k_mean"]),
        ]))
        gain_axis_emb = float(np.mean([
            float(e50b["axis_abs_projection_diff_mean"]) - float(e50a["axis_abs_projection_diff_mean"]),
            float(e100b["axis_abs_projection_diff_mean"]) - float(e100a["axis_abs_projection_diff_mean"]),
        ]))
        gain_axis_ctx = float(np.mean([
            float(c50b["axis_abs_projection_diff_mean"]) - float(c50a["axis_abs_projection_diff_mean"]),
            float(c100b["axis_abs_projection_diff_mean"]) - float(c100a["axis_abs_projection_diff_mean"]),
        ]))

        rows.append(
            {
                "language": lang_label[lang],
                "ratio_50m": ratio_50,
                "ratio_100m": ratio_100,
                "overlap_gain_dnn_emb": gain_nn_emb,
                "overlap_gain_dnn_ctx": gain_nn_ctx,
                "overlap_gain_daxis_emb": gain_axis_emb,
                "overlap_gain_daxis_ctx": gain_axis_ctx,
            }
        )

    if not rows:
        return
    out = pd.DataFrame(rows).sort_values("language").reset_index(drop=True)
    # Persist machine-readable summary for downstream checks.
    out.to_csv(multilingual_root / "multilingual_summary.csv", index=False)

    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Ctx./emb. ratio} & \multicolumn{2}{c}{Overlap gain $D_{NN}$} & \multicolumn{2}{c}{Overlap gain $D_{Axis}$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Lang & 50M & 100M & Emb. & Ctx. & Emb. & Ctx. \\",
        r"\midrule",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"{r['language']} & {r['ratio_50m']:.1f} & {r['ratio_100m']:.1f} & "
            f"{_fmt(r['overlap_gain_dnn_emb'])} & {_fmt(r['overlap_gain_dnn_ctx'])} & "
            f"{_fmt(r['overlap_gain_daxis_emb'])} & {_fmt(r['overlap_gain_daxis_ctx'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "appendix_multilingual_summary.tex")

    ci_rows = []
    for lang in langs:
        ci_path = multilingual_root / f"{lang}_shared_language" / "bli_bootstrap_ci.csv"
        if not ci_path.exists():
            continue
        c = pd.read_csv(ci_path)
        c = c[
            (c["metric"] == "axis_ratio_contextual_over_embedding")
            & (c["model_a"].isin([f"{lang}_50m", f"{lang}_100m"]))
            & (c["model_b"] == f"en_{lang}_a")
        ].copy()
        if c.empty:
            continue
        c["baseline"] = c["model_a"].map({f"{lang}_50m": "50M", f"{lang}_100m": "100M"})
        c["language"] = lang_label[lang]
        ci_rows.append(c[["language", "baseline", "mean", "ci_low", "ci_high"]])

    if ci_rows:
        ci_df = pd.concat(ci_rows, ignore_index=True).sort_values(["language", "baseline"])
        ci_lines = [
            r"\begin{tabular}{@{}llrrr@{}}",
            r"\toprule",
            r"Lang & Baseline & Mean ratio & CI low & CI high \\",
            r"\midrule",
        ]
        for _, r in ci_df.iterrows():
            ci_lines.append(
                f"{r['language']} & {r['baseline']} & {r['mean']:.1f} & {r['ci_low']:.1f} & {r['ci_high']:.1f} \\\\"
            )
        ci_lines += [r"\bottomrule", r"\end{tabular}"]
        write_table(ci_lines, latex_root / "tables" / "appendix_multilingual_ratio_ci.tex")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), gridspec_kw={"wspace": 0.34}, facecolor=PAPER_BG)
    x = np.arange(len(out))
    width = 0.36
    b1 = axes[0].bar(
        x - width / 2,
        out["ratio_50m"].to_numpy(),
        width=width,
        color="#f4a261",
        edgecolor="black",
        linewidth=0.8,
        label="50M baseline",
    )
    b2 = axes[0].bar(
        x + width / 2,
        out["ratio_100m"].to_numpy(),
        width=width,
        color="#2a9d8f",
        edgecolor="black",
        linewidth=0.8,
        label="100M baseline",
    )
    for b in b1:
        b.set_hatch("//")
    for b in b2:
        b.set_hatch("..")
    axes[0].set_ylabel(r"Contextual / embedding $D_{Axis}$ ratio", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(out["language"].tolist())
    axes[0].tick_params(axis="both", labelsize=11)
    axes[0].grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.5, alpha=0.42)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True,
        edgecolor="#9a9a9a",
        fontsize=11,
    )

    b3 = axes[1].bar(
        x - width / 2,
        out["overlap_gain_daxis_emb"].to_numpy(),
        width=width,
        color="#cdcdcd",
        edgecolor="black",
        linewidth=0.8,
        label="Embedding",
    )
    b4 = axes[1].bar(
        x + width / 2,
        out["overlap_gain_daxis_ctx"].to_numpy(),
        width=width,
        color="#9fc4e6",
        edgecolor="black",
        linewidth=0.8,
        label="Contextual",
    )
    for b in b3:
        b.set_hatch("..")
    for b in b4:
        b.set_hatch("//")
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel(r"Overlap gain on $D_{Axis}$", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(out["language"].tolist())
    axes[1].tick_params(axis="both", labelsize=11)
    axes[1].grid(axis="y", linestyle=(0, (3, 2)), linewidth=0.5, alpha=0.42)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True,
        edgecolor="#9a9a9a",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.82, wspace=0.34)
    fig.savefig(latex_root / "figures" / "appendix_multilingual_overview.pdf", dpi=450, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    # Exploratory regression block for cross-language trend discussion.
    # Proxies are intentionally coarse and fully disclosed in the appendix:
    # script_match_with_english: Latin script -> 1, else 0
    # family_match_with_english: Indo-European -> 1, else 0
    proxy = pd.DataFrame(
        [
            {"language": "FAS", "script_match_with_english": 0, "family_match_with_english": 1},
            {"language": "NLD", "script_match_with_english": 1, "family_match_with_english": 1},
            {"language": "UKR", "script_match_with_english": 0, "family_match_with_english": 1},
            {"language": "BUL", "script_match_with_english": 0, "family_match_with_english": 1},
            {"language": "IND", "script_match_with_english": 1, "family_match_with_english": 0},
            {"language": "DEU", "script_match_with_english": 1, "family_match_with_english": 1},
        ]
    )
    reg = out.merge(proxy, on="language", how="inner")
    if len(reg) >= 4:
        long = pd.concat(
            [
                reg[["language", "script_match_with_english", "family_match_with_english", "ratio_50m"]]
                .rename(columns={"ratio_50m": "ratio"})
                .assign(is_100m=0),
                reg[["language", "script_match_with_english", "family_match_with_english", "ratio_100m"]]
                .rename(columns={"ratio_100m": "ratio"})
                .assign(is_100m=1),
            ],
            ignore_index=True,
        )
        y = long["ratio"].to_numpy(dtype=np.float64)
        X = np.column_stack(
            [
                np.ones(len(long), dtype=np.float64),
                long["script_match_with_english"].to_numpy(dtype=np.float64),
                long["family_match_with_english"].to_numpy(dtype=np.float64),
                long["is_100m"].to_numpy(dtype=np.float64),
            ]
        )
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - (ss_res / max(1e-12, ss_tot))

        reg["ratio_mean"] = 0.5 * (reg["ratio_50m"] + reg["ratio_100m"])
        reg["distance_proxy"] = 2 - reg["script_match_with_english"] - reg["family_match_with_english"]
        rho, pval = spearmanr(reg["distance_proxy"].to_numpy(), reg["ratio_mean"].to_numpy())

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter(
            reg["distance_proxy"].to_numpy(),
            reg["ratio_mean"].to_numpy(),
            s=60,
            c="#9fc4e6",
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95,
        )
        for _, rr in reg.iterrows():
            ax.annotate(
                rr["language"],
                (rr["distance_proxy"], rr["ratio_mean"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=10.0,
            )
        xfit = reg["distance_proxy"].to_numpy(dtype=np.float64)
        yfit = reg["ratio_mean"].to_numpy(dtype=np.float64)
        if len(np.unique(xfit)) > 1:
            m, b = np.polyfit(xfit, yfit, 1)
            xs = np.linspace(xfit.min(), xfit.max(), 50)
            ax.plot(xs, m * xs + b, color="#264653", linewidth=1.3, linestyle=(0, (3, 2)))
        ax.set_xlabel("Distance proxy (higher means farther from English)")
        ax.set_ylabel(r"Mean amplification ratio $(D_{Axis}^{Ctx}/D_{Axis}^{Emb})$")
        ax.grid(True, linestyle=(0, (3, 2)), linewidth=0.5, alpha=0.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(latex_root / "figures" / "appendix_multilingual_regression_scatter.pdf", dpi=450, bbox_inches="tight")
        plt.close(fig)

        lines = [
            r"\begin{tabular}{@{}lr@{}}",
            r"\toprule",
            r"Quantity & Value \\",
            r"\midrule",
            f"OLS intercept & {_fmt(coef[0])} \\\\",
            f"OLS coef: script-match (Latin=1) & {_fmt(coef[1])} \\\\",
            f"OLS coef: family-match (Indo-European=1) & {_fmt(coef[2])} \\\\",
            f"OLS coef: baseline(100M=1) & {_fmt(coef[3])} \\\\",
            f"OLS $R^2$ (pooled 50M/100M; $n={len(long)}$) & {_fmt(r2)} \\\\",
            f"Spearman $\\rho$(distance proxy, mean ratio; $n={len(reg)}$) & {_fmt(float(rho))} \\\\",
            f"Spearman $p$-value & {float(pval):.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        write_table(lines, latex_root / "tables" / "appendix_multilingual_regression.tex")


def write_report_meta(output_root: Path) -> None:
    paths = [
        output_root / "en_ablation" / "bli_summary_metrics.csv",
        output_root / "zh_shared_language" / "bli_summary_metrics.csv",
        output_root / "fr_shared_language" / "bli_summary_metrics.csv",
    ]
    meta = {"sources": [str(p) for p in paths], "status": "artifacts_generated"}
    (output_root / "report_meta_v2.json").write_text(pd.Series(meta).to_json(indent=2), encoding="utf-8")


def build_main_multilingual_validation_figure(
    output_root: Path,
    multilingual_root: Path,
    latex_root: Path,
    same_lang_df: pd.DataFrame | None = None,
) -> None:
    summary_path = multilingual_root / "language_ratio_summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path).copy()
    if df.empty:
        return
    df = df.sort_values("lang_code").reset_index(drop=True)

    def _axis(lang: str, baseline: str, repr_type: str) -> float:
        paths = [
            output_root / f"{lang}_shared_language" / "bli_summary_metrics.csv",
            multilingual_root / f"{lang}_shared_language" / "bli_summary_metrics.csv",
        ]
        p = next((x for x in paths if x.exists()), None)
        if p is None:
            return float("nan")
        d = pd.read_csv(p)
        base = f"{lang}_{baseline}"
        a = d[
            (d["repr_type"] == repr_type)
            & (d["model_a"] == base)
            & (d["model_b"] == f"en_{lang}_a")
        ]
        if a.empty:
            return float("nan")
        return float(a["axis_abs_projection_diff_mean"].iloc[0])

    def _plot_multilingual_slice(baseline: str, out_name: str) -> None:
        ctx_col = f"ctx_{baseline}"
        emb_col = f"emb_{baseline}"
        df[ctx_col] = df["lang_code"].astype(str).map(lambda x: _axis(str(x), baseline, "pre_lmhead_contextual"))
        df[emb_col] = df["lang_code"].astype(str).map(lambda x: _axis(str(x), baseline, "embedding_matrix"))
        p = df.dropna(subset=[ctx_col, emb_col]).copy()
        if p.empty:
            return
        rank = {lg: i for i, lg in enumerate(CORE_LANGS)}
        p["lang_rank"] = p["lang_code"].astype(str).map(lambda x: rank.get(x, 999))
        p = p.sort_values("lang_rank").reset_index(drop=True)
        ref = None

        y = np.arange(len(p))
        fig, ax = plt.subplots(figsize=(COL_FIG_W, COL_FIG_H), facecolor=PAPER_BG)
        style_paper_axis(ax, grid_axis="x")
        ctx_vals = p[ctx_col].to_numpy(dtype=float)
        emb_vals = p[emb_col].to_numpy(dtype=float)
        if ref is not None:
            ax.axvspan(ref["low"], ref["high"], color="#9e9e9e", alpha=0.16, zorder=0)
            ax.axvline(ref["mean"], color="#616161", linestyle=(0, (4, 2)), linewidth=1.15, zorder=1)
        for i, (_, rr) in enumerate(p.iterrows()):
            lang = str(rr["lang_code"]).upper()
            c = pastel_lang_color(lang, lighten=0.30)
            ec = blend_hex(LANG_COLORS.get(lang, "#4C78A8"), (0, 0, 0), 0.35)
            ax.plot([rr[emb_col], rr[ctx_col]], [i, i], color="#9aa6b4", linewidth=2.2, alpha=0.95, zorder=1)
            ax.scatter([rr[emb_col]], [i], s=42, marker="o", facecolors="white", edgecolors=ec, linewidths=1.0, zorder=3)
            ax.scatter([rr[ctx_col]], [i], s=52, marker="s", facecolors=c, edgecolors=ec, linewidths=0.9, zorder=4)
            ax.text(
                float(rr[ctx_col]) + 0.025 * max(0.3, float(np.nanmax(ctx_vals))),
                i,
                f"+{float(rr[ctx_col] - rr[emb_col]):.2f}",
                ha="left",
                va="center",
                fontsize=10.0,
                color="#333333",
            )

        ax.set_yticks(y)
        ax.set_yticklabels(p["lang_code"].astype(str).str.upper().tolist(), fontsize=10.0, fontweight="bold")
        color_language_ticklabels(ax, axis="y")
        ax.invert_yaxis()
        ax.set_ylabel("Language", fontsize=10.0)
        ax.set_xlabel(f"$D_{{Axis}}$ in {baseline.upper()} space (mono. vs EN+L2, shared docs)", fontsize=10.0)
        ax.tick_params(axis="x", labelsize=10.0)
        xmax = max(float(np.nanmax(ctx_vals)), float(np.nanmax(emb_vals)))
        ax.set_xlim(0.0, xmax * 1.22)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#4c5a67", label="Embedding", markersize=6),
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#c9dff2", markeredgecolor="#4c5a67", label="Contextual", markersize=6),
        ]
        if ref is not None:
            handles.append(
                plt.Line2D([0], [0], color="#6b6b6b", linestyle=(0, (3, 2)), lw=1.2, label="EN-NULL mean/range")
            )
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#c5ccd7",
            framealpha=0.96,
            fontsize=10.0,
            columnspacing=0.80,
            handletextpad=0.35,
        )
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.20, top=0.82)
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    _plot_multilingual_slice("50m", "combined_multilingual_fig5_fig11.pdf")
    _plot_multilingual_slice("100m", "combined_multilingual_fig5_fig11_100m.pdf")


def build_same_language_controls_table(ctrl_df: pd.DataFrame | None, latex_root: Path) -> None:
    if ctrl_df is None or ctrl_df.empty:
        lines = [
            r"\begin{tabular}{@{}l@{}}",
            r"\toprule",
            r"Same-language control metrics unavailable in current artifact cache. \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        write_table(lines, latex_root / "tables" / "same_language_controls.tex")
        return
    x = ctrl_df.copy()
    x["baseline"] = x["language"].astype(str)
    x["repr"] = x["eval_repr"].map(
        {
            "embedding_matrix": "Embedding",
            "pre_lmhead_contextual": "Contextual",
        }
    ).fillna(x["eval_repr"])
    g = (
        x.groupby(["baseline", "repr"], as_index=False)["axis_abs_projection_diff_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Baseline & Representation & Mean raw $D_{Axis}$ & Std. dev. & EN-seed pairs \\",
        r"\midrule",
    ]
    for _, r in g.sort_values(["baseline", "repr"]).iterrows():
        lines.append(
            f"{r['baseline']} & {r['repr']} & {_fmt(r['mean'])} & {_fmt(r['std'])} & {int(r['count'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "same_language_controls.tex")


def build_framework_holdout_table(framework_df: pd.DataFrame | None, latex_root: Path) -> None:
    if framework_df is None or framework_df.empty:
        lines = [
            r"\begin{tabular}{@{}l@{}}",
            r"\toprule",
            r"Framework-holdout metrics unavailable in current artifact cache. \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        write_table(lines, latex_root / "tables" / "framework_holdout.tex")
        return
    agg = framework_df[framework_df["scope"] == "aggregate"].copy()
    if agg.empty:
        return
    repr_map = {
        "embedding_matrix": "Embedding",
        "pre_lmhead_contextual": "Contextual",
    }
    agg["repr"] = agg["eval_repr"].map(repr_map).fillna(agg["eval_repr"])
    g = agg.groupby("repr", as_index=False)[["daxis_matched", "daxis_heldout", "delta_heldout_minus_matched"]].mean()

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Representation & Matched-axis $D_{Axis}$ & Held-out-axis $D_{Axis}$ & Held-out minus matched \\",
        r"\midrule",
    ]
    for _, r in g.iterrows():
        lines.append(
            f"{r['repr']} & {_fmt(r['daxis_matched'])} & {_fmt(r['daxis_heldout'])} & {_fmt(r['delta_heldout_minus_matched'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "framework_holdout.tex")


def build_progress_sensitivity_table(progress_df: pd.DataFrame | None, latex_root: Path) -> None:
    if progress_df is None or progress_df.empty:
        lines = [
            r"\begin{tabular}{@{}l@{}}",
            r"\toprule",
            r"Progress-sensitivity metrics unavailable in current artifact cache. \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
        write_table(lines, latex_root / "tables" / "progress_sensitivity.tex")
        return
    if {"step", "language", "repr_type", "axis_abs_projection_diff_mean"}.issubset(progress_df.columns):
        p = progress_df.copy()
        order = [code.upper() for code in CORE_LANGS]
        rows = []
        for lang in order:
            sub = p[p["language"].astype(str).str.upper() == lang].copy()
            if sub.empty:
                continue
            emb = sub[sub["repr_type"] == "embedding_matrix"].set_index("step")["axis_abs_projection_diff_mean"]
            ctx = sub[sub["repr_type"] == "pre_lmhead_contextual"].set_index("step")["axis_abs_projection_diff_mean"]
            common = sorted(set(emb.index.tolist()) & set(ctx.index.tolist()))
            if not common:
                continue
            rows.append(
                {
                    "lang": lang,
                    "emb_500": float(emb.get(500, np.nan)),
                    "ctx_500": float(ctx.get(500, np.nan)),
                    "emb_3000": float(emb.get(3000, np.nan)),
                    "ctx_3000": float(ctx.get(3000, np.nan)),
                    "mean_gap": float(np.mean([float(ctx[s] - emb[s]) for s in common])),
                }
            )
        lines = [
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Language & 500 Emb. & 500 Ctx. & 3000 Emb. & 3000 Ctx. & Mean gap \\",
            r"\midrule",
        ]
        for r in rows:
            lines.append(
                f"{r['lang']} & {_fmt(r['emb_500'])} & {_fmt(r['ctx_500'])} & {_fmt(r['emb_3000'])} & {_fmt(r['ctx_3000'])} & {_fmt(r['mean_gap'])} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        write_table(lines, latex_root / "tables" / "progress_sensitivity.tex")
        return
    p = progress_df.copy()
    repr_map = {
        "embedding_matrix": "Embedding",
        "pre_lmhead_contextual": "Contextual",
        "ratio_ctx_over_emb": "Contextual/Embedding ratio",
    }
    p["repr"] = p["repr_type"].map(repr_map).fillna(p["repr_type"])
    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Language & Quantity & 50M & 100M & 100M minus 50M \\",
        r"\midrule",
    ]
    for lang in sorted(p["language"].unique()):
        sub = p[p["language"] == lang]
        first = True
        for key in ["Embedding", "Contextual", "Contextual/Embedding ratio"]:
            s = sub[sub["repr"] == key]
            if s.empty:
                continue
            r = s.iloc[0]
            left = lang if first else ""
            lines.append(
                f"{left} & {key} & {_fmt(r['daxis_50m'])} & {_fmt(r['daxis_100m'])} & {_fmt(r['delta_100m_minus_50m'])} \\\\"
            )
            first = False
        lines.append(r"\cmidrule(lr){1-5}")
    if lines[-1] == r"\cmidrule(lr){1-5}":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "progress_sensitivity.tex")


def build_dense_progress_trajectory_figure(progress_df: pd.DataFrame | None, latex_root: Path) -> None:
    if progress_df is None or progress_df.empty:
        return
    need = {"step", "language", "repr_type", "axis_abs_projection_diff_mean"}
    if not need.issubset(progress_df.columns):
        return

    df = progress_df.copy()
    order = [code.upper() for code in CORE_LANGS]
    df["language"] = df["language"].astype(str).str.upper()
    df = df[df["language"].isin(order)].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(2, 4, figsize=(7.35, 4.55), sharex=True, sharey=True, facecolor=PAPER_BG)
    axes = axes.flatten()
    markers = {"embedding_matrix": "o", "pre_lmhead_contextual": "s"}
    linestyles = {"embedding_matrix": (0, (3, 2)), "pre_lmhead_contextual": "-"}
    labels = {"embedding_matrix": "Embedding", "pre_lmhead_contextual": "Contextual"}

    for ax, lang in zip(axes, order):
        style_paper_axis(ax, grid_axis="y")
        sub = df[df["language"] == lang].copy()
        for repr_type in ["embedding_matrix", "pre_lmhead_contextual"]:
            rsub = sub[sub["repr_type"] == repr_type].sort_values("step")
            if rsub.empty:
                continue
            color = pastel_lang_color(lang, lighten=0.52 if repr_type == "embedding_matrix" else 0.18)
            ax.plot(
                rsub["step"],
                rsub["axis_abs_projection_diff_mean"],
                marker=markers[repr_type],
                linestyle=linestyles[repr_type],
                linewidth=1.8,
                markersize=4.5,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.35,
                label=labels[repr_type],
            )
        ax.set_title(f"EN vs EN+{lang}", fontsize=10.0, color=LANG_COLORS.get(lang, INK), pad=4)
        ax.set_xticks([500, 1500, 3000])
        ax.set_xticklabels(["500", "1500", "3000"], fontsize=10.0, color=INK)
        ax.tick_params(axis="y", labelsize=10.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    for ax in axes[:4]:
        ax.tick_params(axis="x", labelbottom=False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        fontsize=10.0,
    )
    fig.supxlabel("Training step", fontsize=10.0, color=INK, y=0.035)
    fig.supylabel(r"$D_{Axis}$", fontsize=10.0, color=INK, x=0.018)
    fig.subplots_adjust(top=0.86, wspace=0.24, hspace=0.34, bottom=0.13, left=0.085, right=0.99)
    fig.savefig(latex_root / "figures" / "appendix_dense_progress_trajectory.pdf", dpi=450, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def build_dense_progress_summary_figure(progress_df: pd.DataFrame | None, latex_root: Path) -> None:
    if progress_df is None or progress_df.empty:
        return
    need = {"step", "language", "repr_type", "axis_abs_projection_diff_mean"}
    if not need.issubset(progress_df.columns):
        return

    df = progress_df.copy()
    order = [code.upper() for code in CORE_LANGS]
    df["language"] = df["language"].astype(str).str.upper()
    df = df[
        df["language"].isin(order)
        & df["repr_type"].isin(["embedding_matrix", "pre_lmhead_contextual"])
    ].copy()
    if df.empty:
        return

    metric = "axis_abs_projection_diff_mean"
    summary = (
        df.groupby(["step", "repr_type"], as_index=False)[metric]
        .agg(
            mean="mean",
            q20=lambda s: float(s.quantile(0.20)),
            q80=lambda s: float(s.quantile(0.80)),
        )
        .sort_values(["repr_type", "step"])
    )

    fig, ax = plt.subplots(figsize=(COL_FIG_W, 2.04), facecolor=PAPER_BG)
    style_paper_axis(ax, grid_axis="y")
    configs = {
        "embedding_matrix": {
            "label": "Embedding",
            "color": "#5f7890",
            "marker": "o",
            "linestyle": (0, (3, 2)),
            "alpha": 0.16,
            "individual_alpha": 0.20,
        },
        "pre_lmhead_contextual": {
            "label": "Contextual",
            "color": NAVY,
            "marker": "s",
            "linestyle": "-",
            "alpha": 0.18,
            "individual_alpha": 0.16,
        },
    }
    for repr_type, cfg in configs.items():
        for lang in order:
            rsub = df[(df["language"] == lang) & (df["repr_type"] == repr_type)].sort_values("step")
            if rsub.empty:
                continue
            ax.plot(
                rsub["step"],
                rsub[metric],
                color=cfg["color"],
                linewidth=0.85,
                linestyle=cfg["linestyle"],
                alpha=cfg["individual_alpha"],
                zorder=1,
                solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=1.8, foreground=PAPER_BG, alpha=0.35), pe.Normal()],
            )
        s = summary[summary["repr_type"] == repr_type].sort_values("step")
        if s.empty:
            continue
        x = s["step"].to_numpy(dtype=float)
        mean = s["mean"].to_numpy(dtype=float)
        q20 = s["q20"].to_numpy(dtype=float)
        q80 = s["q80"].to_numpy(dtype=float)
        ax.fill_between(x, q20, q80, color=cfg["color"], alpha=cfg["alpha"], linewidth=0, zorder=2)
        ax.plot(
            x,
            mean,
            color=cfg["color"],
            linewidth=1.9,
            marker=cfg["marker"],
            markersize=4.8,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            linestyle=cfg["linestyle"],
            label=cfg["label"],
            zorder=4,
        )
    ax.set_xlim(450, 3050)
    ax.set_xticks([500, 1500, 3000])
    ax.set_xticklabels(["500", "1500", "3000"], fontsize=10.0)
    ax.tick_params(axis="y", labelsize=10.0)
    ax.set_xlabel("Training step", fontsize=10.0, color=INK, labelpad=2)
    ax.set_ylabel(r"Raw $D_{Axis}$", fontsize=10.0, color=INK, labelpad=2)
    ax.legend(loc="upper left", fontsize=10.0, frameon=True, facecolor="white", edgecolor="#cbd1d6", framealpha=0.92)
    ax.text(
        0.98,
        0.08,
        "mean and 20 to 80% range",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.0,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.15, right=0.985, top=0.98, bottom=0.22)
    fig.savefig(latex_root / "figures" / "main_dense_progress_summary.pdf", dpi=450, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def build_anchor_sensitivity_table(anchor_df: pd.DataFrame | None, latex_root: Path) -> None:
    if anchor_df is None or anchor_df.empty:
        return
    g = (
        anchor_df.groupby(["baseline", "repr_type", "subset_size"], as_index=False)[
            ["axis_abs_projection_diff_mean", "anchor_residual_per_anchor"]
        ]
        .mean()
    )
    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Baseline & Representation & Anchor subset & Mean $D_{Axis}$ & Residual / anchor \\",
        r"\midrule",
    ]
    repr_map = {
        "embedding_matrix": "Embedding",
        "pre_lmhead_contextual": "Contextual",
    }
    base_map = {"en_50m": "EN-50M", "en_100m": "EN-100M"}
    for _, r in g.sort_values(["baseline", "repr_type", "subset_size"]).iterrows():
        lines.append(
            f"{base_map.get(str(r['baseline']), str(r['baseline']))} & {repr_map.get(str(r['repr_type']), str(r['repr_type']))} & "
            f"{int(r['subset_size'])} & {_fmt(r['axis_abs_projection_diff_mean'])} & {_fmt(r['anchor_residual_per_anchor'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "anchor_sensitivity.tex")


def build_tokenizer_check_table(tokenizer_df: pd.DataFrame | None, latex_root: Path) -> None:
    if tokenizer_df is None or tokenizer_df.empty:
        return
    use_cols = [
        c for c in [
            "tokenizer.json_sha256",
            "tokenizer_config.json_sha256",
            "special_tokens_map.json_sha256",
        ]
        if c in tokenizer_df.columns
    ]
    rows = []
    for col in use_cols:
        vals = tokenizer_df[col].dropna().astype(str)
        vals = vals[vals != ""]
        rows.append(
            {
                "artifact": col.replace("_sha256", ""),
                "unique_hashes": int(vals.nunique()),
            }
        )
    if not rows:
        return
    lines = [
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"Tokenizer artifact & Unique hashes across runs \\",
        r"\midrule",
    ]
    for r in rows:
        artifact = str(r["artifact"]).replace("_", r"\_")
        lines.append(f"{artifact} & {r['unique_hashes']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "tokenizer_check.tex")


def build_scope_tests_table(scope_df: pd.DataFrame | None, latex_root: Path) -> None:
    if scope_df is None or scope_df.empty:
        return
    keep = scope_df[
        scope_df["test"].isin(
            ["contextual_gt_embedding_axis", "anchor_sensitivity_ctx_gt_emb"]
        )
    ].copy()
    if keep.empty:
        return
    label_map = {
        "contextual_gt_embedding_axis": "Main EN-centered gap",
        "anchor_sensitivity_ctx_gt_emb": "Anchor-subset reruns",
        "en_50m": "EN-50M",
        "en_100m": "EN-100M",
        "all": "All",
    }
    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Test block & Slice & $n$ & Mean diff. & One-sided $p$ \\",
        r"\midrule",
    ]
    order = {
        ("contextual_gt_embedding_axis", "en_50m"): 0,
        ("contextual_gt_embedding_axis", "en_100m"): 1,
        ("contextual_gt_embedding_axis", "all"): 2,
        ("anchor_sensitivity_ctx_gt_emb", "en_50m"): 3,
        ("anchor_sensitivity_ctx_gt_emb", "en_100m"): 4,
        ("anchor_sensitivity_ctx_gt_emb", "all"): 5,
    }
    keep["__order"] = keep.apply(lambda r: order.get((r["test"], r["slice"]), 999), axis=1)
    keep = keep.sort_values("__order")
    for _, r in keep.iterrows():
        p = float(r["p_value_greater"])
        p_str = r"$< 10^{-6}$" if p < 1e-6 else f"{p:.3g}"
        lines.append(
            f"{label_map.get(str(r['test']), str(r['test']))} & {label_map.get(str(r['slice']), str(r['slice']))} & "
            f"{int(r['n'])} & {_fmt(r['mean_difference'])} & {p_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "scope_tests.tex")


def build_main_multilingual_regression_figure_and_table(multilingual_root: Path, latex_root: Path) -> None:
    """Generate main-text multilingual regression figure and compact ratio table.

    Reads language_ratio_summary.csv and generates:
    1. Scatter plot with regression line: Hofstede IDV distance vs Contextual/Embedding ratio
    2. Compact table for main text with 6-language ratio matrix
    """
    summary_path = multilingual_root / "language_ratio_summary.csv"
    if not summary_path.exists():
        print(f"Skipping main multilingual artifacts: {summary_path} not found")
        return

    df = pd.read_csv(summary_path)

    # Add more informative family labels
    family_map = {
        "Chinese": "Sino-Tibetan",
        "French": "Indo-Eur (Romance)",
        "Persian": "Indo-Eur (Iranian)",
        "German": "Indo-Eur (Germanic)",
        "Dutch": "Indo-Eur (Germanic)",
        "Ukrainian": "Indo-Eur (Slavic)",
        "Bulgarian": "Indo-Eur (Slavic)",
        "Indonesian": "Austronesian",
    }
    df["family"] = df["language"].map(family_map)

    # Create family color mapping
    family_colors = {
        "Sino-Tibetan": "#e74c3c",
        "Indo-Eur (Romance)": "#3498db",
        "Indo-Eur (Iranian)": "#e67e22",
        "Indo-Eur (Germanic)": "#2ecc71",
        "Indo-Eur (Slavic)": "#9b59b6",
        "Austronesian": "#f39c12",
    }
    df["color"] = df["family"].map(family_colors)

    # Generate regression figure.
    set_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Plot each language as a colored point with error bars
    for _, row in df.iterrows():
        # Compute symmetric error for visualization (using both 50M and 100M CIs)
        # Average the lower and upper CI differences
        err_low = row["avg_ratio"] - min(row["ci_50m_low"], row["ci_100m_low"])
        err_high = max(row["ci_50m_high"], row["ci_100m_high"]) - row["avg_ratio"]

        # Plot individual point with error bar
        ax.errorbar(
            row["hofstede_dist"],
            row["avg_ratio"],
            yerr=[[err_low], [err_high]],
            fmt="o",
            markersize=8,
            color=row["color"],
            alpha=0.8,
            capsize=5,
            capthick=1.5,
            linewidth=1.5,
            elinewidth=1.5,
            label=row["language"],
        )

    # Compute and plot Spearman correlation with regression line
    valid = df.dropna(subset=["hofstede_dist", "avg_ratio"])
    if len(valid) > 2:
        from scipy import stats as sp_stats
        x = valid["hofstede_dist"].values
        y = valid["avg_ratio"].values

        # Compute Spearman correlation
        rho, p_val = spearmanr(x, y)

        # Linear regression for visualization
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min() - 5, x.max() + 5, 100)
        p_text = "<0.001" if p_val < 1e-3 else f"={p_val:.3f}"
        ax.plot(x_line, p(x_line), color="black", linestyle=(0, (3, 2)), alpha=0.4, linewidth=1.5, label=f"Spearman ρ={rho:.2f}, p{p_text}")

    ax.set_xlabel("Hofstede IDV Distance from English", fontsize=12)
    ax.set_ylabel("Contextual-to-Embedding Ratio", fontsize=12)
    ax.set_title("Typological Distance vs Representation-Dependence Gap", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10.0, framealpha=0.95)
    ax.set_xlim(-5, 80)

    plt.tight_layout()
    fig_path = latex_root / "figures" / "main_multilingual_regression.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {fig_path}")

    # Generate main-text multilingual summary table.
    lines = [
        r"\begin{tabular}{@{}lllrr@{}}",
        r"\toprule",
        r"Language & Family & IDV Dist. & Ratio (50M) & Ratio (100M) \\",
        r"\midrule",
    ]

    # Sort by average ratio descending for prominence
    df_sorted = df.sort_values("avg_ratio", ascending=False)
    for _, row in df_sorted.iterrows():
        family_short = row["family"].replace("Indo-Eur ", "").replace("Sino-Tibetan", "Sino-Tib.")
        lines.append(
            f"{row['language']:12} & {family_short:18} & {row['hofstede_dist']:5.0f} & "
            f"{row['ratio_50m']:6.2f} & {row['ratio_100m']:6.2f} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "main_multilingual_ratios.tex")
    print(f"Generated {latex_root / 'tables' / 'main_multilingual_ratios.tex'}")


def main() -> None:
    global EMIT_TABLES
    args = parse_args()
    if args.main_figures_only:
        args.figures_only = True
    EMIT_TABLES = not args.figures_only
    set_style()
    ensure_dirs(args.latex_root)

    en = add_agreement(pd.read_csv(args.output_root / "en_ablation" / "bli_summary_metrics.csv"), mode="direct")
    zh = add_agreement(pd.read_csv(args.output_root / "zh_shared_language" / "bli_summary_metrics.csv"))
    fr = add_agreement(pd.read_csv(args.output_root / "fr_shared_language" / "bli_summary_metrics.csv"))

    ci_path = args.output_root / "en_ablation" / "bli_bootstrap_ci.csv"
    ci = pd.read_csv(ci_path) if ci_path.exists() else None

    stats_path = args.output_root / "en_ablation" / "bli_wilcoxon_overlap.csv"
    stats_df = pd.read_csv(stats_path) if stats_path.exists() else None
    word_path = args.output_root / "en_ablation" / "bli_word_neighbor_divergence.csv"
    word_df = pd.read_csv(word_path) if word_path.exists() else None
    axis_path = args.output_root / "en_ablation" / "bli_axis_divergence.csv"
    axis_df = pd.read_csv(axis_path) if axis_path.exists() else None
    ctx_align_path = args.output_root / "en_ablation" / "bli_contextual_alignment_variant.csv"
    ctx_align_df = pd.read_csv(ctx_align_path) if ctx_align_path.exists() else None
    aln_method_path = args.output_root / "en_ablation" / "bli_alignment_method_comparison.csv"
    aln_method_df = pd.read_csv(aln_method_path) if aln_method_path.exists() else None
    same_lang_path = args.output_root / "en_ablation" / "bli_same_language_controls.csv"
    same_lang_df = pd.read_csv(same_lang_path) if same_lang_path.exists() else None
    framework_path = args.output_root / "en_ablation" / "bli_framework_holdout_eval.csv"
    framework_df = pd.read_csv(framework_path) if framework_path.exists() else None
    progress_path = args.output_root / "en_ablation" / "bli_progress_sensitivity.csv"
    progress_df = pd.read_csv(progress_path) if progress_path.exists() else None
    dense_progress_path = args.output_root / "en_ablation" / "bli_dense_progress_trajectory.csv"
    dense_progress_df = pd.read_csv(dense_progress_path) if dense_progress_path.exists() else None
    anchor_path = args.output_root / "en_ablation" / "bli_anchor_sensitivity.csv"
    anchor_df = pd.read_csv(anchor_path) if anchor_path.exists() else None
    scope_tests_path = args.output_root / "en_ablation" / "bli_scope_tests.csv"
    scope_tests_df = pd.read_csv(scope_tests_path) if scope_tests_path.exists() else None
    tokenizer_audit_path = args.output_root / "en_ablation" / "bli_tokenizer_audit.csv"
    tokenizer_audit_df = pd.read_csv(tokenizer_audit_path) if tokenizer_audit_path.exists() else None

    strat_path = args.output_root / "en_ablation" / "bli_stratified_metrics.csv"
    strat_df = pd.read_csv(strat_path) if strat_path.exists() else None

    layer_path = args.output_root / "en_ablation" / "bli_layerwise_divergence.csv"
    layer_df = pd.read_csv(layer_path) if layer_path.exists() else None
    norm_control_path = args.output_root / "en_ablation" / "bli_norm_controlled_axis.csv"
    norm_control_df = pd.read_csv(norm_control_path) if norm_control_path.exists() else None
    norm_layer_path = args.output_root / "en_ablation" / "bli_norm_controlled_layerwise.csv"
    norm_layer_df = pd.read_csv(norm_layer_path) if norm_layer_path.exists() else None
    perhead_path = args.output_root / "en_ablation" / "bli_perhead_analysis.csv"
    perhead_df = pd.read_csv(perhead_path) if perhead_path.exists() else None
    neg_path = args.output_root / "en_ablation" / "bli_negative_control_eval.csv"
    neg_df = pd.read_csv(neg_path) if neg_path.exists() else None
    probe_set = json.loads(args.probe_set.read_text(encoding="utf-8")) if args.probe_set.exists() else {}
    translations: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(args.translations_dir.glob("translations_*.csv")):
        lang = csv_path.stem.replace("translations_", "").strip().lower()
        if not lang:
            continue
        translations[lang] = pd.read_csv(csv_path)

    if args.main_figures_only:
        build_control_matrix_figure(en, args.latex_root)
        build_exp1_tables_fig(en, ci, args.latex_root, same_lang_df=same_lang_df)
        build_exp2_table_fig(en, stats_df, args.latex_root, same_lang_df=same_lang_df)
        build_exp4_ratio(en, ci, args.latex_root)
        build_exp5_alignment_method_artifacts(aln_method_df, args.latex_root)
        build_dense_progress_summary_figure(dense_progress_df, args.latex_root)
        build_exp4_signed_axis_scatter(
            axis_df,
            args.latex_root,
            output_root=args.output_root,
            probe_set=probe_set,
        )
        build_layerwise_artifacts(layer_df, args.latex_root)
        build_main_multilingual_validation_figure(
            args.output_root,
            args.multilingual_output_root,
            args.latex_root,
            same_lang_df=same_lang_df,
        )
        publish_numbered_assets(
            args.latex_root,
            include_tables=False,
            figure_names=MAIN_FIGURE_SOURCE_NAMES,
        )
        print("Generated main figure artifacts.")
        return

    build_control_matrix_figure(en, args.latex_root)
    build_exp1_tables_fig(en, ci, args.latex_root, same_lang_df=same_lang_df)
    if word_df is not None:
        build_exp1_hotspots_table(word_df, args.latex_root)
    build_exp1_negative_controls_table(neg_df, args.latex_root)
    build_exp1_procrustes_table(en, args.latex_root)
    build_exp2_table_fig(en, stats_df, args.latex_root, same_lang_df=same_lang_df)
    build_exp2_quality_table(stats_df, args.latex_root)
    build_exp3_table_fig(zh, fr, args.latex_root)
    build_exp4_ratio(en, ci, args.latex_root)
    build_exp5_alignment_method_artifacts(aln_method_df, args.latex_root)
    build_appendix_daxis_interpretation(en, args.latex_root)
    build_exp4_signed_axis_scatter(
        axis_df,
        args.latex_root,
        output_root=args.output_root,
        probe_set=probe_set,
    )
    build_appendix_l2_signed_hotspots(
        output_root=args.output_root,
        multilingual_root=args.multilingual_output_root,
        latex_root=args.latex_root,
        probe_set=probe_set,
    )
    build_layerwise_artifacts(layer_df, args.latex_root)
    build_norm_control_artifacts(norm_control_df, norm_layer_df, args.latex_root)
    build_category_heatmap(strat_df, args.latex_root)
    build_contextual_alignment_variant_artifacts(ctx_align_df, args.latex_root)
    build_perhead_artifacts(perhead_df, args.latex_root)
    build_axis_inventory_table(probe_set, args.latex_root)
    build_axis_grounding_table(probe_set, args.latex_root)
    build_probe_qc_table(translations, args.latex_root)
    build_multilingual_expansion_artifacts(args.multilingual_output_root, args.latex_root)
    build_main_multilingual_validation_figure(
        args.output_root,
        args.multilingual_output_root,
        args.latex_root,
        same_lang_df=same_lang_df,
    )
    build_main_multilingual_regression_figure_and_table(args.multilingual_output_root, args.latex_root)
    build_same_language_controls_table(same_lang_df, args.latex_root)
    build_framework_holdout_table(framework_df, args.latex_root)
    build_progress_sensitivity_table(dense_progress_df if dense_progress_df is not None else progress_df, args.latex_root)
    build_dense_progress_trajectory_figure(dense_progress_df, args.latex_root)
    build_dense_progress_summary_figure(dense_progress_df, args.latex_root)
    build_anchor_sensitivity_table(anchor_df, args.latex_root)
    build_tokenizer_check_table(tokenizer_audit_df, args.latex_root)
    build_scope_tests_table(scope_tests_df, args.latex_root)
    publish_numbered_assets(args.latex_root)

    write_report_meta(args.output_root)
    print("Generated revision artifacts.")


if __name__ == "__main__":
    main()
