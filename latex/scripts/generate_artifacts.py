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
    return p.parse_args()


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12})
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
    "DEU": "--",
}


def blend_hex(hex_color: str, target_rgb: tuple[float, float, float], frac: float) -> tuple[float, float, float]:
    c = np.array(colors.to_rgb(hex_color), dtype=float)
    t = np.array(target_rgb, dtype=float)
    return tuple(((1.0 - frac) * c + frac * t).tolist())


def pastel_lang_color(lang_code: str, lighten: float = 0.35) -> tuple[float, float, float]:
    base = LANG_COLORS.get(str(lang_code).upper(), "#4C78A8")
    return blend_hex(base, (1.0, 1.0, 1.0), lighten)

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
    "appendix_perhead_heatmap.pdf": "fig9-appendix-perhead-heatmap.pdf",
    "category_heatmap.pdf": "fig10-appendix-category-heatmap.pdf",
    "appendix_multilingual_overview.pdf": "fig11-appendix-multilingual-overview.pdf",
    "appendix_multilingual_regression_scatter.pdf": "fig12-typology-regression-scatter.pdf",
    "appendix_l2_signed_hotspots_panel1.pdf": "fig13-appendix-l2-signed-hotspots-panel1.pdf",
    "appendix_l2_signed_hotspots_panel2.pdf": "fig14-appendix-l2-signed-hotspots-panel2.pdf",
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
    "exp4_ratio_ci.tex": "tab14-exp3-ratio-ci.tex",
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


def publish_numbered_assets(latex_root: Path) -> None:
    figures_dir = latex_root / "figures"
    tables_dir = latex_root / "tables"

    for src_name, dst_name in FIGURE_NAME_MAP.items():
        src = figures_dir / src_name
        dst = figures_dir / dst_name
        if src.exists():
            _convert_image_to_pdf(src, dst)

    for src_name, dst_name in TABLE_NAME_MAP.items():
        src = tables_dir / src_name
        dst = tables_dir / dst_name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Remove legacy-named intermediate outputs so the repo keeps a single naming scheme.
    for src_name in FIGURE_NAME_MAP:
        if src_name in LEGACY_FIGURE_SOURCES_TO_KEEP:
            continue
        src = figures_dir / src_name
        if src.exists():
            src.unlink()
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
        if ref is not None:
            pdf = pd.concat(
                [
                    pd.DataFrame(
                        [
                            {
                                "lang": "EN-NULL",
                                "daxis": 0.0,
                                "low": ref["low"] - ref["mean"],
                                "high": ref["high"] - ref["mean"],
                                "is_ref": True,
                            }
                        ]
                    ),
                    pdf.assign(is_ref=False),
                ],
                ignore_index=True,
            )
        else:
            pdf = pdf.assign(is_ref=False)
        lang_order = [x.upper() for x in CORE_LANGS]
        pdf["lang_order"] = pdf["lang"].map(lambda x: lang_order.index(x) if x in lang_order else 999)
        pdf.loc[pdf["lang"] == "EN-NULL", "lang_order"] = -1
        pdf = pdf.sort_values("lang_order").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(7.25, 3.95))
        x = np.arange(len(pdf))
        bars = ax.bar(
            x,
            pdf["daxis"].to_numpy(),
            edgecolor="#222222",
            linewidth=0.95,
            alpha=0.98,
            zorder=2,
        )
        for b, (_, r) in zip(bars, pdf.iterrows()):
            lg = str(r["lang"]).upper()
            if bool(r["is_ref"]):
                b.set_facecolor("#c7c7c7")
                b.set_hatch("////")
            else:
                b.set_facecolor(pastel_lang_color(lg))
                b.set_hatch(LANG_HATCHES.get(lg, "//"))
            b.set_path_effects(
                [
                    pe.SimplePatchShadow(offset=(1.0, -1.0), shadow_rgbFace=(0, 0, 0), alpha=0.10),
                    pe.Normal(),
                ]
            )

        for i, rr in pdf.iterrows():
            if not np.isfinite(float(rr["low"])) or not np.isfinite(float(rr["high"])):
                continue
            lo = float(rr["low"])
            hi = float(rr["high"])
            center = float(rr["daxis"])
            lower = max(0.0, center - lo)
            upper = max(0.0, hi - center)
            ecolor = "#4d4d4d" if bool(rr["is_ref"]) else "#2f3b4a"
            ax.errorbar(
                [i],
                [center],
                yerr=[[lower], [upper]],
                fmt="none",
                ecolor=ecolor,
                elinewidth=1.0,
                capsize=3.5,
                capthick=1.0,
                zorder=4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(pdf["lang"].tolist(), fontsize=8.8, fontweight="bold")
        ax.set_xlabel("Language")
        ax.set_ylabel(r"Centered contextual $\Delta D_{Axis}$")
        ax.grid(axis="y", linestyle="--", linewidth=0.52, alpha=0.32)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.35)
        fig.subplots_adjust(left=0.14)
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_ctx_slice("en_50m", "EN-50M", "exp1_controls.pdf")
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
                        "jaccard_at_k_mean": r"$D_{NN}$",
                        "frobenius_cultural_similarity": r"$D_{Struct}$",
                        "axis_abs_projection_diff_mean": r"$D_{Axis}$",
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
                        "jaccard_at_k_mean": r"$D_{NN}$",
                        "frobenius_cultural_similarity": r"$D_{Struct}$",
                        "axis_abs_projection_diff_mean": r"$D_{Axis}$",
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
        r"Representation & Group & $D_{NN}$ & $D_{Struct}$ & $D_{Axis}$ \\",
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

        fig, ax = plt.subplots(figsize=(6.95, 4.25))
        y = np.arange(len(odf))
        for i, rr in odf.iterrows():
            ax.plot(
                [rr["daxis_c3"], rr["daxis_c4"]],
                [i, i],
                color="#a8b0bc",
                linewidth=1.45,
                alpha=0.95,
                zorder=1,
            )
            lang = str(rr["lang"]).upper()
            c = pastel_lang_color(lang)
            ec = blend_hex(LANG_COLORS.get(lang, "#4C78A8"), (0, 0, 0), 0.30)
            ax.scatter([rr["daxis_c3"]], [i], s=44, marker="o", color=c, edgecolors=ec, linewidths=0.8, zorder=3)
            ax.scatter([rr["daxis_c4"]], [i], s=46, marker="s", facecolors="white", edgecolors=ec, linewidths=1.0, zorder=4)
        if ref is not None:
            ax.axvspan(ref_low, ref_high, color="#bdbdbd", alpha=0.24, zorder=0)
            ax.axvline(0.0, color="#666666", linestyle=(0, (4, 2)), linewidth=1.1, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(odf["lang"].tolist(), fontsize=9, fontweight="bold")
        ax.set_ylabel("Language")
        ax.invert_yaxis()
        ax.set_xlabel(r"Centered contextual $\Delta D_{Axis}$ (value $-$ EN-null mean)")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#c7d8ea", markeredgecolor="#394a5a", label="C3 (Shared)", markersize=6),
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="#394a5a", label="C4 (Disjoint)", markersize=6),
        ]
        if ref is not None:
            handles.append(
                plt.Line2D([0], [0], color="#6b6b6b", linestyle="--", lw=1.2, label="EN-null mean/range")
            )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3 if ref is not None else 2,
            frameon=True,
            edgecolor="gray",
            fontsize=8.8,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_overlap_slice("en_50m", "EN-50M", "exp2_overlap.pdf")
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
    piv["ratio"] = piv["pre_lmhead_contextual"] / np.maximum(piv["embedding_matrix"], 1e-12)
    piv = piv.sort_values("pair")

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Pair & Embedding $D_{Axis}$ & Contextual $D_{Axis}$ & Ratio (Contextual/Embedding) \\",
        r"\midrule",
    ]
    for _, r in piv.iterrows():
        lines.append(
            f"{r['pair']} & {_fmt(r['embedding_matrix'])} & {_fmt(r['pre_lmhead_contextual'])} & {r['ratio']:.1f} \\\\"
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
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="lower right", frameon=True, edgecolor="gray", fontsize=10)
    fig.savefig(latex_root / "figures" / "exp3_shared.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def build_exp4_ratio(en: pd.DataFrame, ci: pd.DataFrame | None, latex_root: Path) -> None:
    # Use setup A for stable headline ratios.
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
            "ratio": c_axis / max(1e-12, e_axis),
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
        plot_df["emb_offset"] = -0.5 * plot_df["delta_ctx_minus_emb"]
        plot_df["ctx_offset"] = +0.5 * plot_df["delta_ctx_minus_emb"]
        y = np.arange(len(plot_df))

        fig, ax = plt.subplots(figsize=(6.85, 4.85))
        ax.axvline(0.0, color="#666666", linestyle=(0, (4, 2)), linewidth=1.15, zorder=0)
        for i, rr in plot_df.iterrows():
            lang = str(rr["lang"]).upper()
            c = pastel_lang_color(lang)
            ec = blend_hex(LANG_COLORS.get(lang, "#4C78A8"), (0, 0, 0), 0.35)
            ax.plot(
                [rr["emb_offset"], rr["ctx_offset"]],
                [i, i],
                color="#a7b2be",
                linewidth=1.4,
                alpha=0.95,
                zorder=1,
            )
            ax.scatter(
                [rr["emb_offset"]],
                [i],
                s=44,
                marker="o",
                facecolors="white",
                edgecolors=ec,
                linewidths=1.0,
                zorder=3,
            )
            ax.scatter(
                [rr["ctx_offset"]],
                [i],
                s=50,
                marker="s",
                facecolors=c,
                edgecolors=ec,
                linewidths=0.9,
                zorder=4,
            )
            ax.text(
                rr["ctx_offset"] + 0.02 * max(0.3, float(plot_df["ctx_offset"].abs().max())),
                i,
                f"{rr['delta_ctx_minus_emb']:.2f}",
                va="center",
                ha="left",
                fontsize=8.0,
                color="#3a3a3a",
            )
        lim = float(np.max(np.abs(np.r_[plot_df["emb_offset"].to_numpy(), plot_df["ctx_offset"].to_numpy()])))
        lim = max(lim * 1.24, 0.35)
        ax.set_xlim(-lim, lim * 1.20)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["lang"].str.upper().tolist(), fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel(
            f"Within-language midpoint at 0 ({base_lbl}, C3)\n"
            "Right-side labels report contextual minus embedding",
            labelpad=4,
        )
        ax.set_ylabel("Language")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.33)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#4c5a67", label="Embedding offset", markersize=6),
                plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#c9dff2", markeredgecolor="#4c5a67", label="Contextual offset", markersize=6),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=True,
            edgecolor="gray",
            fontsize=8.3,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_repr_gap_slice("EN-50M", "exp4_ratio.pdf")
    _plot_repr_gap_slice("EN-100M", "exp4_ratio_100m.pdf")

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Pair & $D_A$ (Embedding) & $D_A$ (Contextual) & Ratio \\",
        r"\midrule",
    ]
    for _, r in rdf.iterrows():
        lines.append(f"{r['pair']} & {_fmt(r['embedding_axis'])} & {_fmt(r['contextual_axis'])} & {r['ratio']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp4_ratio.tex")

    if ci is not None and not ci.empty:
        csub = ci[
            (ci["metric"] == "axis_ratio_contextual_over_embedding")
            & (ci["model_a"].isin(["en_50m", "en_100m"]))
        ]
        csub = csub[
            csub["model_b"].astype(str).map(
                lambda mb: parse_en_target(mb) is not None and parse_en_target(mb)[1] == "a"
            )
        ]
        if not csub.empty:
            ls = [
                r"\begin{tabular}{@{}lrrr@{}}",
                r"\toprule",
                r"Pair & Mean & CI Low & CI High \\",
                r"\midrule",
            ]
            for _, rr in csub.iterrows():
                pretty_pair = en_pair_label(rr["model_a"], rr["model_b"])
                ls.append(
                    f"{pretty_pair} & {rr['mean']:.1f} & {rr['ci_low']:.1f} & {rr['ci_high']:.1f} \\\\")
            ls += [r"\bottomrule", r"\end{tabular}"]
            write_table(ls, latex_root / "tables" / "exp4_ratio_ci.tex")


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
        piv["orth_offset"] = -0.5 * piv["delta_affine_minus_orth"]
        piv["aff_offset"] = +0.5 * piv["delta_affine_minus_orth"]
        y = np.arange(len(piv))

        fig, ax = plt.subplots(figsize=(6.85, 4.45))
        ax.axvline(0.0, color="#666666", linestyle=(0, (4, 2)), linewidth=1.15, zorder=0)
        for i, rr in piv.iterrows():
            lang = str(rr["lang"]).upper()
            c = pastel_lang_color(lang)
            ec = blend_hex(LANG_COLORS.get(lang, "#4C78A8"), (0, 0, 0), 0.35)
            ax.plot(
                [rr["orth_offset"], rr["aff_offset"]],
                [i, i],
                color="#a7b2be",
                linewidth=1.4,
                alpha=0.95,
                zorder=1,
            )
            ax.scatter([rr["orth_offset"]], [i], s=42, marker="o", facecolors="white", edgecolors=ec, linewidths=1.0, zorder=3)
            ax.scatter([rr["aff_offset"]], [i], s=46, marker="s", facecolors=c, edgecolors=ec, linewidths=0.9, zorder=4)
            ax.text(
                rr["aff_offset"] + 0.02 * max(0.2, float(piv["aff_offset"].abs().max())),
                i,
                f"{rr['delta_affine_minus_orth']:.2f}",
                va="center",
                ha="left",
                fontsize=8.0,
                color="#3a3a3a",
            )
        lim = float(np.max(np.abs(np.r_[piv["orth_offset"].to_numpy(), piv["aff_offset"].to_numpy()])))
        lim = max(lim * 1.24, 0.22)
        ax.set_xlim(-lim, lim * 1.20)
        ax.set_yticks(y)
        ax.set_yticklabels(piv["lang"].astype(str).str.upper().tolist(), fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel(f"Within-language method midpoint at 0 ({model_a.replace('en_', 'EN-').upper()}, contextual, C3)")
        ax.set_ylabel("Language")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#4c5a67", label="Orthogonal offset", markersize=6),
                plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#c9dff2", markeredgecolor="#4c5a67", label="Affine offset", markersize=6),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=True,
            edgecolor="gray",
            fontsize=8.3,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)

    _plot_alignment_slice("en_50m", "exp5_alignment_methods.pdf")
    _plot_alignment_slice("en_100m", "exp5_alignment_methods_100m.pdf")

    # Appendix table with compact numeric summary.
    metrics = [
        ("jaccard_at_k_mean", r"$D_{NN}$"),
        ("frobenius_cultural_similarity", r"$D_{Struct}$"),
        ("axis_abs_projection_diff_mean", r"$D_{Axis}$"),
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
                                top_n = 20
                                # Axis-balanced selection so one axis does not monopolize rows.
                                per_axis_cap = 2
                                keep: list[str] = []
                                axis_counts: dict[str, int] = {}
                                cand_index = cand.index.tolist()

                                # Pin human-interpretable exemplar if available.
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
                                h = mat.loc[keep].copy()

                                # Shorten row labels for visual scan in one figure.
                                def _short_label(s: str) -> str:
                                    probe, axis = s.split(" | ", 1)
                                    probe_s = probe if len(probe) <= 20 else probe[:19] + "…"
                                    axis_s = axis if len(axis) <= 22 else axis[:21] + "…"
                                    return f"{probe_s} | {axis_s}"

                                # Save original labels before index reassignment (used for
                                # domain lookup and for the supporting CSV below).
                                orig_labels = keep[:]
                                h.index = [_short_label(x) for x in h.index]
                                q = np.nanpercentile(np.abs(h.to_numpy()), 95)
                                vmax = float(max(0.25, q))

                                langs_ordered = list(h.columns)  # ZH, FR, FAS, …
                                n_rows = len(h)
                                eps_icon = 0.05

                                # ── Range-Ribbon + Language-Beads figure (cleaner readability) ─
                                row_h = 0.34
                                fig_h = max(5.2, n_rows * row_h + 1.8)
                                fig, ax = plt.subplots(figsize=(7.1, fig_h))
                                # Reserve extra headroom so the legend stays above the first data row.
                                fig.subplots_adjust(left=0.44, right=0.96, bottom=0.12, top=0.89)

                                xpad = vmax * 0.14
                                ax.axvspan(-eps_icon, eps_icon, color="#f2f2f2", alpha=0.8, zorder=0)
                                ax.axvline(0.0, color="#444444", lw=1.0, ls="--", zorder=2, alpha=0.75)

                                for i, (row_label, orig_lbl) in enumerate(zip(h.index, orig_labels)):
                                    row_vals = h.iloc[i]
                                    vals = row_vals.dropna().astype(float)
                                    if vals.empty:
                                        continue
                                    vmin_i = float(vals.min())
                                    vmax_i = float(vals.max())

                                    # Ribbon shows cross-language spread for this row.
                                    ax.plot(
                                        [vmin_i, vmax_i], [i, i],
                                        color="#aeb9c9", lw=6.2, alpha=0.55,
                                        solid_capstyle="round", zorder=1,
                                    )

                                    # Language beads
                                    for lg, val in zip(langs_ordered, row_vals):
                                        if pd.isna(val):
                                            continue
                                        c = LANG_COLORS.get(lg, "#888888")
                                        ax.scatter(
                                            [float(val)], [i],
                                            color=c, s=30, zorder=3,
                                            edgecolors="white", linewidths=0.45,
                                        )

                                    # subtle row separator
                                    ax.axhline(i, color="#eceff3", lw=0.55, zorder=0)

                                ax.set_xlim(-vmax - xpad, vmax + xpad * 0.45)
                                ax.set_ylim(-0.6, n_rows - 0.35)
                                ax.invert_yaxis()
                                ax.set_yticks(range(n_rows))
                                ax.set_yticklabels(h.index, fontsize=6.8, fontfamily="monospace")
                                ax.set_xlabel(
                                    "Signed shift $\\Delta$ (EN-50M minus EN+L2)\nLeft: EN+L2 higher    Right: EN-50M higher",
                                    fontsize=8.0,
                                    labelpad=4,
                                )
                                ax.set_xticks([-vmax, -vmax / 2, 0.0, vmax / 2, vmax])
                                ax.set_xticklabels(
                                    [f"{-vmax:.2f}", f"{-vmax/2:.2f}", "0", f"{vmax/2:.2f}", f"{vmax:.2f}"],
                                    fontsize=7.6,
                                )
                                ax.spines["top"].set_visible(False)
                                ax.spines["right"].set_visible(False)
                                ax.spines["left"].set_visible(False)
                                ax.tick_params(axis="y", length=0, pad=3)
                                ax.tick_params(axis="x", labelsize=7.6)

                                # Legend for language beads
                                leg_handles = [
                                    plt.scatter(
                                        [], [], color=LANG_COLORS.get(lg, "#888"),
                                        s=30, label=lg, edgecolors="white", linewidths=0.4
                                    )
                                    for lg in langs_ordered
                                ]
                                ax.legend(
                                    handles=leg_handles,
                                    loc="lower center",
                                    bbox_to_anchor=(0.5, 1.01),
                                    ncol=max(1, len(langs_ordered)),
                                    fontsize=7.1,
                                    frameon=True,
                                    edgecolor="#9a9a9a",
                                    framealpha=0.95,
                                    handletextpad=0.3,
                                    columnspacing=0.65,
                                )

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
                                support.to_csv(
                                    latex_root / "tables" / "fig5_signflip_hotspots.csv",
                                    index=False,
                                )
                                # Full probe-axis matrix (not only top rows) for direct querying
                                # of arbitrary probes/axes from the same computation.
                                full_support = mat.copy().reset_index().rename(columns={"index": "probe_axis"})
                                full_support.to_csv(
                                    latex_root / "tables" / "fig5_probe_axis_signed_full.csv",
                                    index=False,
                                )

                                # 100M companion for Figure 5: recompute signed-shift values
                                # using EN-100M, then plot the same selected probe-axis rows.
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
                                            vmax100 = float(max(0.25, q100))
                                            langs100 = list(h100.columns)
                                            n_rows100 = len(h100)
                                            fig_h100 = max(5.2, n_rows100 * 0.34 + 1.8)
                                            fig100, ax100 = plt.subplots(figsize=(7.1, fig_h100))
                                            # Match main panel: keep legend above rows without overlap.
                                            fig100.subplots_adjust(left=0.44, right=0.96, bottom=0.12, top=0.89)
                                            xpad100 = vmax100 * 0.14
                                            ax100.axvspan(-eps_icon, eps_icon, color="#f2f2f2", alpha=0.8, zorder=0)
                                            ax100.axvline(0.0, color="#444444", lw=1.0, ls="--", zorder=2, alpha=0.75)
                                            for i_row in range(n_rows100):
                                                row_vals = h100.iloc[i_row]
                                                vals = row_vals.dropna().astype(float)
                                                if vals.empty:
                                                    continue
                                                vmin_i = float(vals.min())
                                                vmax_i = float(vals.max())
                                                ax100.plot(
                                                    [vmin_i, vmax_i],
                                                    [i_row, i_row],
                                                    color="#aeb9c9",
                                                    lw=6.2,
                                                    alpha=0.55,
                                                    solid_capstyle="round",
                                                    zorder=1,
                                                )
                                                for lg, val in zip(langs100, row_vals):
                                                    if pd.isna(val):
                                                        continue
                                                    c = LANG_COLORS.get(lg, "#888888")
                                                    ax100.scatter(
                                                        [float(val)],
                                                        [i_row],
                                                        color=c,
                                                        s=30,
                                                        zorder=3,
                                                        edgecolors="white",
                                                        linewidths=0.45,
                                                    )
                                                ax100.axhline(i_row, color="#eceff3", lw=0.55, zorder=0)
                                            ax100.set_xlim(-vmax100 - xpad100, vmax100 + xpad100 * 0.45)
                                            ax100.set_ylim(-0.6, n_rows100 - 0.35)
                                            ax100.invert_yaxis()
                                            ax100.set_yticks(range(n_rows100))
                                            ax100.set_yticklabels(h100.index, fontsize=6.8, fontfamily="monospace")
                                            ax100.set_xlabel(
                                                "Signed shift $\\Delta$ (EN-100M minus EN+L2)\nLeft: EN+L2 higher    Right: EN-100M higher",
                                                fontsize=8.0,
                                                labelpad=4,
                                            )
                                            ax100.set_xticks([-vmax100, -vmax100 / 2, 0.0, vmax100 / 2, vmax100])
                                            ax100.set_xticklabels(
                                                [f"{-vmax100:.2f}", f"{-vmax100/2:.2f}", "0", f"{vmax100/2:.2f}", f"{vmax100:.2f}"],
                                                fontsize=7.6,
                                            )
                                            ax100.spines["top"].set_visible(False)
                                            ax100.spines["right"].set_visible(False)
                                            ax100.spines["left"].set_visible(False)
                                            ax100.tick_params(axis="y", length=0, pad=3)
                                            ax100.tick_params(axis="x", labelsize=7.6)
                                            leg_handles100 = [
                                                plt.scatter([], [], color=LANG_COLORS.get(lg, "#888"), s=30, label=lg, edgecolors="white", linewidths=0.4)
                                                for lg in langs100
                                            ]
                                            ax100.legend(
                                                handles=leg_handles100,
                                                loc="lower center",
                                                bbox_to_anchor=(0.5, 1.01),
                                                ncol=max(1, len(langs100)),
                                                fontsize=7.1,
                                                frameon=True,
                                                edgecolor="#9a9a9a",
                                                framealpha=0.95,
                                                handletextpad=0.3,
                                                columnspacing=0.65,
                                            )
                                            fig100.savefig(
                                                latex_root / "figures" / "exp4_signed_axes_100m.pdf",
                                                dpi=450,
                                                bbox_inches="tight",
                                                pad_inches=0.05,
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
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
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
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.2), sharex=True)
        axes = axes.flatten()
        for ax, lg in zip(axes, lang_list):
            sub = per_lang.get(lg, pd.DataFrame(columns=["signed"]))
            if sub.empty:
                ax.text(0.5, 0.5, f"{lg}: no rows", ha="center", va="center", fontsize=9)
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

            ax.axvspan(-eps, eps, color="#f2f2f2", alpha=0.8, zorder=0)
            ax.axvline(0.0, color="#444444", lw=0.9, ls="--", zorder=1, alpha=0.75)
            ax.hlines(y, 0.0, vals, color="#bcc5d1", lw=1.8, zorder=1)
            ax.scatter(vals, y, s=26, color=LANG_COLORS.get(lg, "#4c78a8"), edgecolors="white", linewidths=0.45, zorder=2)

            ax.set_yticks(y)
            ax.set_yticklabels(labels_unique, fontsize=7.0, fontfamily="monospace")
            ax.set_title(f"{lg} (L2-50M vs EN+L2)", fontsize=10, pad=4)
            ax.set_xlim(-xlim, xlim)
            ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=2)
            ax.tick_params(axis="x", labelsize=8)

        for ax in axes[len(lang_list):]:
            ax.axis("off")
        fig.supxlabel("Signed shift (L2-50M minus EN+L2): left = EN+L2 higher, right = L2-50M higher", fontsize=9)
        fig.tight_layout(rect=(0.02, 0.03, 0.995, 0.995))
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
            r"\begin{tabular}{@{}rlllll@{}}",
            r"\toprule",
            r"\# & Endpoint 1 & Endpoint 2 & Category & Framework basis & Citation(s) \\",
            r"\midrule",
        ]
        for row in rows:
            idx = int(row.get("index", 0))
            left = str(row.get("endpoint_1", ""))
            right = str(row.get("endpoint_2", ""))
            cat = str(row.get("category", ""))
            cat_info = cat_meta.get(cat, {})
            cat_label = str(cat_info.get("display_name", cat))
            framework = str(cat_info.get("framework_basis", ""))
            cites = row.get("citations", [])
            cite_txt = f"\\cite{{{','.join(cites)}}}" if cites else "---"
            lines.append(f"{idx} & {left} & {right} & {cat_label} & {framework} & {cite_txt} \\\\")
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
    # Shorten column names for readability
    col_rename = {}
    for c in pvt.columns:
        if "__vs__" in str(c):
            left, right = str(c).split("__vs__", 1)
            col_rename[c] = en_pair_label(left, right).replace("EN-", "").replace(" vs EN+", "\nvs ")
        else:
            col_rename[c] = str(c)
    pvt = pvt.rename(columns=col_rename)

    fig, ax = plt.subplots(figsize=(11.4, 4.8))
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
    ax.tick_params(axis="x", labelsize=8.2)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.collections[0].colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(latex_root / "figures" / "category_heatmap.pdf", dpi=450, bbox_inches="tight")
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
        fig, ax = plt.subplots(figsize=(7.55, 3.9))
        vals = []
        layers_ref = None
        for lang in CORE_LANGS:
            sub = dbase[dbase["model_b"] == f"en_{lang}_a"].sort_values("layer")
            if sub.empty:
                continue
            yv = sub["axis_abs_projection_diff_mean"].to_numpy(dtype=float)
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
            ax.plot(
                layers_ref,
                y_mean,
                color="#1f1f1f",
                linewidth=2.35,
                marker="o",
                markersize=3.6,
                markerfacecolor="white",
                markeredgecolor="#1f1f1f",
                markeredgewidth=0.9,
                label=f"{model_a.replace('en_', 'EN-').upper()} mean across L2",
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel(r"Contextual $D_{Axis}$ (log)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.38)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="#7f8ea1", lw=1.4, marker="o", markersize=3, label="Per-language curves"),
                plt.Line2D([0], [0], color="#1f1f1f", lw=2.2, marker="o", markersize=4, markerfacecolor="white", label="Mean across languages"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=2,
            frameon=True,
            edgecolor="gray",
            fontsize=8.6,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        fig.subplots_adjust(left=0.14)
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
        r"Pair & Alignment source & $D_{NN}$ & $D_{Struct}$ & $D_{Axis}$ & Residual/anchor \\",
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

    fig, ax = plt.subplots(figsize=(12, 5.8))
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
    ax.tick_params(axis="x", labelsize=8)
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    if len(ytick_labels) > 32:
        show_every = 2
        ax.set_yticks(np.arange(0.5, len(ytick_labels), show_every))
        ax.set_yticklabels(ytick_labels[::show_every], fontsize=7.5)
    else:
        ax.tick_params(axis="y", labelsize=8)
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
        r"Lang & Ratio@50M & Ratio@100M & Gain $D_{NN}$ (Embedding) & Gain $D_{NN}$ (Contextual) & Gain $D_{Axis}$ (Embedding) & Gain $D_{Axis}$ (Contextual) \\",
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

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8), gridspec_kw={"wspace": 0.34})
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
    axes[0].set_ylabel(r"$D_{Axis}$(Contextual) / $D_{Axis}$(Embedding) (higher means larger contextual amplification)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(out["language"].tolist())
    axes[0].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(loc="upper right", frameon=True, edgecolor="gray")

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
    axes[1].set_ylabel(r"Overlap gain on $D_{Axis}$ (No-overlap $-$ Overlap; higher means larger overlap effect)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(out["language"].tolist())
    axes[1].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(loc="upper right", frameon=True, edgecolor="gray")
    fig.savefig(latex_root / "figures" / "appendix_multilingual_overview.pdf", dpi=450, bbox_inches="tight")
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
                fontsize=9,
            )
        xfit = reg["distance_proxy"].to_numpy(dtype=np.float64)
        yfit = reg["ratio_mean"].to_numpy(dtype=np.float64)
        if len(np.unique(xfit)) > 1:
            m, b = np.polyfit(xfit, yfit, 1)
            xs = np.linspace(xfit.min(), xfit.max(), 50)
            ax.plot(xs, m * xs + b, color="#264653", linewidth=1.3, linestyle="--")
        ax.set_xlabel("Distance proxy (higher means farther from English)")
        ax.set_ylabel(r"Mean amplification ratio $(D_{Axis}^{Ctx}/D_{Axis}^{Emb})$")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
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

    def _ctx_axis(lang: str, baseline: str) -> float:
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
            (d["repr_type"] == "pre_lmhead_contextual")
            & (d["model_a"] == base)
            & (d["model_b"] == f"en_{lang}_a")
        ]
        if a.empty:
            return float("nan")
        return float(a["axis_abs_projection_diff_mean"].iloc[0])

    def _plot_multilingual_slice(baseline: str, out_name: str) -> None:
        col = f"ctx_{baseline}"
        df[col] = df["lang_code"].astype(str).map(lambda x: _ctx_axis(str(x), baseline))
        p = df.dropna(subset=[col]).copy()
        if p.empty:
            return
        rank = {lg: i for i, lg in enumerate(CORE_LANGS)}
        p["lang_rank"] = p["lang_code"].astype(str).map(lambda x: rank.get(x, 999))
        p = p.sort_values("lang_rank").reset_index(drop=True)
        ref = seed_null_metric_stats(
            same_lang_df,
            f"EN-{baseline.upper()}",
            eval_repr="pre_lmhead_contextual",
            metric_col="axis_abs_projection_diff_mean",
        )

        x = np.arange(len(p))
        fig, ax = plt.subplots(figsize=(7.35, 4.25))
        vals = p[col].to_numpy(dtype=float)
        bars = ax.bar(
            x,
            vals,
            width=0.66,
            edgecolor="#2d2d2d",
            linewidth=0.95,
            alpha=0.98,
            zorder=2,
        )
        for rect, lang in zip(bars, p["lang_code"].astype(str).str.upper().tolist()):
            rect.set_facecolor(pastel_lang_color(lang))
            rect.set_hatch(LANG_HATCHES.get(lang, "//"))
            rect.set_path_effects(
                [
                    pe.SimplePatchShadow(offset=(1.1, -1.1), shadow_rgbFace=(0, 0, 0), alpha=0.10),
                    pe.Normal(),
                ]
            )
            x0 = rect.get_x() + 0.10 * rect.get_width()
            x1 = rect.get_x() + 0.90 * rect.get_width()
            yv = rect.get_height()
            ax.plot([x0, x1], [yv, yv], color="white", alpha=0.52, linewidth=1.0, zorder=3)

        if ref is not None:
            ax.axhspan(ref["low"], ref["high"], color="#9e9e9e", alpha=0.16, zorder=0)
            ax.axhline(ref["mean"], color="#616161", linestyle=(0, (4, 2)), linewidth=1.25, zorder=1)
        for rect, value in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                float(value) + 0.05,
                f"{float(value):.2f}",
                ha="center",
                va="bottom",
                fontsize=7.7,
                color="#222222",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(p["lang_code"].astype(str).str.upper().tolist(), fontsize=9.2, fontweight="bold")
        ax.set_ylabel(r"Contextual $D_{Axis}$", fontsize=11.5)
        ax.set_xlabel(f"Target language (L2 monolingual {baseline.upper()} vs EN+L2, C3)", fontsize=10.2)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.32)
        ax.set_axisbelow(True)
        ymax = max(vals.max() * 1.16, vals.max() + 0.35)
        ax.set_ylim(bottom=0.0, top=ymax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#d7e5f3",
                edgecolor="#2d2d2d",
                hatch="//",
                label=f"L2 bars ({baseline.upper()}, color/hatch by language)",
            )
        ]
        if ref is not None:
            handles.append(
                plt.Line2D([0], [0], color="#6b6b6b", linestyle="--", lw=1.2, label="EN-NULL mean/range")
            )
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2 if ref is not None else 1,
            frameon=True,
            edgecolor="#8a8a8a",
            framealpha=0.96,
            fontsize=8.6,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        fig.savefig(latex_root / "figures" / out_name, dpi=450, bbox_inches="tight")
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
    repr_map = {
        "embedding_matrix": "Embedding",
        "pre_lmhead_contextual": "Contextual",
    }
    x["repr"] = x["eval_repr"].map(repr_map).fillna(x["eval_repr"])

    piv = x.pivot_table(index=["language", "model_a", "model_b"], columns="repr", values="axis_abs_projection_diff_mean").reset_index()
    if "Embedding" not in piv.columns or "Contextual" not in piv.columns:
        return
    piv["ratio_ctx_over_emb"] = piv["Contextual"] / np.maximum(piv["Embedding"], 1e-12)

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Language control pair & $D_{Axis}$ (Embedding) & $D_{Axis}$ (Contextual) & Contextual/Embedding ratio \\",
        r"\midrule",
    ]
    for _, r in piv.sort_values("language").iterrows():
        pair_lbl = f"{r['model_a']} vs {r['model_b']}".replace("_", r"\_")
        lines.append(
            f"{r['language']}: {pair_lbl} & {_fmt(r['Embedding'])} & {_fmt(r['Contextual'])} & {r['ratio_ctx_over_emb']:.1f} \\\\"
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
        r"Representation & Matched-axis $D_{Axis}$ & Held-out-axis $D_{Axis}$ & Held-out $-$ matched \\",
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
        r"Language & Quantity & 50M & 100M & 100M$-$50M \\",
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

    # --- Generate regression figure ---
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
        ax.plot(x_line, p(x_line), "k--", alpha=0.4, linewidth=1.5, label=f"Spearman ρ={rho:.2f}, p{p_text}")

    ax.set_xlabel("Hofstede IDV Distance from English", fontsize=12)
    ax.set_ylabel("Contextual-to-Embedding Ratio", fontsize=12)
    ax.set_title("Typological Distance vs Representation-Dependence Gap", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.set_xlim(-5, 80)

    plt.tight_layout()
    fig_path = latex_root / "figures" / "main_multilingual_regression.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {fig_path}")

    # --- Generate main-text multilingual summary table ---
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
    args = parse_args()
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

    strat_path = args.output_root / "en_ablation" / "bli_stratified_metrics.csv"
    strat_df = pd.read_csv(strat_path) if strat_path.exists() else None

    layer_path = args.output_root / "en_ablation" / "bli_layerwise_divergence.csv"
    layer_df = pd.read_csv(layer_path) if layer_path.exists() else None
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
    build_progress_sensitivity_table(progress_df, args.latex_root)
    publish_numbered_assets(args.latex_root)

    write_report_meta(args.output_root)
    print("Generated revision artifacts.")


if __name__ == "__main__":
    main()
