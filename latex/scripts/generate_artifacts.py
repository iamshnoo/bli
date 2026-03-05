#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
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
    "embedding_matrix": "Emb.",
    "pre_lmhead_contextual": "Ctx.",
}

PAIR_LABEL = {
    ("en_50m", "en_zh_a"): "EN-50M vs EN+ZH",
    ("en_50m", "en_fr_a"): "EN-50M vs EN+FR",
    ("en_100m", "en_zh_a"): "EN-100M vs EN+ZH",
    ("en_100m", "en_fr_a"): "EN-100M vs EN+FR",
}


def add_agreement(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["pair"] = x.apply(lambda r: f"{r['model_a']}__vs__{r['model_b']}", axis=1)
    x["nn_agree"] = 1.0 - x["jaccard_at_k_mean"]
    x["struct_agree"] = 1.0 / (1.0 + x["frobenius_cultural_similarity"])
    x["axis_agree"] = 1.0 / (1.0 + x["axis_abs_projection_diff_mean"])
    return x


def ensure_dirs(latex_root: Path) -> None:
    (latex_root / "tables").mkdir(parents=True, exist_ok=True)
    (latex_root / "figures").mkdir(parents=True, exist_ok=True)


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
    return PAIR_LABEL.get((left, right), f"{left} vs {right}")


def build_exp1_tables_fig(en: pd.DataFrame, ci: pd.DataFrame | None, latex_root: Path) -> None:
    specs = [
        ("en_50m", "en_zh", "EN-50M vs EN+ZH"),
        ("en_100m", "en_zh", "EN-100M vs EN+ZH"),
        ("en_50m", "en_fr", "EN-50M vs EN+FR"),
        ("en_100m", "en_fr", "EN-100M vs EN+FR"),
    ]
    rows = []
    for repr_type in ["embedding_matrix", "pre_lmhead_contextual"]:
        for left, fam, label in specs:
            sub = en[(en["repr_type"] == repr_type) & (en["model_a"] == left) & (en["model_b"].str.startswith(fam))]
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

    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Repr. & Comparison & $A_{NN}$ & $A_{Struct}$ & $A_{Axis}$ \\",
        r"\midrule",
    ]
    for repr_name in ["Emb.", "Ctx."]:
        sub = tdf[tdf["repr"] == repr_name].reset_index(drop=True)
        for i, r in sub.iterrows():
            left = repr_name if i == 0 else ""
            lines.append(f"{left} & {r['comparison']} & {_fmt(r['nn'])} & {_fmt(r['struct'])} & {_fmt(r['axis'])} \\\\")
        if repr_name == "Embedding matrix":
            lines.append(r"\cmidrule(lr){1-5}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp1_results.tex")

    metric_cols = [
        ("nn", r"$D_{NN}$ $(\uparrow$ worse$)$"),
        ("struct", r"$D_{Struct}$ $(\uparrow$ worse$)$"),
        ("axis", r"$D_{Axis}$ $(\uparrow$ worse$)$"),
    ]
    short_xlabels = ["50M\nvs EN+ZH", "50M\nvs EN+FR", "100M\nvs EN+ZH", "100M\nvs EN+FR"]
    xlabels = [s[2] for s in specs]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"wspace": 0.38})
    xpos = np.arange(len(xlabels))
    width = 0.36
    for ax, (mcol, ylab) in zip(axes, metric_cols):
        for j, repr_type in enumerate(["Emb.", "Ctx."]):
            short_label = repr_type
            vals = [tdf[(tdf["repr"] == repr_type) & (tdf["comparison"] == l)][mcol].iloc[0] for l in xlabels]
            bars = ax.bar(
                xpos + (j - 0.5) * width,
                vals,
                width,
                color=COLORS["embedding_matrix" if j == 0 else "pre_lmhead_contextual"],
                edgecolor="black",
                linewidth=0.7,
                label=short_label,
            )
            for b in bars:
                b.set_hatch(HATCH["embedding_matrix" if j == 0 else "pre_lmhead_contextual"])
        ax.set_ylabel(ylab)
        ax.set_xticks(xpos)
        ax.set_xticklabels(short_xlabels, rotation=0, ha="center", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper right", ncol=1, frameon=True, edgecolor="gray", fontsize=10)
    fig.savefig(latex_root / "figures" / "exp1_controls.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    if ci is not None and not ci.empty:
        # compact CI table for exp1 pairs only
        keep_pairs = [("en_50m", "en_zh_a"), ("en_50m", "en_fr_a"), ("en_100m", "en_zh_a"), ("en_100m", "en_fr_a")]
        csub = ci[ci["metric"].isin(["jaccard_at_k_mean", "frobenius_cultural_similarity", "axis_abs_projection_diff_mean"])].copy()
        lines = [
            r"\begin{tabular}{@{}llll@{}}",
            r"\toprule",
            r"Repr. & Pair & Metric & Mean [95\% CI] \\",
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
                pretty_pair = PAIR_LABEL.get((ma, mb), f"{ma} vs {mb}")
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


def build_exp1_hotspots_table(word_df: pd.DataFrame, latex_root: Path) -> None:
    sub = word_df[
        (word_df["repr_type"] == "pre_lmhead_contextual")
        & (word_df["pair"].isin(["en_50m__vs__en_zh_a", "en_50m__vs__en_fr_a", "en_100m__vs__en_zh_a", "en_100m__vs__en_fr_a"]))
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
        r"Repr. & Group & $D_{NN}$ & $D_{Struct}$ & $D_{Axis}$ \\",
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
    core = en[
        en["model_a"].isin(["en_50m", "en_100m"])
        & en["model_b"].isin(["en_zh_a", "en_fr_a"])
    ].copy()
    if core.empty:
        return
    lines = [
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Repr. & Pair & Anchor residual / anchor & Anchor residual (Frobenius) \\",
        r"\midrule",
    ]
    for rt in ["embedding_matrix", "pre_lmhead_contextual"]:
        sub = core[core["repr_type"] == rt].copy()
        sub["pair_pretty"] = sub.apply(lambda r: PAIR_LABEL.get((r["model_a"], r["model_b"]), f"{r['model_a']} vs {r['model_b']}"), axis=1)
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


def build_exp2_table_fig(en: pd.DataFrame, stats_df: pd.DataFrame | None, latex_root: Path) -> None:
    rows = []
    families = [("en_zh", "EN+ZH"), ("en_fr", "EN+FR")]
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

    lines = [
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Repr. & Family & Baseline & $\Delta D_{NN}$ & $\Delta D_{Struct}$ & $\Delta D_{Axis}$ \\",
        r"\midrule",
    ]
    for repr_name in ["Emb.", "Ctx."]:
        sub = df[df["repr"] == repr_name]
        for i, r in sub.reset_index(drop=True).iterrows():
            left = repr_name if i == 0 else ""
            lines.append(f"{left} & {r['family']} & {r['baseline']} & {_fmt(r['delta_nn'])} & {_fmt(r['delta_struct'])} & {_fmt(r['delta_axis'])} \\\\")
        if repr_name == "Emb.":
            lines.append(r"\cmidrule(lr){1-6}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp2_overlap.tex")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), gridspec_kw={"wspace": 0.38})
    order = [("EN+ZH", "EN-50M"), ("EN+ZH", "EN-100M"), ("EN+FR", "EN-50M"), ("EN+FR", "EN-100M")]
    metrics = [("delta_nn", r"$\Delta D_{NN}$"), ("delta_struct", r"$\Delta D_{Struct}$"), ("delta_axis", r"$\Delta D_{Axis}$")]
    short_titles = {"Embedding matrix": "Emb.", "Pre-LM-head contextual": "Ctx."}
    for ax, repr_name in zip(axes, ["Emb.", "Ctx."]):
        sub = df[df["repr"] == repr_name]
        x = np.arange(len(order))
        width = 0.24
        for j, (mcol, mlabel) in enumerate(metrics):
            vals = []
            for fam, base in order:
                v = sub[(sub["family"] == fam) & (sub["baseline"] == base)][mcol].iloc[0]
                vals.append(v)
            bars = ax.bar(x + (j - 1) * width, vals, width, label=mlabel, edgecolor="black", linewidth=0.7)
            if j == 0:
                for b in bars:
                    b.set_facecolor("#f4a261")
                    b.set_hatch("//")
            elif j == 1:
                for b in bars:
                    b.set_facecolor("#2a9d8f")
                    b.set_hatch("..")
            else:
                for b in bars:
                    b.set_facecolor("#457b9d")
                    b.set_hatch("xx")
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(repr_name, fontsize=12)
        ax.set_ylabel(r"$\Delta$ (Setup A $-$ Setup B)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{fam}\n{base}" for fam, base in order], fontsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper right", ncol=1, frameon=True, edgecolor="gray", fontsize=10)
    fig.savefig(latex_root / "figures" / "exp2_overlap.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    if stats_df is not None and not stats_df.empty:
        base_label_map = {"en_50m": "EN-50M", "en_100m": "EN-100M"}
        fam_label_map = {"en_zh": "EN+ZH", "en_fr": "EN+FR"}
        if "quality_tier" in stats_df.columns:
            stats_df = stats_df[stats_df["quality_tier"] == "all"].copy()
        ls = [
            r"\begin{tabular}{@{}llllrr@{}}",
            r"\toprule",
            r"Repr. & Baseline & Family & $n$ & Med.\ $\Delta D_{NN}$ & $p$ \\",
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
    s["family"] = s["family"].map({"en_zh": "EN+ZH", "en_fr": "EN+FR"}).fillna(s["family"])
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
        & en["model_b"].isin(["en_zh_a", "en_fr_a"])
        & en["repr_type"].isin(["embedding_matrix", "pre_lmhead_contextual"])
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

    piv["pair"] = piv.apply(lambda r: PAIR_LABEL.get((r["model_a"], r["model_b"]), f"{r['model_a']} vs {r['model_b']}"), axis=1)
    piv["ratio"] = piv["pre_lmhead_contextual"] / np.maximum(piv["embedding_matrix"], 1e-12)
    piv = piv.sort_values("pair")

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Pair & Embedding $D_{Axis}$ & Contextual $D_{Axis}$ & Ratio (Ctx./Emb.) \\",
        r"\midrule",
    ]
    for _, r in piv.iterrows():
        lines.append(
            f"{r['pair']} & {_fmt(r['embedding_matrix'])} & {_fmt(r['pre_lmhead_contextual'])} & {r['ratio']:.1f}$\\times$ \\\\"
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
        r"Target & Repr. & Setup & $A_{NN}$ & $A_{Struct}$ & $A_{Axis}$ \\",
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
        for j, repr_name in enumerate(["Emb.", "Ctx."]):
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
    pairs = [
        ("en_50m", "en_zh_a", "EN-50M vs EN+ZH"),
        ("en_50m", "en_fr_a", "EN-50M vs EN+FR"),
        ("en_100m", "en_zh_a", "EN-100M vs EN+ZH"),
        ("en_100m", "en_fr_a", "EN-100M vs EN+FR"),
    ]
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

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(rdf))
    colors_ratio = ["#f4a261", "#2a9d8f", "#e76f51", "#264653"]
    bars = ax.bar(x, rdf["ratio"].to_numpy(), color=colors_ratio[:len(rdf)], edgecolor="black", linewidth=0.8)
    for b, val in zip(bars, rdf["ratio"].to_numpy()):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.3, f"{val:.1f}$\\times$",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(rdf["pair"].tolist(), rotation=12, ha="right", fontsize=10)
    ax.set_ylabel(r"$D_{Axis}$(Ctx.) / $D_{Axis}$(Emb.)", fontsize=11)
    ax.set_ylim(0, max(rdf["ratio"].to_numpy()) * 1.2)
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--", label="1$\\times$ (no gap)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(latex_root / "figures" / "exp4_ratio.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Pair & $D_A$ (Emb.) & $D_A$ (Ctx.) & Ratio \\",
        r"\midrule",
    ]
    for _, r in rdf.iterrows():
        lines.append(f"{r['pair']} & {_fmt(r['embedding_axis'])} & {_fmt(r['contextual_axis'])} & {r['ratio']:.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "exp4_ratio.tex")

    if ci is not None and not ci.empty:
        csub = ci[
            (ci["metric"] == "axis_ratio_contextual_over_embedding")
            & (ci["model_a"].isin(["en_50m", "en_100m"]))
            & (ci["model_b"].isin(["en_zh_a", "en_fr_a"]))
        ]
        if not csub.empty:
            ls = [
                r"\begin{tabular}{@{}lrrr@{}}",
                r"\toprule",
                r"Pair & Mean & CI Low & CI High \\",
                r"\midrule",
            ]
            for _, rr in csub.iterrows():
                pretty_pair = PAIR_LABEL.get((rr['model_a'], rr['model_b']), f"{rr['model_a']} vs {rr['model_b']}")
                ls.append(
                    f"{pretty_pair} & {rr['mean']:.1f}$\\times$ & {rr['ci_low']:.1f}$\\times$ & {rr['ci_high']:.1f}$\\times$ \\\\")
            ls += [r"\bottomrule", r"\end{tabular}"]
            write_table(ls, latex_root / "tables" / "exp4_ratio_ci.tex")


def build_exp4_signed_axis_scatter(axis_df: pd.DataFrame | None, latex_root: Path) -> None:
    if axis_df is None or axis_df.empty or "mean_signed_projection_diff" not in axis_df.columns:
        return
    a = axis_df[(axis_df["repr_type"] == "pre_lmhead_contextual") & (axis_df["pair"] == "en_100m__vs__en_zh_a")][
        ["axis", "mean_signed_projection_diff"]
    ].rename(columns={"mean_signed_projection_diff": "signed_zh"})
    b = axis_df[(axis_df["repr_type"] == "pre_lmhead_contextual") & (axis_df["pair"] == "en_100m__vs__en_fr_a")][
        ["axis", "mean_signed_projection_diff"]
    ].rename(columns={"mean_signed_projection_diff": "signed_fr"})
    m = a.merge(b, on="axis", how="inner")
    if m.empty:
        return

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.scatter(
        m["signed_zh"].to_numpy(),
        m["signed_fr"].to_numpy(),
        s=45,
        c="#9fc4e6",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
    )
    ax.axhline(0.0, color="gray", linewidth=0.9)
    ax.axvline(0.0, color="gray", linewidth=0.9)
    ax.set_xlabel("EN+ZH direction (shift toward endpoint 2)", fontsize=10)
    ax.set_ylabel("EN+FR direction (shift toward endpoint 2)", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    # annotate a few largest-magnitude axes for readability
    m["mag"] = np.abs(m["signed_zh"]) + np.abs(m["signed_fr"])
    ann = m.sort_values("mag", ascending=False).head(6)
    for _, r in ann.iterrows():
        ax.annotate(r["axis"], (r["signed_zh"], r["signed_fr"]), fontsize=8, xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(latex_root / "figures" / "exp4_signed_axes.png", dpi=450, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # quadrant summary table
    q1 = ((m["signed_zh"] > 0) & (m["signed_fr"] > 0)).sum()
    q2 = ((m["signed_zh"] < 0) & (m["signed_fr"] > 0)).sum()
    q3 = ((m["signed_zh"] < 0) & (m["signed_fr"] < 0)).sum()
    q4 = ((m["signed_zh"] > 0) & (m["signed_fr"] < 0)).sum()
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
    lines = [
        r"\begin{tabular}{@{}rlllll@{}}",
        r"\toprule",
        r"\# & Endpoint 1 & Endpoint 2 & Category & Framework basis & Citation(s) \\",
        r"\midrule",
    ]
    for row in axis_meta:
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
    write_table(lines, latex_root / "tables" / "appendix_axis_grounding.tex")


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
        col_rename[c] = c.replace("en_50m vs en_zh", "50M\nvs EN+ZH").replace("en_50m vs en_fr", "50M\nvs EN+FR").replace("en_100m vs en_zh", "100M\nvs EN+ZH").replace("en_100m vs en_fr", "100M\nvs EN+FR")
    pvt = pvt.rename(columns=col_rename)

    fig, ax = plt.subplots(figsize=(11, 4.8))
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
        cbar_kws={"label": r"Mean $D_{Axis}$ (↑ worse)", "shrink": 0.8},
        ax=ax,
    )
    _annotate_heatmap_adaptive(ax, pvt.values, "YlOrBr", norm, fontsize=8)
    ax.collections[0].colorbar.ax.locator = MaxNLocator(4)
    ax.collections[0].colorbar.update_ticks()
    ax.set_xlabel("Model pair", fontsize=11)
    ax.set_ylabel("Probe category", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.collections[0].colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(latex_root / "figures" / "category_heatmap.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def build_layerwise_artifacts(layerwise_df: pd.DataFrame | None, latex_root: Path) -> None:
    if layerwise_df is None or layerwise_df.empty:
        return

    df = layerwise_df.copy()
    pair_map = {
        ("eng_only", "eng_zho"): "EN vs EN+ZH",
        ("eng_only", "eng_fra"): "EN vs EN+FR",
        ("eng_zho", "eng_fra"): "EN+ZH vs EN+FR",
    }
    df["pair"] = df.apply(lambda r: pair_map.get((r["model_a"], r["model_b"]), f"{r['model_a']} vs {r['model_b']}"), axis=1)
    df = df.sort_values(["pair", "layer"]).reset_index(drop=True)

    # Figure: log-scale line plot to handle strong dynamic range after layer 0.
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    styles = {
        "EN vs EN+ZH": dict(color="#e76f51", marker="o", linestyle="-"),
        "EN vs EN+FR": dict(color="#2a9d8f", marker="s", linestyle="-"),
        "EN+ZH vs EN+FR": dict(color="#457b9d", marker="^", linestyle="--"),
    }
    for pair in ["EN vs EN+ZH", "EN vs EN+FR", "EN+ZH vs EN+FR"]:
        sub = df[df["pair"] == pair]
        if sub.empty:
            continue
        st = styles[pair]
        ax.plot(
            sub["layer"].to_numpy(),
            sub["axis_abs_projection_diff_mean"].to_numpy(),
            label=pair,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=2.0,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(r"Mean $D_{Axis}$ (log scale)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, edgecolor="gray", fontsize=10)
    fig.tight_layout()
    fig.savefig(latex_root / "figures" / "exp4_layerwise.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    # Table: peak and last-layer summaries.
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Pair & Peak & $D_A$(peak) & $D_A$(final) & F/P \\",
        r"\midrule",
    ]
    for pair in ["EN vs EN+ZH", "EN vs EN+FR", "EN+ZH vs EN+FR"]:
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
    d["pair"] = d.apply(lambda r: PAIR_LABEL.get((r["model_a"], r["model_b"]), f"{r['model_a']} vs {r['model_b']}"), axis=1)
    d["align_lbl"] = d["alignment_source"].map(
        {
            "embedding_matrix": "Align on Emb. anchors",
            "pre_lmhead_contextual": "Align on Ctx. anchors",
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
    d["pair"] = d.apply(lambda r: PAIR_LABEL.get((r["model_a"], r["model_b"]), f"{r['model_a']} vs {r['model_b']}"), axis=1)
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
        linewidths=0.8,
        linecolor="black",
        cbar_kws={"label": r"Mean $D_{Axis}$ (↑ worse)", "shrink": 0.85},
        ax=ax,
    )
    _annotate_heatmap_adaptive(ax, pivot.values, "YlOrBr", norm, fontsize=6)
    ax.collections[0].colorbar.ax.locator = MaxNLocator(4)
    ax.collections[0].colorbar.update_ticks()
    ax.set_xlabel("Attention head index")
    ax.set_ylabel("Pair / Layer")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(latex_root / "figures" / "appendix_perhead_heatmap.png", dpi=450, bbox_inches="tight")
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
        r"Lang & Ratio@50M & Ratio@100M & Gain $D_{NN}$ (Emb.) & Gain $D_{NN}$ (Ctx.) & Gain $D_{Axis}$ (Emb.) & Gain $D_{Axis}$ (Ctx.) \\",
        r"\midrule",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"{r['language']} & {r['ratio_50m']:.1f}$\\times$ & {r['ratio_100m']:.1f}$\\times$ & "
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
                f"{r['language']} & {r['baseline']} & {r['mean']:.1f}$\\times$ & {r['ci_low']:.1f}$\\times$ & {r['ci_high']:.1f}$\\times$ \\\\"
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
    axes[0].set_ylabel(r"$D_{Axis}$(Ctx.) / $D_{Axis}$(Emb.) (higher means larger contextual amplification)")
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
        label="Emb.",
    )
    b4 = axes[1].bar(
        x + width / 2,
        out["overlap_gain_daxis_ctx"].to_numpy(),
        width=width,
        color="#9fc4e6",
        edgecolor="black",
        linewidth=0.8,
        label="Ctx.",
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
    fig.savefig(latex_root / "figures" / "appendix_multilingual_overview.png", dpi=450, bbox_inches="tight")
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
        fig.savefig(latex_root / "figures" / "appendix_multilingual_regression_scatter.png", dpi=450, bbox_inches="tight")
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
            f"{row['ratio_50m']:6.2f}$\\times$ & {row['ratio_100m']:6.2f}$\\times$ \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(lines, latex_root / "tables" / "main_multilingual_ratios.tex")
    print(f"Generated {latex_root / 'tables' / 'main_multilingual_ratios.tex'}")


def main() -> None:
    args = parse_args()
    set_style()
    ensure_dirs(args.latex_root)

    en = add_agreement(pd.read_csv(args.output_root / "en_ablation" / "bli_summary_metrics.csv"))
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

    build_exp1_tables_fig(en, ci, args.latex_root)
    if word_df is not None:
        build_exp1_hotspots_table(word_df, args.latex_root)
    build_exp1_negative_controls_table(neg_df, args.latex_root)
    build_exp1_procrustes_table(en, args.latex_root)
    build_exp2_table_fig(en, stats_df, args.latex_root)
    build_exp2_quality_table(stats_df, args.latex_root)
    build_exp3_table_fig(zh, fr, args.latex_root)
    build_exp4_ratio(en, ci, args.latex_root)
    build_appendix_daxis_interpretation(en, args.latex_root)
    build_exp4_signed_axis_scatter(axis_df, args.latex_root)
    build_layerwise_artifacts(layer_df, args.latex_root)
    build_category_heatmap(strat_df, args.latex_root)
    build_contextual_alignment_variant_artifacts(ctx_align_df, args.latex_root)
    build_perhead_artifacts(perhead_df, args.latex_root)
    build_axis_inventory_table(probe_set, args.latex_root)
    build_axis_grounding_table(probe_set, args.latex_root)
    build_probe_qc_table(translations, args.latex_root)
    build_multilingual_expansion_artifacts(args.multilingual_output_root, args.latex_root)
    build_main_multilingual_regression_figure_and_table(args.multilingual_output_root, args.latex_root)

    write_report_meta(args.output_root)
    print("Generated revision artifacts.")


if __name__ == "__main__":
    main()
