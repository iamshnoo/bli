#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shared_utils import (
    align_source_to_target,
    case_signed_shift,
    load_probe_set,
    load_repr,
    load_tokenizer_and_model,
    resolve_device,
    safe_mean,
    safe_std,
    score_completion,
)

CORE_LANGS = ["zh", "fr", "fas", "nld", "ukr", "bul", "ind", "deu"]
TEMPLATES = [
    'Question: Which is more associated with "{probe}", "{left}" or "{right}"? Answer:',
    'For "{probe}", the closer association is',
    'In ordinary usage, "{probe}" is more related to',
    'When describing "{probe}", one would choose',
    'The concept "{probe}" is closer to',
]


@dataclass(frozen=True)
class ProbeCase:
    probe: str
    left: str
    right: str
    axis_index: str = ""
    category: str = ""
    case_source: str = ""

    @property
    def probe_axis(self) -> str:
        return f"{self.probe} | {self.left}->{self.right}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Connect signed representation shifts to forced-choice likelihood differences."
    )
    p.add_argument("--probe-set", type=Path, default=Path("data/probes/probe_sets.json"))
    p.add_argument(
        "--cases-csv",
        type=Path,
        default=Path("data/probes/output_likelihood_association_cases_expanded.csv"),
    )
    p.add_argument("--out-csv", type=Path, required=True, help="Case-level output CSV.")
    p.add_argument("--template-out-csv", type=Path, default=None, help="Per-template audit CSV.")
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--languages", default=",".join(CORE_LANGS), help="Comma-separated language codes.")
    p.add_argument("--baselines", default="en_50m,en_100m", help="Comma-separated EN baselines.")
    return p.parse_args()


def rep_roots() -> list[Path]:
    roots = [Path("outputs/revision/en_ablation/representations")]
    for lang in CORE_LANGS:
        roots.append(Path(f"outputs/multilingual_expansion/{lang}_shared_language/representations"))
        roots.append(Path(f"outputs/revision/{lang}_shared_language/representations"))
    return [p for p in roots if p.exists()]


def default_template_out(path: Path) -> Path:
    return path.with_name(f"{path.stem}_templates{path.suffix}")


def parse_cases(path: Path) -> list[ProbeCase]:
    df = pd.read_csv(path)
    rows: list[ProbeCase] = []
    seen: set[tuple[str, str, str]] = set()

    if {"probe", "left_endpoint", "right_endpoint"}.issubset(df.columns):
        for raw in df.to_dict("records"):
            case = ProbeCase(
                probe=str(raw["probe"]).strip(),
                left=str(raw["left_endpoint"]).strip(),
                right=str(raw["right_endpoint"]).strip(),
                axis_index=str(raw.get("axis_index", "")).strip(),
                category=str(raw.get("category", "")).strip(),
                case_source=str(raw.get("case_source", "")).strip(),
            )
            key = (case.probe, case.left, case.right)
            if all(key) and key not in seen:
                seen.add(key)
                rows.append(case)
        return rows

    if "probe_axis" in df.columns:
        for raw in df["probe_axis"].astype(str):
            probe, axis = [x.strip() for x in raw.split("|", 1)]
            left, right = [x.strip() for x in axis.split("->", 1)]
            key = (probe, left, right)
            if all(key) and key not in seen:
                seen.add(key)
                rows.append(ProbeCase(probe=probe, left=left, right=right))
        return rows

    raise ValueError(f"{path} must contain probe/left_endpoint/right_endpoint columns or probe_axis")


def build_random_controls(probe: dict, cases: list[ProbeCase], seed: int) -> list[ProbeCase]:
    rng = np.random.default_rng(seed)
    axis_words = sorted({w for axis in probe["semantic_axes"] for w in axis})
    random_cases: list[ProbeCase] = []
    for case in cases:
        for _ in range(1000):
            rand_left, rand_right = rng.choice(axis_words, size=2, replace=False).tolist()
            if (rand_left, rand_right) != (case.left, case.right):
                random_cases.append(
                    ProbeCase(
                        probe=case.probe,
                        left=str(rand_left),
                        right=str(rand_right),
                        axis_index=case.axis_index,
                        category=case.category,
                        case_source="random_axis_control",
                    )
                )
                break
        else:
            random_cases.append(case)
    return random_cases


def finite_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    vals = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 3 or vals["x"].nunique() < 2 or vals["y"].nunique() < 2:
        return float("nan"), float("nan")
    rho, p = spearmanr(vals["x"], vals["y"])
    return float(rho), float(p)


def delta_rho(valid: pd.DataFrame) -> float:
    ctx_rho, _ = finite_spearman(
        np.abs(valid["repr_signed_contextual"]),
        np.abs(valid["output_shift_left_minus_right"]),
    )
    emb_rho, _ = finite_spearman(
        np.abs(valid["repr_signed_embedding"]),
        np.abs(valid["output_shift_left_minus_right"]),
    )
    return float(ctx_rho - emb_rho) if np.isfinite(ctx_rho) and np.isfinite(emb_rho) else float("nan")


def bootstrap_delta_ci(valid: pd.DataFrame, iters: int, seed: int) -> tuple[float, float]:
    if iters <= 0 or valid.empty or "probe_axis" not in valid:
        return float("nan"), float("nan")
    groups = {k: g for k, g in valid.groupby("probe_axis", sort=False)}
    keys = np.array(list(groups.keys()), dtype=object)
    if len(keys) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(iters):
        sample_keys = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[k] for k in sample_keys], ignore_index=True)
        d = delta_rho(sample)
        if np.isfinite(d):
            vals.append(float(d))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(np.array(vals, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def sign_agreement(repr_shift: float, output_shift: float) -> float:
    if not np.isfinite(repr_shift) or not np.isfinite(output_shift) or abs(output_shift) <= 1e-9:
        return float("nan")
    return float(int(np.sign(repr_shift) == np.sign(output_shift)))


def template_sign_consistency(shifts: list[float], mean_shift: float) -> float:
    vals = [x for x in shifts if np.isfinite(x)]
    if not vals or not np.isfinite(mean_shift) or abs(mean_shift) <= 1e-9:
        return float("nan")
    nonzero = [x for x in vals if abs(x) > 1e-9]
    if not nonzero:
        return float("nan")
    return float(np.mean([int(np.sign(x) == np.sign(mean_shift)) for x in nonzero]))


def build_summary(out: pd.DataFrame, bootstrap_iters: int, seed: int) -> pd.DataFrame:
    summary_rows = []
    grouped = [((baseline, condition), sub) for (baseline, condition), sub in out.groupby(["baseline", "condition"])]
    for condition in sorted(out["condition"].unique()):
        grouped.append((("all", condition), out[out["condition"] == condition]))

    for group_i, ((baseline, condition), sub) in enumerate(grouped):
        valid = sub.dropna(
            subset=["output_shift_left_minus_right", "repr_signed_contextual", "repr_signed_embedding"]
        ).copy()
        if valid.empty:
            continue
        ctx_rho, ctx_p = finite_spearman(
            np.abs(valid["repr_signed_contextual"]),
            np.abs(valid["output_shift_left_minus_right"]),
        )
        emb_rho, emb_p = finite_spearman(
            np.abs(valid["repr_signed_embedding"]),
            np.abs(valid["output_shift_left_minus_right"]),
        )
        d_rho = float(ctx_rho - emb_rho) if np.isfinite(ctx_rho) and np.isfinite(emb_rho) else float("nan")
        ci_lo, ci_hi = bootstrap_delta_ci(valid, bootstrap_iters, seed + group_i)
        summary_rows.append(
            {
                "baseline": baseline,
                "condition": condition,
                "n_cases": int(len(valid)),
                "n_probe_axes": int(valid["probe_axis"].nunique()),
                "n_templates": int(valid["n_templates"].max()) if "n_templates" in valid else len(TEMPLATES),
                "contextual_sign_agreement": float(valid["sign_agree_contextual"].mean()),
                "embedding_sign_agreement": float(valid["sign_agree_embedding"].mean()),
                "contextual_abs_rho": ctx_rho,
                "contextual_abs_p": ctx_p,
                "embedding_abs_rho": emb_rho,
                "embedding_abs_p": emb_p,
                "delta_abs_rho": d_rho,
                "delta_abs_rho_ci_low": ci_lo,
                "delta_abs_rho_ci_high": ci_hi,
                "mean_abs_output_shift": float(np.mean(np.abs(valid["output_shift_left_minus_right"]))),
                "template_sign_consistency": float(valid["template_sign_consistency"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    probe = load_probe_set(args.probe_set)
    w2i = probe["_w2i"]
    roots = rep_roots()
    theory_cases = parse_cases(args.cases_csv)
    random_cases = build_random_controls(probe, theory_cases, args.seed)
    template_out_csv = args.template_out_csv or default_template_out(args.out_csv)

    rep_cache: dict[tuple[str, str], np.ndarray] = {}

    def get_repr(model_name: str, repr_type: str) -> np.ndarray:
        key = (model_name, repr_type)
        if key not in rep_cache:
            rep_cache[key] = load_repr(model_name, repr_type, roots)
        return rep_cache[key]

    case_rows = []
    template_rows = []
    for base in baselines:
        en_tok, en_model = load_tokenizer_and_model(Path("models/hf") / base, device)
        en_pref_cache: dict[tuple[str, str, str, int], float] = {}
        try:
            for lang in languages:
                model_b = f"en_{lang}_a"
                bi_tok, bi_model = load_tokenizer_and_model(Path("models/hf") / model_b, device)
                try:
                    emb_en = get_repr(base, "embedding_matrix")
                    emb_bi = get_repr(model_b, "embedding_matrix")
                    ctx_en = get_repr(base, "pre_lmhead_contextual")
                    ctx_bi = get_repr(model_b, "pre_lmhead_contextual")
                    emb_aligned, _emb_w, _ = align_source_to_target(emb_en, emb_bi, probe["_neutral_idx"])
                    ctx_aligned, _ctx_w, _ = align_source_to_target(ctx_en, ctx_bi, probe["_neutral_idx"])

                    for condition, cases in [("theory", theory_cases), ("random", random_cases)]:
                        for case in cases:
                            if case.probe not in w2i or case.left not in w2i or case.right not in w2i:
                                continue
                            pidx = w2i[case.probe]
                            lidx = w2i[case.left]
                            ridx = w2i[case.right]
                            repr_signed_emb = case_signed_shift(emb_aligned, emb_bi, pidx, lidx, ridx)
                            repr_signed_ctx = case_signed_shift(ctx_aligned, ctx_bi, pidx, lidx, ridx)

                            shifts = []
                            for template_id, tmpl in enumerate(TEMPLATES):
                                prompt = tmpl.format(probe=case.probe, left=case.left, right=case.right)
                                en_key = (condition, case.probe_axis, prompt, template_id)
                                if en_key not in en_pref_cache:
                                    en_pref_cache[en_key] = score_completion(
                                        en_model, en_tok, prompt, f" {case.left}", device
                                    ) - score_completion(en_model, en_tok, prompt, f" {case.right}", device)
                                pref_en = en_pref_cache[en_key]
                                pref_bi = score_completion(bi_model, bi_tok, prompt, f" {case.left}", device) - score_completion(
                                    bi_model, bi_tok, prompt, f" {case.right}", device
                                )
                                delta_out = float(pref_bi - pref_en)
                                shifts.append(delta_out)
                                template_rows.append(
                                    {
                                        "baseline": base,
                                        "language": lang.upper(),
                                        "model_b": model_b,
                                        "condition": condition,
                                        "probe": case.probe,
                                        "left_endpoint": case.left,
                                        "right_endpoint": case.right,
                                        "probe_axis": case.probe_axis,
                                        "axis_index": case.axis_index,
                                        "category": case.category,
                                        "case_source": case.case_source,
                                        "template_id": template_id,
                                        "template": tmpl,
                                        "repr_signed_embedding": repr_signed_emb,
                                        "repr_signed_contextual": repr_signed_ctx,
                                        "output_shift_left_minus_right": delta_out,
                                    }
                                )

                            mean_shift = safe_mean(shifts)
                            case_rows.append(
                                {
                                    "baseline": base,
                                    "language": lang.upper(),
                                    "model_b": model_b,
                                    "condition": condition,
                                    "probe": case.probe,
                                    "left_endpoint": case.left,
                                    "right_endpoint": case.right,
                                    "probe_axis": case.probe_axis,
                                    "axis_index": case.axis_index,
                                    "category": case.category,
                                    "case_source": case.case_source,
                                    "repr_signed_embedding": repr_signed_emb,
                                    "repr_signed_contextual": repr_signed_ctx,
                                    "output_shift_left_minus_right": mean_shift,
                                    "template_shift_mean": mean_shift,
                                    "template_shift_std": safe_std(shifts),
                                    "n_templates": int(len([x for x in shifts if np.isfinite(x)])),
                                    "template_sign_consistency": template_sign_consistency(shifts, mean_shift),
                                    "sign_agree_contextual": sign_agreement(repr_signed_ctx, mean_shift),
                                    "sign_agree_embedding": sign_agreement(repr_signed_emb, mean_shift),
                                }
                            )

                    if case_rows:
                        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(case_rows).sort_values(
                            ["baseline", "condition", "language", "probe_axis"]
                        ).to_csv(args.out_csv, index=False)
                        pd.DataFrame(template_rows).sort_values(
                            ["baseline", "condition", "language", "probe_axis", "template_id"]
                        ).to_csv(template_out_csv, index=False)
                finally:
                    del bi_model
                    if device.type == "cuda":
                        import torch

                        torch.cuda.empty_cache()
        finally:
            del en_model
            if device.type == "cuda":
                import torch

                torch.cuda.empty_cache()

    out = pd.DataFrame(case_rows).sort_values(["baseline", "condition", "language", "probe_axis"]).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    pd.DataFrame(template_rows).sort_values(["baseline", "condition", "language", "probe_axis", "template_id"]).to_csv(
        template_out_csv, index=False
    )

    summary = build_summary(out, args.bootstrap_iters, args.seed)
    summary.to_csv(args.summary_csv, index=False)
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {template_out_csv}")
    print(f"Wrote: {args.summary_csv}")


if __name__ == "__main__":
    main()
