#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


LANG_CODES = {
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "fas": "pes_Arab",
    "nld": "nld_Latn",
    "ukr": "ukr_Cyrl",
    "bul": "bul_Cyrl",
    "ind": "ind_Latn",
    "deu": "deu_Latn",
}

SOURCE_LANG = "eng_Latn"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate BLI probes into target languages and compute back-translation QC")
    p.add_argument(
        "--probe-set",
        type=Path,
        default=Path("data/probes/probe_sets.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/probes"),
    )
    p.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--languages", type=str, default="zh,fr,fas,nld,ukr,bul,ind,deu")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--disable-comet", action="store_true")
    p.add_argument("--comet-model", type=str, default="Unbabel/wmt22-cometkiwi-da")
    p.add_argument("--comet-batch-size", type=int, default=64)
    return p.parse_args()


def batched(seq: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def similarity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_comet_model(model_name: str, device: torch.device):
    try:
        from comet import download_model, load_from_checkpoint
    except Exception as exc:  # pragma: no cover
        print(
            "[warn] COMET import failed. Install with "
            "'pip install \"unbabel-comet>=2.0.0\"' in nanotron-env. "
            f"Reason: {exc}"
        )
        return None

    try:
        checkpoint = download_model(model_name)
        model = load_from_checkpoint(checkpoint)
        if device.type == "cuda":
            model.to("cuda")
        return model
    except Exception as exc:  # pragma: no cover
        print(f"[warn] COMET model load failed for {model_name}: {exc}")
        return None


def score_comet_kiwi(
    comet_model,
    source_texts: Sequence[str],
    translated_texts: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> list[float | None]:
    if comet_model is None:
        return [None] * len(source_texts)
    data = [{"src": src, "mt": mt} for src, mt in zip(source_texts, translated_texts)]
    gpus = 1 if device.type == "cuda" else 0
    result = comet_model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
    scores = getattr(result, "scores", None)
    if scores is None and isinstance(result, dict):
        scores = result.get("scores")
    if scores is None:
        return [None] * len(source_texts)
    return [float(x) for x in scores]


def comet_tier(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def translate(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    device: torch.device,
) -> List[str]:
    tokenizer.src_lang = src_lang
    forced_bos = resolve_lang_token_id(tokenizer, tgt_lang)

    outputs: List[str] = []
    for batch in batched(texts, batch_size):
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            gen = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=64,
                num_beams=4,
            )
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

    return outputs


def resolve_lang_token_id(tokenizer: AutoTokenizer, lang_code: str) -> int:
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(mapping, dict) and lang_code in mapping:
        return int(mapping[lang_code])

    get_lang_id = getattr(tokenizer, "get_lang_id", None)
    if callable(get_lang_id):
        try:
            return int(get_lang_id(lang_code))
        except Exception:
            pass

    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"Unable to resolve language token id for '{lang_code}'")
    return int(token_id)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.probe_set.read_text(encoding="utf-8"))
    source_terms = [str(x).strip().lower() for x in payload.get("cultural_probe_words", []) if str(x).strip()]

    if not source_terms:
        raise RuntimeError("No cultural_probe_words found in probe set")

    requested_languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    for lang in requested_languages:
        if lang not in LANG_CODES:
            raise ValueError(f"Unsupported language code '{lang}'. Supported: {sorted(LANG_CODES)}")

    device = resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    comet_model = None if args.disable_comet else load_comet_model(args.comet_model, device)
    if comet_model is None and not args.disable_comet:
        print("[warn] Proceeding without COMETKiwi QE scores.")

    summary_rows = []

    for lang in requested_languages:
        tgt = LANG_CODES[lang]
        print(f"[translate] {lang}: {SOURCE_LANG} -> {tgt}")
        translated = translate(
            model=model,
            tokenizer=tokenizer,
            texts=source_terms,
            src_lang=SOURCE_LANG,
            tgt_lang=tgt,
            batch_size=args.batch_size,
            device=device,
        )

        print(f"[back-translate] {lang}: {tgt} -> {SOURCE_LANG}")
        back_translated = translate(
            model=model,
            tokenizer=tokenizer,
            texts=translated,
            src_lang=tgt,
            tgt_lang=SOURCE_LANG,
            batch_size=args.batch_size,
            device=device,
        )
        comet_scores = score_comet_kiwi(
            comet_model=comet_model,
            source_texts=source_terms,
            translated_texts=translated,
            batch_size=args.comet_batch_size,
            device=device,
        )

        rows = []
        translated_norm = [normalize_text(x) for x in translated]
        dup_count_map: Dict[str, int] = {}
        for t in translated_norm:
            dup_count_map[t] = dup_count_map.get(t, 0) + 1

        for src, tgt_text, back_text, qe_score in zip(source_terms, translated, back_translated, comet_scores):
            sim = similarity(src, back_text)
            dup = int(dup_count_map.get(normalize_text(tgt_text), 0) > 1)
            tier = comet_tier(qe_score)
            manual = int(sim < 0.55 or dup == 1 or tier == "low")
            rows.append(
                {
                    "source_term_en": src,
                    "target_lang": lang,
                    "target_lang_nllb": tgt,
                    "translated_term": tgt_text,
                    "back_translated_en": back_text,
                    "back_similarity": sim,
                    "comet_kiwi_score": qe_score,
                    "qe_tier": tier,
                    "needs_manual_review": manual,
                    "duplicate_translation": dup,
                }
            )

        out_df = pd.DataFrame(rows)
        out_path = args.output_dir / f"translations_{lang}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[ok] Wrote {out_path}")

        if lang == "zh":
            out_df.to_csv(args.output_dir / "translations_zh.csv", index=False)
        if lang == "fr":
            out_df.to_csv(args.output_dir / "translations_fr.csv", index=False)

        summary_rows.append(
            {
                "lang": lang,
                "n_terms": int(len(out_df)),
                "mean_back_similarity": float(out_df["back_similarity"].mean()),
                "mean_comet_kiwi": float(pd.to_numeric(out_df["comet_kiwi_score"], errors="coerce").mean())
                if "comet_kiwi_score" in out_df.columns
                else None,
                "high_quality": int((out_df["back_similarity"] > 0.80).sum()),
                "medium_quality": int(((out_df["back_similarity"] >= 0.55) & (out_df["back_similarity"] <= 0.80)).sum()),
                "low_quality": int((out_df["back_similarity"] < 0.55).sum()),
                "qe_high": int((out_df["qe_tier"] == "high").sum()) if "qe_tier" in out_df.columns else 0,
                "qe_medium": int((out_df["qe_tier"] == "medium").sum()) if "qe_tier" in out_df.columns else 0,
                "qe_low": int((out_df["qe_tier"] == "low").sum()) if "qe_tier" in out_df.columns else 0,
                "manual_review": int(out_df["needs_manual_review"].sum()),
                "duplicate_translation": int(out_df["duplicate_translation"].sum()),
            }
        )

    summary_path = args.output_dir / "translation_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"[ok] Wrote {summary_path}")


if __name__ == "__main__":
    main()
