#!/usr/bin/env python3
"""Query probe-axis sign flips from Figure 5 supporting CSV.

Example:
  python3 latex/scripts/query_signflip_rows.py \
    --csv latex/tables/fig5_probe_axis_signed_full.csv \
    --lang-a ZH --lang-b FR --top 20 --contains "wedding dress"
"""

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("latex/tables/fig5_probe_axis_signed_full.csv"),
        help="Path to probe-axis signed CSV generated for Figure 5.",
    )
    p.add_argument("--lang-a", required=True, help="First language column (e.g., ZH).")
    p.add_argument("--lang-b", required=True, help="Second language column (e.g., FR).")
    p.add_argument("--top", type=int, default=20, help="Number of rows to print.")
    p.add_argument(
        "--contains",
        default="",
        help="Optional case-insensitive substring filter on probe_axis.",
    )
    p.add_argument(
        "--min-abs",
        type=float,
        default=0.0,
        help="Optional minimum abs value per language for stronger effects.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lang_a = args.lang_a.upper()
    lang_b = args.lang_b.upper()
    needle = args.contains.lower().strip()

    rows: list[tuple[str, float, float, float]] = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ["probe_axis", lang_a, lang_b] if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing required columns: {missing}")

        for r in reader:
            name = r["probe_axis"]
            if needle and needle not in name.lower():
                continue
            try:
                a = float(r[lang_a])
                b = float(r[lang_b])
            except (TypeError, ValueError):
                continue
            if abs(a) < args.min_abs and abs(b) < args.min_abs:
                continue
            if a == 0.0 or b == 0.0:
                continue
            if a * b >= 0.0:
                continue
            score = abs(a - b)
            rows.append((name, a, b, score))

    rows.sort(key=lambda x: x[3], reverse=True)
    top = rows[: max(1, args.top)]

    print(f"csv={args.csv}")
    print(f"pair={lang_a} vs {lang_b} | sign-flip rows={len(rows)} | showing={len(top)}")
    print("probe_axis\t" + lang_a + "\t" + lang_b + "\tabs_diff")
    for name, a, b, score in top:
        print(f"{name}\t{a:.6f}\t{b:.6f}\t{score:.6f}")


if __name__ == "__main__":
    main()
