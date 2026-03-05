#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine two multilingual figures side-by-side")
    p.add_argument("--left", type=Path, default=Path("latex/figures/main_multilingual_regression.png"))
    p.add_argument("--right", type=Path, default=Path("latex/figures/appendix_multilingual_overview.png"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("latex/figures/combined_multilingual_fig5_fig11.png"),
    )
    p.add_argument("--pad", type=int, default=24)
    return p.parse_args()


def resize_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height:
        return img
    width = int(round(img.width * (height / img.height)))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()

    left = Image.open(args.left).convert("RGB")
    right = Image.open(args.right).convert("RGB")

    target_h = max(left.height, right.height)
    left = resize_to_height(left, target_h)
    right = resize_to_height(right, target_h)

    out_w = left.width + right.width + args.pad
    out_h = target_h
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + args.pad, 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
