#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _candidate_roots() -> list[Path]:
    user = os.environ.get("USER", "USER")
    explicit = os.environ.get("NANOTRON_ROOT")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path(f"/scratch/{user}/pretrain/nanotron_full"),
            Path(f"/scratch/{user}/pretrain/nanotron"),
        ]
    )
    return candidates


def _usable_nanotron_root() -> Path:
    for root in _candidate_roots():
        if (root / "src" / "nanotron" / "config" / "lighteval_config.py").exists():
            return root
    raise FileNotFoundError(
        "Could not find a usable Nanotron checkout with src/nanotron/config/lighteval_config.py. "
        "Set NANOTRON_ROOT to a complete checkout."
    )


NANOTRON_ROOT = _usable_nanotron_root()
for candidate in [NANOTRON_ROOT, NANOTRON_ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Nanotron conversion expects distributed env vars even for world-size 1.
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
if "MASTER_PORT" not in os.environ:
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if slurm_job_id.isdigit():
        os.environ["MASTER_PORT"] = str(10000 + (int(slurm_job_id) % 50000))
    else:
        os.environ["MASTER_PORT"] = "29500"

from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.config import Qwen2Config as NanotronQwen2Config
from examples.llama.convert_nanotron_to_hf import convert_checkpoint_and_save


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a Nanotron checkpoint to local HF format")
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--save-path", type=Path, required=True)
    p.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--config-cls", choices=["LlamaConfig", "Qwen2Config"], default="LlamaConfig")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.save_path.exists() and (args.save_path / "config.json").exists() and not args.overwrite:
        print(f"[skip] HF model already exists: {args.save_path}")
        return

    args.save_path.mkdir(parents=True, exist_ok=True)

    config_cls = NanotronLlamaConfig if args.config_cls == "LlamaConfig" else NanotronQwen2Config
    convert_checkpoint_and_save(
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        tokenizer_name=args.tokenizer_name,
        config_cls=config_cls,
    )


if __name__ == "__main__":
    main()
