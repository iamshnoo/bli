#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

USER_NAME = os.environ.get("USER", "USER")
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NANOTRON_ROOT = Path(f"/scratch/{USER_NAME}/langsense/nanotron")

ROOT = Path(os.environ.get("BLI_ROOT", str(DEFAULT_ROOT)))
NANOTRON_ROOT = Path(os.environ.get("NANOTRON_ROOT", str(DEFAULT_NANOTRON_ROOT)))
NANOTRON_ENV = Path(os.environ.get("NANOTRON_ENV", str(Path.home() / "nanotron-env")))
HF_HOME_ROOT = Path(os.environ.get("HF_HOME", f"/scratch/{USER_NAME}/cache/hf_cache"))
ENV_FILE = ROOT / ".env"

LOG_ROOT = ROOT / "logs"
SLURM_LOG_ROOT = LOG_ROOT / "slurm_logs"
SUBMISSION_ROOT = LOG_ROOT / "submissions"
CONFIG_MODEL_ROOT = ROOT / "config/models"
HF_MODEL_ROOT = ROOT / "models/hf"

OUTPUT_REV = ROOT / "outputs/revision"
OUTPUT_MULTI = ROOT / "outputs/multilingual_expansion"

TOKENIZER = "meta-llama/Llama-3.2-1B"
HF_EXTRA_ENV = (
    f"set -a; source {shlex.quote(str(ENV_FILE))}; set +a; "
    f"export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; export HF_HOME={shlex.quote(str(HF_HOME_ROOT))}"
)

LANG_DATASET_CODE = {
    "zh": "zho",
    "fr": "fra",
    "fas": "fas",
    "nld": "nld",
    "ukr": "ukr",
    "bul": "bul",
    "ind": "ind",
    "deu": "deu",
}

CORE_LANGS = list(LANG_DATASET_CODE.keys())
MULTI_LANGS = [l for l in CORE_LANGS if l not in {"zh", "fr"}]
SEED_NULL_SEEDS = [101, 202, 303]


def resolve_tokenizer_reference() -> str:
    env_tok = os.environ.get("BLI_TOKENIZER")
    if env_tok:
        return env_tok
    return TOKENIZER


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit BLI training -> conversion -> analysis Slurm DAG")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def ensure_dirs() -> None:
    for path in [SLURM_LOG_ROOT, SUBMISSION_ROOT, CONFIG_MODEL_ROOT, HF_MODEL_ROOT, OUTPUT_REV, OUTPUT_MULTI]:
        path.mkdir(parents=True, exist_ok=True)


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_cmd(cmd: List[str], cwd: Path | None = None, dry_run: bool = False) -> str:
    print("[cmd]", shell_join(cmd))
    if dry_run:
        return ""

    # Slurm submit calls can fail transiently (e.g., controller hiccups).
    # Retry launcher submits a few times before failing hard.
    is_launcher_submit = len(cmd) >= 2 and cmd[1] == "slurm_launcher.py"
    max_attempts = 3 if is_launcher_submit else 1

    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            env = os.environ.copy()
            if is_launcher_submit:
                # Avoid HF API throttling during config generation;
                # launcher can load tokenizer metadata from local cache.
                env.setdefault("HF_HOME", str(HF_HOME_ROOT))
                env.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME_ROOT / "hub"))
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"

            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)
            return proc.stdout + "\n" + proc.stderr
        except subprocess.CalledProcessError as e:
            last_exc = e
            print(f"[error] command failed (attempt {attempt}/{max_attempts}) rc={e.returncode}")
            if e.stdout:
                print("[error stdout]")
                print(e.stdout)
            if e.stderr:
                print("[error stderr]")
                print(e.stderr)

            if attempt < max_attempts:
                sleep_s = 5 * attempt
                print(f"[retry] sleeping {sleep_s}s then retrying...")
                time.sleep(sleep_s)
                continue

    assert last_exc is not None
    raise last_exc


def parse_job_id(output: str) -> str:
    m = re.search(r"JOBID:\s*(\d+)", output)
    if m:
        return m.group(1)
    m = re.search(r"Submitted batch job\s+(\d+)", output)
    if m:
        return m.group(1)
    raise RuntimeError(f"Could not parse job id from output:\n{output}")


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def submit_sbatch(script_text: str, dry_run: bool, dependency_ids: List[str] | None = None) -> str:
    dep_arg: List[str] = []
    if dependency_ids:
        dep_arg = [f"--dependency=afterok:{':'.join(dependency_ids)}"]

    if dry_run:
        print("[sbatch script]\n" + script_text)
        return "DRYRUN"

    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as f:
        f.write(script_text)
        temp_path = f.name

    cmd = ["sbatch", *dep_arg, temp_path]
    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(1, 4):
        try:
            out = run_cmd(cmd, dry_run=False)
            return parse_job_id(out)
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if attempt < 3:
                sleep_s = 5 * attempt
                print(f"[retry] sbatch failed (attempt {attempt}/3), retrying in {sleep_s}s...")
                time.sleep(sleep_s)
                continue
            break
    assert last_exc is not None
    raise last_exc


def required_data_paths() -> List[Path]:
    paths = [
        ROOT / "data/processed/partitions/eng_shared/train",
        ROOT / "data/processed/partitions/eng_disjoint/train",
        ROOT / "data/processed/babylm-eng/train",
    ]
    for code in LANG_DATASET_CODE.values():
        paths.append(ROOT / f"data/processed/babylm-{code}/train")
    return paths


def validate_data_ready() -> None:
    missing = []
    for p in required_data_paths():
        if not p.exists() or not any(p.glob("*.ds")):
            missing.append(str(p))
    if missing:
        raise RuntimeError("Missing tokenized datasets:\n" + "\n".join(missing))


def latest_checkpoint_step(run_name: str) -> int | None:
    ckpt_root = ROOT / "logs/checkpoints" / run_name
    if not ckpt_root.exists():
        return None

    latest_file = ckpt_root / "latest.txt"
    if latest_file.exists():
        raw = latest_file.read_text(encoding="utf-8").strip()
        if raw.isdigit():
            return int(raw)

    steps = []
    for child in ckpt_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(int(child.name))
    if not steps:
        return None
    return max(steps)


def is_training_complete(spec: dict) -> bool:
    step = latest_checkpoint_step(spec["run_name"])
    return step is not None and step >= int(spec["steps"])


def dependency_ids(values: Iterable[str | None]) -> List[str]:
    return [v for v in values if v and v != "DRYRUN"]


def active_jobs_by_name() -> Dict[str, dict]:
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j|%T|%R"], text=True
        )
    except Exception:
        return {}

    active: Dict[str, dict] = {}
    for line in out.splitlines():
        job_id, job_name, state, reason = (line.strip().split("|", 3) + ["", "", "", ""])[:4]
        if not job_id or not job_name:
            continue
        active[job_name] = {"job_id": job_id, "state": state, "reason": reason}
    return active


def is_reusable_active_job(job: dict) -> bool:
    if not job:
        return False
    state = job.get("state", "")
    reason = job.get("reason", "")
    if state not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
        return False
    if reason.startswith("DependencyNever"):
        return False
    return True


def active_training_jobs() -> Dict[str, str]:
    active_all = active_jobs_by_name()
    active: Dict[str, str] = {}
    for job_name, payload in active_all.items():
        if not job_name.startswith("babylm_160m_"):
            continue
        if not is_reusable_active_job(payload):
            continue
        active[job_name] = payload["job_id"]
    return active


def alternating_stage_specs(en_dataset: Path, l2_dataset: Path) -> List[str]:
    specs = []
    stage_idx = 0
    for start in range(1, 3001, 50):
        use_en = stage_idx % 2 == 0
        dataset = en_dataset if use_en else l2_dataset
        role = "en" if use_en else "l2"
        spec = f"start={start};dataset={dataset};weights=1;seq=512;name={role}_stage_{stage_idx+1:02d}"
        specs.append(spec)
        stage_idx += 1
    return specs


def mono_train_specs() -> List[dict]:
    specs = [
        {
            "short_name": "en_50m",
            "run_name": "babylm_160m_en_50m",
            "dataset": ROOT / "data/processed/partitions/eng_shared/train",
            "steps": 1500,
            "warmup": 150,
        },
        {
            "short_name": "en_100m",
            "run_name": "babylm_160m_en_100m",
            "dataset": ROOT / "data/processed/babylm-eng/train",
            "steps": 3000,
            "warmup": 300,
        },
    ]

    # True seed-matched EN-only null controls (same data + budget, different seeds only).
    for idx, seed in enumerate(SEED_NULL_SEEDS, start=1):
        specs.append(
            {
                "short_name": f"en_50m_s{idx}",
                "run_name": f"babylm_160m_en_50m_s{idx}",
                "dataset": ROOT / "data/processed/partitions/eng_shared/train",
                "steps": 1500,
                "warmup": 150,
                "seed": seed,
            }
        )
        specs.append(
            {
                "short_name": f"en_100m_s{idx}",
                "run_name": f"babylm_160m_en_100m_s{idx}",
                "dataset": ROOT / "data/processed/babylm-eng/train",
                "steps": 3000,
                "warmup": 300,
                "seed": seed,
            }
        )

    for short, iso3 in LANG_DATASET_CODE.items():
        for tokens, steps, warmup in [("50m", 1500, 150), ("100m", 3000, 300)]:
            specs.append(
                {
                    "short_name": f"{short}_{tokens}",
                    "run_name": f"babylm_160m_{short}_{tokens}",
                    "dataset": ROOT / f"data/processed/babylm-{iso3}/train",
                    "steps": steps,
                    "warmup": warmup,
                }
            )
    return specs


def bilingual_train_specs() -> List[dict]:
    specs: List[dict] = []
    for short, iso3 in LANG_DATASET_CODE.items():
        l2_dataset = ROOT / f"data/processed/babylm-{iso3}/train"
        for setup, en_dataset in [
            ("a", ROOT / "data/processed/partitions/eng_shared/train"),
            ("b", ROOT / "data/processed/partitions/eng_disjoint/train"),
        ]:
            specs.append(
                {
                    "short_name": f"en_{short}_{setup}",
                    "run_name": f"babylm_160m_en_{short}_setup_{setup}",
                    "stages": alternating_stage_specs(en_dataset=en_dataset, l2_dataset=l2_dataset),
                    "steps": 3000,
                    "warmup": 300,
                }
            )
    return specs


def submit_training_jobs(dry_run: bool) -> Dict[str, dict]:
    train_specs = mono_train_specs() + bilingual_train_specs()
    submitted: Dict[str, dict] = {}
    tokenizer_ref = resolve_tokenizer_reference()
    print(f"[info] tokenizer reference: {tokenizer_ref}")
    active_jobs = active_training_jobs()

    for spec in train_specs:
        run_name = spec["run_name"]
        if run_name in active_jobs:
            print(f"[skip] training already queued/running for {run_name} (job {active_jobs[run_name]})")
            submitted[spec["short_name"]] = {
                **spec,
                "job_id": active_jobs[run_name],
                "status": "already_active",
            }
            continue

        if is_training_complete(spec):
            print(f"[skip] training complete for {run_name}")
            submitted[spec["short_name"]] = {
                **spec,
                "job_id": None,
                "status": "checkpoint_complete",
            }
            continue

        cmd = [
            sys.executable,
            "slurm_launcher.py",
            "--run",
            spec["run_name"],
            "--gpus_per_node",
            "1",
            "--partition",
            "contrib-gpuq",
            "--qos",
            "cs_dept",
            "--gpu-type",
            "A100.80gb",
            "--cpus-per-task",
            "16",
            "--mem",
            "81920M",
            "--time_limit",
            "3-00:00:00",
            "--enable-wandb",
            "--no-sanity",
            "--dp",
            "1",
            "--tp",
            "1",
            "--pp",
            "1",
            "--cp",
            "1",
            "--ep",
            "1",
            "--mbs",
            "8",
            "--acc",
            "8",
            "--model",
            "160m",
            "--vocab-size",
            "128256",
            "--tokenizer",
            tokenizer_ref,
            "--save-interval",
            "500",
            "--grad-clip",
            "0.1",
            "--learning-rate",
            "0.003",
            "--min-lr",
            "0.0003",
            "--weight-decay",
            "0.033",
            "--warmup-steps",
            str(spec["warmup"]),
            "--seed",
            str(spec.get("seed", 42)),
            "--seq",
            "512",
            "--steps",
            str(spec["steps"]),
            "--val-check-interval",
            "-1",
            "--slurm-logs-path",
            str(ROOT / "logs/slurm_logs"),
            "--checkpoints-path",
            str(ROOT / "logs/checkpoints"),
            "--configs-path",
            str(ROOT / "logs/configs"),
            "--slurm-scripts-dir",
            str(ROOT / "logs/slurm_scripts"),
            "--auto-resume",
            "--extra_env",
            HF_EXTRA_ENV,
        ]

        if "dataset" in spec:
            cmd += ["--dataset", str(spec["dataset"])]
        else:
            for stage_spec in spec["stages"]:
                cmd += ["--stage", stage_spec]

        out = run_cmd(cmd, cwd=NANOTRON_ROOT, dry_run=dry_run)
        job_id = "DRYRUN" if dry_run else parse_job_id(out)

        submitted[spec["short_name"]] = {
            **spec,
            "job_id": job_id,
            "status": "submitted",
        }

    return submitted


def submit_conversion_jobs(train_jobs: Dict[str, dict], dry_run: bool) -> Dict[str, str | None]:
    conv_ids: Dict[str, str | None] = {}
    log_dir = SLURM_LOG_ROOT / "conversion"
    log_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_ref = resolve_tokenizer_reference()
    active_jobs = active_jobs_by_name()

    for short_name, spec in train_jobs.items():
        run_name = spec["run_name"]
        expected_step = spec["steps"]
        out_dir = HF_MODEL_ROOT / short_name

        script = f"""#!/bin/bash
#SBATCH --job-name=conv_{short_name}
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=48G
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time=0-08:00:00
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err
set -euo pipefail

BLI_ROOT={shlex.quote(str(ROOT))}
NANOTRON_ROOT={shlex.quote(str(NANOTRON_ROOT))}
NANOTRON_ENV={shlex.quote(str(NANOTRON_ENV))}
source "${{NANOTRON_ENV}}/bin/activate"
cd "${{NANOTRON_ROOT}}"

set +x
set -a; source "${{BLI_ROOT}}/.env"; set +a
export HUGGING_FACE_HUB_TOKEN="${{HF_TOKEN:-${{HUGGING_FACE_HUB_TOKEN:-}}}}"
export HF_HOME={shlex.quote(str(HF_HOME_ROOT))}
set -x

        if [ -f {out_dir}/config.json ]; then
  echo "HF model already exists at {out_dir}; skipping conversion"
  exit 0
fi

CKPT_ROOT=${{BLI_ROOT}}/logs/checkpoints/{run_name}
EXPECTED_STEP={expected_step}

if [ -d "$CKPT_ROOT/$EXPECTED_STEP" ]; then
  CKPT_PATH="$CKPT_ROOT/$EXPECTED_STEP"
elif [ -f "$CKPT_ROOT/latest.txt" ]; then
  CKPT_PATH="$CKPT_ROOT/$(cat "$CKPT_ROOT/latest.txt")"
else
  echo "Missing checkpoint for {run_name} under $CKPT_ROOT"
  exit 1
fi

python "${{BLI_ROOT}}/src/training/convert_checkpoint_to_hf.py" \\
  --checkpoint-path "$CKPT_PATH" \\
  --save-path {out_dir} \\
  --tokenizer-name {tokenizer_ref}
"""
        if (out_dir / "config.json").exists():
            print(f"[skip] conversion complete for {short_name}")
            conv_ids[short_name] = None
            continue

        active = active_jobs.get(f"conv_{short_name}")
        if is_reusable_active_job(active):
            print(f"[skip] conversion already queued/running for {short_name} (job {active['job_id']})")
            conv_ids[short_name] = active["job_id"]
            continue

        job_id = spec.get("job_id")
        dep_ids = [str(job_id)] if job_id and job_id != "DRYRUN" else None
        conv_ids[short_name] = submit_sbatch(script, dry_run=dry_run, dependency_ids=dep_ids)

    return conv_ids


def write_models_json_files() -> Dict[str, Path]:
    CONFIG_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    def model_path(name: str) -> str:
        return str(Path("models/hf") / name)

    files: Dict[str, Path] = {}

    en_ablation = {
        "en_50m": model_path("en_50m"),
        "en_100m": model_path("en_100m"),
    }
    for lang in CORE_LANGS:
        en_ablation[f"en_{lang}_a"] = model_path(f"en_{lang}_a")
        en_ablation[f"en_{lang}_b"] = model_path(f"en_{lang}_b")

    layerwise = {
        "en_50m": model_path("en_50m"),
        "en_100m": model_path("en_100m"),
    }
    for lang in CORE_LANGS:
        layerwise[f"en_{lang}_a"] = model_path(f"en_{lang}_a")

    mapping = {
        "models_en_ablation.json": en_ablation,
        "models_layerwise.json": layerwise,
        "models_seed_null.json": {
            f"en_50m_s{i}": model_path(f"en_50m_s{i}") for i in range(1, 4)
        }
    }
    mapping["models_seed_null.json"].update({f"en_100m_s{i}": model_path(f"en_100m_s{i}") for i in range(1, 4)})

    for lang in CORE_LANGS:
        mapping[f"models_{lang}_shared.json"] = {
            f"{lang}_50m": model_path(f"{lang}_50m"),
            f"{lang}_100m": model_path(f"{lang}_100m"),
            f"en_{lang}_a": model_path(f"en_{lang}_a"),
            f"en_{lang}_b": model_path(f"en_{lang}_b"),
        }

    for filename, payload in mapping.items():
        path = CONFIG_MODEL_ROOT / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        files[filename] = path

    return files


def analysis_job_header(name: str, log_dir: Path, time_limit: str = "1-12:00:00") -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=80G
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --time={time_limit}
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err
set -euo pipefail

BLI_ROOT={shlex.quote(str(ROOT))}
NANOTRON_ENV={shlex.quote(str(NANOTRON_ENV))}
source "${{NANOTRON_ENV}}/bin/activate"
cd "${{BLI_ROOT}}"

set +x
set -a; source "${{BLI_ROOT}}/.env"; set +a
export HUGGING_FACE_HUB_TOKEN="${{HF_TOKEN:-${{HUGGING_FACE_HUB_TOKEN:-}}}}"
export HF_HOME={shlex.quote(str(HF_HOME_ROOT))}
set -x
"""


def submit_analysis_jobs(conv_ids: Dict[str, str | None], model_jsons: Dict[str, Path], dry_run: bool) -> Dict[str, str]:
    analysis_ids: Dict[str, str] = {}
    log_dir = SLURM_LOG_ROOT / "analysis"
    log_dir.mkdir(parents=True, exist_ok=True)
    active_jobs = active_jobs_by_name()

    en_model_keys = ["en_50m", "en_100m"]
    for lang in CORE_LANGS:
        en_model_keys.extend([f"en_{lang}_a", f"en_{lang}_b"])
    en_deps = dependency_ids([conv_ids[x] for x in en_model_keys])

    en_pair_specs = []
    for base in ["en_50m", "en_100m"]:
        for lang in CORE_LANGS:
            for setup in ["a", "b"]:
                en_pair_specs.append(f"{base},en_{lang}_{setup}")

    layer_pair_specs = []
    for base in ["en_50m", "en_100m"]:
        for lang in CORE_LANGS:
            layer_pair_specs.append(f"{base},en_{lang}_a")

    en_run_cmd = [
        "python",
        str(ROOT / "src/bli_analysis/run_bli_pipeline.py"),
        "--models-json",
        str(model_jsons["models_en_ablation.json"]),
        "--output-dir",
        str(OUTPUT_REV / "en_ablation"),
        "--device",
        "cuda",
        "--batch-size",
        "64",
    ]
    for pair in en_pair_specs:
        en_run_cmd += ["--pair", pair]

    layer_cmd = [
        "python",
        str(ROOT / "src/bli_analysis/run_layerwise_analysis.py"),
        "--models-json",
        str(model_jsons["models_layerwise.json"]),
        "--output-csv",
        str(OUTPUT_REV / "en_ablation/bli_layerwise_divergence.csv"),
        "--batch-size",
        "32",
        "--device",
        "cuda",
    ]
    for pair in layer_pair_specs:
        layer_cmd += ["--pair", pair]

    en_cmds = [
        shell_join(en_run_cmd),
        shell_join([
            "python",
            str(ROOT / "src/bli_analysis/compute_bootstrap_ci.py"),
            "--summary-csv",
            str(OUTPUT_REV / "en_ablation/bli_summary_metrics.csv"),
            "--word-csv",
            str(OUTPUT_REV / "en_ablation/bli_word_neighbor_divergence.csv"),
            "--axis-csv",
            str(OUTPUT_REV / "en_ablation/bli_axis_divergence.csv"),
            "--repr-dir",
            str(OUTPUT_REV / "en_ablation/representations"),
            "--out-csv",
            str(OUTPUT_REV / "en_ablation/bli_bootstrap_ci.csv"),
        ]),
        shell_join([
            "python",
            str(ROOT / "src/bli_analysis/run_statistical_tests.py"),
            "--word-csv",
            str(OUTPUT_REV / "en_ablation/bli_word_neighbor_divergence.csv"),
            "--translations-dir",
            str(ROOT / "data/probes"),
            "--out-csv",
            str(OUTPUT_REV / "en_ablation/bli_wilcoxon_overlap.csv"),
        ]),
        shell_join([
            "python",
            str(ROOT / "src/bli_analysis/run_stratified_analysis.py"),
            "--word-csv",
            str(OUTPUT_REV / "en_ablation/bli_word_neighbor_divergence.csv"),
            "--axis-csv",
            str(OUTPUT_REV / "en_ablation/bli_axis_divergence.csv"),
            "--out-csv",
            str(OUTPUT_REV / "en_ablation/bli_stratified_metrics.csv"),
        ]),
        shell_join([
            "python",
            str(ROOT / "src/bli_analysis/run_negative_control_eval.py"),
            "--probe-set",
            str(ROOT / "data/probes/probe_sets.json"),
            "--rep-dir",
            str(OUTPUT_REV / "en_ablation/representations"),
            "--out-csv",
            str(OUTPUT_REV / "en_ablation/bli_negative_control_eval.csv"),
        ]),
        shell_join([
            "python",
            str(ROOT / "src/bli_analysis/run_contextual_alignment_variant.py"),
            "--probe-set",
            str(ROOT / "data/probes/probe_sets.json"),
            "--rep-dir",
            str(OUTPUT_REV / "en_ablation/representations"),
            "--out-csv",
            str(OUTPUT_REV / "en_ablation/bli_contextual_alignment_variant.csv"),
        ]),
        shell_join([
            "python",
            str(ROOT / "src/bli_analysis/run_perhead_analysis.py"),
            "--models-json",
            str(model_jsons["models_en_ablation.json"]),
            "--out-csv",
            str(OUTPUT_REV / "en_ablation/bli_perhead_analysis.csv"),
            "--layers",
            "4,5,6",
            "--batch-size",
            "64",
            "--device",
            "cuda",
        ]),
        shell_join(layer_cmd),
    ]
    en_script = analysis_job_header("bli2_en_analysis", log_dir, time_limit="2-00:00:00") + "\n".join(en_cmds) + "\n"
    active = active_jobs.get("bli2_en_analysis")
    if active and active.get("state") == "RUNNING":
        analysis_ids["en_ablation"] = active["job_id"]
    else:
        analysis_ids["en_ablation"] = submit_sbatch(en_script, dry_run=dry_run, dependency_ids=en_deps or None)

    # Dedicated seed-null analysis: compute representations and same-language controls
    seed_model_keys = [f"en_50m_s{i}" for i in range(1, 4)] + [f"en_100m_s{i}" for i in range(1, 4)]
    seed_deps = dependency_ids([conv_ids.get(x) for x in seed_model_keys])
    seed_pairs = []
    for prefix in ["en_50m_s", "en_100m_s"]:
        names = [f"{prefix}{i}" for i in range(1, 4)]
        for a, b in itertools.combinations(names, 2):
            seed_pairs.append(f"{a},{b}")

    seed_run_cmd = [
        "python",
        str(ROOT / "src/bli_analysis/run_bli_pipeline.py"),
        "--models-json",
        str(model_jsons["models_seed_null.json"]),
        "--output-dir",
        str(OUTPUT_REV / "en_seed_null"),
        "--device",
        "cuda",
        "--batch-size",
        "64",
    ]
    for pair in seed_pairs:
        seed_run_cmd += ["--pair", pair]

    seed_cmds = [
        shell_join(seed_run_cmd),
        shell_join(
            [
                "python",
                str(ROOT / "src/bli_analysis/run_same_language_controls.py"),
                "--probe-set",
                str(ROOT / "data/probes/probe_sets.json"),
                "--rep-dir",
                str(OUTPUT_REV / "en_seed_null/representations"),
                "--out-csv",
                str(OUTPUT_REV / "en_ablation/bli_same_language_controls.csv"),
            ]
        ),
    ]
    seed_script = analysis_job_header("bli2_seed_null", log_dir, time_limit="1-00:00:00") + "\n".join(seed_cmds) + "\n"
    active = active_jobs.get("bli2_seed_null")
    if active and active.get("state") == "RUNNING":
        analysis_ids["seed_null"] = active["job_id"]
    else:
        analysis_ids["seed_null"] = submit_sbatch(seed_script, dry_run=dry_run, dependency_ids=seed_deps or None)

    for lang in CORE_LANGS:
        deps = dependency_ids([conv_ids[f"{lang}_50m"], conv_ids[f"{lang}_100m"], conv_ids[f"en_{lang}_a"], conv_ids[f"en_{lang}_b"]])
        models_json = model_jsons[f"models_{lang}_shared.json"]
        out_root = OUTPUT_REV if lang in {"zh", "fr"} else OUTPUT_MULTI
        out_dir = out_root / f"{lang}_shared_language"
        pairs = [
            f"{lang}_50m,en_{lang}_a",
            f"{lang}_50m,en_{lang}_b",
            f"{lang}_100m,en_{lang}_a",
            f"{lang}_100m,en_{lang}_b",
        ]

        run_cmd_parts = [
            "python",
            str(ROOT / "src/bli_analysis/run_bli_pipeline.py"),
            "--models-json",
            str(models_json),
            "--output-dir",
            str(out_dir),
            "--device",
            "cuda",
            "--batch-size",
            "64",
        ]
        for pair in pairs:
            run_cmd_parts += ["--pair", pair]

        cmds = [
            shell_join(run_cmd_parts),
            shell_join([
                "python",
                str(ROOT / "src/bli_analysis/compute_bootstrap_ci.py"),
                "--summary-csv",
                str(out_dir / "bli_summary_metrics.csv"),
                "--word-csv",
                str(out_dir / "bli_word_neighbor_divergence.csv"),
                "--axis-csv",
                str(out_dir / "bli_axis_divergence.csv"),
                "--repr-dir",
                str(out_dir / "representations"),
                "--out-csv",
                str(out_dir / "bli_bootstrap_ci.csv"),
            ]),
        ]

        job_key = f"{lang}_shared"
        script = analysis_job_header(f"bli2_{job_key}", log_dir, time_limit="1-00:00:00") + "\n".join(cmds) + "\n"
        active = active_jobs.get(f"bli2_{job_key}")
        if active and active.get("state") == "RUNNING":
            analysis_ids[job_key] = active["job_id"]
        else:
            analysis_ids[job_key] = submit_sbatch(script, dry_run=dry_run, dependency_ids=deps or None)

    return analysis_ids


def submit_postprocess_job(analysis_ids: Dict[str, str], dry_run: bool) -> str:
    deps = [x for x in analysis_ids.values() if x != "DRYRUN"]
    log_dir = SLURM_LOG_ROOT / "postprocess"
    log_dir.mkdir(parents=True, exist_ok=True)
    active = active_jobs_by_name().get("bli2_post")
    if active and active.get("state") == "RUNNING":
        return active["job_id"]

    script = f"""#!/bin/bash
#SBATCH --job-name=bli2_post
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=normal
#SBATCH --time=0-12:00:00
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err
set -euo pipefail

BLI_ROOT={shlex.quote(str(ROOT))}
NANOTRON_ENV={shlex.quote(str(NANOTRON_ENV))}
source "${{NANOTRON_ENV}}/bin/activate"
cd "${{BLI_ROOT}}"

python src/bli_analysis/center_en_subspace_metrics.py \\
  --seed-summary-csv "${{BLI_ROOT}}/outputs/revision/en_seed_null/bli_summary_metrics.csv" \\
  --en-ablation-dir "${{BLI_ROOT}}/outputs/revision/en_ablation"

python src/bli_analysis/run_progress_sensitivity.py \\
  --summary-csv "${{BLI_ROOT}}/outputs/revision/en_ablation/bli_summary_metrics.csv" \\
  --out-csv "${{BLI_ROOT}}/outputs/revision/en_ablation/bli_progress_sensitivity.csv"

python src/pipeline/build_language_ratio_summary.py \\
  --revision-root "${{BLI_ROOT}}/outputs/revision" \\
  --multilingual-root "${{BLI_ROOT}}/outputs/multilingual_expansion" \\
  --output "${{BLI_ROOT}}/outputs/multilingual_expansion/language_ratio_summary.csv"

python latex/scripts/generate_artifacts.py \\
  --output-root "${{BLI_ROOT}}/outputs/revision" \\
  --multilingual-output-root "${{BLI_ROOT}}/outputs/multilingual_expansion" \\
  --latex-root "${{BLI_ROOT}}/latex"

bash latex/scripts/sync_report.sh
"""

    return submit_sbatch(script, dry_run=dry_run, dependency_ids=deps if deps else None)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not args.dry_run:
        validate_data_ready()

    model_jsons = write_models_json_files()

    train_jobs = submit_training_jobs(dry_run=args.dry_run)
    conv_jobs = submit_conversion_jobs(train_jobs, dry_run=args.dry_run)
    analysis_jobs = submit_analysis_jobs(conv_jobs, model_jsons=model_jsons, dry_run=args.dry_run)
    post_job = submit_postprocess_job(analysis_jobs, dry_run=args.dry_run)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "train_jobs": train_jobs,
        "conversion_jobs": conv_jobs,
        "analysis_jobs": analysis_jobs,
        "postprocess_job": post_job,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUBMISSION_ROOT / f"submission_manifest_{ts}.json"
    out_path.write_text(json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8")

    print(f"[ok] Wrote submission manifest: {out_path}")
    print("[summary]")
    print(f"  train jobs: {len(train_jobs)}")
    print(f"  conversion jobs: {len(conv_jobs)}")
    print(f"  analysis jobs: {len(analysis_jobs)}")
    print(f"  postprocess job: {post_job}")


if __name__ == "__main__":
    main()
