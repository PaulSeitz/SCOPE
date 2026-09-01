#!/usr/bin/env python3
"""
Run every experiment YAML under ``configs/experiment/feature_eval``.

Configs exercise recent feature work (ego LoS/weapon flags, CLIP_REWARD off,
NAVIGATE skill head, Seaquest/Kangaroo heuristics, skill-success + info logging)
with ego skill observations. Filter W&B with ``suite:feature_eval``.

Usage (from project root):
  python scripts/run_feature_eval.py --list
  python scripts/run_feature_eval.py
  python scripts/run_feature_eval.py --only seaquest_baseline_ego seaquest_no_clip
  CUDA_VISIBLE_DEVICES=0 python scripts/run_feature_eval.py --shard 1/2 &
  CUDA_VISIBLE_DEVICES=1 python scripts/run_feature_eval.py --shard 2/2 &
  python scripts/run_feature_eval.py --trial
  python scripts/run_feature_eval.py --shard 1/3 --trial --wandb-mode online
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as train_main  # noqa: E402
from concept_based_learning.training.rtpt_progress import (  # noqa: E402
    num_updates_from_timesteps,
    start_rtpt,
)

CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "feature_eval"
RTPT_BASE = "features"
DEFAULT_SEED = 44
TRIAL_STEPS_DEFAULT = 32768


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_experiment_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    alg = raw.pop("alg", {})
    return {**raw, **alg}


def _parse_shard(spec: str) -> tuple[int, int]:
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid shard {spec!r}; expected INDEX/TOTAL (e.g. 1/2)")
    shard_one_based = int(parts[0])
    num_shards = int(parts[1])
    if num_shards < 1:
        raise ValueError(f"shard total must be >= 1, got {num_shards}")
    if not 1 <= shard_one_based <= num_shards:
        raise ValueError(
            f"shard index must be in [1, {num_shards}], got {shard_one_based}"
        )
    return shard_one_based - 1, num_shards


def _discover_configs(
    *,
    only: list[str] | None = None,
) -> list[Path]:
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    paths = [p for p in paths if not p.name.startswith("manifest")]
    if only:
        wanted = {name.removesuffix(".yaml") for name in only}
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(
                f"Config(s) not found in {CONFIG_DIR}: {sorted(missing)}"
            )
    return paths


def _shard_paths(paths: list[Path], *, shard_idx: int, num_shards: int) -> list[Path]:
    return [p for i, p in enumerate(paths) if i % num_shards == shard_idx]


def _rtpt_name(shard: str | None) -> str:
    if shard is None:
        return RTPT_BASE
    shard_idx, _ = _parse_shard(shard)
    return f"{RTPT_BASE}{shard_idx + 1}"


def _rollout_batch(num_envs: int, num_steps: int) -> int:
    return int(num_envs) * int(num_steps)


def _align_trial_timesteps(cfg: dict[str, Any]) -> None:
    batch = _rollout_batch(int(cfg.get("NUM_ENVS", 512)), int(cfg.get("NUM_STEPS", 64)))
    if batch <= 0:
        return
    total = int(cfg.get("TOTAL_TIMESTEPS", 0))
    updates = max(1, total // batch)
    cfg["TOTAL_TIMESTEPS"] = updates * batch


def _apply_run_defaults(
    cfg: dict[str, Any],
    *,
    wandb_mode: str | None,
    trial_steps: int | None,
    record_video: bool,
) -> str:
    if wandb_mode is not None:
        cfg["WANDB_MODE"] = wandb_mode
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        cfg.setdefault("WANDB_TAGS", [])
        if "trial" not in cfg["WANDB_TAGS"]:
            cfg["WANDB_TAGS"] = [*cfg["WANDB_TAGS"], "trial"]
        _align_trial_timesteps(cfg)
    cfg.setdefault("ENTITY", "paul-seitz-tu-darmstadt")
    cfg.setdefault("PROJECT", "concept_based_learning")
    cfg.setdefault("SEED", DEFAULT_SEED)
    cfg["RECORD_VIDEO"] = record_video
    cfg["VIDEO_AT_END"] = record_video
    return str(cfg.get("WANDB_RUN_NAME", "run"))


def _updates_for_config_path(path: Path, *, trial_steps: int | None) -> int:
    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        _align_trial_timesteps(cfg)
    return num_updates_from_timesteps(
        int(cfg.get("TOTAL_TIMESTEPS", 0)),
        num_steps=int(cfg.get("NUM_STEPS", 64)),
        num_envs=int(cfg.get("NUM_ENVS", 512)),
    )


def _init_rtpt_for_paths(
    paths: list[Path],
    *,
    experiment_name: str,
    trial_steps: int | None,
) -> None:
    total = sum(_updates_for_config_path(p, trial_steps=trial_steps) for p in paths)
    total = max(1, total)
    if start_rtpt(
        {
            "RTPT_ENABLED": True,
            "RTPT_NAME_INITIALS": "PS",
            "RTPT_EXPERIMENT_NAME": experiment_name,
            "RTPT_STEP_EVERY": 10,
            "TOTAL_TIMESTEPS": total,
            "NUM_STEPS": 1,
            "NUM_ENVS": 1,
        }
    ):
        print(
            f"RTPT initialized: {total:,} total updates across {len(paths)} run(s) "
            f"({experiment_name}).\n"
        )


def _clear_jax_memory() -> None:
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def _release_run_memory() -> None:
    gc.collect()
    _clear_jax_memory()


def _print_matrix(paths: list[Path]) -> None:
    print(f"\n=== feature_eval ({CONFIG_DIR.relative_to(PROJECT_ROOT)}) ===")
    print(f"{'Config':<36} {'WANDB_RUN_NAME':<28} WANDB_TAGS")
    print("-" * 110)
    for path in paths:
        raw = _load_yaml(path)
        print(
            f"{path.stem:<36} "
            f"{str(raw.get('WANDB_RUN_NAME', path.stem)):<28} "
            f"{list(raw.get('WANDB_TAGS', []))}"
        )
    print("\nW&B filter: suite:feature_eval")
    print(
        "Shard example:\n"
        "  CUDA_VISIBLE_DEVICES=0 python scripts/run_feature_eval.py --shard 1/2 &\n"
        "  CUDA_VISIBLE_DEVICES=1 python scripts/run_feature_eval.py --shard 2/2 &\n"
    )


def _run_experiment(
    path: Path,
    *,
    wandb_mode: str | None,
    trial_steps: int | None,
    record_video: bool,
) -> None:
    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    _apply_run_defaults(
        cfg,
        wandb_mode=wandb_mode,
        trial_steps=trial_steps,
        record_video=record_video,
    )

    print(f"\n{'=' * 70}")
    print(f"  START: {path.name}")
    print(
        f"  ENV={cfg.get('ENV_NAME')}  MODE={cfg.get('TRAIN_MODE')}  "
        f"STEPS={cfg.get('TOTAL_TIMESTEPS', 0):,}"
    )
    print(f"{'=' * 70}\n")

    try:
        train_main.single_run(cfg)
        print(f"\n  DONE: {path.name}\n")
    finally:
        _release_run_memory()


def _resolve_wandb_mode(*, trial: bool, wandb_mode: str | None) -> str:
    if wandb_mode is not None:
        return wandb_mode
    return "disabled" if trial else "online"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="NAME",
        help="Subset of config stems (e.g. seaquest_baseline_ego); default: all",
    )
    parser.add_argument(
        "--shard",
        metavar="INDEX/TOTAL",
        help="Run every TOTAL-th config starting at INDEX (1-based), e.g. 1/2 and 2/2",
    )
    parser.add_argument("--list", action="store_true", help="Print configs and exit")
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Short smoke run; wandb disabled by default",
    )
    parser.add_argument("--trial-steps", type=int, default=TRIAL_STEPS_DEFAULT)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip end-of-run video recording",
    )
    parser.add_argument(
        "--abort-on-failure",
        action="store_true",
        help="Stop the shard on the first failed run (default: continue)",
    )
    args = parser.parse_args()

    if not CONFIG_DIR.is_dir():
        raise SystemExit(f"Config directory missing: {CONFIG_DIR}")

    paths = _discover_configs(only=args.only)
    if not paths:
        raise SystemExit(f"No YAML configs found in {CONFIG_DIR}")

    if args.shard is not None:
        shard_idx, num_shards = _parse_shard(args.shard)
        paths = _shard_paths(paths, shard_idx=shard_idx, num_shards=num_shards)
        if not paths:
            raise SystemExit(f"Shard {args.shard} selected 0 configs")

    if args.list:
        _print_matrix(paths)
        return

    wandb_mode = _resolve_wandb_mode(trial=args.trial, wandb_mode=args.wandb_mode)
    trial_steps = args.trial_steps if args.trial else None
    record_video = not args.no_video

    print(f"feature_eval: {len(paths)} run(s)")
    for p in paths:
        print(f"  - {p.stem}")
    print()

    _init_rtpt_for_paths(
        paths,
        experiment_name=_rtpt_name(args.shard),
        trial_steps=trial_steps,
    )

    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            _run_experiment(
                path,
                wandb_mode=wandb_mode,
                trial_steps=trial_steps,
                record_video=record_video,
            )
        except Exception as exc:
            traceback.print_exc()
            failures.append((path.name, str(exc)))
            if args.abort_on_failure:
                break
            print(f"\n  FAILED (continuing): {path.name}: {exc}\n")

    if failures:
        print("\nFailed runs:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        raise SystemExit(1)

    print("\nAll feature_eval runs finished successfully.")


if __name__ == "__main__":
    main()
