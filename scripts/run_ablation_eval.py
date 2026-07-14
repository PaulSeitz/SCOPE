#!/usr/bin/env python3
"""
Run input-mode and Q-stickiness ablation suites.

Configs live under ``configs/experiment/input_ablation`` and
``configs/experiment/q_stickiness_eval`` with ``WANDB_TAGS`` / ``WANDB_RUN_NAME``
embedded for W&B filtering.

Configs are run sequentially in one Python process (same pattern as
``run_sequential_eval.py``). RTPT is initialized once with the total update count
for the shard/manifest so nvidia-smi shows one continuous ETA.

Usage (from project root):
  python scripts/run_ablation_eval.py --list-matrix
  python scripts/run_ablation_eval.py --suite input_ablation
  python scripts/run_ablation_eval.py --suite input_ablation --manifest manifest_multi_env.yaml
  CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation_eval.py --suite input_ablation --manifest manifest_multi_env.yaml --manifest-shard 1/2 &
  CUDA_VISIBLE_DEVICES=1 python scripts/run_ablation_eval.py --suite input_ablation --manifest manifest_multi_env.yaml --manifest-shard 2/2 &
  python scripts/run_ablation_eval.py --suite q_stickiness
  python scripts/run_ablation_eval.py --suite q_stickiness --phases 1
  python scripts/run_ablation_eval.py --suite input_ablation --variants mi_flat_si_ego --trial
  python scripts/materialize_ablation_configs.py   # regenerate YAMLs after editing manifest
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

# Reduce cudnn autotune memory spikes for pixel_embed tandem runs.
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

INPUT_DIR = PROJECT_ROOT / "configs" / "experiment" / "input_ablation"
QSTICK_DIR = PROJECT_ROOT / "configs" / "experiment" / "q_stickiness_eval"
DEFAULT_INPUT_MANIFEST = INPUT_DIR / "manifest_multi_env.yaml"
INPUT_ABLATION_RTPT_BASE = "HRL_input_ablation"

INPUT_ABLATION_ENVS = ("seaquest", "spaceinvaders", "freeway", "beamrider")
INPUT_VARIANT_KEYS = [
    f"mi_{meta}_si_{skill}"
    for meta in ("flat", "oc_embed", "pixel_embed")
    for skill in ("ego", "oc_embed", "pixel_embed")
]

QSTICK_PHASE1 = "phase1_skills"
QSTICK_PHASE2_KEYS = [
    "baseline_no_qsticky",
    "qsticky_t0",
    "qsticky_t025",
    "qsticky_t0_switch_penalty",
    "qsticky_t_neg1",
]

# Trade parallel envs for longer rollouts (same batch size, same update count).
_SAME_BATCH_PAIRS: tuple[tuple[int, int], ...] = (
    (512, 64),
    (256, 128),
    (128, 256),
)
# Last-resort tiers: smaller rollout batches, update count preserved via TOTAL_TIMESTEPS.
_SMALLER_BATCH_PAIRS: tuple[tuple[int, int], ...] = (
    (512, 32),
    (256, 64),
    (256, 32),
    (128, 64),
    (128, 32),
    (64, 64),
    (64, 32),
    (32, 64),
    (32, 32),
    (16, 64),
    (16, 32),
)
_MIN_NUM_ENVS = 16
_MIN_NUM_STEPS = 32
_MAX_NUM_STEPS = 128


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_manifest_path(manifest: str | Path | None) -> Path | None:
    if manifest is None:
        return None
    path = Path(manifest)
    if not path.is_absolute():
        candidate = INPUT_DIR / path
        if candidate.is_file():
            return candidate
        candidate = PROJECT_ROOT / path
        if candidate.is_file():
            return candidate
    return path


def _load_input_manifest(path: Path) -> list[tuple[str, str]]:
    raw = _load_yaml(path)
    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"Manifest {path} must contain a non-empty 'runs' list")

    entries: list[tuple[str, str]] = []
    for idx, entry in enumerate(runs):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest run #{idx} must be a mapping, got {type(entry)!r}")
        env_name = str(entry.get("env", "")).strip()
        variant = str(entry.get("variant", "")).strip()
        if not env_name or not variant:
            raise ValueError(f"Manifest run #{idx} needs both 'env' and 'variant'")
        entries.append((env_name, variant))
    return entries


def _parse_manifest_shard(spec: str) -> tuple[int, int]:
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid manifest shard {spec!r}; expected INDEX/TOTAL (e.g. 1/2)")
    shard_one_based = int(parts[0])
    num_shards = int(parts[1])
    if num_shards < 1:
        raise ValueError(f"manifest shard total must be >= 1, got {num_shards}")
    if not 1 <= shard_one_based <= num_shards:
        raise ValueError(
            f"manifest shard index must be in [1, {num_shards}], got {shard_one_based}"
        )
    return shard_one_based - 1, num_shards


def _shard_manifest_entries(
    entries: list[tuple[str, str]], *, shard_idx: int, num_shards: int
) -> list[tuple[str, str]]:
    return [entry for i, entry in enumerate(entries) if i % num_shards == shard_idx]


def _input_ablation_rtpt_name(manifest_shard: str | None = None) -> str:
    """RTPT label on the cluster: HRL_input_ablation, HRL_input_ablation1, …"""
    if manifest_shard is None:
        return INPUT_ABLATION_RTPT_BASE
    shard_idx, _ = _parse_manifest_shard(manifest_shard)
    return f"{INPUT_ABLATION_RTPT_BASE}{shard_idx + 1}"


def _flatten_experiment_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    alg = raw.pop("alg", {})
    return {**raw, **alg}


def _experiment_path(suite: str, key: str, *, env_name: str = "seaquest") -> Path:
    if suite == "input_ablation":
        return INPUT_DIR / f"{env_name}_{key}.yaml"
    if suite == "q_stickiness":
        if key == QSTICK_PHASE1:
            return QSTICK_DIR / f"{key}.yaml"
        return QSTICK_DIR / f"seaquest_{key}.yaml"
    raise ValueError(f"Unknown suite={suite!r}")


def _list_rows(
    suite: str,
    *,
    env_name: str | None = None,
    manifest_entries: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str, list[str]]]:
    rows: list[tuple[str, str, list[str]]] = []
    if suite == "input_ablation":
        if manifest_entries is not None:
            for env, key in manifest_entries:
                path = _experiment_path(suite, key, env_name=env)
                if not path.is_file():
                    rows.append((f"{env}/{key}", f"<missing: {path.name}>", []))
                    continue
                raw = _load_yaml(path)
                rows.append(
                    (
                        f"{env}/{key}",
                        str(raw.get("WANDB_RUN_NAME", key)),
                        list(raw.get("WANDB_TAGS", [])),
                    )
                )
            return rows
        envs = (env_name,) if env_name else INPUT_ABLATION_ENVS
        for env in envs:
            for key in INPUT_VARIANT_KEYS:
                path = _experiment_path(suite, key, env_name=env)
                if not path.is_file():
                    continue
                raw = _load_yaml(path)
                rows.append(
                    (
                        f"{env}/{key}",
                        str(raw.get("WANDB_RUN_NAME", key)),
                        list(raw.get("WANDB_TAGS", [])),
                    )
                )
        return rows
    raw_p1 = _load_yaml(_experiment_path(suite, QSTICK_PHASE1))
    rows.append(
        (
            QSTICK_PHASE1,
            str(raw_p1.get("WANDB_RUN_NAME", QSTICK_PHASE1)),
            list(raw_p1.get("WANDB_TAGS", [])),
        )
    )
    for key in QSTICK_PHASE2_KEYS:
        raw = _load_yaml(_experiment_path(suite, key))
        rows.append((key, str(raw.get("WANDB_RUN_NAME", key)), list(raw.get("WANDB_TAGS", []))))
    return rows


def print_matrix(
    suite: str | None = None,
    *,
    manifest_entries: list[tuple[str, str]] | None = None,
) -> None:
    suites = ["input_ablation", "q_stickiness"] if suite is None else [suite]
    for s in suites:
        print(f"\n=== {s} ===")
        print(f"{'Variant':<34} {'WANDB_RUN_NAME':<24} WANDB_TAGS")
        print("-" * 100)
        for key, run_name, tags in _list_rows(
            s, manifest_entries=manifest_entries if s == "input_ablation" else None
        ):
            print(f"{key:<34} {run_name:<24} {tags}")
        print("\nW&B filter examples:")
        if s == "input_ablation":
            print("  suite:input_ablation")
            print("  meta_input:pixel_embed skill_input:pixel_embed")
        else:
            print("  suite:q_stickiness_eval")
            print("  q_sticky:on q_thresh:0 switch_penalty:on")


def _rollout_batch(num_envs: int, num_steps: int) -> int:
    return int(num_envs) * int(num_steps)


def _num_updates(total_timesteps: int, num_envs: int, num_steps: int) -> int:
    batch = _rollout_batch(num_envs, num_steps)
    if batch <= 0:
        return 0
    return int(total_timesteps) // batch


def _valid_rollout_pair(
    num_envs: int,
    num_steps: int,
    *,
    num_minibatches: int,
) -> bool:
    if num_envs < _MIN_NUM_ENVS or num_steps < _MIN_NUM_STEPS:
        return False
    if num_steps > _MAX_NUM_STEPS:
        return False
    batch = _rollout_batch(num_envs, num_steps)
    return batch > 0 and batch % num_minibatches == 0


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "out of memory",
        "resource_exhausted",
        "cuda_error_memory_allocation",
        "failed to allocate",
        "xla allocation",
        "oom",
    )
    return any(needle in msg for needle in needles)


def _oom_fallback_tiers(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build lower-memory rollout tiers, preserving update count when possible."""
    num_minibatches = int(cfg.get("NUM_MINIBATCHES", 32))
    total = int(cfg["TOTAL_TIMESTEPS"])
    envs = int(cfg["NUM_ENVS"])
    steps = int(cfg["NUM_STEPS"])
    target_updates = _num_updates(total, envs, steps)
    initial_batch = _rollout_batch(envs, steps)

    tiers: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = {(envs, steps)}

    def _append_tier(
        num_envs: int,
        num_steps: int,
        *,
        preserve_total_timesteps: bool,
        mode: str,
    ) -> None:
        pair = (num_envs, num_steps)
        if pair in seen_pairs:
            return
        if not _valid_rollout_pair(
            num_envs, num_steps, num_minibatches=num_minibatches
        ):
            return
        if num_envs >= envs and num_steps >= steps:
            return
        if num_envs >= envs and _rollout_batch(num_envs, num_steps) >= initial_batch:
            return

        tier = copy.deepcopy(cfg)
        tier["NUM_ENVS"] = num_envs
        tier["NUM_STEPS"] = num_steps
        if preserve_total_timesteps:
            tier["TOTAL_TIMESTEPS"] = total
        else:
            tier["TOTAL_TIMESTEPS"] = target_updates * _rollout_batch(num_envs, num_steps)
        tier["OOM_FALLBACK_MODE"] = mode
        tier["OOM_FALLBACK_ROLLOUT"] = f"{num_envs}x{num_steps}"
        tiers.append(tier)
        seen_pairs.add(pair)

    for num_envs, num_steps in _SAME_BATCH_PAIRS:
        if _rollout_batch(num_envs, num_steps) != initial_batch:
            continue
        _append_tier(
            num_envs,
            num_steps,
            preserve_total_timesteps=True,
            mode="same_batch",
        )

    for num_envs, num_steps in _SMALLER_BATCH_PAIRS:
        if _rollout_batch(num_envs, num_steps) >= initial_batch:
            continue
        _append_tier(
            num_envs,
            num_steps,
            preserve_total_timesteps=False,
            mode="smaller_batch",
        )

    tiers.sort(key=lambda t: (int(t["NUM_ENVS"]), int(t["NUM_STEPS"])))
    return tiers


def _mark_oom_fallback(cfg: dict[str, Any], *, base_run_name: str) -> None:
    rollout = str(cfg.get("OOM_FALLBACK_ROLLOUT", "unknown"))
    mode = str(cfg.get("OOM_FALLBACK_MODE", "unknown"))
    cfg["OOM_FALLBACK"] = True
    cfg.setdefault("WANDB_TAGS", [])
    extra_tags = [
        "oom_fallback:on",
        f"rollout:{rollout}",
        f"oom_fallback_mode:{mode}",
    ]
    cfg["WANDB_TAGS"] = [
        *cfg["WANDB_TAGS"],
        *(tag for tag in extra_tags if tag not in cfg["WANDB_TAGS"]),
    ]
    suffix = f"_oom{rollout}"
    run_name = str(cfg.get("WANDB_RUN_NAME", base_run_name))
    if not run_name.endswith(suffix):
        cfg["WANDB_RUN_NAME"] = f"{run_name}{suffix}"


def _align_trial_timesteps(cfg: dict[str, Any]) -> None:
    """Ensure trial runs execute at least one PPO update."""
    batch = int(cfg.get("NUM_ENVS", 512)) * int(cfg.get("NUM_STEPS", 64))
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
) -> str:
    if wandb_mode is not None:
        cfg["WANDB_MODE"] = wandb_mode
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        cfg.setdefault("WANDB_TAGS", [])
        if "trial" not in cfg["WANDB_TAGS"]:
            cfg["WANDB_TAGS"] = [*cfg["WANDB_TAGS"], "trial"]
        meta_mode = str(cfg.get("META_INPUT_MODE", "flat")).lower().strip()
        skill_mode = str(cfg.get("SKILL_INPUT_MODE", "ego")).lower().strip()
        dual_pixel = meta_mode == "pixel_embed" and skill_mode == "pixel_embed"
        if meta_mode == "pixel_embed" or skill_mode == "pixel_embed":
            cap_envs = 64 if dual_pixel else 128
            cfg["NUM_ENVS"] = min(int(cfg.get("NUM_ENVS", 512)), cap_envs)
            cfg["NUM_STEPS"] = min(int(cfg.get("NUM_STEPS", 64)), 32)
        _align_trial_timesteps(cfg)
    cfg.setdefault("ENTITY", "paul-seitz-tu-darmstadt")
    cfg.setdefault("PROJECT", "concept_based_learning")
    cfg.setdefault("SEED", 44)
    cfg["RECORD_VIDEO"] = False
    cfg["VIDEO_AT_END"] = False
    return str(cfg.get("WANDB_RUN_NAME", "run"))


def _clear_jax_memory() -> None:
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def _updates_for_config_path(path: Path, *, trial_steps: int | None) -> int:
    """PPO update count for one experiment YAML (after trial overrides)."""
    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        meta_mode = str(cfg.get("META_INPUT_MODE", "flat")).lower().strip()
        skill_mode = str(cfg.get("SKILL_INPUT_MODE", "ego")).lower().strip()
        dual_pixel = meta_mode == "pixel_embed" and skill_mode == "pixel_embed"
        if meta_mode == "pixel_embed" or skill_mode == "pixel_embed":
            cap_envs = 64 if dual_pixel else 128
            cfg["NUM_ENVS"] = min(int(cfg.get("NUM_ENVS", 512)), cap_envs)
            cfg["NUM_STEPS"] = min(int(cfg.get("NUM_STEPS", 64)), 32)
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
    """One RTPT bar for the whole suite (same pattern as run_sequential_eval.py)."""
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


def _release_run_memory() -> None:
    """Best-effort JAX cache drop between sequential runs in one process."""
    gc.collect()
    _clear_jax_memory()


def _run_experiment(
    path: Path,
    *,
    wandb_mode: str | None,
    trial_steps: int | None,
    oom_retry: bool,
) -> None:
    raw = _load_yaml(path)
    base_cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    base_run_name = _apply_run_defaults(
        base_cfg,
        wandb_mode=wandb_mode,
        trial_steps=trial_steps,
    )

    print(f"\n{'=' * 70}")
    print(f"  START: {path.name}")
    print(
        f"  ENV={base_cfg.get('ENV_NAME')}  MODE={base_cfg.get('TRAIN_MODE')}  "
        f"STEPS={base_cfg.get('TOTAL_TIMESTEPS', 0):,}"
    )
    print(f"{'=' * 70}\n")

    tier_cfgs = [base_cfg]
    if oom_retry:
        tier_cfgs.extend(_oom_fallback_tiers(base_cfg))

    last_exc: BaseException | None = None
    for tier_idx, tier_cfg in enumerate(tier_cfgs):
        if tier_idx > 0:
            _mark_oom_fallback(tier_cfg, base_run_name=base_run_name)
            print(
                "\n>>> OOM retry "
                f"(tier {tier_idx}/{len(tier_cfgs) - 1}): "
                f"NUM_ENVS={tier_cfg['NUM_ENVS']}, "
                f"NUM_STEPS={tier_cfg['NUM_STEPS']}, "
                f"TOTAL_TIMESTEPS={tier_cfg['TOTAL_TIMESTEPS']}, "
                f"mode={tier_cfg.get('OOM_FALLBACK_MODE')}"
            )
        try:
            train_main.single_run(tier_cfg)
            print(f"\n  DONE: {path.name}\n")
            _release_run_memory()
            return
        except Exception as exc:
            last_exc = exc
            can_retry = tier_idx < len(tier_cfgs) - 1 and _is_oom_error(exc)
            if not can_retry:
                raise
            print(f"\n>>> OOM detected for {path.name}; trying lower-memory tier...")
            traceback.print_exc()
            _release_run_memory()

    if last_exc is not None:
        raise last_exc


def _collect_input_ablation_paths(
    *,
    manifest_entries: list[tuple[str, str]] | None,
    variants: list[str] | None,
    env_name: str | None,
) -> list[Path]:
    paths: list[Path] = []
    if manifest_entries is not None:
        for env, key in manifest_entries:
            path = _experiment_path("input_ablation", key, env_name=env)
            if path.is_file():
                paths.append(path)
        return paths

    keys = variants or INPUT_VARIANT_KEYS
    envs = (env_name,) if env_name else INPUT_ABLATION_ENVS
    for env in envs:
        for key in keys:
            path = _experiment_path("input_ablation", key, env_name=env)
            if path.is_file():
                paths.append(path)
            else:
                print(f"\n>>> Skipping missing config {path.name}")
    return paths


def run_suite(
    suite: str,
    *,
    variants: list[str] | None,
    phases: list[int],
    wandb_mode: str | None,
    trial_steps: int | None,
    env_name: str | None = None,
    manifest_entries: list[tuple[str, str]] | None = None,
    oom_retry: bool = True,
    rtpt_experiment_name: str | None = None,
) -> None:
    if suite == "input_ablation":
        paths = _collect_input_ablation_paths(
            manifest_entries=manifest_entries,
            variants=variants,
            env_name=env_name,
        )
        if manifest_entries is not None:
            for env, key in manifest_entries:
                path = _experiment_path(suite, key, env_name=env)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Manifest entry {env}/{key} has no config at {path}"
                    )
        if rtpt_experiment_name is not None and paths:
            _init_rtpt_for_paths(
                paths,
                experiment_name=rtpt_experiment_name,
                trial_steps=trial_steps,
            )
        for path in paths:
            try:
                _run_experiment(
                    path,
                    wandb_mode=wandb_mode,
                    trial_steps=trial_steps,
                    oom_retry=oom_retry,
                )
            except Exception:
                traceback.print_exc()
                raise
        return

    if suite == "q_stickiness":
        if 1 in phases:
            path = _experiment_path(suite, QSTICK_PHASE1)
            print(f"\n>>> Running {suite}/{path.name}")
            _run_experiment(
                path,
                wandb_mode=wandb_mode,
                trial_steps=trial_steps,
                oom_retry=oom_retry,
            )
        if 2 in phases:
            keys = variants or QSTICK_PHASE2_KEYS
            for key in keys:
                if key == QSTICK_PHASE1:
                    continue
                path = _experiment_path(suite, key)
                print(f"\n>>> Running {suite}/{path.name}")
                _run_experiment(
                    path,
                    wandb_mode=wandb_mode,
                    trial_steps=trial_steps,
                    oom_retry=oom_retry,
                )
        return

    raise ValueError(f"Unknown suite={suite!r}")


def _resolve_wandb_mode(*, trial: bool, wandb_mode: str | None) -> str:
    if wandb_mode is not None:
        return wandb_mode
    return "disabled" if trial else "online"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["input_ablation", "q_stickiness"],
        help="Which ablation suite to run",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        help="Subset of variant keys (default: full suite)",
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="*",
        default=[1, 2],
        help="For q_stickiness only: 1=skills pretrain, 2=meta tandem variants",
    )
    parser.add_argument(
        "--env",
        choices=list(INPUT_ABLATION_ENVS),
        help="For input_ablation only: run configs for one environment (default: all present)",
    )
    parser.add_argument(
        "--manifest",
        nargs="?",
        const=str(DEFAULT_INPUT_MANIFEST.relative_to(INPUT_DIR)),
        default=None,
        metavar="YAML",
        help=(
            "For input_ablation: YAML list of env/variant runs "
            f"(default file: {DEFAULT_INPUT_MANIFEST.name})"
        ),
    )
    parser.add_argument(
        "--manifest-shard",
        metavar="INDEX/TOTAL",
        help=(
            "With --manifest: run every TOTAL-th entry starting at INDEX "
            "(e.g. 1/2 and 2/2 split across two GPUs)"
        ),
    )
    parser.add_argument("--list-matrix", action="store_true", help="Print run names and tags")
    parser.add_argument("--trial", action="store_true", help="Short smoke run; wandb disabled by default")
    parser.add_argument("--trial-steps", type=int, default=32768)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument(
        "--oom-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On GPU OOM, retry with lower NUM_ENVS / adjusted NUM_STEPS (default: on)",
    )
    args = parser.parse_args()

    manifest_entries: list[tuple[str, str]] | None = None
    if args.manifest is not None:
        if args.suite != "input_ablation":
            parser.error("--manifest is only supported with --suite input_ablation")
        manifest_path = _resolve_manifest_path(args.manifest)
        if manifest_path is None or not manifest_path.is_file():
            parser.error(f"Manifest not found: {args.manifest!r}")
        manifest_entries = _load_input_manifest(manifest_path)
        if args.variants:
            variant_set = set(args.variants)
            manifest_entries = [
                (env, key) for env, key in manifest_entries if key in variant_set
            ]
            if not manifest_entries:
                parser.error(
                    f"No manifest runs match --variants {args.variants!r} in "
                    f"{manifest_path.name}"
                )
        if args.env is not None:
            manifest_entries = [
                (env, key) for env, key in manifest_entries if env == args.env
            ]
            if not manifest_entries:
                parser.error(
                    f"No manifest runs for env={args.env!r} in {manifest_path.name}"
                )
        if args.manifest_shard is not None:
            shard_idx, num_shards = _parse_manifest_shard(args.manifest_shard)
            manifest_entries = _shard_manifest_entries(
                manifest_entries, shard_idx=shard_idx, num_shards=num_shards
            )
            if not manifest_entries:
                parser.error(
                    f"Manifest shard {args.manifest_shard!r} matched no runs in "
                    f"{manifest_path.name}"
                )
            print(
                f"\n>>> Manifest shard {shard_idx + 1}/{num_shards}: "
                f"{len(manifest_entries)} run(s)"
            )
    elif args.manifest_shard is not None:
        parser.error("--manifest-shard requires --manifest")

    if args.list_matrix:
        print_matrix(args.suite, manifest_entries=manifest_entries)
        return

    if args.suite is None:
        parser.error("--suite is required unless --list-matrix is set")

    wandb_mode = _resolve_wandb_mode(trial=args.trial, wandb_mode=args.wandb_mode)
    trial_steps = args.trial_steps if args.trial else None
    rtpt_experiment_name = (
        _input_ablation_rtpt_name(args.manifest_shard)
        if args.suite == "input_ablation"
        else None
    )
    if rtpt_experiment_name is not None:
        print(f"RTPT experiment name: {rtpt_experiment_name}")
    run_suite(
        args.suite,
        variants=args.variants,
        phases=list(args.phases),
        wandb_mode=wandb_mode,
        trial_steps=trial_steps,
        env_name=args.env,
        manifest_entries=manifest_entries,
        oom_retry=args.oom_retry,
        rtpt_experiment_name=rtpt_experiment_name,
    )


if __name__ == "__main__":
    main()
