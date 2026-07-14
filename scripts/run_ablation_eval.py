#!/usr/bin/env python3
"""
Run input-mode and Q-stickiness ablation suites.

Configs live under ``configs/experiment/input_ablation`` and
``configs/experiment/q_stickiness_eval`` with ``WANDB_TAGS`` / ``WANDB_RUN_NAME``
embedded for W&B filtering.

Checkpoints are written under each YAML's ``SAVE_PATH`` (see ``main.single_run``).
W&B uses the same ``PROJECT`` / ``ENTITY`` as normal training runs; ablations are
distinguished by ``WANDB_TAGS`` (e.g. ``suite:input_ablation``) and
``WANDB_RUN_NAME``. On GPU OOM, ``--oom-retry`` (default) retries with lower
``NUM_ENVS`` / higher ``NUM_STEPS`` while keeping the update count fixed when
possible, tagging the run with ``oom_fallback:on``.

Usage (from project root):
  python scripts/run_ablation_eval.py --list-matrix
  python scripts/run_ablation_eval.py --suite input_ablation
  python scripts/run_ablation_eval.py --suite q_stickiness
  python scripts/run_ablation_eval.py --suite q_stickiness --phases 1
  python scripts/run_ablation_eval.py --suite input_ablation --variants mi_flat_si_ego --trial
  python scripts/materialize_ablation_configs.py   # regenerate YAMLs after editing manifest
"""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as train_main  # noqa: E402

INPUT_DIR = PROJECT_ROOT / "configs" / "experiment" / "input_ablation"
QSTICK_DIR = PROJECT_ROOT / "configs" / "experiment" / "q_stickiness_eval"

INPUT_ABLATION_ENVS = ("seaquest", "spaceinvaders")
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
)
_MIN_NUM_ENVS = 128
_MIN_NUM_STEPS = 32
_MAX_NUM_STEPS = 128


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def _list_rows(suite: str, *, env_name: str | None = None) -> list[tuple[str, str, list[str]]]:
    rows: list[tuple[str, str, list[str]]] = []
    if suite == "input_ablation":
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


def print_matrix(suite: str | None = None) -> None:
    suites = ["input_ablation", "q_stickiness"] if suite is None else [suite]
    for s in suites:
        print(f"\n=== {s} ===")
        print(f"{'Variant':<34} {'WANDB_RUN_NAME':<24} WANDB_TAGS")
        print("-" * 100)
        for key, run_name, tags in _list_rows(s):
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
        base_cfg, wandb_mode=wandb_mode, trial_steps=trial_steps
    )

    tier_cfgs = [base_cfg]
    if oom_retry and trial_steps is None:
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
            return
        except Exception as exc:
            last_exc = exc
            can_retry = tier_idx < len(tier_cfgs) - 1 and _is_oom_error(exc)
            if not can_retry:
                raise
            print(f"\n>>> OOM detected for {path.name}; trying lower-memory tier...")
            traceback.print_exc()
            _clear_jax_memory()

    if last_exc is not None:
        raise last_exc


def run_suite(
    suite: str,
    *,
    variants: list[str] | None,
    phases: list[int],
    wandb_mode: str | None,
    trial_steps: int | None,
    env_name: str | None = None,
    oom_retry: bool = True,
) -> None:
    if suite == "input_ablation":
        keys = variants or INPUT_VARIANT_KEYS
        envs = (env_name,) if env_name else INPUT_ABLATION_ENVS
        for env in envs:
            for key in keys:
                path = _experiment_path(suite, key, env_name=env)
                if not path.is_file():
                    print(f"\n>>> Skipping missing config {path.name}")
                    continue
                print(f"\n>>> Running {suite}/{path.name}")
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

    if args.list_matrix:
        print_matrix(args.suite)
        return

    if args.suite is None:
        parser.error("--suite is required unless --list-matrix is set")

    wandb_mode = _resolve_wandb_mode(trial=args.trial, wandb_mode=args.wandb_mode)
    trial_steps = args.trial_steps if args.trial else None
    run_suite(
        args.suite,
        variants=args.variants,
        phases=list(args.phases),
        wandb_mode=wandb_mode,
        trial_steps=trial_steps,
        env_name=args.env,
        oom_retry=args.oom_retry,
    )


if __name__ == "__main__":
    main()
