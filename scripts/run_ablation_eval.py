#!/usr/bin/env python3
"""
Run input-mode and Q-stickiness ablation suites.

Configs live under ``configs/experiment/input_ablation`` and
``configs/experiment/q_stickiness_eval`` with ``WANDB_TAGS`` / ``WANDB_RUN_NAME``
embedded for W&B filtering.

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


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_experiment_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    alg = raw.pop("alg", {})
    return {**raw, **alg}


def _experiment_path(suite: str, key: str) -> Path:
    if suite == "input_ablation":
        return INPUT_DIR / f"seaquest_{key}.yaml"
    if suite == "q_stickiness":
        if key == QSTICK_PHASE1:
            return QSTICK_DIR / f"{key}.yaml"
        return QSTICK_DIR / f"seaquest_{key}.yaml"
    raise ValueError(f"Unknown suite={suite!r}")


def _list_rows(suite: str) -> list[tuple[str, str, list[str]]]:
    rows: list[tuple[str, str, list[str]]] = []
    if suite == "input_ablation":
        for key in INPUT_VARIANT_KEYS:
            raw = _load_yaml(_experiment_path(suite, key))
            rows.append((key, str(raw.get("WANDB_RUN_NAME", key)), list(raw.get("WANDB_TAGS", []))))
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


def _run_experiment(path: Path, *, wandb_mode: str | None, trial_steps: int | None) -> None:
    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
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
    train_main.single_run(cfg)


def run_suite(
    suite: str,
    *,
    variants: list[str] | None,
    phases: list[int],
    wandb_mode: str | None,
    trial_steps: int | None,
) -> None:
    if suite == "input_ablation":
        keys = variants or INPUT_VARIANT_KEYS
        for key in keys:
            path = _experiment_path(suite, key)
            print(f"\n>>> Running {suite}/{path.name}")
            try:
                _run_experiment(path, wandb_mode=wandb_mode, trial_steps=trial_steps)
            except Exception:
                traceback.print_exc()
                raise
        return

    if suite == "q_stickiness":
        if 1 in phases:
            path = _experiment_path(suite, QSTICK_PHASE1)
            print(f"\n>>> Running {suite}/{path.name}")
            _run_experiment(path, wandb_mode=wandb_mode, trial_steps=trial_steps)
        if 2 in phases:
            keys = variants or QSTICK_PHASE2_KEYS
            for key in keys:
                if key == QSTICK_PHASE1:
                    continue
                path = _experiment_path(suite, key)
                print(f"\n>>> Running {suite}/{path.name}")
                _run_experiment(path, wandb_mode=wandb_mode, trial_steps=trial_steps)
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
    parser.add_argument("--list-matrix", action="store_true", help="Print run names and tags")
    parser.add_argument("--trial", action="store_true", help="Short smoke run; wandb disabled by default")
    parser.add_argument("--trial-steps", type=int, default=32768)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
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
    )


if __name__ == "__main__":
    main()
