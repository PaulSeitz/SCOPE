"""
Sequential multi-environment evaluation runner.

Runs three phases across 11 Atari environments in a single Python process:
  Phase 1: Skill pretrain on Seaquest (from scratch)
  Phase 2: Meta-only training per env (frozen Phase-1 skills)
  Phase 3: Tandem finetune per env (low SKILL_LR, skills re-init from Phase-1,
           meta randomly initialized — fresh start per env)

RTPT is initialized once before Phase 1 with the total update count across all
phases, so nvidia-smi shows a single continuous ETA for the entire run.

wandb: each run gets its own entry tagged with the phase label:
  skill_train  — Phase 1 (Seaquest skill pretrain)
  frozen_meta  — Phase 2 (frozen skills, meta learns)
  finetune     — Phase 3 (joint meta + skill, low skill LR)

Usage (from the project root):
  python scripts/run_sequential_eval.py
  python scripts/run_sequential_eval.py --phases 1 2      # run only phases 1 and 2
  python scripts/run_sequential_eval.py --phases 2 3 --skip-envs skiing tennis
  python scripts/run_sequential_eval.py --trial           # 10k steps per phase (smoke test)
  python scripts/run_sequential_eval.py --abort-on-failure
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import main.single_run without hydra.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import main as train_main  # noqa: E402  (after sys.path setup)
from concept_based_learning.training.rtpt_progress import (  # noqa: E402
    num_updates_from_timesteps,
    start_rtpt,
)

# ---------------------------------------------------------------------------
# Environment sequence for Phase 2 & 3
# ---------------------------------------------------------------------------
PHASE2_3_ENVS = [
    "seaquest",
    "pong",
    "breakout",
    "spaceinvaders",
    "phoenix",
    "freeway",
    "choppercommand",
    "beamrider",
    "kangaroo",
    "skiing",
    "tennis",
]

CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "sequential_eval"

SEED = 44
WANDB_ENTITY = "paul-seitz-tu-darmstadt"

# Minimum steps that produce ≥1 update for any config in this run.
# Phase 1:   NUM_ENVS=512, NUM_STEPS=32 → min=16384
# Phase 2/3: NUM_ENVS=512, NUM_STEPS=64 → min=32768
TRIAL_STEPS_DEFAULT = 32768

# wandb tags per phase
PHASE_TAGS = {
    1: "skill_train",
    2: "frozen_meta",
    3: "finetune",
}

# ---------------------------------------------------------------------------
# Checkpoint path helpers — must match main.py naming convention:
#   skills_only  → {save}/{env}/hierarchical_pqn_{env}_seed{N}.safetensors
#   meta_only    → {save}/{env}/hierarchical_pqn_{env}_seed{N}_meta.safetensors
#                  (skills also saved as …_skills.safetensors alongside)
#   tandem       → same split as meta_only
# ---------------------------------------------------------------------------

def _ckpt_dir(save_path: str, env: str) -> Path:
    return Path(save_path) / env


def _skills_ckpt(save_path: str, env: str, seed: int = SEED) -> Path:
    """skills_only output (no meta head saved alongside)."""
    return _ckpt_dir(save_path, env) / f"hierarchical_pqn_{env}_seed{seed}.safetensors"



# ---------------------------------------------------------------------------
# Config loading — flattens the nested alg: block just like main.single_run
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    alg = raw.pop("alg", {})
    raw.update(alg)
    raw["SEED"] = SEED
    return raw


def _load_config(env: str, phase_suffix: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{env}_{phase_suffix}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return _load_yaml(path)


# ---------------------------------------------------------------------------
# RTPT total-update pre-computation
# ---------------------------------------------------------------------------

def _compute_total_updates(
    phases: list[int],
    envs: list[str],
    trial_steps: int | None,
) -> int:
    suffix_map = {1: "phase1_skills", 2: "phase2_meta", 3: "phase3_tandem"}
    total = 0
    for phase in phases:
        env_list = ["seaquest"] if phase == 1 else envs
        for env in env_list:
            try:
                cfg = _load_config(env, suffix_map[phase])
                ts = trial_steps if trial_steps else int(cfg.get("TOTAL_TIMESTEPS", 0))
                total += num_updates_from_timesteps(
                    ts,
                    num_steps=int(cfg.get("NUM_STEPS", 32)),
                    num_envs=int(cfg.get("NUM_ENVS", 512)),
                )
            except FileNotFoundError:
                pass
    return max(1, total)


# ---------------------------------------------------------------------------
# Single run wrapper
# ---------------------------------------------------------------------------

def _run(cfg: dict[str, Any], label: str, abort_on_failure: bool) -> bool:
    """Calls single_run; returns True on success, False (or raises) on failure."""
    print(f"\n{'='*70}")
    print(f"  START: {label}")
    print(f"  ENV={cfg.get('ENV_NAME')}  MODE={cfg.get('TRAIN_MODE')}  "
          f"STEPS={cfg.get('TOTAL_TIMESTEPS', 0):,}  "
          f"INIT={cfg.get('INIT_MODEL_PATH', 'scratch')}")
    print(f"{'='*70}\n")
    try:
        train_main.single_run(cfg)
        print(f"\n  DONE: {label}\n")
        return True
    except Exception:
        print(f"\n{'!'*70}")
        print(f"  FAILED: {label}")
        traceback.print_exc()
        print(f"{'!'*70}\n")
        if abort_on_failure:
            raise
        return False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_sequential_eval(
    phases: list[int],
    envs: list[str],
    abort_on_failure: bool,
    trial_steps: int | None,
    wandb_mode: str = "online",
) -> None:
    results: list[tuple[str, bool]] = []

    # Initialize RTPT once for the entire run so nvidia-smi ETA is global.
    total_updates = _compute_total_updates(phases, envs, trial_steps)
    start_rtpt({
        "RTPT_ENABLED": True,
        "RTPT_NAME_INITIALS": "PS",
        "RTPT_EXPERIMENT_NAME": "HRL_fulleval",
        "RTPT_STEP_EVERY": 10,
        # start_rtpt computes max_iterations from TOTAL_TIMESTEPS/NUM_STEPS/NUM_ENVS;
        # pass total_updates directly by faking 1-step-per-env-step.
        "TOTAL_TIMESTEPS": total_updates,
        "NUM_STEPS": 1,
        "NUM_ENVS": 1,
    })
    print(f"RTPT initialized: {total_updates:,} total updates across all phases/envs.\n")

    # Resolve Phase-1 skill checkpoint path whether or not Phase 1 is in this run.
    # Both Phase 2 and Phase 3 need it (frozen skills init and finetune init).
    _p1_cfg = _load_config("seaquest", "phase1_skills")
    phase1_save: str = _p1_cfg["SAVE_PATH"]

    def _apply_common(cfg: dict, phase: int, run_name: str, trial_steps: int | None) -> dict:
        """Inject wandb settings and optional trial step override into a config."""
        if trial_steps:
            cfg["TOTAL_TIMESTEPS"] = trial_steps
        cfg["WANDB_MODE"] = wandb_mode
        cfg["ENTITY"] = WANDB_ENTITY
        cfg["PROJECT"] = "concept_based_learning"
        cfg["WANDB_TAGS"] = [PHASE_TAGS[phase], "sequential_eval"]
        cfg["WANDB_RUN_NAME"] = run_name
        # Video recording requires a display; disable on headless cluster nodes.
        cfg["RECORD_VIDEO"] = False
        cfg["VIDEO_AT_END"] = False
        return cfg

    def _wire_phase1_ckpt(cfg: dict) -> dict:
        ckpt = _skills_ckpt(phase1_save, "seaquest")
        if ckpt.exists():
            cfg["INIT_MODEL_PATH"] = str(ckpt)
        else:
            print(f"  [warn] Phase-1 checkpoint not found at {ckpt}; "
                  f"using config default: {cfg.get('INIT_MODEL_PATH')}")
        return cfg

    # -----------------------------------------------------------------------
    # Phase 1: Skill pretrain on Seaquest (from scratch)
    # -----------------------------------------------------------------------
    if 1 in phases:
        cfg = _apply_common(_p1_cfg, 1, "seaquest_skill_train", trial_steps)
        ok = _run(cfg, "Phase 1 — Seaquest skill pretrain", abort_on_failure)
        results.append(("Phase1/seaquest/skill_train", ok))

    # -----------------------------------------------------------------------
    # Phase 2: Frozen-skill meta training per env
    # -----------------------------------------------------------------------
    if 2 in phases:
        for env in envs:
            cfg = _load_config(env, "phase2_meta")
            _wire_phase1_ckpt(cfg)
            _apply_common(cfg, 2, f"{env}_frozen_meta", trial_steps)
            ok = _run(cfg, f"Phase 2 — {env} frozen_meta", abort_on_failure)
            results.append((f"Phase2/{env}/frozen_meta", ok))

    # -----------------------------------------------------------------------
    # Phase 3: Joint finetune — skills from Phase-1 Seaquest, meta fresh.
    # -----------------------------------------------------------------------
    if 3 in phases:
        for env in envs:
            cfg = _load_config(env, "phase3_tandem")
            _wire_phase1_ckpt(cfg)
            _apply_common(cfg, 3, f"{env}_finetune", trial_steps)
            ok = _run(cfg, f"Phase 3 — {env} finetune (low SKILL_LR)", abort_on_failure)
            results.append((f"Phase3/{env}/finetune", ok))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("  SEQUENTIAL EVAL COMPLETE")
    print("="*70)
    passed = sum(1 for _, ok in results if ok)
    print(f"  {passed}/{len(results)} runs succeeded\n")
    for label, ok in results:
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {label}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases", nargs="+", type=int, default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which phases to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--envs", nargs="+", default=PHASE2_3_ENVS,
        help="Environments for Phase 2/3 (default: all 11)",
    )
    parser.add_argument(
        "--skip-envs", nargs="+", default=[],
        dest="skip_envs",
        help="Environments to skip from Phase 2/3",
    )
    parser.add_argument(
        "--abort-on-failure", action="store_true",
        help="Stop the whole run if any single training phase fails",
    )
    parser.add_argument(
        "--trial", action="store_true",
        help=f"Smoke-test mode: 1 update per env (= {TRIAL_STEPS_DEFAULT} steps, overrides TOTAL_TIMESTEPS)",
    )
    parser.add_argument(
        "--trial-steps", type=int, default=TRIAL_STEPS_DEFAULT,
        dest="trial_steps",
        help=f"Steps per phase in trial mode (default: {TRIAL_STEPS_DEFAULT})",
    )
    parser.add_argument(
        "--wandb-mode", default="online", choices=["online", "offline", "disabled"],
        dest="wandb_mode",
        help="wandb logging mode (default: online)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    envs = [e for e in args.envs if e not in args.skip_envs]
    run_sequential_eval(
        phases=sorted(set(args.phases)),
        envs=envs,
        abort_on_failure=args.abort_on_failure,
        trial_steps=args.trial_steps if args.trial else None,
        wandb_mode=args.wandb_mode,
    )
