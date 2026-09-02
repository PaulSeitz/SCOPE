#!/usr/bin/env python3
"""Materialize facing-frame ablation configs (clip / angle encoding sweeps).

Arms (each: Seaquest train → eval_full → Pong + Phoenix zero-shot):
  - clip20 + φ-bins8
  - noclip + φ-bins8
  - clip50 + sin/cos(φ)
  - clip50 + rounded degree scalar φ

Usage:
    python scripts/materialize_pov_facing_ablation_schedule.py
    python scripts/materialize_pov_facing_ablation_schedule.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "pov_facing_ablation"

_TRAIN_ALG = """
  NUM_SKILLS: 4
  SKILL_NAMES:
  - DESTROY
  - COLLECT
  - AVOID
  - NAVIGATE
  MAX_OBJECTS: 35
  EGO_ACTION_DIM: 18
  HIDDEN_SIZE: 128
  NUM_LAYERS: 3
  SKILL_EMBED_DIM: 256
  SKILL_LR: 0.0005
  SKILL_LR_FINISH: 0.0001
  SKILL_LR_DECAY: 0.5
  SKILL_REWARD_LAMBDA: 0.0
  CLIP_REWARD: true
  RESTRICT_FIRE_FOR_NON_DESTROY: false
  FIRST_FIRE: true
  STICKY_ACTIONS: false
  EPISODIC_LIFE: true
  FRAME_STACK_SIZE: 4
  FRAME_SKIP: 4
  TOTAL_TIMESTEPS: 24985600
  NUM_ENVS: 128
  NUM_STEPS: 128
  NUM_EPOCHS: 4
  NUM_MINIBATCHES: 64
  GAMMA: 0.99
  SKILL_GAMMA: 0.99
  LAMBDA: 0.65
  SKILL_LAMBDA: 0.65
  LR: 0.0001
  MAX_GRAD_NORM: 0.8
  OFF_POLICY: true
  RELABEL_ALL_OBJECTS: false
  RELABEL_K_OBJECTS: null
  SKILL_BOOTSTRAP: q_lambda
  EPS_START: 0.2
  EPS_FINISH: 0.01
  EPS_DECAY: 0.2
  HEURISTIC_MODE: env_specific
  SKILL_INPUT_MODE: ego
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
"""

ARMS: tuple[dict, ...] = (
    {
        "suffix": "clip20_facing_psibins8_v2",
        "clip": 20,
        "psi_bins": 8,
        "psi_scalar": False,
        "ego_tag": "phi_bins8",
        "probe": "pov_clip20_facing_psibins8_full",
        "extra_wandb": [],
    },
    {
        "suffix": "noclip_facing_psibins8_v2",
        "clip": None,
        "psi_bins": 8,
        "psi_scalar": False,
        "ego_tag": "phi_bins8",
        "probe": "pov_noclip_facing_psibins8_full",
        "extra_wandb": ["- ego:noclip"],
    },
    {
        "suffix": "clip50_facing_sincos_v2",
        "clip": 50,
        "psi_bins": None,
        "psi_scalar": False,
        "ego_tag": "phi_sincos",
        "probe": "pov_clip50_facing_sincos_full",
        "extra_wandb": [],
    },
    {
        "suffix": "clip50_facing_phideg_v2",
        "clip": 50,
        "psi_bins": None,
        "psi_scalar": True,
        "ego_tag": "phi_deg_scalar",
        "probe": "pov_clip50_facing_phideg_full",
        "extra_wandb": [],
    },
)


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)} ({len(text)} bytes)")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _clip_line(clip: int | None) -> str:
    return "null" if clip is None else str(clip)


def _psi_lines(*, psi_bins: int | None, psi_scalar: bool) -> tuple[str, str]:
    psi_line = "null" if psi_bins is None else str(psi_bins)
    scalar_line = "true" if psi_scalar else "false"
    return psi_line, scalar_line


def _seaquest_train(arm: dict) -> str:
    stem = f"seaquest_skills_ego_pov_{arm['suffix']}"
    clip_line = _clip_line(arm["clip"])
    psi_line, scalar_line = _psi_lines(
        psi_bins=arm["psi_bins"], psi_scalar=arm["psi_scalar"]
    )
    extra = "\n".join(arm["extra_wandb"])
    extra_block = f"\n{extra}" if extra else ""
    return f"""# Facing FPS ablation — {arm['suffix']}.
PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: seaquest
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: {clip_line}
EGO_ANGLE_BINS: null
EGO_PSI_ANGLE_BINS: {psi_line}
EGO_PSI_ANGLE_SCALAR: {scalar_line}
GAME_MODS: null
EVAL_GAME_MODS: null
ENV_NAME: seaquest
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_{arm['suffix']}
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- train_mode:skills_only
- skill_input:ego
- mapper:pov_facing
- ego:{arm['ego_tag']}
- ego:blocking
- ego:weapon
- ablation:collect_death
- destroy:contact_fail
- destroy:living_penalty{extra_block}
SKILL_INPUT_MODE: ego
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
VIDEO_AT_END: false
VIDEO_MAX_FRAMES: 600
VIDEO_FRAME_STRIDE: 4
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
USE_EGO_SHOT_FEATURES: false
alg:{_TRAIN_ALG}
"""


def _seaquest_eval(arm: dict) -> str:
    stem = f"seaquest_skills_ego_pov_{arm['suffix']}"
    clip_line = _clip_line(arm["clip"])
    psi_line, scalar_line = _psi_lines(
        psi_bins=arm["psi_bins"], psi_scalar=arm["psi_scalar"]
    )
    return f"""# Full Seaquest mod eval — {arm['suffix']} @ 27k.
PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {stem}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{stem}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: seaquest
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_{arm['suffix'].replace('_v2', '')}_full_v2
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- phase:mod_eval
- skill_input:ego
- mapper:pov_facing
- ego:{arm['ego_tag']}
- probe:{arm['probe']}
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: {clip_line}
EGO_PSI_ANGLE_BINS: {psi_line}
EGO_PSI_ANGLE_SCALAR: {scalar_line}
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
"""


def _transfer(
    *,
    env: str,
    arm: dict,
    max_objects: int,
    heuristic: str,
) -> str:
    train_stem = f"seaquest_skills_ego_pov_{arm['suffix']}"
    stem = f"{env}_transfer_ego_pov_{arm['suffix']}"
    clip_line = _clip_line(arm["clip"])
    psi_line, scalar_line = _psi_lines(
        psi_bins=arm["psi_bins"], psi_scalar=arm["psi_scalar"]
    )
    abbr = {"pong": "po", "phoenix": "ph"}[env]
    return f"""# {env.title()} zero-shot — {train_stem}.
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: transfer
PAPER_EVAL_MODS: {env}
PAPER_EVAL_MOD_KEYS:
- base
PAPER_EVAL_REQUIRES_RQ1:
- {train_stem}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{train_stem}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {train_stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: {env}
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {abbr}_paper_rq2_transfer_ego_pov_{arm['suffix']}
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:zero_shot
- skill_input:ego
- mapper:pov_facing
- ego:{arm['ego_tag']}
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: {clip_line}
EGO_PSI_ANGLE_BINS: {psi_line}
EGO_PSI_ANGLE_SCALAR: {scalar_line}
EGO_LEGAL_ACTION_MASK: true
GAME_MODS: null
EVAL_GAME_MODS: null
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: {heuristic}
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
VIDEO_AT_END: false
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
alg:
  NUM_SKILLS: 4
  SKILL_NAMES:
  - DESTROY
  - COLLECT
  - AVOID
  - NAVIGATE
  MAX_OBJECTS: {max_objects}
  EGO_ACTION_DIM: 18
  HIDDEN_SIZE: 128
  NUM_LAYERS: 3
  SKILL_EMBED_DIM: 256
  FRAME_SKIP: 4
  FRAME_STACK_SIZE: 4
  HEURISTIC_MODE: {heuristic}
  SKILL_INPUT_MODE: ego
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
"""


def _all_stems() -> list[str]:
    stems: list[str] = []
    for arm in ARMS:
        train = f"seaquest_skills_ego_pov_{arm['suffix']}"
        stems.extend(
            [
                train,
                f"{train}_eval_full",
                f"pong_transfer_ego_pov_{arm['suffix']}",
                f"phoenix_transfer_ego_pov_{arm['suffix']}",
            ]
        )
    return stems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = bool(args.dry_run)

    files: list[tuple[Path, str]] = []
    for arm in ARMS:
        train = f"seaquest_skills_ego_pov_{arm['suffix']}"
        files.append((_RQ1 / f"{train}.yaml", _seaquest_train(arm)))
        files.append((_RQ1 / f"{train}_eval_full.yaml", _seaquest_eval(arm)))
        files.append(
            (
                _RQ2 / f"pong_transfer_ego_pov_{arm['suffix']}.yaml",
                _transfer(env="pong", arm=arm, max_objects=6, heuristic="nearest"),
            )
        )
        files.append(
            (
                _RQ2 / f"phoenix_transfer_ego_pov_{arm['suffix']}.yaml",
                _transfer(env="phoenix", arm=arm, max_objects=21, heuristic="env_specific"),
            )
        )

    for path, text in files:
        _write(path, text, dry=dry)

    print(f"\nSuite `{SUITE}` stems ({len(_all_stems())}):")
    for s in _all_stems():
        print(f"  - {s}")


if __name__ == "__main__":
    main()
