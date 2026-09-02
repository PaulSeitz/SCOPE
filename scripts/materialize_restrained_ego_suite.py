#!/usr/bin/env python3
"""Materialize restrained-ego facing-v2 configs (Arm3: L/C/R × near/far).

Uses v2 reward spine (approach-shaped COLLECT, not sparse). Legal ego mask on
train + transfer for honest zero-shot eval.

Usage:
    python scripts/materialize_restrained_ego_suite.py
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "restrained_ego"
BASE_SUFFIX = "facing_lcr_nearfar_v2"

ARMS = {
    "lcr": {
        "train": f"seaquest_skills_ego_pov_{BASE_SUFFIX}",
        "eval": f"seaquest_skills_ego_pov_{BASE_SUFFIX}_eval_full",
        "pong": f"pong_transfer_ego_pov_{BASE_SUFFIX}",
        "restrained": "lcr_nearfar",
        "ego_tag": "lcr_nearfar",
        "probe": "restrained_lcr",
        "comment": (
            "Arm3: facing-frame restrained pose L/C/R × near/far (band=8px, far=50px). "
            "v2 reward spine + legal ego mask."
        ),
    },
}


def _train_yaml(arm: dict) -> str:
    stem = arm["train"]
    return f"""# {arm["comment"]}
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
EGO_RESTRAINED_POSE: {arm["restrained"]}
EGO_RESTRAINED_LATERAL_BAND_PX: 8
EGO_RESTRAINED_NEAR_FAR_PX: 50
EGO_DISTANCE_CLIP_MAX_PX: null
EGO_ANGLE_BINS: null
EGO_PSI_ANGLE_BINS: null
EGO_LEGAL_ACTION_MASK: true
GAME_MODS: null
EVAL_GAME_MODS: null
ENV_NAME: seaquest
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
WANDB_RUN_NAME: sq_paper_rq1_skills_{stem}
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- train_mode:skills_only
- skill_input:ego
- mapper:pov_facing
- ego:{arm["ego_tag"]}
- ego:blocking
- ego:weapon
- ego:legal_mask
- ablation:collect_death
- destroy:contact_fail
- destroy:living_penalty
- probe:{arm["probe"]}
SKILL_INPUT_MODE: ego
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
VIDEO_AT_END: false
VIDEO_MAX_FRAMES: 600
VIDEO_FRAME_STRIDE: 4
alg:
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
  EGO_LEGAL_ACTION_MASK: true
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
USE_EGO_SHOT_FEATURES: false
"""


def _eval_yaml(arm: dict) -> str:
    stem = arm["train"]
    eval_stem = arm["eval"]
    return f"""# Full Seaquest mod eval — {arm["probe"]} @ 27k.
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
WANDB_RUN_NAME: sq_paper_rq1_eval_{eval_stem}
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- phase:mod_eval
- skill_input:ego
- mapper:pov_facing
- ego:{arm["ego_tag"]}
- ego:legal_mask
- probe:{arm["probe"]}_full
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
USE_EGO_SHOT_FEATURES: false
EGO_RESTRAINED_POSE: {arm["restrained"]}
EGO_RESTRAINED_LATERAL_BAND_PX: 8
EGO_RESTRAINED_NEAR_FAR_PX: 50
EGO_DISTANCE_CLIP_MAX_PX: null
EGO_PSI_ANGLE_BINS: null
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
EGO_LEGAL_ACTION_MASK: true
"""


def _pong_yaml(arm: dict) -> str:
    stem = arm["train"]
    pong = arm["pong"]
    return f"""# Pong zero-shot — {arm["probe"]} (Seaquest-trained, legal ego mask).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: transfer
PAPER_EVAL_MODS: pong
PAPER_EVAL_MOD_KEYS:
- base
PAPER_EVAL_REQUIRES_RQ1:
- {stem}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{stem}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: pong
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
SAVE_PATH: ./outputs/models/paper_eval/rq2/{pong}
WANDB_RUN_NAME: po_paper_rq2_transfer_{pong}
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:pong
- phase:zero_shot
- skill_input:ego
- mapper:pov_facing
- ego:{arm["ego_tag"]}
- ego:legal_mask
- probe:{arm["probe"]}
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
EGO_RESTRAINED_POSE: {arm["restrained"]}
EGO_RESTRAINED_LATERAL_BAND_PX: 8
EGO_RESTRAINED_NEAR_FAR_PX: 50
EGO_DISTANCE_CLIP_MAX_PX: null
EGO_PSI_ANGLE_BINS: null
EGO_LEGAL_ACTION_MASK: true
GAME_MODS: null
EVAL_GAME_MODS: null
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: nearest
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
  MAX_OBJECTS: 2
  EGO_ACTION_DIM: 18
  HIDDEN_SIZE: 128
  NUM_LAYERS: 3
  SKILL_EMBED_DIM: 256
  FRAME_SKIP: 4
  FRAME_STACK_SIZE: 4
  HEURISTIC_MODE: nearest
  SKILL_INPUT_MODE: ego
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
  EGO_LEGAL_ACTION_MASK: true
"""


def main() -> None:
    for arm in ARMS.values():
        for path, text in (
            (_RQ1 / f"{arm['train']}.yaml", _train_yaml(arm)),
            (_RQ1 / f"{arm['eval']}.yaml", _eval_yaml(arm)),
            (_RQ2 / f"{arm['pong']}.yaml", _pong_yaml(arm)),
        ):
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(_ROOT)}")
    print("stems (sequential suite order):")
    for arm in ARMS.values():
        for k in ("train", "eval", "pong"):
            print(f"  {arm[k]}")


if __name__ == "__main__":
    main()
