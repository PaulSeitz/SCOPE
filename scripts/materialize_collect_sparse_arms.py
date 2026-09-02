#!/usr/bin/env python3
"""Materialize Arm1/Arm2 sparse-reward facing-v2 configs + register suite stems.

Arm 1: COLLECT_SPARSE_ONLY (keep COLLECT death + DESTROY living/contact).
Arm 2: Arm1 + DESTROY dense off (no living tax / contact-fail).

Usage:
    python scripts/materialize_collect_sparse_arms.py
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "collect_sparse_arms"
BASE = "seaquest_skills_ego_pov_clip50_facing_psibins8_v2"

ARMS = {
    "arm1": {
        "train": "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_collect_sparse",
        "eval": "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_collect_sparse_eval_full",
        "pong": "pong_transfer_ego_pov_clip50_facing_psibins8_v2_collect_sparse",
        "tag": "collect:sparse_only",
        "probe": "collect_sparse_arm1",
        "destroy_contact": True,
        "destroy_living": True,
        "comment": (
            "Arm1: COLLECT sparse only (+1 contact / death). "
            "DESTROY keeps contact-fail + living tax. Facing φ8 clip50 v2 spine."
        ),
    },
    "arm2": {
        "train": "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_collect_sparse_destroy_sparse",
        "eval": "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_collect_sparse_destroy_sparse_eval_full",
        "pong": "pong_transfer_ego_pov_clip50_facing_psibins8_v2_collect_sparse_destroy_sparse",
        "tag": "destroy:sparse_only",
        "probe": "collect_sparse_arm2",
        "destroy_contact": False,
        "destroy_living": False,
        "comment": (
            "Arm2: COLLECT sparse + DESTROY sparse (kill +1 only; no living tax / "
            "contact-fail). COLLECT death penalty kept. Facing φ8 clip50 v2 spine."
        ),
    },
}


def _train_yaml(arm: dict) -> str:
    stem = arm["train"]
    dc = "true" if arm["destroy_contact"] else "false"
    dl = "true" if arm["destroy_living"] else "false"
    destroy_tags = []
    if arm["destroy_contact"]:
        destroy_tags.append("- destroy:contact_fail")
    if arm["destroy_living"]:
        destroy_tags.append("- destroy:living_penalty")
    else:
        destroy_tags.append("- destroy:sparse_only")
    destroy_tags_s = "\n".join(destroy_tags)
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
EGO_DISTANCE_CLIP_MAX_PX: 50
EGO_ANGLE_BINS: null
EGO_PSI_ANGLE_BINS: 8
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
- ego:phi_bins8
- ego:blocking
- ego:weapon
- ablation:collect_death
- {arm["tag"]}
- probe:{arm["probe"]}
{destroy_tags_s}
SKILL_INPUT_MODE: ego
SKILL_DEATH_PENALIZE_COLLECT: true
COLLECT_SPARSE_ONLY: true
COLLECT_LIVING_PENALTY: false
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
  COLLECT_SPARSE_ONLY: true
  COLLECT_LIVING_PENALTY: false
  DESTROY_CONTACT_FAIL: {dc}
  DESTROY_LIVING_PENALTY: {dl}
  USE_EGO_SHOT_FEATURES: false
DESTROY_CONTACT_FAIL: {dc}
DESTROY_LIVING_PENALTY: {dl}
USE_EGO_SHOT_FEATURES: false
"""


def _eval_yaml(arm: dict) -> str:
    stem = arm["train"]
    eval_stem = arm["eval"]
    dc = "true" if arm["destroy_contact"] else "false"
    dl = "true" if arm["destroy_living"] else "false"
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
- ego:phi_bins8
- probe:{arm["probe"]}_full
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
COLLECT_SPARSE_ONLY: true
DESTROY_CONTACT_FAIL: {dc}
DESTROY_LIVING_PENALTY: {dl}
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: 50
EGO_PSI_ANGLE_BINS: 8
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
"""


def _pong_yaml(arm: dict) -> str:
    stem = arm["train"]
    pong = arm["pong"]
    dc = "true" if arm["destroy_contact"] else "false"
    dl = "true" if arm["destroy_living"] else "false"
    return f"""# Pong zero-shot — {arm["probe"]} (Seaquest-trained).
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
- ego:phi_bins8
- {arm["tag"]}
- probe:{arm["probe"]}
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: 50
EGO_PSI_ANGLE_BINS: 8
EGO_LEGAL_ACTION_MASK: false
GAME_MODS: null
EVAL_GAME_MODS: null
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: nearest
SKILL_DEATH_PENALIZE_COLLECT: true
COLLECT_SPARSE_ONLY: true
RECORD_VIDEO: false
VIDEO_AT_END: false
DESTROY_CONTACT_FAIL: {dc}
DESTROY_LIVING_PENALTY: {dl}
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
  COLLECT_SPARSE_ONLY: true
  DESTROY_CONTACT_FAIL: {dc}
  DESTROY_LIVING_PENALTY: {dl}
  USE_EGO_SHOT_FEATURES: false
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
