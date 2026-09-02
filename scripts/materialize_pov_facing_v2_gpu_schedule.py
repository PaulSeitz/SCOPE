#!/usr/bin/env python3
"""Materialize facing-v2 transfer / from-scratch / noclip-xy configs for the 3-GPU schedule.

GPU 0: φ-bins8 zero-shot BR+PH → Phoenix from-scratch facing φ-bins8
GPU 1: φ-bins8+mid zero-shot BR+PH → Beamrider from-scratch facing φ-bins8
GPU 2: Seaquest noclip facing_xy train → SQ mods + Pong/PH/BR zero-shot

Usage:
    python scripts/materialize_pov_facing_v2_gpu_schedule.py
    python scripts/materialize_pov_facing_v2_gpu_schedule.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "pov_facing_v2"


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)} ({len(text)} bytes)")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _transfer(
    *,
    env: str,
    train_stem: str,
    ego_tag: str,
    pose_xy: bool,
    clip: int | None,
    psi_bins: int | None,
    mid: bool,
    max_objects: int,
    heuristic: str,
    extra_top: str = "",
    extra_alg: str = "",
) -> str:
    stem = f"{env}_transfer_ego_" + train_stem.removeprefix("seaquest_skills_ego_")
    clip_line = "null" if clip is None else str(clip)
    psi_line = "null" if psi_bins is None else str(psi_bins)
    mid_tag = "\n- heuristic:collect_midfield" if mid else ""
    mid_margin = "\nCOLLECT_EDGE_MARGIN_PX: 24.0" if mid else ""
    mid_alg = "\n  COLLECT_EDGE_MARGIN_PX: 24.0" if mid else ""
    abbr = {"pong": "po", "phoenix": "ph", "beamrider": "br"}[env]
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
EGO_POSE_XY: {"true" if pose_xy else "false"}
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {abbr}_paper_rq2_transfer_{stem.removeprefix(env + "_transfer_")}
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:zero_shot
- skill_input:ego
- mapper:pov_facing
- ego:{ego_tag}{mid_tag}
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: {clip_line}
EGO_PSI_ANGLE_BINS: {psi_line}{mid_margin}
EGO_LEGAL_ACTION_MASK: true
{extra_top}GAME_MODS: null
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
  USE_EGO_SHOT_FEATURES: false{mid_alg}{extra_alg}
"""


def _from_scratch(*, env: str, max_objects: int, extra_top: str = "") -> str:
    stem = f"{env}_from_scratch_ego_pov_facing_psibins8"
    abbr = {"phoenix": "ph", "beamrider": "be"}[env]
    return f"""# Native {env} skills ceiling — facing FPS φ-bins8 + clip50 (~25M).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: {env}
PAPER_EVAL_REQUIRES_RQ1: []
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: {env}
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {abbr}_paper_rq2_scratch_ego_pov_facing_psibins8
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:from_scratch
- skill_input:ego
- mapper:pov_facing
- ego:phi_bins8
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
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
VIDEO_AT_END: false
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
{extra_top}alg:
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


def _seaquest_noclip_xy_train() -> str:
    stem = "seaquest_skills_ego_pov_noclip_facing_xy_v2"
    return f"""# Facing FPS xy + noclip (post-facing-fix v2 retrain).
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
EGO_POSE_XY: true
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: null
EGO_ANGLE_BINS: null
EGO_PSI_ANGLE_BINS: null
GAME_MODS: null
EVAL_GAME_MODS: null
ENV_NAME: seaquest
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_noclip_facing_xy_v2
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- train_mode:skills_only
- skill_input:ego
- mapper:pov_facing
- ego:pose_xy
- ego:noclip
- ego:blocking
- ego:weapon
- ablation:collect_death
- destroy:contact_fail
- destroy:living_penalty
SKILL_INPUT_MODE: ego
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
VIDEO_AT_END: false
VIDEO_MAX_FRAMES: 600
VIDEO_FRAME_STRIDE: 4
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
USE_EGO_SHOT_FEATURES: false
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
"""


def _seaquest_noclip_xy_eval() -> str:
    stem = "seaquest_skills_ego_pov_noclip_facing_xy_v2"
    return f"""# Full Seaquest mod eval — noclip facing xy v2 @ 27k.
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
EGO_POSE_XY: true
SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_noclip_facing_xy_full_v2
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- phase:mod_eval
- skill_input:ego
- mapper:pov_facing
- ego:pose_xy
- ego:noclip
- probe:pov_noclip_facing_xy_full
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: null
EGO_PSI_ANGLE_BINS: null
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = bool(args.dry_run)

    files: list[tuple[Path, str]] = []

    # --- GPU0 / GPU1 zero-shots ---
    for mid, train in [
        (False, "seaquest_skills_ego_pov_clip50_facing_psibins8_v2"),
        (True, "seaquest_skills_ego_pov_clip50_facing_psibins8_mid_v2"),
    ]:
        for env, max_obj, heur, extra in [
            ("beamrider", 31, "env_specific", "NO_TARGET_GAME_ACTION: 1\n"),
            ("phoenix", 21, "env_specific", ""),
        ]:
            text = _transfer(
                env=env,
                train_stem=train,
                ego_tag="phi_bins8",
                pose_xy=False,
                clip=50,
                psi_bins=8,
                mid=mid,
                max_objects=max_obj,
                heuristic=heur,
                extra_top=extra,
            )
            stem = f"{env}_transfer_ego_" + train.removeprefix("seaquest_skills_ego_")
            files.append((_RQ2 / f"{stem}.yaml", text))

    # --- from-scratch ceilings ---
    files.append(
        (
            _RQ2 / "phoenix_from_scratch_ego_pov_facing_psibins8.yaml",
            _from_scratch(env="phoenix", max_objects=21),
        )
    )
    files.append(
        (
            _RQ2 / "beamrider_from_scratch_ego_pov_facing_psibins8.yaml",
            _from_scratch(
                env="beamrider",
                max_objects=31,
                extra_top="NO_TARGET_GAME_ACTION: 1\n",
            ),
        )
    )

    # --- GPU2 noclip xy ---
    train = "seaquest_skills_ego_pov_noclip_facing_xy_v2"
    files.append((_RQ1 / f"{train}.yaml", _seaquest_noclip_xy_train()))
    files.append((_RQ1 / f"{train}_eval_full.yaml", _seaquest_noclip_xy_eval()))
    for env, max_obj, heur, extra in [
        ("pong", 2, "nearest", ""),
        ("phoenix", 21, "env_specific", ""),
        ("beamrider", 31, "env_specific", "NO_TARGET_GAME_ACTION: 1\n"),
    ]:
        text = _transfer(
            env=env,
            train_stem=train,
            ego_tag="pose_xy",
            pose_xy=True,
            clip=None,
            psi_bins=None,
            mid=False,
            max_objects=max_obj,
            heuristic=heur,
            extra_top=extra,
        )
        stem = f"{env}_transfer_ego_" + train.removeprefix("seaquest_skills_ego_")
        files.append((_RQ2 / f"{stem}.yaml", text))

    for path, text in files:
        _write(path, text, dry=dry)

    print(f"\n{len(files)} config(s)")


if __name__ == "__main__":
    main()
