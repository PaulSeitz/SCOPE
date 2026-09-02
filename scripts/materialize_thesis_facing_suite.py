#!/usr/bin/env python3
"""Materialize overnight thesis suite on facing φ8 + clip50.

Creates:
  - seaquest_skills_{pixel,oc}_pov trains + Seaquest mod evals
  - pixel no-outline eval (reuses pixel_pov ckpt)
  - ego / pixel_pov / oc_pov zero-shot transfers for all RQ2 envs

All three skill baselines use fresh stems. The ego architecture is copied from
the proven facing φ8/clip50 v2 arm, but must retrain because Seaquest's surface
waypoint is now COLLECT instead of NAVIGATE.

Usage:
    python scripts/materialize_thesis_facing_suite.py
    python scripts/materialize_thesis_facing_suite.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "thesis_facing"
EGO_SOURCE = "seaquest_skills_ego_pov_clip50_facing_psibins8_v2"
EGO_TRAIN = "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface"
PIXEL_TRAIN = "seaquest_skills_pixel_pov"
OC_TRAIN = "seaquest_skills_oc_pov"

RQ2_ENVS: tuple[dict, ...] = (
    {"env": "pong", "abbr": "po", "max_objects": 2, "heuristic": "nearest"},
    {"env": "tennis", "abbr": "te", "max_objects": 2, "heuristic": "nearest"},
    {"env": "beamrider", "abbr": "br", "max_objects": 31, "heuristic": "env_specific"},
    {"env": "freeway", "abbr": "fr", "max_objects": 12, "heuristic": "env_specific"},
    {"env": "asterix", "abbr": "as", "max_objects": 17, "heuristic": "env_specific"},
    {"env": "phoenix", "abbr": "ph", "max_objects": 21, "heuristic": "env_specific"},
)


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)} ({len(text)} bytes)")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _ego_configs() -> tuple[str, str]:
    """Fork the proven φ8/clip50 config under a fresh, non-reusable stem."""
    train = (_RQ1 / f"{EGO_SOURCE}.yaml").read_text(encoding="utf-8")
    eval_full = (_RQ1 / f"{EGO_SOURCE}_eval_full.yaml").read_text(encoding="utf-8")
    train = train.replace(EGO_SOURCE, EGO_TRAIN)
    eval_full = eval_full.replace(EGO_SOURCE, EGO_TRAIN)
    train = train.replace("suite:pov_facing_v2", f"suite:{SUITE}")
    eval_full = eval_full.replace("suite:pov_facing_v2", f"suite:{SUITE}")
    train = train.replace(
        "# Facing-locked FPS + clip50 + 8 full-circle φ bins",
        "# Fresh RQ1: facing-locked FPS + clip50 + φ8; surface relabeled COLLECT",
    )
    eval_full = eval_full.replace(
        "# Full Seaquest mod eval for facing-locked FPS + clip50 + φ-bins8",
        "# Full Seaquest mod eval for fresh φ8/clip50 COLLECT-surface skills",
    )
    return train, eval_full


def _pixel_train() -> str:
    return f"""# RQ1 pixel skills with facing-relative action mapper (no FORWARD-collapse cheat).
# New SAVE_PATH so --resume cannot reuse pre-mapper-fix checkpoints.
PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: seaquest
NUM_SEEDS: 1

ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
USE_LEGACY_EGO_OBS: false
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
USE_POV_RELATIVE_MAPPER: true
USE_POV_FACING_FRAME: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
EGO_DISTANCE_CLIP_MAX_PX: null
GAME_MODS: null
EVAL_GAME_MODS: null
ENV_NAME: seaquest
META_PIXEL_DO_RESIZE: true
META_PIXEL_USE_NATIVE_DOWNSCALING: true
META_PIXEL_RESIZE_SHAPE:
- 84
- 84
META_PIXEL_GRAYSCALE: true
SAVE_PATH: ./outputs/models/paper_eval/rq1/{PIXEL_TRAIN}
WANDB_RUN_NAME: sq_paper_rq1_skills_pixel_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- train_mode:skills_only
- skill_input:pixel_embed
- mapper:pov_facing
- variant:skills_pixel_pov
SKILL_INPUT_MODE: pixel_embed
SKILL_DEATH_PENALIZE_COLLECT: true
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
RECORD_VIDEO: true
VIDEO_AT_END: true
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
  SKILL_BOOTSTRAP: q_lambda
  EPS_START: 0.2
  EPS_FINISH: 0.01
  EPS_DECAY: 0.2
  HEURISTIC_MODE: env_specific
  SKILL_INPUT_MODE: pixel_embed
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
  SKILL_PIXEL_CNN_CHANNELS:
  - 16
  - 32
  SKILL_PIXEL_PAINT_TARGET_OUTLINE: true
"""


def _oc_train() -> str:
    return f"""# RQ1 OC skills with facing-relative action mapper.
PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: seaquest
NUM_SEEDS: 1

ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
USE_LEGACY_EGO_OBS: false
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
USE_POV_RELATIVE_MAPPER: true
USE_POV_FACING_FRAME: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
EGO_DISTANCE_CLIP_MAX_PX: null
GAME_MODS: null
EVAL_GAME_MODS: null
ENV_NAME: seaquest
SAVE_PATH: ./outputs/models/paper_eval/rq1/{OC_TRAIN}
WANDB_RUN_NAME: sq_paper_rq1_skills_oc_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- train_mode:skills_only
- skill_input:oc_embed
- mapper:pov_facing
- variant:skills_oc_pov
SKILL_INPUT_MODE: oc_embed
SKILL_OC_PSI_FEATURES: true
SKILL_DEATH_PENALIZE_COLLECT: true
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
RECORD_VIDEO: true
VIDEO_AT_END: true
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
  SKILL_BOOTSTRAP: q_lambda
  EPS_START: 0.2
  EPS_FINISH: 0.01
  EPS_DECAY: 0.2
  HEURISTIC_MODE: env_specific
  SKILL_INPUT_MODE: oc_embed
  SKILL_OC_PSI_FEATURES: true
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
"""


def _skills_eval(*, train_stem: str, skill_input: str, wandb_suffix: str) -> str:
    outline = ""
    if skill_input == "pixel_embed":
        outline = """
META_PIXEL_DO_RESIZE: true
META_PIXEL_USE_NATIVE_DOWNSCALING: true
META_PIXEL_RESIZE_SHAPE:
- 84
- 84
META_PIXEL_GRAYSCALE: true
"""
    return f"""# Full Seaquest mod eval for {train_stem}.
PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {train_stem}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{train_stem}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {train_stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: seaquest
USE_LEGACY_EGO_OBS: false
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
USE_POV_RELATIVE_MAPPER: true
USE_POV_FACING_FRAME: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
EGO_DISTANCE_CLIP_MAX_PX: null
SAVE_PATH: ./outputs/models/paper_eval/rq1/{train_stem}
WANDB_RUN_NAME: sq_paper_rq1_eval_{wandb_suffix}
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- phase:mod_eval
- skill_input:{skill_input}
- mapper:pov_facing
GAME_MODS: null
SKILL_INPUT_MODE: {skill_input}
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
{outline}alg:
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
  FRAME_SKIP: 4
  FRAME_STACK_SIZE: 4
  HEURISTIC_MODE: env_specific
  SKILL_INPUT_MODE: {skill_input}
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
"""


def _pixel_nooutline_eval() -> str:
    return f"""# Pixel no-outline ablation: same {PIXEL_TRAIN} ckpt, outline paint OFF.
PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {PIXEL_TRAIN}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{PIXEL_TRAIN}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {PIXEL_TRAIN}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: seaquest
USE_LEGACY_EGO_OBS: false
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
USE_POV_RELATIVE_MAPPER: true
USE_POV_FACING_FRAME: false
EGO_DISTANCE_CLIP_MAX_PX: null
SAVE_PATH: ./outputs/models/paper_eval/rq1/{PIXEL_TRAIN}_nooutline
WANDB_RUN_NAME: sq_paper_rq1_eval_pixel_pov_nooutline
WANDB_TAGS:
- suite:{SUITE}
- rq:1
- env:seaquest
- phase:mod_eval
- skill_input:pixel_embed
- mapper:pov_facing
- probe:nooutline
GAME_MODS: null
SKILL_INPUT_MODE: pixel_embed
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
META_PIXEL_DO_RESIZE: true
META_PIXEL_USE_NATIVE_DOWNSCALING: true
META_PIXEL_RESIZE_SHAPE:
- 84
- 84
META_PIXEL_GRAYSCALE: true
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
  FRAME_SKIP: 4
  FRAME_STACK_SIZE: 4
  HEURISTIC_MODE: env_specific
  SKILL_INPUT_MODE: pixel_embed
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  SKILL_PIXEL_CNN_CHANNELS:
  - 16
  - 32
  SKILL_PIXEL_PAINT_TARGET_OUTLINE: false
"""


def _ego_transfer(spec: dict) -> str:
    env = spec["env"]
    stem = f"{env}_transfer_ego_pov_clip50_facing_psibins8_collectsurface"
    return f"""# {env.title()} zero-shot ego — facing φ8 clip50 (Seaquest-trained).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: transfer
PAPER_EVAL_MODS: {env}
PAPER_EVAL_REQUIRES_RQ1:
- {EGO_TRAIN}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{EGO_TRAIN}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {EGO_TRAIN}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: {env}
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {spec["abbr"]}_paper_rq2_transfer_ego_pov_clip50_facing_psibins8_v2
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:zero_shot
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
EGO_PSI_ANGLE_BINS: 8
EGO_LEGAL_ACTION_MASK: true
GAME_MODS: null
EVAL_GAME_MODS: null
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: {spec["heuristic"]}
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
  MAX_OBJECTS: {spec["max_objects"]}
  EGO_ACTION_DIM: 18
  HIDDEN_SIZE: 128
  NUM_LAYERS: 3
  SKILL_EMBED_DIM: 256
  FRAME_SKIP: 4
  FRAME_STACK_SIZE: 4
  HEURISTIC_MODE: {spec["heuristic"]}
  SKILL_INPUT_MODE: ego
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
"""


def _mode_transfer(spec: dict, *, mode: str, train_stem: str) -> str:
    env = spec["env"]
    stem = f"{env}_transfer_{mode}_pov"
    skill_tag = "pixel_embed" if mode == "pixel" else "oc_embed"
    pixel_block = ""
    if mode == "pixel":
        pixel_block = """
META_PIXEL_DO_RESIZE: true
META_PIXEL_USE_NATIVE_DOWNSCALING: true
META_PIXEL_RESIZE_SHAPE:
- 84
- 84
META_PIXEL_GRAYSCALE: true
"""
    return f"""# {env.title()} zero-shot {mode} — {train_stem} (facing-relative mapper).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: transfer
PAPER_EVAL_MODS: {env}
PAPER_EVAL_REQUIRES_RQ1:
- {train_stem}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{train_stem}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {train_stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: {env}
USE_LEGACY_EGO_OBS: false
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
USE_POV_RELATIVE_MAPPER: true
USE_POV_FACING_FRAME: false
EGO_DISTANCE_CLIP_MAX_PX: null
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {spec["abbr"]}_paper_rq2_transfer_{mode}_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:zero_shot
- skill_input:{skill_tag}
- mapper:pov_facing
GAME_MODS: null
EVAL_GAME_MODS: null
SKILL_INPUT_MODE: {skill_tag}
HEURISTIC_MODE: {spec["heuristic"]}
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
VIDEO_AT_END: false
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
{pixel_block}alg:
  NUM_SKILLS: 4
  SKILL_NAMES:
  - DESTROY
  - COLLECT
  - AVOID
  - NAVIGATE
  MAX_OBJECTS: {spec["max_objects"]}
  EGO_ACTION_DIM: 18
  HIDDEN_SIZE: 128
  NUM_LAYERS: 3
  SKILL_EMBED_DIM: 256
  FRAME_SKIP: 4
  FRAME_STACK_SIZE: 4
  HEURISTIC_MODE: {spec["heuristic"]}
  SKILL_INPUT_MODE: {skill_tag}
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = bool(args.dry_run)

    ego_train, ego_eval = _ego_configs()
    _write(_RQ1 / f"{EGO_TRAIN}.yaml", ego_train, dry=dry)
    _write(_RQ1 / f"{EGO_TRAIN}_eval_full.yaml", ego_eval, dry=dry)
    _write(_RQ1 / f"{PIXEL_TRAIN}.yaml", _pixel_train(), dry=dry)
    _write(_RQ1 / f"{OC_TRAIN}.yaml", _oc_train(), dry=dry)
    _write(
        _RQ1 / f"{PIXEL_TRAIN}_eval_full.yaml",
        _skills_eval(train_stem=PIXEL_TRAIN, skill_input="pixel_embed", wandb_suffix="pixel_pov_full"),
        dry=dry,
    )
    _write(
        _RQ1 / f"{OC_TRAIN}_eval_full.yaml",
        _skills_eval(train_stem=OC_TRAIN, skill_input="oc_embed", wandb_suffix="oc_pov_full"),
        dry=dry,
    )
    _write(_RQ1 / f"{PIXEL_TRAIN}_eval_nooutline.yaml", _pixel_nooutline_eval(), dry=dry)

    for spec in RQ2_ENVS:
        env = spec["env"]
        ego_stem = f"{env}_transfer_ego_pov_clip50_facing_psibins8_collectsurface"
        # Always (re)write so tennis/freeway/asterix match the locked ego arm.
        _write(_RQ2 / f"{ego_stem}.yaml", _ego_transfer(spec), dry=dry)
        _write(
            _RQ2 / f"{env}_transfer_pixel_pov.yaml",
            _mode_transfer(spec, mode="pixel", train_stem=PIXEL_TRAIN),
            dry=dry,
        )
        _write(
            _RQ2 / f"{env}_transfer_oc_pov.yaml",
            _mode_transfer(spec, mode="oc", train_stem=OC_TRAIN),
            dry=dry,
        )

    print("done.")


if __name__ == "__main__":
    main()
