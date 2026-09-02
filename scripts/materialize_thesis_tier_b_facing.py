#!/usr/bin/env python3
"""Materialize Tier B finetune YAMLs aligned with thesis_facing RQ1 ckpts.

Creates 18 configs (6 envs × ego/pixel/oc): 5M skill finetune on the target
env, initialized from:
  - ego  → seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface
  - pixel → seaquest_skills_pixel_pov
  - oc   → seaquest_skills_oc_pov

Post-eval mod tables: run with --post-eval on suite thesis_tier_b_facing.

Usage:
    python scripts/materialize_thesis_tier_b_facing.py
    python scripts/materialize_thesis_tier_b_facing.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "thesis_tier_b_facing"
EGO_TRAIN = "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface"
PIXEL_TRAIN = "seaquest_skills_pixel_pov"
OC_TRAIN = "seaquest_skills_oc_pov"
CKPT = "./outputs/models/paper_eval/rq1/{train}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
FINETUNE_STEPS = 4_997_120  # ~5M (128×128 rollout chunks)

RQ2_ENVS: tuple[dict, ...] = (
    {
        "env": "pong",
        "abbr": "po",
        "max_objects": 2,
        "heuristic": "nearest",
        "episodic_life": False,
        "no_target_fire": False,
    },
    {
        "env": "tennis",
        "abbr": "te",
        "max_objects": 2,
        "heuristic": "nearest",
        "episodic_life": False,
        "no_target_fire": False,
    },
    {
        "env": "beamrider",
        "abbr": "br",
        "max_objects": 31,
        "heuristic": "env_specific",
        "episodic_life": True,
        "no_target_fire": True,
    },
    {
        "env": "freeway",
        "abbr": "fr",
        "max_objects": 12,
        "heuristic": "env_specific",
        "episodic_life": False,
        "no_target_fire": False,
    },
    {
        "env": "asterix",
        "abbr": "as",
        "max_objects": 17,
        "heuristic": "env_specific",
        "episodic_life": True,
        "no_target_fire": False,
    },
    {
        "env": "phoenix",
        "abbr": "ph",
        "max_objects": 21,
        "heuristic": "env_specific",
        "episodic_life": True,
        "no_target_fire": False,
    },
)


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)} ({len(text)} bytes)")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _stem(env: str, mode: str) -> str:
    if mode == "ego":
        return f"{env}_finetune_ego_pov_collectsurface"
    return f"{env}_finetune_{mode}_pov"


def _train_stem(mode: str) -> str:
    if mode == "ego":
        return EGO_TRAIN
    if mode == "pixel":
        return PIXEL_TRAIN
    return OC_TRAIN


def _skill_input(mode: str) -> str:
    if mode == "ego":
        return "ego"
    if mode == "pixel":
        return "pixel_embed"
    return "oc_embed"


def _finetune_yaml(spec: dict, *, mode: str) -> str:
    env = spec["env"]
    train_stem = _train_stem(mode)
    stem = _stem(env, mode)
    init_path = CKPT.format(train=train_stem)
    skill_input = _skill_input(mode)
    skill_tag = skill_input.replace("_embed", "")

    no_target_line = ""
    if spec.get("no_target_fire"):
        no_target_line = "NO_TARGET_GAME_ACTION: 1\n"

    pov_top = ""
    pov_alg = ""
    pixel_block = ""
    ego_top = ""
    ego_alg = ""

    if mode == "ego":
        ego_top = """
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
EGO_DISTANCE_CLIP_MAX_PX: 50
EGO_PSI_ANGLE_BINS: 8
EGO_LEGAL_ACTION_MASK: true
"""
        ego_alg = """
  EGO_DISTANCE_CLIP_MAX_PX: 50
  EGO_PSI_ANGLE_BINS: 8
"""
    else:
        pov_top = """
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
USE_POV_RELATIVE_MAPPER: true
USE_POV_FACING_FRAME: false
"""
        if mode == "pixel":
            pixel_block = """
META_PIXEL_DO_RESIZE: true
META_PIXEL_USE_NATIVE_DOWNSCALING: true
META_PIXEL_RESIZE_SHAPE:
- 84
- 84
META_PIXEL_GRAYSCALE: true
"""

    return f"""# {env.title()} 5M finetune — {mode} from {train_stem} (facing POV spine).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: finetune
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: {env}
PAPER_EVAL_REQUIRES_RQ1:
- {train_stem}
PAPER_EVAL_ARCHITECTURE_OF: {train_stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: {env}
INIT_MODEL_PATH: {init_path}
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {spec["abbr"]}_paper_rq2_finetune_{mode}_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:finetune
- skill_input:{skill_tag}
- mapper:pov_facing
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
{pov_top}{ego_top}{no_target_line}{pixel_block}GAME_MODS: null
EVAL_GAME_MODS: null
SKILL_INPUT_MODE: {skill_input}
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
  SKILL_LR: 0.0005
  SKILL_LR_FINISH: 0.0001
  SKILL_LR_DECAY: 0.5
  SKILL_REWARD_LAMBDA: 0.0
  CLIP_REWARD: true
  RESTRICT_FIRE_FOR_NON_DESTROY: false
  FIRST_FIRE: true
  STICKY_ACTIONS: false
  EPISODIC_LIFE: {str(spec["episodic_life"]).lower()}
  FRAME_STACK_SIZE: 4
  FRAME_SKIP: 4
  TOTAL_TIMESTEPS: {FINETUNE_STEPS}
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
  HEURISTIC_MODE: {spec["heuristic"]}
  SKILL_INPUT_MODE: {skill_input}
  SKILL_DEATH_PENALIZE_COLLECT: true
  SKILL_PIXEL_CNN_CHANNELS:
  - 16
  - 32
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false{ego_alg}
  INIT_MODEL_PATH: {init_path}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = bool(args.dry_run)

    for spec in RQ2_ENVS:
        for mode in ("ego", "pixel", "oc"):
            stem = _stem(spec["env"], mode)
            _write(_RQ2 / f"{stem}.yaml", _finetune_yaml(spec, mode=mode), dry=dry)

    print("done.")


if __name__ == "__main__":
    main()
