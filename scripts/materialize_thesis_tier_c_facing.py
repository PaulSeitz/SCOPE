#!/usr/bin/env python3
"""Materialize Tier C on the facing POV spine (+ optional RQ2 tandem-from-scratch).

Creates:
  - RQ2 from-scratch skills (6 envs × ego/pixel, ~25M) with POV mapper + facing φ8 ego
  - RQ3 Seaquest meta (pixel / OC meta, frozen ego skills) + tandem OC, init from RQ1
  - RQ2 tandem-from-scratch (beamrider / freeway / asterix × pixel|OC meta + ego skills)

Post-eval mod tables: ``--suite thesis_tier_c_facing --post-eval``.

Usage:
    python scripts/materialize_thesis_tier_c_facing.py
    python scripts/materialize_thesis_tier_c_facing.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"
_RQ3 = _ROOT / "configs/experiment/paper_eval/rq3"

SUITE = "thesis_tier_c_facing"
EGO_TRAIN = "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface"
EGO_CKPT = (
    f"./outputs/models/paper_eval/rq1/{EGO_TRAIN}/seaquest/"
    "hierarchical_pqn_seaquest_seed44.safetensors"
)
SCRATCH_STEPS = 24_985_600  # ~25M (128×128 rollout chunks)

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

TANDEM_SCRATCH_ENVS = ("beamrider", "freeway", "asterix")


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _spec(env: str) -> dict:
    return next(s for s in RQ2_ENVS if s["env"] == env)


def _no_target_line(spec: dict) -> str:
    return "NO_TARGET_GAME_ACTION: 1\n" if spec.get("no_target_fire") else ""


def _ego_facing_top() -> str:
    return """
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
EGO_DISTANCE_CLIP_MAX_PX: 50
EGO_PSI_ANGLE_BINS: 8
EGO_LEGAL_ACTION_MASK: false
"""


def _ego_facing_alg() -> str:
    return """
  EGO_DISTANCE_CLIP_MAX_PX: 50
  EGO_PSI_ANGLE_BINS: 8
"""


def _pixel_pov_top() -> str:
    return """
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
USE_POV_RELATIVE_MAPPER: true
USE_POV_FACING_FRAME: false
EGO_DISTANCE_CLIP_MAX_PX: null
"""


def _pixel_block() -> str:
    return """
META_PIXEL_DO_RESIZE: true
META_PIXEL_USE_NATIVE_DOWNSCALING: true
META_PIXEL_RESIZE_SHAPE:
- 84
- 84
META_PIXEL_GRAYSCALE: true
SKILL_PIXEL_CNN_CHANNELS:
- 16
- 32
"""


def _skills_alg_tail(spec: dict, *, skill_input: str, steps: int) -> str:
    return f"""  FRAME_STACK_SIZE: 4
  FRAME_SKIP: 4
  TOTAL_TIMESTEPS: {steps}
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
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false{_ego_facing_alg() if skill_input == "ego" else ""}
"""


def _from_scratch_skills(spec: dict, *, mode: str) -> str:
    env = spec["env"]
    if mode == "ego":
        stem = f"{env}_from_scratch_ego_pov_collectsurface"
        skill_input = "ego"
        extra_top = _ego_facing_top()
        pixel_extra = ""
    else:
        stem = f"{env}_from_scratch_pixel_pov"
        skill_input = "pixel_embed"
        extra_top = _pixel_pov_top()
        pixel_extra = _pixel_block()
    skill_tag = skill_input.replace("_embed", "")
    return f"""# {env.title()} from-scratch skills — {mode} on facing POV spine (~25M).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: {env}
PAPER_EVAL_REQUIRES_RQ1: []
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: {env}
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {spec["abbr"]}_paper_rq2_scratch_{skill_tag}_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:from_scratch
- skill_input:{skill_tag}
- mapper:pov_facing
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: {"true" if mode == "ego" else "false"}
USE_EGO_SHOT_FEATURES: false
{extra_top}{_no_target_line(spec)}{pixel_extra}GAME_MODS: null
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
{_skills_alg_tail(spec, skill_input=skill_input, steps=SCRATCH_STEPS)}
"""


def _meta_train(*, meta_mode: str) -> str:
    meta_input = "pixel_embed" if meta_mode == "pixel" else "oc_embed"
    stem = f"seaquest_meta_{meta_mode}_ego_pov"
    meta_tag = meta_input.replace("_embed", "")
    return f"""# Seaquest meta-only — {meta_mode} meta, frozen facing ego skills (POV spine).
PAPER_EVAL_RQ: 3
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: meta
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {EGO_TRAIN}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: meta_only
META_ARCHITECTURE: per_object_mlp
ENV_NAME: seaquest
INIT_MODEL_PATH: {EGO_CKPT}
SAVE_PATH: ./outputs/models/paper_eval/rq3/{stem}
WANDB_RUN_NAME: sq_paper_rq3_meta_{meta_mode}_ego_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:3
- env:seaquest
- train_mode:meta_only
- meta_input:{meta_tag}
- skill_input:ego
- mapper:pov_facing
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
{_ego_facing_top()}{_pixel_block()}
GAME_MODS: null
EVAL_GAME_MODS: null
META_INPUT_MODE: {meta_input}
SKILL_INPUT_MODE: ego
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
alg:
  INIT_MODEL_PATH: {EGO_CKPT}
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
  META_HIDDEN_SIZE: 256
  META_PER_OBJECT_NUM_LAYERS: 3
  META_FLAT_NUM_LAYERS: 3
  META_EMBED_DIM: 256
  SKILL_EMBED_DIM: 256
  META_LR: 0.0005
  SKILL_LR: 0.0
  META_USE_NOOP_TARGET: true
  META_NOOP_TARGET_IDX: 0
  META_SWITCH_PENALTY: -0.05
  META_DEATH_PENALTY: -10.0
  META_REWARD_LAMBDA: 1.0
  CLIP_REWARD: true
  FIRST_FIRE: true
  STICKY_ACTIONS: false
  EPISODIC_LIFE: true
  FRAME_STACK_SIZE: 4
  FRAME_SKIP: 4
  TOTAL_TIMESTEPS: {SCRATCH_STEPS}
  NUM_ENVS: 128
  NUM_STEPS: 128
  NUM_EPOCHS: 4
  NUM_MINIBATCHES: 64
  GAMMA: 0.99
  LAMBDA: 0.65
  LR: 0.0001
  MAX_GRAD_NORM: 0.8
  OFF_POLICY: false
  RELABEL_ALL_OBJECTS: false
  EPS_START: 0.1
  EPS_FINISH: 0.01
  EPS_DECAY: 0.2
  META_INPUT_MODE: {meta_input}
  SKILL_INPUT_MODE: ego
  META_PIXEL_CNN_CHANNELS:
  - 16
  - 32
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false{_ego_facing_alg()}
"""


def _meta_eval(*, meta_mode: str) -> str:
    meta_input = "pixel_embed" if meta_mode == "pixel" else "oc_embed"
    stem = f"seaquest_meta_{meta_mode}_ego_pov"
    meta_tag = meta_input.replace("_embed", "")
    return f"""# Full Seaquest mod eval for {stem}.
PAPER_EVAL_RQ: 3
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: meta
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {stem}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq3/{stem}/seaquest/hierarchical_pqn_seaquest_seed44_skills.safetensors
PAPER_EVAL_META_CKPT: ./outputs/models/paper_eval/rq3/{stem}/seaquest/hierarchical_pqn_seaquest_seed44_meta.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: meta_only
ENV_NAME: seaquest
{_ego_facing_top()}SAVE_PATH: ./outputs/models/paper_eval/rq3/{stem}
WANDB_RUN_NAME: sq_paper_rq3_eval_meta_{meta_mode}_ego_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:3
- env:seaquest
- phase:mod_eval
- train_mode:meta_only
- meta_input:{meta_tag}
- skill_input:ego
- mapper:pov_facing
SKILL_INPUT_MODE: ego
META_INPUT_MODE: {meta_input}
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
USE_EGO_SHOT_FEATURES: false
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
alg:
  SKILL_INPUT_MODE: ego
  META_INPUT_MODE: {meta_input}
  HEURISTIC_MODE: env_specific
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false{_ego_facing_alg()}
"""


def _tandem_seaquest_train() -> str:
    stem = "seaquest_tandem_oc_ego_pov"
    return f"""# Seaquest tandem — OC meta + facing ego skills (init from RQ1 ego ckpt).
PAPER_EVAL_RQ: 3
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: tandem
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {EGO_TRAIN}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: tandem
META_ARCHITECTURE: per_object_mlp
ENV_NAME: seaquest
INIT_MODEL_PATH: {EGO_CKPT}
SAVE_PATH: ./outputs/models/paper_eval/rq3/{stem}
WANDB_RUN_NAME: sq_paper_rq3_tandem_oc_ego_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:3
- env:seaquest
- train_mode:tandem
- meta_input:oc
- skill_input:ego
- mapper:pov_facing
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
{_ego_facing_top()}
GAME_MODS: null
EVAL_GAME_MODS: null
META_INPUT_MODE: oc_embed
SKILL_INPUT_MODE: ego
SKILL_DEATH_PENALIZE_COLLECT: true
{_pixel_block()}
RECORD_VIDEO: false
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
alg:
  INIT_MODEL_PATH: {EGO_CKPT}
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
  META_HIDDEN_SIZE: 256
  META_PER_OBJECT_NUM_LAYERS: 3
  META_FLAT_NUM_LAYERS: 3
  META_EMBED_DIM: 256
  SKILL_EMBED_DIM: 256
  META_LR: 0.0005
  SKILL_LR: 0.0005
  META_USE_NOOP_TARGET: true
  META_NOOP_TARGET_IDX: 0
  META_SWITCH_PENALTY: -0.05
  META_DEATH_PENALTY: -10.0
  META_REWARD_LAMBDA: 1.0
  CLIP_REWARD: true
  FIRST_FIRE: true
  STICKY_ACTIONS: false
  EPISODIC_LIFE: true
  FRAME_STACK_SIZE: 4
  FRAME_SKIP: 4
  TOTAL_TIMESTEPS: {SCRATCH_STEPS}
  NUM_ENVS: 128
  NUM_STEPS: 128
  NUM_EPOCHS: 4
  NUM_MINIBATCHES: 64
  GAMMA: 0.99
  LAMBDA: 0.65
  LR: 0.0001
  MAX_GRAD_NORM: 0.8
  OFF_POLICY: true
  RELABEL_ALL_OBJECTS: false
  EPS_START: 0.1
  EPS_FINISH: 0.01
  EPS_DECAY: 0.2
  META_INPUT_MODE: oc_embed
  SKILL_INPUT_MODE: ego
  META_PIXEL_CNN_CHANNELS:
  - 16
  - 32
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
  SKILL_LR_FINISH: 0.0001
  SKILL_LR_DECAY: 0.5
  SKILL_BOOTSTRAP: q_lambda
  SKILL_LAMBDA: 0.65
  SKILL_GAMMA: 0.99{_ego_facing_alg()}
"""


def _tandem_seaquest_eval() -> str:
    stem = "seaquest_tandem_oc_ego_pov"
    return f"""# Full Seaquest mod eval for {stem}.
PAPER_EVAL_RQ: 3
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: tandem
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {stem}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq3/{stem}/seaquest/hierarchical_pqn_seaquest_seed44_skills.safetensors
PAPER_EVAL_META_CKPT: ./outputs/models/paper_eval/rq3/{stem}/seaquest/hierarchical_pqn_seaquest_seed44_meta.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: tandem
ENV_NAME: seaquest
{_ego_facing_top()}SAVE_PATH: ./outputs/models/paper_eval/rq3/{stem}
WANDB_RUN_NAME: sq_paper_rq3_eval_tandem_oc_ego_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:3
- env:seaquest
- phase:mod_eval
- train_mode:tandem
- meta_input:oc
- skill_input:ego
- mapper:pov_facing
SKILL_INPUT_MODE: ego
META_INPUT_MODE: oc_embed
HEURISTIC_MODE: env_specific
SKILL_DEATH_PENALIZE_COLLECT: true
USE_EGO_SHOT_FEATURES: false
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
alg:
  SKILL_INPUT_MODE: ego
  META_INPUT_MODE: oc_embed
  HEURISTIC_MODE: env_specific
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false{_ego_facing_alg()}
"""


def _tandem_scratch_train(spec: dict, *, meta_mode: str) -> str:
    env = spec["env"]
    meta_input = "pixel_embed" if meta_mode == "pixel" else "oc_embed"
    stem = f"{env}_from_scratch_tandem_{meta_mode}_ego_pov"
    meta_tag = meta_input.replace("_embed", "")
    return f"""# {env.title()} tandem from-scratch — {meta_mode} meta + facing ego skills (~25M).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: tandem
PAPER_EVAL_MODS: {env}
PAPER_EVAL_REQUIRES_RQ1: []
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: tandem
META_ARCHITECTURE: per_object_mlp
ENV_NAME: {env}
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {spec["abbr"]}_paper_rq2_scratch_tandem_{meta_mode}_ego_pov
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:from_scratch
- train_mode:tandem
- meta_input:{meta_tag}
- skill_input:ego
- mapper:pov_facing
- probe:tandem_scratch
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
{_ego_facing_top()}{_no_target_line(spec)}{_pixel_block()}
GAME_MODS: null
EVAL_GAME_MODS: null
META_INPUT_MODE: {meta_input}
SKILL_INPUT_MODE: ego
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
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
  META_HIDDEN_SIZE: 256
  META_PER_OBJECT_NUM_LAYERS: 3
  META_FLAT_NUM_LAYERS: 3
  META_EMBED_DIM: 256
  SKILL_EMBED_DIM: 256
  META_LR: 0.0005
  SKILL_LR: 0.0005
  META_USE_NOOP_TARGET: true
  META_NOOP_TARGET_IDX: 0
  META_SWITCH_PENALTY: -0.05
  META_DEATH_PENALTY: -10.0
  META_REWARD_LAMBDA: 1.0
  CLIP_REWARD: true
  FIRST_FIRE: true
  STICKY_ACTIONS: false
  EPISODIC_LIFE: {str(spec["episodic_life"]).lower()}
  FRAME_STACK_SIZE: 4
  FRAME_SKIP: 4
  TOTAL_TIMESTEPS: {SCRATCH_STEPS}
  NUM_ENVS: 128
  NUM_STEPS: 128
  NUM_EPOCHS: 4
  NUM_MINIBATCHES: 64
  GAMMA: 0.99
  LAMBDA: 0.65
  LR: 0.0001
  MAX_GRAD_NORM: 0.8
  OFF_POLICY: true
  RELABEL_ALL_OBJECTS: false
  EPS_START: 0.1
  EPS_FINISH: 0.01
  EPS_DECAY: 0.2
  META_INPUT_MODE: {meta_input}
  SKILL_INPUT_MODE: ego
  META_PIXEL_CNN_CHANNELS:
  - 16
  - 32
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
  SKILL_LR_FINISH: 0.0001
  SKILL_LR_DECAY: 0.5
  SKILL_BOOTSTRAP: q_lambda
  SKILL_LAMBDA: 0.65
  SKILL_GAMMA: 0.99{_ego_facing_alg()}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = bool(args.dry_run)

    for spec in RQ2_ENVS:
        env = spec["env"]
        _write(
            _RQ2 / f"{env}_from_scratch_ego_pov_collectsurface.yaml",
            _from_scratch_skills(spec, mode="ego"),
            dry=dry,
        )
        _write(
            _RQ2 / f"{env}_from_scratch_pixel_pov.yaml",
            _from_scratch_skills(spec, mode="pixel"),
            dry=dry,
        )

    _write(_RQ3 / "seaquest_meta_pixel_ego_pov.yaml", _meta_train(meta_mode="pixel"), dry=dry)
    _write(_RQ3 / "seaquest_meta_pixel_ego_pov_eval.yaml", _meta_eval(meta_mode="pixel"), dry=dry)
    _write(_RQ3 / "seaquest_meta_oc_ego_pov.yaml", _meta_train(meta_mode="oc"), dry=dry)
    _write(_RQ3 / "seaquest_meta_oc_ego_pov_eval.yaml", _meta_eval(meta_mode="oc"), dry=dry)
    _write(_RQ3 / "seaquest_tandem_oc_ego_pov.yaml", _tandem_seaquest_train(), dry=dry)
    _write(_RQ3 / "seaquest_tandem_oc_ego_pov_eval.yaml", _tandem_seaquest_eval(), dry=dry)

    for env_name in TANDEM_SCRATCH_ENVS:
        spec = _spec(env_name)
        for meta_mode in ("pixel", "oc"):
            stem = f"{env_name}_from_scratch_tandem_{meta_mode}_ego_pov"
            _write(_RQ2 / f"{stem}.yaml", _tandem_scratch_train(spec, meta_mode=meta_mode), dry=dry)

    print("done.")


if __name__ == "__main__":
    main()
