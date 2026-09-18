#!/usr/bin/env python3
"""ALE clip10 heuristic smoke (+ tandem / SQ-10M append).

Locked spine for most envs: facing φ8, clip10, noblocking, noweapon, surfacesub5
(AVOID ranges kept for config parity; only Seaquest uses them).

All scratch trains (including Asteroids / Time Pilot) use the facing spine.
SQ→AS/TP zero-shot stays on the same facing spine so the Seaquest ckpt can load.

Skills trains (heuristic usability, ~25M):
  phoenix, namethisgame, timepilot, asteroids,  # EVAL_PLAN bases (+ TP/AS smoke)
  pong, breakout, beamrider, spaceinvaders, freeway  # transfer-family scratch

ZS transfers from SQ clip10 **skillmech** ckpt (ungated multi-skill rewards):
  spaceinvaders, beamrider, timepilot, asteroids, pong, breakout,
  freeway, namethisgame, phoenix

Appended:
  - Montezuma skills scratch (~25M; no tandem / no ZS)
  - Seaquest skills-only **10M** (skillmech) + full mod eval
  - Seaquest skills-only **10M sparse C/D/A** (COLLECT/DESTROY/AVOID sparse) + eval
  - Δxy negative-control SQ train (~25M skillmech spine) + eval + Pong/BR/PH ZS
  - Tandem from-scratch (pixel meta + ego skills) on Seaquest + scratch envs
    except Montezuma; home mod eval; ZS only from Seaquest tandem skills

Mod allowlists are still mostly base-only outside Seaquest; PAPER_EVAL_MODS is
wired so fuller strips run when the YAML grows.

    python scripts/materialize_ale_clip10_heu_smoke.py
    python scripts/run_paper_eval.py --suite ale_clip10_heu_smoke \\
        --gpus 0,1,2 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"
_RQ3 = _ROOT / "configs/experiment/paper_eval/rq3"
_MODS = _ROOT / "configs/experiment/paper_eval/mods"

SUITE = "ale_clip10_heu_smoke"
SPINE = "clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
# SQ ZS source: ungated skill-mechanics clip10 (intended off-policy rewards).
SQ_TRAIN = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech"
)
SQ_10M_TRAIN = f"{SQ_TRAIN}_10m"
SQ_SPARSE_10M_TRAIN = f"{SQ_TRAIN}_sparse_cda_10m"
# Δxy negative control on current skillmech spine (noclip distance block).
DELTAXY_TRAIN = (
    "seaquest_skills_ego_pov_deltaxy_facing_psibins8_v2_noblocking_noweapon_"
    "surfacesub5_skillmech"
)
DELTAXY_TRANSFER_ENVS = ("pong", "beamrider", "phoenix")
TIMESTEPS = 24_985_600  # ~25M (128×128 rollout chunks; suite standard)
TIMESTEPS_10M = 9_994_240  # 610 × 128 × 128
SQ_MAX_OBJECTS = 35

# env -> (max_objects, heuristic_mode, wandb_prefix, role)
# role: base | transfer | both (base+transfer family)
ENVS: dict[str, dict] = {
    "phoenix": {
        "max_objects": 21,
        "heu": "env_specific",
        "prefix": "ph",
        "scratch": True,
        "transfer": True,
    },
    "namethisgame": {
        "max_objects": 11,
        "heu": "env_specific",
        "prefix": "ntg",
        "scratch": True,
        "transfer": True,
    },
    "timepilot": {
        "max_objects": 7,
        "heu": "env_specific",
        "prefix": "tp",
        "scratch": True,
        "transfer": True,
    },
    "asteroids": {
        "max_objects": 18,
        "heu": "env_specific",
        "prefix": "as",
        "scratch": True,
        "transfer": True,
    },
    "pong": {
        "max_objects": 2,
        "heu": "nearest",
        "prefix": "po",
        "scratch": True,
        "transfer": True,
    },
    "breakout": {
        "max_objects": 3,
        "heu": "nearest",
        "prefix": "bo",
        "scratch": True,
        "transfer": True,
    },
    "beamrider": {
        "max_objects": 31,
        "heu": "env_specific",
        "prefix": "br",
        "scratch": True,
        "transfer": True,
    },
    "spaceinvaders": {
        "max_objects": 41,
        "heu": "env_specific",
        "prefix": "si",
        "scratch": True,
        "transfer": True,
    },
    "freeway": {
        "max_objects": 12,
        "heu": "env_specific",
        "prefix": "fw",
        "scratch": True,
        "transfer": True,
    },
    "montezumarevenge": {
        "max_objects": 32,
        "heu": "env_specific",
        "prefix": "mz",
        "scratch": True,
        "transfer": False,
        "tandem": False,
        # Showcase: COLLECT approach + JUMP_OVER chasm (NAVIGATE head idle).
        "num_skills": 5,
        "skill_names": ["DESTROY", "COLLECT", "AVOID", "NAVIGATE", "JUMP_OVER"],
    },
}

# Seaquest: tandem + 10M skills only (not a ~25M scratch in this suite).
SEAQUEST_META = {
    "max_objects": SQ_MAX_OBJECTS,
    "heu": "env_specific",
    "prefix": "sq",
}

# Tandem from-scratch: SQ + every skills-scratch env (no Montezuma).
TANDEM_ENVS: tuple[str, ...] = ("seaquest",) + tuple(
    env
    for env, meta in ENVS.items()
    if meta.get("scratch") and meta.get("tandem", True)
)

# ZS target order (matches run_paper_eval._ALE_CLIP10_TRANSFER_ENVS).
TRANSFER_ENV_ORDER: tuple[str, ...] = (
    "spaceinvaders",
    "beamrider",
    "timepilot",
    "asteroids",
    "pong",
    "breakout",
    "freeway",
    "namethisgame",
    "phoenix",
)


# Existing SQ→env clip10 transfers under the facing spine — rewrite so they
# depend on the skillmech SQ ckpt (do not keep stale lock REQUIRES).
EXISTING_TRANSFERS: set[str] = set()


def _scratch_spine(env: str) -> str:
    del env
    return SPINE


def _tandem_spine(env: str) -> str:
    del env
    return SPINE


def _tandem_stem(env: str) -> str:
    return f"{env}_from_scratch_tandem_pixel_ego_pov_{_tandem_spine(env)}"


def _env_meta(env: str) -> dict:
    if env == "seaquest":
        return SEAQUEST_META
    return ENVS[env]


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _facing_scratch_yaml(env: str, meta: dict) -> str:
    stem = f"{env}_from_scratch_ego_pov_{SPINE}"
    heu = meta["heu"]
    num_skills = int(meta.get("num_skills", 4))
    skill_names = meta.get(
        "skill_names",
        ["DESTROY", "COLLECT", "AVOID", "NAVIGATE"][:num_skills],
    )
    skill_names_yaml = "\n".join(f"          - {n}" for n in skill_names)
    return dedent(
        f"""\
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
        WANDB_RUN_NAME: {meta['prefix']}_paper_rq2_scratch_ego_pov_{SPINE}
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:2
        - env:{env}
        - phase:from_scratch
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - ablation:ale_heu_smoke
        USE_LEGACY_EGO_OBS: false
        USE_EGO_SIZE_FEATURES: false
        USE_EGO_VELOCITY_FEATURES: false
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        USE_EGO_SHOT_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_ANGLE_BINS: null
        EGO_PSI_ANGLE_BINS: 8
        GAME_MODS: null
        EVAL_GAME_MODS: null
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: {heu}
        SKILL_DEATH_PENALIZE_COLLECT: true
        RECORD_VIDEO: false
        VIDEO_AT_END: false
        DESTROY_CONTACT_FAIL: true
        DESTROY_LIVING_PENALTY: true
        alg:
          NUM_SKILLS: {num_skills}
          SKILL_NAMES:
{skill_names_yaml}
          MAX_OBJECTS: {meta['max_objects']}
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
          TOTAL_TIMESTEPS: {TIMESTEPS}
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
          HEURISTIC_MODE: {heu}
          SKILL_INPUT_MODE: ego
          SKILL_DEATH_PENALIZE_COLLECT: true
          DESTROY_CONTACT_FAIL: true
          DESTROY_LIVING_PENALTY: true
          USE_EGO_SHOT_FEATURES: false
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def _transfer_yaml(env: str, meta: dict) -> str:
    stem = f"{env}_transfer_ego_pov_{SPINE}"
    heu = meta["heu"]
    return dedent(
        f"""\
        PAPER_EVAL_RQ: 2
        PAPER_EVAL_PHASE: eval
        PAPER_EVAL_KIND: transfer
        PAPER_EVAL_MODS: {env}
        PAPER_EVAL_REQUIRES_RQ1:
        - {SQ_TRAIN}
        PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{SQ_TRAIN}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
        PAPER_EVAL_ARCHITECTURE_OF: {SQ_TRAIN}
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: skills_only
        ENV_NAME: {env}
        USE_POV_RELATIVE_MAPPER: true
        USE_POV_FIXED_BILATERAL: false
        USE_POV_FACING_FRAME: true
        EGO_POSE_XY: false
        SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
        WANDB_RUN_NAME: {meta['prefix']}_paper_rq2_transfer_ego_pov_{SPINE}
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:2
        - env:{env}
        - phase:zero_shot
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ablation:ale_heu_smoke
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        USE_LEGACY_EGO_OBS: false
        USE_EGO_SIZE_FEATURES: false
        USE_EGO_VELOCITY_FEATURES: false
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        USE_EGO_SHOT_FEATURES: false
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_PSI_ANGLE_BINS: 8
        EGO_LEGAL_ACTION_MASK: false
        GAME_MODS: null
        EVAL_GAME_MODS: null
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: {heu}
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
          MAX_OBJECTS: {meta['max_objects']}
          EGO_ACTION_DIM: 18
          HIDDEN_SIZE: 128
          NUM_LAYERS: 3
          SKILL_EMBED_DIM: 256
          FRAME_SKIP: 4
          FRAME_STACK_SIZE: 4
          HEURISTIC_MODE: {heu}
          SKILL_INPUT_MODE: ego
          SKILL_DEATH_PENALIZE_COLLECT: true
          DESTROY_CONTACT_FAIL: true
          DESTROY_LIVING_PENALTY: true
          USE_EGO_SHOT_FEATURES: false
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def _mods_base_only(env: str) -> str:
    return dedent(
        f"""\
        # Minimal allowlist for {env} — base only until mods workshop.
        env: {env}
        status: draft

        eval_mods:
          - key: base
            mods: []
            category: control

        paper_core:
          - base
        """
    )


def _sq_10m_train_yaml() -> str:
    """Clone skillmech lock knobs at 10M for a cheap SQ skills baseline."""
    stem = SQ_10M_TRAIN
    return dedent(
        f"""\
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
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        USE_EGO_SHOT_FEATURES: false
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_ANGLE_BINS: null
        EGO_PSI_ANGLE_BINS: 8
        GAME_MODS: null
        EVAL_GAME_MODS: null
        ENV_NAME: seaquest
        USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
        USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
        USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
        SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
        WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_{SPINE}_skillmech_10m
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:1
        - env:seaquest
        - train_mode:skills_only
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ablation:skillmech
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - budget:10m
        SKILL_INPUT_MODE: ego
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
          MAX_OBJECTS: {SQ_MAX_OBJECTS}
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
          TOTAL_TIMESTEPS: {TIMESTEPS_10M}
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
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def _sq_10m_eval_yaml() -> str:
    stem = SQ_10M_TRAIN
    return dedent(
        f"""\
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
        WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_{SPINE}_skillmech_10m
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:1
        - env:seaquest
        - phase:mod_eval
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ablation:skillmech
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - budget:10m
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: env_specific
        SKILL_DEATH_PENALIZE_COLLECT: true
        DESTROY_CONTACT_FAIL: true
        DESTROY_LIVING_PENALTY: true
        USE_EGO_SHOT_FEATURES: false
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_PSI_ANGLE_BINS: 8
        USE_LEGACY_EGO_OBS: false
        USE_EGO_SIZE_FEATURES: false
        USE_EGO_VELOCITY_FEATURES: false
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def _tandem_facing_yaml(env: str, meta: dict) -> str:
    spine = _tandem_spine(env)
    stem = _tandem_stem(env)
    heu = meta["heu"]
    rq_dir = "rq3" if env == "seaquest" else "rq2"
    return dedent(
        f"""\
        # {env} tandem from-scratch — pixel meta + facing ego skills (~25M).
        PAPER_EVAL_RQ: {3 if env == "seaquest" else 2}
        PAPER_EVAL_PHASE: train
        PAPER_EVAL_KIND: tandem
        PAPER_EVAL_MODS: {env}
        PAPER_EVAL_REQUIRES_RQ1: []
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: tandem
        META_ARCHITECTURE: per_object_mlp
        ENV_NAME: {env}
        USE_POV_RELATIVE_MAPPER: true
        USE_POV_FIXED_BILATERAL: false
        USE_POV_FACING_FRAME: true
        EGO_POSE_XY: false
        SAVE_PATH: ./outputs/models/paper_eval/{rq_dir}/{stem}
        WANDB_RUN_NAME: {meta['prefix']}_paper_tandem_pixel_ego_pov_{spine}
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:{3 if env == "seaquest" else 2}
        - env:{env}
        - phase:from_scratch
        - train_mode:tandem
        - meta_input:pixel
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - probe:tandem_scratch
        USE_LEGACY_EGO_OBS: false
        USE_EGO_SIZE_FEATURES: false
        USE_EGO_VELOCITY_FEATURES: false
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        USE_EGO_SHOT_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_ANGLE_BINS: null
        EGO_PSI_ANGLE_BINS: 8
        EGO_LEGAL_ACTION_MASK: false
        META_PIXEL_DO_RESIZE: true
        META_PIXEL_USE_NATIVE_DOWNSCALING: true
        META_PIXEL_RESIZE_SHAPE:
        - 84
        - 84
        META_PIXEL_GRAYSCALE: true
        SKILL_PIXEL_CNN_CHANNELS:
        - 32
        - 64
        GAME_MODS: null
        EVAL_GAME_MODS: null
        META_INPUT_MODE: pixel_embed
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: {heu}
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
          MAX_OBJECTS: {meta['max_objects']}
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
          META_NOOP_TARGET_IDX: null
          META_SWITCH_PENALTY: -0.05
          META_DEATH_PENALTY: 0.0
          META_REWARD_LAMBDA: 1.0
          CLIP_REWARD: true
          FIRST_FIRE: true
          STICKY_ACTIONS: false
          EPISODIC_LIFE: true
          FRAME_STACK_SIZE: 4
          FRAME_SKIP: 4
          TOTAL_TIMESTEPS: {TIMESTEPS}
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
          RELABEL_K_OBJECTS: null
          EPS_START: 0.1
          EPS_FINISH: 0.01
          EPS_DECAY: 0.2
          META_INPUT_MODE: pixel_embed
          SKILL_INPUT_MODE: ego
          META_PIXEL_CNN_CHANNELS:
          - 32
          - 64
          HEURISTIC_MODE: {heu}
          SKILL_DEATH_PENALIZE_COLLECT: true
          DESTROY_CONTACT_FAIL: true
          DESTROY_LIVING_PENALTY: true
          USE_EGO_SHOT_FEATURES: false
          SKILL_LR_FINISH: 0.0001
          SKILL_LR_DECAY: 0.5
          SKILL_BOOTSTRAP: q_lambda
          SKILL_LAMBDA: 0.65
          SKILL_GAMMA: 0.99
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def _tandem_eval_yaml(env: str, meta: dict) -> str:
    spine = _tandem_spine(env)
    stem = _tandem_stem(env)
    heu = meta["heu"]
    rq_dir = "rq3" if env == "seaquest" else "rq2"
    mapper_tags = "- mapper:pov_facing\n        - ego:phi_bins8"
    return dedent(
        f"""\
        # Home-env mod eval for {stem} (skills + meta ckpts).
        PAPER_EVAL_RQ: {3 if env == "seaquest" else 2}
        PAPER_EVAL_PHASE: eval
        PAPER_EVAL_KIND: tandem
        PAPER_EVAL_MODS: {env}
        PAPER_EVAL_REQUIRES_RQ1:
        - {stem}
        PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/{rq_dir}/{stem}/{env}/hierarchical_pqn_{env}_seed44_skills.safetensors
        PAPER_EVAL_META_CKPT: ./outputs/models/paper_eval/{rq_dir}/{stem}/{env}/hierarchical_pqn_{env}_seed44_meta.safetensors
        PAPER_EVAL_ARCHITECTURE_OF: {stem}
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: tandem
        ENV_NAME: {env}
        SAVE_PATH: ./outputs/models/paper_eval/{rq_dir}/{stem}
        WANDB_RUN_NAME: {meta['prefix']}_paper_eval_tandem_pixel_ego_pov_{spine}
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:{3 if env == "seaquest" else 2}
        - env:{env}
        - phase:mod_eval
        - train_mode:tandem
        - meta_input:pixel
        - skill_input:ego
        {mapper_tags}
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - probe:tandem_scratch
        SKILL_INPUT_MODE: ego
        META_INPUT_MODE: pixel_embed
        HEURISTIC_MODE: {heu}
        SKILL_DEATH_PENALIZE_COLLECT: true
        USE_EGO_SHOT_FEATURES: false
        DESTROY_CONTACT_FAIL: true
        DESTROY_LIVING_PENALTY: true
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_PSI_ANGLE_BINS: 8
        alg:
          SKILL_INPUT_MODE: ego
          META_INPUT_MODE: pixel_embed
          HEURISTIC_MODE: {heu}
          SKILL_DEATH_PENALIZE_COLLECT: true
          DESTROY_CONTACT_FAIL: true
          DESTROY_LIVING_PENALTY: true
          USE_EGO_SHOT_FEATURES: false
          EGO_DISTANCE_CLIP_MAX_PX: 10
          EGO_PSI_ANGLE_BINS: 8
        SEED: 44
        """
    )


def _tandem_transfer_yaml(env: str, meta: dict) -> str:
    """ZS from Seaquest tandem *skills* ckpt (meta stays home)."""
    sq_stem = _tandem_stem("seaquest")
    stem = f"{env}_transfer_tandem_pixel_ego_pov_{SPINE}"
    heu = meta["heu"]
    return dedent(
        f"""\
        PAPER_EVAL_RQ: 2
        PAPER_EVAL_PHASE: eval
        PAPER_EVAL_KIND: transfer
        PAPER_EVAL_MODS: {env}
        PAPER_EVAL_REQUIRES_RQ1:
        - {sq_stem}
        PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq3/{sq_stem}/seaquest/hierarchical_pqn_seaquest_seed44_skills.safetensors
        PAPER_EVAL_ARCHITECTURE_OF: {sq_stem}
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: skills_only
        ENV_NAME: {env}
        USE_POV_RELATIVE_MAPPER: true
        USE_POV_FIXED_BILATERAL: false
        USE_POV_FACING_FRAME: true
        EGO_POSE_XY: false
        SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
        WANDB_RUN_NAME: {meta['prefix']}_paper_rq2_transfer_tandem_pixel_ego_pov_{SPINE}
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:2
        - env:{env}
        - phase:zero_shot
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - probe:tandem_skills_zs
        USE_LEGACY_EGO_OBS: false
        USE_EGO_SIZE_FEATURES: false
        USE_EGO_VELOCITY_FEATURES: false
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        USE_EGO_SHOT_FEATURES: false
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_PSI_ANGLE_BINS: 8
        EGO_LEGAL_ACTION_MASK: false
        GAME_MODS: null
        EVAL_GAME_MODS: null
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: {heu}
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
          MAX_OBJECTS: {meta['max_objects']}
          EGO_ACTION_DIM: 18
          HIDDEN_SIZE: 128
          NUM_LAYERS: 3
          SKILL_EMBED_DIM: 256
          FRAME_SKIP: 4
          FRAME_STACK_SIZE: 4
          HEURISTIC_MODE: {heu}
          SKILL_INPUT_MODE: ego
          SKILL_DEATH_PENALIZE_COLLECT: true
          DESTROY_CONTACT_FAIL: true
          DESTROY_LIVING_PENALTY: true
          USE_EGO_SHOT_FEATURES: false
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def _sq_sparse_10m_train_yaml() -> str:
    """10M skillmech arm with sparse COLLECT / DESTROY / AVOID (NAVIGATE shaped)."""
    stem = SQ_SPARSE_10M_TRAIN
    body = _sq_10m_train_yaml()
    body = body.replace(SQ_10M_TRAIN, stem)
    body = body.replace("_skillmech_10m", "_skillmech_sparse_cda_10m")
    body = body.replace("- budget:10m\n", "- budget:10m\n- ablation:sparse_cda\n")
    # Top-level + alg DESTROY living tax → off; add sparse flags.
    body = body.replace(
        "DESTROY_CONTACT_FAIL: true\nDESTROY_LIVING_PENALTY: true\n",
        "DESTROY_CONTACT_FAIL: true\nDESTROY_LIVING_PENALTY: false\n"
        "COLLECT_SPARSE_ONLY: true\nAVOID_SPARSE_ONLY: true\n",
    )
    body = body.replace(
        "  DESTROY_CONTACT_FAIL: true\n  DESTROY_LIVING_PENALTY: true\n",
        "  DESTROY_CONTACT_FAIL: true\n  DESTROY_LIVING_PENALTY: false\n"
        "  COLLECT_SPARSE_ONLY: true\n  AVOID_SPARSE_ONLY: true\n",
    )
    if "COLLECT_SPARSE_ONLY: true" not in body:
        raise RuntimeError("sparse C/D/A knobs failed to apply to 10M template")
    return body


def _sq_sparse_10m_eval_yaml() -> str:
    body = _sq_10m_eval_yaml()
    body = body.replace(SQ_10M_TRAIN, SQ_SPARSE_10M_TRAIN)
    body = body.replace("_skillmech_10m", "_skillmech_sparse_cda_10m")
    body = body.replace("- budget:10m\n", "- budget:10m\n- ablation:sparse_cda\n")
    return body


def _deltaxy_train_yaml() -> str:
    stem = DELTAXY_TRAIN
    return dedent(
        f"""\
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
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        USE_EGO_SHOT_FEATURES: false
        EGO_DISTANCE_CLIP_MAX_PX: null
        EGO_DISTANCE_MODE: delta_xy
        EGO_ANGLE_BINS: null
        EGO_PSI_ANGLE_BINS: 8
        GAME_MODS: null
        EVAL_GAME_MODS: null
        ENV_NAME: seaquest
        USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
        USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
        USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
        SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
        WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_deltaxy_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:1
        - env:seaquest
        - train_mode:skills_only
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ablation:skillmech
        - ego:noblocking
        - ego:noweapon
        - ego:noclip
        - ego:deltaxy
        SKILL_INPUT_MODE: ego
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
          MAX_OBJECTS: {SQ_MAX_OBJECTS}
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
          TOTAL_TIMESTEPS: {TIMESTEPS}
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
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def _deltaxy_eval_yaml() -> str:
    stem = DELTAXY_TRAIN
    return dedent(
        f"""\
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
        WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_deltaxy_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:1
        - env:seaquest
        - phase:mod_eval
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ablation:skillmech
        - ego:noblocking
        - ego:noweapon
        - ego:noclip
        - ego:deltaxy
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: env_specific
        SKILL_DEATH_PENALIZE_COLLECT: true
        DESTROY_CONTACT_FAIL: true
        DESTROY_LIVING_PENALTY: true
        USE_EGO_SHOT_FEATURES: false
        EGO_DISTANCE_CLIP_MAX_PX: null
        EGO_DISTANCE_MODE: delta_xy
        EGO_PSI_ANGLE_BINS: 8
        USE_LEGACY_EGO_OBS: false
        USE_EGO_SIZE_FEATURES: false
        USE_EGO_VELOCITY_FEATURES: false
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def _deltaxy_transfer_yaml(env: str, meta: dict) -> str:
    stem = f"{env}_transfer_ego_pov_deltaxy_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech"
    heu = meta["heu"]
    return dedent(
        f"""\
        PAPER_EVAL_RQ: 2
        PAPER_EVAL_PHASE: eval
        PAPER_EVAL_KIND: transfer
        PAPER_EVAL_MODS: {env}
        PAPER_EVAL_REQUIRES_RQ1:
        - {DELTAXY_TRAIN}
        PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{DELTAXY_TRAIN}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
        PAPER_EVAL_ARCHITECTURE_OF: {DELTAXY_TRAIN}
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: skills_only
        ENV_NAME: {env}
        USE_POV_RELATIVE_MAPPER: true
        USE_POV_FIXED_BILATERAL: false
        USE_POV_FACING_FRAME: true
        EGO_POSE_XY: false
        SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
        WANDB_RUN_NAME: {meta['prefix']}_paper_rq2_transfer_ego_pov_deltaxy_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:2
        - env:{env}
        - phase:zero_shot
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ego:noblocking
        - ego:noweapon
        - ego:noclip
        - ego:deltaxy
        - ablation:skillmech
        USE_LEGACY_EGO_OBS: false
        USE_EGO_SIZE_FEATURES: false
        USE_EGO_VELOCITY_FEATURES: false
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        AVOID_INTERRUPT_RANGE: 55
        AVOID_INTERRUPT_RANGE_Y: 55
        SURFACE_SUB_AVOID_RANGE_X: 5
        SURFACE_SUB_AVOID_RANGE_Y: 5
        USE_EGO_SHOT_FEATURES: false
        EGO_DISTANCE_CLIP_MAX_PX: null
        EGO_DISTANCE_MODE: delta_xy
        EGO_PSI_ANGLE_BINS: 8
        EGO_LEGAL_ACTION_MASK: false
        GAME_MODS: null
        EVAL_GAME_MODS: null
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: {heu}
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
          MAX_OBJECTS: {meta['max_objects']}
          EGO_ACTION_DIM: 18
          HIDDEN_SIZE: 128
          NUM_LAYERS: 3
          SKILL_EMBED_DIM: 256
          FRAME_SKIP: 4
          FRAME_STACK_SIZE: 4
          HEURISTIC_MODE: {heu}
          SKILL_INPUT_MODE: ego
          SKILL_DEATH_PENALIZE_COLLECT: true
          DESTROY_CONTACT_FAIL: true
          DESTROY_LIVING_PENALTY: true
          USE_EGO_SHOT_FEATURES: false
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    )


def materialize(*, dry: bool) -> None:
    for env, meta in ENVS.items():
        mods_path = _MODS / f"{env}.yaml"
        if not mods_path.exists():
            _write(mods_path, _mods_base_only(env), dry=dry)

        if meta.get("scratch"):
            spine = _scratch_spine(env)
            stem = f"{env}_from_scratch_ego_pov_{spine}"
            _write(_RQ2 / f"{stem}.yaml", _facing_scratch_yaml(env, meta), dry=dry)

        if meta.get("transfer"):
            stem = f"{env}_transfer_ego_pov_{SPINE}"
            path = _RQ2 / f"{stem}.yaml"
            if env in EXISTING_TRANSFERS and path.exists():
                print(f"OK keep existing transfer {path.relative_to(_ROOT)}")
                continue
            _write(path, _transfer_yaml(env, meta), dry=dry)

    # --- append: SQ 10M skills + sparse 10M + full mod evals ---
    _write(_RQ1 / f"{SQ_10M_TRAIN}.yaml", _sq_10m_train_yaml(), dry=dry)
    _write(_RQ1 / f"{SQ_10M_TRAIN}_eval_full.yaml", _sq_10m_eval_yaml(), dry=dry)
    _write(_RQ1 / f"{SQ_SPARSE_10M_TRAIN}.yaml", _sq_sparse_10m_train_yaml(), dry=dry)
    _write(
        _RQ1 / f"{SQ_SPARSE_10M_TRAIN}_eval_full.yaml",
        _sq_sparse_10m_eval_yaml(),
        dry=dry,
    )

    # --- append: Δxy negative control (current skillmech spine) ---
    _write(_RQ1 / f"{DELTAXY_TRAIN}.yaml", _deltaxy_train_yaml(), dry=dry)
    _write(_RQ1 / f"{DELTAXY_TRAIN}_eval_full.yaml", _deltaxy_eval_yaml(), dry=dry)
    for env in DELTAXY_TRANSFER_ENVS:
        meta = ENVS[env]
        stem = (
            f"{env}_transfer_ego_pov_deltaxy_facing_psibins8_v2_"
            "noblocking_noweapon_surfacesub5_skillmech"
        )
        _write(_RQ2 / f"{stem}.yaml", _deltaxy_transfer_yaml(env, meta), dry=dry)

    # --- append: tandem trains + home mod evals ---
    for env in TANDEM_ENVS:
        meta = _env_meta(env)
        stem = _tandem_stem(env)
        rq_dir = _RQ3 if env == "seaquest" else _RQ2
        _write(rq_dir / f"{stem}.yaml", _tandem_facing_yaml(env, meta), dry=dry)
        _write(rq_dir / f"{stem}_eval.yaml", _tandem_eval_yaml(env, meta), dry=dry)

    # --- append: ZS from SQ tandem skills ---
    for env, meta in ENVS.items():
        if not meta.get("transfer"):
            continue
        stem = f"{env}_transfer_tandem_pixel_ego_pov_{SPINE}"
        _write(_RQ2 / f"{stem}.yaml", _tandem_transfer_yaml(env, meta), dry=dry)


def stem_lists() -> list[str]:
    """Suite order: trains first, then evals/transfers (wait-deps friendly)."""
    skills_scratch = [
        f"{env}_from_scratch_ego_pov_{_scratch_spine(env)}"
        for env, meta in ENVS.items()
        if meta.get("scratch")
    ]
    skills_zs = [f"{env}_transfer_ego_pov_{SPINE}" for env in TRANSFER_ENV_ORDER]
    tandem_trains = [_tandem_stem(env) for env in TANDEM_ENVS]
    tandem_evals = [f"{_tandem_stem(env)}_eval" for env in TANDEM_ENVS]
    tandem_zs = [
        f"{env}_transfer_tandem_pixel_ego_pov_{SPINE}" for env in TRANSFER_ENV_ORDER
    ]
    deltaxy_zs = [
        f"{env}_transfer_ego_pov_deltaxy_facing_psibins8_v2_"
        "noblocking_noweapon_surfacesub5_skillmech"
        for env in DELTAXY_TRANSFER_ENVS
    ]
    return (
        skills_scratch
        + [SQ_10M_TRAIN, SQ_SPARSE_10M_TRAIN, DELTAXY_TRAIN]
        + tandem_trains
        + skills_zs
        + [
            f"{SQ_10M_TRAIN}_eval_full",
            f"{SQ_SPARSE_10M_TRAIN}_eval_full",
            f"{DELTAXY_TRAIN}_eval_full",
        ]
        + deltaxy_zs
        + tandem_evals
        + tandem_zs
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument(
        "--print-stems",
        action="store_true",
        help="Print suite stem order and exit",
    )
    args = parser.parse_args()
    if args.print_stems:
        for s in stem_lists():
            print(s)
        return
    materialize(dry=args.dry)
    stems = stem_lists()
    n_scratch = sum(1 for e, m in ENVS.items() if m.get("scratch"))
    n_zs = sum(1 for e, m in ENVS.items() if m.get("transfer"))
    print(
        f"\n{n_scratch} skills scratch + 2× SQ-10M + 1 Δxy + "
        f"{len(TANDEM_ENVS)} tandem trains"
    )
    print(
        f"{n_zs} skillmech ZS + SQ/sparse/Δxy evals + {len(DELTAXY_TRANSFER_ENVS)} "
        f"Δxy ZS + {len(TANDEM_ENVS)} tandem evals + {n_zs} tandem ZS"
    )
    print(f"suite stems (trains first): {len(stems)}")


if __name__ == "__main__":
    main()
