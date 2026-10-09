#!/usr/bin/env python3
"""Materialize sparse C/D/A clean-story batch (Seaquest + Beamrider + Montezuma + tandem).

Everything-sparse skill rewards (NAVIGATE stays shaped — same contract as
``sparse_cda`` in ale_clip10):

  COLLECT_SPARSE_ONLY / AVOID_SPARSE_ONLY
  DESTROY_LIVING_PENALTY: false (kill +1 only; contact-fail still on)

Seaquest uses the spawn-fixed / standard instance (OFF_POLICY=true, ~25M) with
full home mod eval + full ZS transfer panel. Beamrider + Montezuma are 10M
from-scratch probes. Seaquest pixel-meta tandem is 20M + home mod eval.

GPU layout (separate suites so uneven pipelines stay sequential per GPU)::

  sparse_cda_clean_gpu0  — Seaquest train → eval_full → transfers
  sparse_cda_clean_gpu1  — Beamrider 10M → Montezuma 10M
  sparse_cda_clean_gpu2  — Seaquest tandem 20M → tandem eval

    python scripts/materialize_sparse_cda_clean.py
    bash scripts/run_sparse_cda_clean.sh --gpus 0,1,2

Meta-OC ablation (tomorrow, not this batch)::

    python scripts/materialize_seaquest_meta_oc_ablation.py
    python scripts/run_paper_eval.py --suite seaquest_meta_oc_ablation \\
        --gpus 0,1,2 --wait-deps --lean --skip-rq-gate
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"
_RQ3 = _ROOT / "configs/experiment/paper_eval/rq3"

SUITE = "sparse_cda_clean"
SPINE = "clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SPARSE_TAG = "sparse_cda"

TIMESTEPS_25M = 24_985_600
TIMESTEPS_20M = 19_988_480  # 1220 × 128 × 128
TIMESTEPS_10M = 9_994_240  # 610 × 128 × 128

SQ_TRAIN = f"seaquest_skills_ego_pov_{SPINE}_skillmech_{SPARSE_TAG}"
SQ_EVAL = f"{SQ_TRAIN}_eval_full"
BR_TRAIN = f"beamrider_from_scratch_ego_pov_{SPINE}_{SPARSE_TAG}_10m"
MZ_TRAIN = f"montezumarevenge_from_scratch_ego_pov_{SPINE}_{SPARSE_TAG}_10m"
TANDEM_TRAIN = f"seaquest_from_scratch_tandem_pixel_ego_pov_{SPINE}_{SPARSE_TAG}_20m"
TANDEM_EVAL = f"{TANDEM_TRAIN}_eval"

TRANSFER_ENVS: dict[str, dict] = {
    "spaceinvaders": {"max_objects": 41, "heu": "sweep", "prefix": "si"},
    "beamrider": {"max_objects": 31, "heu": "env_specific", "prefix": "br"},
    "timepilot": {"max_objects": 7, "heu": "env_specific", "prefix": "tp"},
    "asteroids": {"max_objects": 18, "heu": "env_specific", "prefix": "as"},
    "pong": {"max_objects": 2, "heu": "nearest", "prefix": "po"},
    "breakout": {"max_objects": 3, "heu": "nearest", "prefix": "bo"},
    "freeway": {"max_objects": 12, "heu": "env_specific", "prefix": "fw"},
    "namethisgame": {"max_objects": 11, "heu": "env_specific", "prefix": "ntg"},
    "phoenix": {"max_objects": 21, "heu": "env_specific", "prefix": "ph"},
}


def _transfer_stem(env: str) -> str:
    return f"{env}_transfer_ego_pov_{SPINE}_skillmech_{SPARSE_TAG}"


def _sparse_reward_block(*, indent: int = 0) -> str:
    """YAML keys for sparse C/D/A; ``indent`` is spaces for every line."""
    pad = " " * indent
    return "\n".join(
        f"{pad}{line}"
        for line in (
            "DESTROY_CONTACT_FAIL: true",
            "DESTROY_LIVING_PENALTY: false",
            "COLLECT_SPARSE_ONLY: true",
            "AVOID_SPARSE_ONLY: true",
        )
    )


def _write(path: Path, body: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = body if body.endswith("\n") else body + "\n"
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text)
    print(f"wrote {path.relative_to(_ROOT)}")


def _fill_sparse(body: str) -> str:
    # Placeholder lines keep their YAML indent; replacement must not double-indent.
    return body.replace("__SPARSE_TOP__", _sparse_reward_block(indent=0)).replace(
        "  __SPARSE_ALG__", _sparse_reward_block(indent=2)
    )


def _sq_train_yaml() -> str:
    return _fill_sparse(dedent(
        f"""\
        # Seaquest skills-only ~25M — sparse C/D/A on spawn-fixed Seaquest (OFF_POLICY).
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
        SAVE_PATH: ./outputs/models/paper_eval/rq1/{SQ_TRAIN}
        WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_{SPINE}_skillmech_{SPARSE_TAG}
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
        - ablation:{SPARSE_TAG}
        - probe:newseaquest_spawnfix
        - budget:25m
        SKILL_INPUT_MODE: ego
        SKILL_DEATH_PENALIZE_COLLECT: true
        RECORD_VIDEO: false
        VIDEO_AT_END: false
        VIDEO_MAX_FRAMES: 600
        VIDEO_FRAME_STRIDE: 4
        __SPARSE_TOP__
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
          TOTAL_TIMESTEPS: {TIMESTEPS_25M}
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
          __SPARSE_ALG__
          USE_EGO_SHOT_FEATURES: false
        SEED: 44
        EVAL_SEED: 10044
        EVAL_NUM_SEEDS: 3
        SURFACE_SUB_NEAR_RANGE: 5
        """
    ))


def _sq_eval_yaml() -> str:
    return _fill_sparse(dedent(
        f"""\
        PAPER_EVAL_RQ: 1
        PAPER_EVAL_PHASE: eval
        PAPER_EVAL_KIND: skills
        PAPER_EVAL_MODS: seaquest
        PAPER_EVAL_REQUIRES_RQ1:
        - {SQ_TRAIN}
        PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{SQ_TRAIN}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
        PAPER_EVAL_ARCHITECTURE_OF: {SQ_TRAIN}
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: skills_only
        ENV_NAME: seaquest
        USE_POV_RELATIVE_MAPPER: true
        USE_POV_FIXED_BILATERAL: false
        USE_POV_FACING_FRAME: true
        EGO_POSE_XY: false
        SAVE_PATH: ./outputs/models/paper_eval/rq1/{SQ_TRAIN}
        WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_{SPINE}_skillmech_{SPARSE_TAG}
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
        - ablation:{SPARSE_TAG}
        - probe:newseaquest_spawnfix
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: env_specific
        SKILL_DEATH_PENALIZE_COLLECT: true
        __SPARSE_TOP__
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
        EVAL_SEED: 10044
        EVAL_NUM_SEEDS: 3
        SURFACE_SUB_NEAR_RANGE: 5
        """
    ))


def _transfer_yaml(env: str, meta: dict) -> str:
    stem = _transfer_stem(env)
    return _fill_sparse(dedent(
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
        WANDB_RUN_NAME: {meta['prefix']}_paper_rq2_transfer_ego_pov_{SPINE}_skillmech_{SPARSE_TAG}
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
        - ablation:{SPARSE_TAG}
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
        HEURISTIC_MODE: {meta['heu']}
        SKILL_DEATH_PENALIZE_COLLECT: true
        RECORD_VIDEO: false
        VIDEO_AT_END: false
        __SPARSE_TOP__
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
          HEURISTIC_MODE: {meta['heu']}
          SKILL_INPUT_MODE: ego
          SKILL_DEATH_PENALIZE_COLLECT: true
          __SPARSE_ALG__
          USE_EGO_SHOT_FEATURES: false
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    ))


def _scratch_train_yaml(
    *,
    env: str,
    stem: str,
    prefix: str,
    max_objects: int,
    timesteps: int,
    budget_tag: str,
    num_skills: int,
    skill_names: list[str],
) -> str:
    skill_block = "\n".join(f"          - {n}" for n in skill_names)
    return _fill_sparse(dedent(
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
        WANDB_RUN_NAME: {prefix}_paper_rq2_scratch_ego_pov_{SPINE}_{SPARSE_TAG}_{budget_tag}
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
        - ablation:{SPARSE_TAG}
        - budget:{budget_tag}
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
        HEURISTIC_MODE: env_specific
        SKILL_DEATH_PENALIZE_COLLECT: true
        RECORD_VIDEO: false
        VIDEO_AT_END: false
        VIDEO_MAX_FRAMES: 600
        VIDEO_FRAME_STRIDE: 4
        __SPARSE_TOP__
        alg:
          NUM_SKILLS: {num_skills}
          SKILL_NAMES:
{skill_block}
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
          TOTAL_TIMESTEPS: {timesteps}
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
          __SPARSE_ALG__
          USE_EGO_SHOT_FEATURES: false
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    ))


def _tandem_train_yaml() -> str:
    return _fill_sparse(dedent(
        f"""\
        # seaquest tandem from-scratch — pixel meta + sparse C/D/A ego skills (~20M).
        PAPER_EVAL_RQ: 3
        PAPER_EVAL_PHASE: train
        PAPER_EVAL_KIND: tandem
        PAPER_EVAL_MODS: seaquest
        PAPER_EVAL_REQUIRES_RQ1: []
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: tandem
        META_ARCHITECTURE: per_object_mlp
        ENV_NAME: seaquest
        USE_POV_RELATIVE_MAPPER: true
        USE_POV_FIXED_BILATERAL: false
        USE_POV_FACING_FRAME: true
        EGO_POSE_XY: false
        SAVE_PATH: ./outputs/models/paper_eval/rq3/{TANDEM_TRAIN}
        WANDB_RUN_NAME: sq_paper_tandem_pixel_ego_pov_{SPINE}_{SPARSE_TAG}_20m
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:3
        - env:seaquest
        - phase:from_scratch
        - train_mode:tandem
        - meta_input:pixel
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - ablation:{SPARSE_TAG}
        - budget:20m
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
        HEURISTIC_MODE: env_specific
        SKILL_DEATH_PENALIZE_COLLECT: true
        RECORD_VIDEO: false
        VIDEO_AT_END: false
        VIDEO_MAX_FRAMES: 600
        VIDEO_FRAME_STRIDE: 4
        __SPARSE_TOP__
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
          TOTAL_TIMESTEPS: {TIMESTEPS_20M}
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
          HEURISTIC_MODE: env_specific
          SKILL_DEATH_PENALIZE_COLLECT: true
          __SPARSE_ALG__
          USE_EGO_SHOT_FEATURES: false
          SKILL_LR_FINISH: 0.0001
          SKILL_LR_DECAY: 0.5
          SKILL_BOOTSTRAP: q_lambda
          SKILL_LAMBDA: 0.65
          SKILL_GAMMA: 0.99
        SEED: 44
        SURFACE_SUB_NEAR_RANGE: 5
        """
    ))


def _tandem_eval_yaml() -> str:
    return _fill_sparse(dedent(
        f"""\
        # Home-env mod eval for {TANDEM_TRAIN} (skills + meta ckpts).
        PAPER_EVAL_RQ: 3
        PAPER_EVAL_PHASE: eval
        PAPER_EVAL_KIND: tandem
        PAPER_EVAL_MODS: seaquest
        PAPER_EVAL_REQUIRES_RQ1:
        - {TANDEM_TRAIN}
        PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq3/{TANDEM_TRAIN}/seaquest/hierarchical_pqn_seaquest_seed44_skills.safetensors
        PAPER_EVAL_META_CKPT: ./outputs/models/paper_eval/rq3/{TANDEM_TRAIN}/seaquest/hierarchical_pqn_seaquest_seed44_meta.safetensors
        PAPER_EVAL_ARCHITECTURE_OF: {TANDEM_TRAIN}
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: tandem
        ENV_NAME: seaquest
        SAVE_PATH: ./outputs/models/paper_eval/rq3/{TANDEM_TRAIN}
        WANDB_RUN_NAME: sq_paper_eval_tandem_pixel_ego_pov_{SPINE}_{SPARSE_TAG}_20m
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:3
        - env:seaquest
        - phase:mod_eval
        - train_mode:tandem
        - meta_input:pixel
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - ablation:{SPARSE_TAG}
        - budget:20m
        - probe:tandem_scratch
        SKILL_INPUT_MODE: ego
        META_INPUT_MODE: pixel_embed
        HEURISTIC_MODE: env_specific
        SKILL_DEATH_PENALIZE_COLLECT: true
        USE_EGO_SHOT_FEATURES: false
        __SPARSE_TOP__
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_PSI_ANGLE_BINS: 8
        alg:
          SKILL_INPUT_MODE: ego
          META_INPUT_MODE: pixel_embed
          HEURISTIC_MODE: env_specific
          SKILL_DEATH_PENALIZE_COLLECT: true
          __SPARSE_ALG__
          USE_EGO_SHOT_FEATURES: false
          EGO_DISTANCE_CLIP_MAX_PX: 10
          EGO_PSI_ANGLE_BINS: 8
        SEED: 44
        """
    ))


def stems_gpu0() -> tuple[str, ...]:
    return (SQ_TRAIN, SQ_EVAL) + tuple(_transfer_stem(e) for e in TRANSFER_ENVS)


def stems_gpu1() -> tuple[str, ...]:
    return (BR_TRAIN, MZ_TRAIN)


def stems_gpu2() -> tuple[str, ...]:
    return (TANDEM_TRAIN, TANDEM_EVAL)


def all_stems() -> tuple[str, ...]:
    return stems_gpu0() + stems_gpu1() + stems_gpu2()


def main(dry: bool = False) -> None:
    _write(_RQ1 / f"{SQ_TRAIN}.yaml", _sq_train_yaml(), dry=dry)
    _write(_RQ1 / f"{SQ_EVAL}.yaml", _sq_eval_yaml(), dry=dry)
    for env, meta in TRANSFER_ENVS.items():
        _write(_RQ2 / f"{_transfer_stem(env)}.yaml", _transfer_yaml(env, meta), dry=dry)
    _write(
        _RQ2 / f"{BR_TRAIN}.yaml",
        _scratch_train_yaml(
            env="beamrider",
            stem=BR_TRAIN,
            prefix="br",
            max_objects=31,
            timesteps=TIMESTEPS_10M,
            budget_tag="10m",
            num_skills=4,
            skill_names=["DESTROY", "COLLECT", "AVOID", "NAVIGATE"],
        ),
        dry=dry,
    )
    _write(
        _RQ2 / f"{MZ_TRAIN}.yaml",
        _scratch_train_yaml(
            env="montezumarevenge",
            stem=MZ_TRAIN,
            prefix="mz",
            max_objects=32,
            timesteps=TIMESTEPS_10M,
            budget_tag="10m",
            num_skills=5,
            skill_names=["DESTROY", "COLLECT", "AVOID", "NAVIGATE", "JUMP_OVER"],
        ),
        dry=dry,
    )
    _write(_RQ3 / f"{TANDEM_TRAIN}.yaml", _tandem_train_yaml(), dry=dry)
    _write(_RQ3 / f"{TANDEM_EVAL}.yaml", _tandem_eval_yaml(), dry=dry)
    print(
        f"\n{SUITE}: {len(all_stems())} stems "
        f"(gpu0={len(stems_gpu0())}, gpu1={len(stems_gpu1())}, gpu2={len(stems_gpu2())})"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    main(dry=bool(args.dry))
