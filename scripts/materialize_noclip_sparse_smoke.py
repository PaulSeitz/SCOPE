#!/usr/bin/env python3
"""Materialize noclip + sparse C/D/A full-eval smoke (20M hard budget).

Per env (bases + transfers): ego skills scratch, flat PQN, from-scratch
pixel-meta tandem — each ``19_988_480`` steps — plus home mod evals.

Bases: seaquest, phoenix, namethisgame, asteroids
Transfers: timepilot, pong, breakout, beamrider, spaceinvaders, freeway

Asteroids / Time Pilot use turn primitives (dim 6) on skills *and* tandem.

    python scripts/materialize_noclip_sparse_smoke.py
    # three terminals:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
        --suite noclip_sparse_smoke_gpu0 --wait-deps --skip-rq-gate
    CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py \\
        --suite noclip_sparse_smoke_gpu1 --wait-deps --skip-rq-gate
    CUDA_VISIBLE_DEVICES=2 python scripts/run_paper_eval.py \\
        --suite noclip_sparse_smoke_gpu2 --wait-deps --skip-rq-gate
    # fill gaps after a cascade abort (single GPU, skips done work):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
        --suite noclip_sparse_smoke_rerun --wait-deps --skip-rq-gate --resume
    # AS/TP tandem catch-up (omitted from the first smoke while turn was skills-only):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
        --suite noclip_sparse_smoke_as_tp_tandem --wait-deps --skip-rq-gate
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"
_RQ3 = _ROOT / "configs/experiment/paper_eval/rq3"

SUITE = "noclip_sparse_smoke"
SPINE = "noclip_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
TURN_SPINE = "noclip_turn_psibins8_v2_noblocking_noweapon_surfacesub5"
TURN_ENVS = frozenset({"asteroids", "timepilot"})
SPARSE_TAG = "sparse_cda"
BUDGET_TAG = "20m"
TIMESTEPS = 19_988_480

BASE_ENVS = frozenset({"seaquest", "phoenix", "namethisgame", "asteroids"})

ENV_ORDER: tuple[str, ...] = (
    "seaquest",
    "phoenix",
    "namethisgame",
    "asteroids",
    "timepilot",
    "pong",
    "breakout",
    "beamrider",
    "spaceinvaders",
    "freeway",
)

ENVS: dict[str, dict] = {
    "seaquest": {"max_objects": 35, "heu": "env_specific", "prefix": "sq"},
    "phoenix": {"max_objects": 21, "heu": "env_specific", "prefix": "ph"},
    "namethisgame": {"max_objects": 11, "heu": "env_specific", "prefix": "ntg"},
    "asteroids": {"max_objects": 18, "heu": "env_specific", "prefix": "as"},
    "timepilot": {"max_objects": 7, "heu": "env_specific", "prefix": "tp"},
    "pong": {"max_objects": 2, "heu": "nearest", "prefix": "po"},
    "breakout": {"max_objects": 3, "heu": "nearest", "prefix": "bo"},
    "beamrider": {"max_objects": 31, "heu": "env_specific", "prefix": "br"},
    "spaceinvaders": {"max_objects": 41, "heu": "env_specific", "prefix": "si"},
    "freeway": {"max_objects": 12, "heu": "env_specific", "prefix": "fw"},
}


def _role(env: str) -> str:
    return "base" if env in BASE_ENVS else "transfer"


def _spine(env: str) -> str:
    return TURN_SPINE if env in TURN_ENVS else SPINE


def _skills_stem(env: str) -> str:
    spine = _spine(env)
    if env == "seaquest":
        return f"seaquest_skills_ego_pov_{spine}_{SPARSE_TAG}_{BUDGET_TAG}"
    return f"{env}_from_scratch_ego_pov_{spine}_{SPARSE_TAG}_{BUDGET_TAG}"


def _pqn_stem(env: str) -> str:
    return f"{env}_pqn_pixel_{BUDGET_TAG}"


def _tandem_stem(env: str) -> str:
    return (
        f"{env}_from_scratch_tandem_pixel_ego_pov_{_spine(env)}"
        f"_{SPARSE_TAG}_{BUDGET_TAG}"
    )


def _skills_rq_dir(env: str) -> str:
    return "rq1" if env == "seaquest" else "rq2"


def _tandem_rq_dir(env: str) -> str:
    return "rq3" if env == "seaquest" else "rq2"


def _skills_rq(env: str) -> int:
    return 1 if env == "seaquest" else 2


def _tandem_rq(env: str) -> int:
    return 3 if env == "seaquest" else 2


def _lines(*parts: str | list[str]) -> str:
    out: list[str] = []
    for p in parts:
        if isinstance(p, list):
            out.extend(p)
        elif p:
            out.extend(p.rstrip("\n").split("\n"))
    return "\n".join(out) + "\n"


def _write(path: Path, body: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(body if body.endswith("\n") else body + "\n")
    print(f"wrote {path.relative_to(_ROOT)}")


def _ego_lines(env: str) -> list[str]:
    if env in TURN_ENVS:
        step = 22.5 if env == "asteroids" else 45.0
        return [
            "USE_POV_RELATIVE_MAPPER: true",
            "USE_POV_FIXED_BILATERAL: false",
            "USE_POV_FACING_FRAME: false",
            "USE_POV_TURN_PRIMITIVES: true",
            f"POV_TURN_STEP_DEG: {step}",
            "POV_TURN_INITIAL_HEADING_DEG: null",
            "EGO_POSE_XY: false",
            "USE_LEGACY_EGO_OBS: false",
            "USE_EGO_SIZE_FEATURES: false",
            "USE_EGO_VELOCITY_FEATURES: false",
            "USE_EGO_BLOCKING_FEATURES: false",
            "USE_EGO_WEAPON_FEATURES: false",
            "USE_EGO_SHOT_FEATURES: false",
            "AVOID_INTERRUPT_RANGE: 55",
            "AVOID_INTERRUPT_RANGE_Y: 55",
            "SURFACE_SUB_AVOID_RANGE_X: 5",
            "SURFACE_SUB_AVOID_RANGE_Y: 5",
            "EGO_DISTANCE_CLIP_MAX_PX: null",
            "EGO_ANGLE_BINS: null",
            "EGO_PSI_ANGLE_BINS: 8",
            "EGO_LEGAL_ACTION_MASK: false",
            "SURFACE_SUB_NEAR_RANGE: 5",
        ]
    return [
        "USE_POV_RELATIVE_MAPPER: true",
        "USE_POV_FIXED_BILATERAL: false",
        "USE_POV_FACING_FRAME: true",
        "EGO_POSE_XY: false",
        "USE_LEGACY_EGO_OBS: false",
        "USE_EGO_SIZE_FEATURES: false",
        "USE_EGO_VELOCITY_FEATURES: false",
        "USE_EGO_BLOCKING_FEATURES: false",
        "USE_EGO_WEAPON_FEATURES: false",
        "USE_EGO_SHOT_FEATURES: false",
        "AVOID_INTERRUPT_RANGE: 55",
        "AVOID_INTERRUPT_RANGE_Y: 55",
        "SURFACE_SUB_AVOID_RANGE_X: 5",
        "SURFACE_SUB_AVOID_RANGE_Y: 5",
        "EGO_DISTANCE_CLIP_MAX_PX: null",
        "EGO_ANGLE_BINS: null",
        "EGO_PSI_ANGLE_BINS: 8",
        "EGO_LEGAL_ACTION_MASK: false",
        "SURFACE_SUB_NEAR_RANGE: 5",
    ]


def _seaquest_legacy_lines() -> list[str]:
    return [
        "USE_LEGACY_SEAQUEST_ACTION_MAPPER: false",
        "USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false",
        "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false",
    ]


def _mapper_tags(env: str) -> list[str]:
    if env in TURN_ENVS:
        return [
            "- mapper:pov_relative",
            "- mapper:turn_primitives",
            "- ego:psi_bins8",
        ]
    return ["- mapper:pov_facing", "- ego:phi_bins8"]


def _sparse_top() -> list[str]:
    return [
        "DESTROY_CONTACT_FAIL: true",
        "DESTROY_LIVING_PENALTY: false",
        "COLLECT_SPARSE_ONLY: true",
        "AVOID_SPARSE_ONLY: true",
    ]


def _sparse_alg() -> list[str]:
    return [
        "  DESTROY_CONTACT_FAIL: true",
        "  DESTROY_LIVING_PENALTY: false",
        "  COLLECT_SPARSE_ONLY: true",
        "  AVOID_SPARSE_ONLY: true",
    ]


def _common_tags(env: str, *, rq: int) -> list[str]:
    return [
        f"- suite:{SUITE}",
        f"- rq:{rq}",
        f"- env:{env}",
        f"- role:{_role(env)}",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        f"- ablation:{SPARSE_TAG}",
        f"- budget:{BUDGET_TAG}",
    ]


def _skills_train_yaml(env: str) -> str:
    meta = ENVS[env]
    stem = _skills_stem(env)
    spine = _spine(env)
    rq = _skills_rq(env)
    rq_dir = _skills_rq_dir(env)
    turn = env in TURN_ENVS
    ego_dim = 6 if turn else 18
    phase = "skills_only" if env == "seaquest" else "from_scratch"
    lines: list[str] = [
        f"# {env} ego skills — noclip + sparse C/D/A (~20M).",
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: train",
        "PAPER_EVAL_KIND: skills",
        f"PAPER_EVAL_MODS: {env}",
        "PAPER_EVAL_REQUIRES_RQ1: []",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        f"ENV_NAME: {env}",
    ]
    lines += _ego_lines(env)
    if env == "seaquest":
        lines += _seaquest_legacy_lines()
    lines += [
        f"SAVE_PATH: ./outputs/models/paper_eval/{rq_dir}/{stem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_paper_skills_ego_pov_{spine}_{SPARSE_TAG}_{BUDGET_TAG}",
        "WANDB_TAGS:",
    ]
    lines += _common_tags(env, rq=rq)
    lines += [
        f"- phase:{phase}",
        "- skill_input:ego",
    ]
    lines += _mapper_tags(env)
    lines += [
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        "SKILL_INPUT_MODE: ego",
        f"HEURISTIC_MODE: {meta['heu']}",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
        "RECORD_VIDEO: false",
        "VIDEO_AT_END: false",
    ]
    lines += _sparse_top()
    lines += [
        "alg:",
        "  NUM_SKILLS: 4",
        "  SKILL_NAMES:",
        "  - DESTROY",
        "  - COLLECT",
        "  - AVOID",
        "  - NAVIGATE",
        f"  MAX_OBJECTS: {meta['max_objects']}",
        f"  EGO_ACTION_DIM: {ego_dim}",
        "  HIDDEN_SIZE: 128",
        "  NUM_LAYERS: 3",
        "  SKILL_EMBED_DIM: 256",
        "  SKILL_LR: 0.0005",
        "  SKILL_REWARD_LAMBDA: 0.0",
        "  CLIP_REWARD: true",
        "  RESTRICT_FIRE_FOR_NON_DESTROY: false",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {TIMESTEPS}",
        "  NUM_ENVS: 128",
        "  NUM_STEPS: 128",
        "  NUM_EPOCHS: 4",
        "  NUM_MINIBATCHES: 64",
        "  GAMMA: 0.99",
        "  SKILL_GAMMA: 0.99",
        "  LAMBDA: 0.65",
        "  SKILL_LAMBDA: 0.65",
        "  LR: 0.0001",
        "  MAX_GRAD_NORM: 0.8",
        "  OFF_POLICY: true",
        "  RELABEL_ALL_OBJECTS: false",
        "  RELABEL_K_OBJECTS: null",
        "  SKILL_BOOTSTRAP: q_lambda",
        "  EPS_START: 0.2",
        "  EPS_FINISH: 0.01",
        "  EPS_DECAY: 0.2",
        f"  HEURISTIC_MODE: {meta['heu']}",
        "  SKILL_INPUT_MODE: ego",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_alg()
    lines += [
        "  USE_EGO_SHOT_FEATURES: false",
        "SEED: 44",
    ]
    return _lines(lines)


def _skills_eval_yaml(env: str) -> str:
    meta = ENVS[env]
    stem = _skills_stem(env)
    spine = _spine(env)
    rq = _skills_rq(env)
    rq_dir = _skills_rq_dir(env)
    ckpt = (
        f"./outputs/models/paper_eval/{rq_dir}/{stem}/{env}/"
        f"hierarchical_pqn_{env}_seed44.safetensors"
    )
    lines: list[str] = [
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: skills",
        f"PAPER_EVAL_MODS: {env}",
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {ckpt}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        f"ENV_NAME: {env}",
    ]
    lines += _ego_lines(env)
    lines += [
        f"SAVE_PATH: ./outputs/models/paper_eval/{rq_dir}/{stem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_paper_eval_skills_ego_pov_{spine}_{SPARSE_TAG}_{BUDGET_TAG}",
        "WANDB_TAGS:",
    ]
    lines += _common_tags(env, rq=rq)
    lines += ["- phase:mod_eval", "- skill_input:ego"]
    lines += _mapper_tags(env)
    lines += [
        "SKILL_INPUT_MODE: ego",
        f"HEURISTIC_MODE: {meta['heu']}",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_top()
    lines += [
        "SEED: 44",
        "EVAL_SEED: 10044",
        "EVAL_NUM_SEEDS: 3",
    ]
    return _lines(lines)


def _pqn_train_yaml(env: str) -> str:
    meta = ENVS[env]
    stem = _pqn_stem(env)
    return _lines(
        [
            "# Flat PQN pixel baseline — paper Table 5 HPs, matched 20M budget.",
            "PAPER_EVAL_RQ: 1",
            "PAPER_EVAL_PHASE: train",
            "PAPER_EVAL_KIND: flat_pqn",
            f"PAPER_EVAL_MODS: {env}",
            "PAPER_EVAL_REQUIRES_RQ1: []",
            "NUM_SEEDS: 1",
            "ALG_NAME: flat_pqn",
            f"ENV_NAME: {env}",
            "TRAIN_MODE: flat_pqn",
            "OBJECT_CENTRIC: false",
            "PIXEL_RESIZE: true",
            "PIXEL_RESIZE_SHAPE:",
            "- 84",
            "- 84",
            "PIXEL_GRAYSCALE: true",
            "USE_NATIVE_DOWNSCALING: true",
            "NORM_TYPE: layer_norm",
            "NORM_INPUT: false",
            "LR_LINEAR_DECAY: false",
            f"SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}",
            f"WANDB_RUN_NAME: {meta['prefix']}_paper_pqn_pixel_{BUDGET_TAG}",
            "WANDB_TAGS:",
            f"- suite:{SUITE}",
            "- rq:1",
            f"- env:{env}",
            f"- role:{_role(env)}",
            "- train_mode:flat_pqn",
            "- variant:pqn_pixel_paper_hps",
            f"- budget:{BUDGET_TAG}",
            f"TOTAL_TIMESTEPS: {TIMESTEPS}",
            f"TOTAL_TIMESTEPS_DECAY: {TIMESTEPS}",
            "NUM_ENVS: 128",
            "NUM_STEPS: 32",
            "NUM_EPOCHS: 2",
            "NUM_MINIBATCHES: 32",
            "FRAME_SKIP: 4",
            "FRAME_STACK_SIZE: 4",
            "FIRST_FIRE: true",
            "STICKY_ACTIONS: false",
            "EPISODIC_LIFE: true",
            "CLIP_REWARD: true",
            "GAMMA: 0.99",
            "LAMBDA: 0.65",
            "LR: 0.00025",
            "MAX_GRAD_NORM: 10",
            "HIDDEN_SIZE: 128",
            "NUM_LAYERS: 2",
            "EPS_START: 1.0",
            "EPS_FINISH: 0.001",
            "EPS_DECAY: 0.1",
            "GAME_MODS: null",
            "EVAL_GAME_MODS: null",
            "SEED: 44",
        ]
    )


def _pqn_eval_yaml(env: str) -> str:
    meta = ENVS[env]
    stem = _pqn_stem(env)
    ckpt = (
        f"./outputs/models/paper_eval/rq1/{stem}/{env}/"
        f"flat_pqn_{env}_pixel_seed44_vmap0.safetensors"
    )
    return _lines(
        [
            "PAPER_EVAL_RQ: 1",
            "PAPER_EVAL_PHASE: eval",
            "PAPER_EVAL_KIND: flat_pqn",
            f"PAPER_EVAL_MODS: {env}",
            "PAPER_EVAL_REQUIRES_RQ1:",
            f"- {stem}",
            f"PAPER_EVAL_FLAT_PQN_CKPT: {ckpt}",
            "PAPER_EVAL_OBJECT_CENTRIC: false",
            "NUM_SEEDS: 1",
            "ALG_NAME: flat_pqn",
            f"ENV_NAME: {env}",
            "OBJECT_CENTRIC: false",
            f"SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}",
            f"WANDB_RUN_NAME: {meta['prefix']}_paper_eval_pqn_pixel_{BUDGET_TAG}",
            "WANDB_TAGS:",
            f"- suite:{SUITE}",
            "- rq:1",
            f"- env:{env}",
            f"- role:{_role(env)}",
            "- phase:mod_eval",
            "- train_mode:flat_pqn",
            f"- budget:{BUDGET_TAG}",
            "SEED: 44",
            "EVAL_SEED: 10044",
            "EVAL_NUM_SEEDS: 3",
        ]
    )


def _tandem_train_yaml(env: str) -> str:
    meta = ENVS[env]
    stem = _tandem_stem(env)
    spine = _spine(env)
    rq = _tandem_rq(env)
    rq_dir = _tandem_rq_dir(env)
    turn = env in TURN_ENVS
    ego_dim = 6 if turn else 18
    lines: list[str] = [
        f"# {env} tandem from-scratch — pixel meta + ego skills (~20M).",
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: train",
        "PAPER_EVAL_KIND: tandem",
        f"PAPER_EVAL_MODS: {env}",
        "PAPER_EVAL_REQUIRES_RQ1: []",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: tandem",
        "META_ARCHITECTURE: per_object_mlp",
        f"ENV_NAME: {env}",
    ]
    lines += _ego_lines(env)
    if env == "seaquest":
        lines += _seaquest_legacy_lines()
    lines += [
        f"SAVE_PATH: ./outputs/models/paper_eval/{rq_dir}/{stem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_paper_tandem_pixel_ego_pov_{spine}_{SPARSE_TAG}_{BUDGET_TAG}",
        "WANDB_TAGS:",
    ]
    lines += _common_tags(env, rq=rq)
    lines += [
        "- phase:from_scratch",
        "- train_mode:tandem",
        "- meta_input:pixel",
        "- skill_input:ego",
    ]
    lines += _mapper_tags(env)
    lines += [
        "- probe:tandem_scratch",
        "META_PIXEL_DO_RESIZE: true",
        "META_PIXEL_USE_NATIVE_DOWNSCALING: true",
        "META_PIXEL_RESIZE_SHAPE:",
        "- 84",
        "- 84",
        "META_PIXEL_GRAYSCALE: true",
        "SKILL_PIXEL_CNN_CHANNELS:",
        "- 16",
        "- 32",
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        "META_INPUT_MODE: pixel_embed",
        "SKILL_INPUT_MODE: ego",
        f"HEURISTIC_MODE: {meta['heu']}",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
        "RECORD_VIDEO: false",
        "VIDEO_AT_END: false",
    ]
    lines += _sparse_top()
    lines += [
        "alg:",
        "  NUM_SKILLS: 4",
        "  SKILL_NAMES:",
        "  - DESTROY",
        "  - COLLECT",
        "  - AVOID",
        "  - NAVIGATE",
        f"  MAX_OBJECTS: {meta['max_objects']}",
        f"  EGO_ACTION_DIM: {ego_dim}",
        "  HIDDEN_SIZE: 128",
        "  NUM_LAYERS: 3",
        "  META_HIDDEN_SIZE: 256",
        "  META_PER_OBJECT_NUM_LAYERS: 3",
        "  META_FLAT_NUM_LAYERS: 3",
        "  META_EMBED_DIM: 256",
        "  SKILL_EMBED_DIM: 256",
        "  META_LR: 0.0005",
        "  SKILL_LR: 0.0005",
        "  META_USE_NOOP_TARGET: true",
        "  META_NOOP_TARGET_IDX: 0",
        "  META_SWITCH_PENALTY: -0.05",
        "  META_DEATH_PENALTY: -10.0",
        "  META_REWARD_LAMBDA: 1.0",
        "  CLIP_REWARD: true",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {TIMESTEPS}",
        "  NUM_ENVS: 128",
        "  NUM_STEPS: 128",
        "  NUM_EPOCHS: 4",
        "  NUM_MINIBATCHES: 64",
        "  GAMMA: 0.99",
        "  LAMBDA: 0.65",
        "  LR: 0.0001",
        "  MAX_GRAD_NORM: 0.8",
        "  OFF_POLICY: true",
        "  RELABEL_ALL_OBJECTS: false",
        "  RELABEL_K_OBJECTS: null",
        "  EPS_START: 0.1",
        "  EPS_FINISH: 0.01",
        "  EPS_DECAY: 0.2",
        "  META_INPUT_MODE: pixel_embed",
        "  SKILL_INPUT_MODE: ego",
        "  META_PIXEL_CNN_CHANNELS:",
        "  - 16",
        "  - 32",
        f"  HEURISTIC_MODE: {meta['heu']}",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_alg()
    lines += [
        "  USE_EGO_SHOT_FEATURES: false",
        "  SKILL_BOOTSTRAP: q_lambda",
        "  SKILL_LAMBDA: 0.65",
        "  SKILL_GAMMA: 0.99",
        "SEED: 44",
    ]
    return _lines(lines)


def _tandem_eval_yaml(env: str) -> str:
    meta = ENVS[env]
    stem = _tandem_stem(env)
    spine = _spine(env)
    rq = _tandem_rq(env)
    rq_dir = _tandem_rq_dir(env)
    skills_ckpt = (
        f"./outputs/models/paper_eval/{rq_dir}/{stem}/{env}/"
        f"hierarchical_pqn_{env}_seed44_skills.safetensors"
    )
    meta_ckpt = (
        f"./outputs/models/paper_eval/{rq_dir}/{stem}/{env}/"
        f"hierarchical_pqn_{env}_seed44_meta.safetensors"
    )
    lines: list[str] = [
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: tandem",
        f"PAPER_EVAL_MODS: {env}",
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {skills_ckpt}",
        f"PAPER_EVAL_META_CKPT: {meta_ckpt}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: tandem",
        f"ENV_NAME: {env}",
        f"SAVE_PATH: ./outputs/models/paper_eval/{rq_dir}/{stem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_paper_eval_tandem_pixel_ego_pov_{spine}_{SPARSE_TAG}_{BUDGET_TAG}",
        "WANDB_TAGS:",
    ]
    lines += _common_tags(env, rq=rq)
    lines += [
        "- phase:mod_eval",
        "- train_mode:tandem",
        "- meta_input:pixel",
        "- skill_input:ego",
    ]
    lines += _mapper_tags(env)
    lines += [
        "- probe:tandem_scratch",
        "SKILL_INPUT_MODE: ego",
        "META_INPUT_MODE: pixel_embed",
        f"HEURISTIC_MODE: {meta['heu']}",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_top()
    lines += [
        "EGO_DISTANCE_CLIP_MAX_PX: null",
        "EGO_PSI_ANGLE_BINS: 8",
        "alg:",
        "  SKILL_INPUT_MODE: ego",
        "  META_INPUT_MODE: pixel_embed",
        f"  HEURISTIC_MODE: {meta['heu']}",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_alg()
    lines += [
        "  USE_EGO_SHOT_FEATURES: false",
        "  EGO_DISTANCE_CLIP_MAX_PX: null",
        "  EGO_PSI_ANGLE_BINS: 8",
        "SEED: 44",
        "EVAL_SEED: 10044",
        "EVAL_NUM_SEEDS: 3",
    ]
    return _lines(lines)


def env_pipeline_stems(env: str) -> tuple[str, ...]:
    """Train→eval order for one env (skills → PQN → tandem, each with home eval)."""
    return (
        _skills_stem(env),
        f"{_skills_stem(env)}_eval",
        _pqn_stem(env),
        f"{_pqn_stem(env)}_eval",
        _tandem_stem(env),
        f"{_tandem_stem(env)}_eval",
    )


def as_tp_tandem_stems() -> tuple[str, ...]:
    """Catch-up: Asteroids + Time Pilot tandem train/eval only."""
    return (
        _tandem_stem("asteroids"),
        f"{_tandem_stem('asteroids')}_eval",
        _tandem_stem("timepilot"),
        f"{_tandem_stem('timepilot')}_eval",
    )


# Single-GPU fill for work the dead gpu1 worker never reached.
# Use with --resume.
#
# Leave gpu0 alone to finish Phoenix tandem → NTG → Asteroids. This suite
# only re-schedules Pong + Breakout (full skills/PQN/tandem pipelines).
RERUN_STEMS: tuple[str, ...] = (
    *env_pipeline_stems("pong"),
    *env_pipeline_stems("breakout"),
)


def rerun_stems() -> tuple[str, ...]:
    return RERUN_STEMS


# 3-GPU split (10 envs → 4+3+3). SQ full-mod pipeline stays on gpu0.
GPU0_ENVS: tuple[str, ...] = ("seaquest", "phoenix", "namethisgame", "asteroids")
GPU1_ENVS: tuple[str, ...] = ("timepilot", "pong", "breakout")
GPU2_ENVS: tuple[str, ...] = ("beamrider", "spaceinvaders", "freeway")


def gpu0_envs() -> tuple[str, ...]:
    return GPU0_ENVS


def gpu1_envs() -> tuple[str, ...]:
    return GPU1_ENVS


def gpu2_envs() -> tuple[str, ...]:
    return GPU2_ENVS


def _stems_for(envs: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for env in envs:
        out.extend(env_pipeline_stems(env))
    return tuple(out)


def gpu0_stems() -> tuple[str, ...]:
    return _stems_for(gpu0_envs())


def gpu1_stems() -> tuple[str, ...]:
    return _stems_for(gpu1_envs())


def gpu2_stems() -> tuple[str, ...]:
    return _stems_for(gpu2_envs())


def all_stems() -> tuple[str, ...]:
    return gpu0_stems() + gpu1_stems() + gpu2_stems()


def materialize(*, dry: bool) -> None:
    for env in ENV_ORDER:
        skills = _skills_stem(env)
        pqn = _pqn_stem(env)
        skills_dir = _RQ1 if env == "seaquest" else _RQ2
        _write(skills_dir / f"{skills}.yaml", _skills_train_yaml(env), dry=dry)
        _write(skills_dir / f"{skills}_eval.yaml", _skills_eval_yaml(env), dry=dry)
        _write(_RQ1 / f"{pqn}.yaml", _pqn_train_yaml(env), dry=dry)
        _write(_RQ1 / f"{pqn}_eval.yaml", _pqn_eval_yaml(env), dry=dry)
        tandem = _tandem_stem(env)
        tandem_dir = _RQ3 if env == "seaquest" else _RQ2
        _write(tandem_dir / f"{tandem}.yaml", _tandem_train_yaml(env), dry=dry)
        _write(tandem_dir / f"{tandem}_eval.yaml", _tandem_eval_yaml(env), dry=dry)

    assert set(ENV_ORDER) == set(GPU0_ENVS + GPU1_ENVS + GPU2_ENVS)
    print(
        f"\nenvs={len(ENV_ORDER)} stems={len(all_stems())} "
        f"gpu0={len(gpu0_stems())} gpu1={len(gpu1_stems())} "
        f"gpu2={len(gpu2_stems())} rerun={len(rerun_stems())} "
        f"as_tp_tandem={len(as_tp_tandem_stems())}"
    )
    print(f"gpu0 envs: {', '.join(gpu0_envs())}")
    print(f"gpu1 envs: {', '.join(gpu1_envs())}")
    print(f"gpu2 envs: {', '.join(gpu2_envs())}")
    print(f"as_tp_tandem suite: {', '.join(as_tp_tandem_stems())}")
    print(f"rerun suite: {', '.join(rerun_stems())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    materialize(dry=args.dry_run)


if __name__ == "__main__":
    main()
