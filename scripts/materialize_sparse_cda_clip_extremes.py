#!/usr/bin/env python3
"""Materialize sparse C/D/A clip extremes (clip0 / noclip) + warmstarted tandem.

Same reward contract as ``sparse_cda_clean``. Arms:

  * **clip0** — ``EGO_DISTANCE_CLIP_MAX_PX: 0`` (direction only; no distance)
  * **noclip** — ``EGO_DISTANCE_CLIP_MAX_PX: null`` (full continuous distance)
  * **tandem from skills** — 20M pixel-meta tandem, identical HPs to the
    from-scratch sparse tandem 20M, but ``INIT_MODEL_PATH`` = clip10 sparse
    skills-only ckpt (skills keep training; meta from scratch)

Suites (3-GPU overnight — plain ``run_paper_eval.py``, no shell wrappers)::

  sparse_cda_clip_extremes_gpu0  — clip0 train → eval → ZS transfers
  sparse_cda_clip_extremes_gpu1  — noclip train → eval → ZS transfers
  sparse_cda_clip_extremes_gpu2  — tandem-from-skills 20M → home eval

    python scripts/materialize_sparse_cda_clip_extremes.py
    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
        --suite sparse_cda_clip_extremes_gpu0 --wait-deps --skip-rq-gate
    CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py \\
        --suite sparse_cda_clip_extremes_gpu1 --wait-deps --skip-rq-gate
    CUDA_VISIBLE_DEVICES=2 python scripts/run_paper_eval.py \\
        --suite sparse_cda_clip_extremes_gpu2 --wait-deps --skip-rq-gate
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"
_RQ3 = _ROOT / "configs/experiment/paper_eval/rq3"

SUITE = "sparse_cda_clip_extremes"
SPARSE_TAG = "sparse_cda"
TIMESTEPS_25M = 24_985_600
TIMESTEPS_20M = 19_988_480  # match sparse_cda_clean tandem 20M

# (token used in stems/tags, YAML clip value — None → null)
CLIP_ARMS: tuple[tuple[str, int | None], ...] = (
    ("clip0", 0),
    ("noclip", None),
)

CLIP10_SKILLS = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
    f"surfacesub5_skillmech_{SPARSE_TAG}"
)
CLIP10_SKILLS_CKPT = (
    f"./outputs/models/paper_eval/rq1/{CLIP10_SKILLS}/seaquest/"
    "hierarchical_pqn_seaquest_seed44.safetensors"
)
TANDEM_TRAIN = (
    "seaquest_tandem_pixel_from_skills_ego_pov_clip10_facing_psibins8_v2_"
    f"noblocking_noweapon_surfacesub5_{SPARSE_TAG}_20m"
)
TANDEM_EVAL = f"{TANDEM_TRAIN}_eval"

TRANSFER_ENVS: dict[str, dict] = {
    "spaceinvaders": {"max_objects": 41, "heu": "env_specific", "prefix": "si"},
    "beamrider": {"max_objects": 31, "heu": "env_specific", "prefix": "br"},
    "timepilot": {"max_objects": 7, "heu": "env_specific", "prefix": "tp"},
    "asteroids": {"max_objects": 18, "heu": "env_specific", "prefix": "as"},
    "pong": {"max_objects": 2, "heu": "nearest", "prefix": "po"},
    "breakout": {"max_objects": 3, "heu": "nearest", "prefix": "bo"},
    "freeway": {"max_objects": 12, "heu": "env_specific", "prefix": "fw"},
    "namethisgame": {"max_objects": 11, "heu": "env_specific", "prefix": "ntg"},
    "phoenix": {"max_objects": 21, "heu": "env_specific", "prefix": "ph"},
}


def _spine(tok: str) -> str:
    return f"{tok}_facing_psibins8_v2_noblocking_noweapon_surfacesub5"


def _train_stem(tok: str) -> str:
    return f"seaquest_skills_ego_pov_{_spine(tok)}_skillmech_{SPARSE_TAG}"


def _eval_stem(tok: str) -> str:
    return f"{_train_stem(tok)}_eval_full"


def _transfer_stem(env: str, tok: str) -> str:
    return f"{env}_transfer_ego_pov_{_spine(tok)}_skillmech_{SPARSE_TAG}"


def _clip_yaml(val: int | None) -> str:
    return "null" if val is None else str(int(val))


def _sparse_reward_block(*, indent: int = 0) -> str:
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


def _fill_sparse(body: str) -> str:
    return body.replace("__SPARSE_TOP__", _sparse_reward_block(indent=0)).replace(
        "  __SPARSE_ALG__", _sparse_reward_block(indent=2)
    )


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if body.endswith("\n") else body + "\n")
    print(f"wrote {path.relative_to(_ROOT)}")


def _train_yaml(tok: str, clip_val: int | None) -> str:
    stem = _train_stem(tok)
    spine = _spine(tok)
    clip_s = _clip_yaml(clip_val)
    return _fill_sparse(
        dedent(
            f"""\
            # Seaquest skills-only ~25M — sparse C/D/A, ego {tok}.
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
            EGO_DISTANCE_CLIP_MAX_PX: {clip_s}
            EGO_ANGLE_BINS: null
            EGO_PSI_ANGLE_BINS: 8
            GAME_MODS: null
            EVAL_GAME_MODS: null
            ENV_NAME: seaquest
            USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
            USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
            USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
            SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
            WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_{spine}_skillmech_{SPARSE_TAG}
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
            - ego:{tok}
            - ablation:{SPARSE_TAG}
            - probe:clip_extremes
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
        )
    )


def _eval_yaml(tok: str, clip_val: int | None) -> str:
    stem = _train_stem(tok)
    spine = _spine(tok)
    clip_s = _clip_yaml(clip_val)
    return _fill_sparse(
        dedent(
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
            WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_{spine}_skillmech_{SPARSE_TAG}
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
            - ego:{tok}
            - ablation:{SPARSE_TAG}
            - probe:clip_extremes
            SKILL_INPUT_MODE: ego
            HEURISTIC_MODE: env_specific
            SKILL_DEATH_PENALIZE_COLLECT: true
            __SPARSE_TOP__
            USE_EGO_SHOT_FEATURES: false
            EGO_DISTANCE_CLIP_MAX_PX: {clip_s}
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
        )
    )


def _transfer_yaml(env: str, meta: dict, tok: str, clip_val: int | None) -> str:
    train = _train_stem(tok)
    stem = _transfer_stem(env, tok)
    spine = _spine(tok)
    clip_s = _clip_yaml(clip_val)
    return _fill_sparse(
        dedent(
            f"""\
            PAPER_EVAL_RQ: 2
            PAPER_EVAL_PHASE: eval
            PAPER_EVAL_KIND: transfer
            PAPER_EVAL_MODS: {env}
            PAPER_EVAL_REQUIRES_RQ1:
            - {train}
            PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{train}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
            PAPER_EVAL_ARCHITECTURE_OF: {train}
            NUM_SEEDS: 1
            ALG_NAME: hierarchical_pqn
            TRAIN_MODE: skills_only
            ENV_NAME: {env}
            USE_POV_RELATIVE_MAPPER: true
            USE_POV_FIXED_BILATERAL: false
            USE_POV_FACING_FRAME: true
            EGO_POSE_XY: false
            SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
            WANDB_RUN_NAME: {meta['prefix']}_paper_rq2_transfer_ego_pov_{spine}_skillmech_{SPARSE_TAG}
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
            - ego:{tok}
            - ablation:{SPARSE_TAG}
            - probe:clip_extremes
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
            EGO_DISTANCE_CLIP_MAX_PX: {clip_s}
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
        )
    )


def _tandem_train_yaml() -> str:
    """Pixel-meta tandem 20M warm-started from clip10 sparse skills-only."""
    return _fill_sparse(
        dedent(
            f"""\
            # Seaquest tandem — pixel meta from scratch + skills init from clip10
            # sparse skills-only; skills keep training (~20M, same HPs as scratch tandem).
            PAPER_EVAL_RQ: 3
            PAPER_EVAL_PHASE: train
            PAPER_EVAL_KIND: tandem
            PAPER_EVAL_MODS: seaquest
            PAPER_EVAL_REQUIRES_RQ1:
            - {CLIP10_SKILLS}
            NUM_SEEDS: 1
            ALG_NAME: hierarchical_pqn
            TRAIN_MODE: tandem
            META_ARCHITECTURE: per_object_mlp
            ENV_NAME: seaquest
            INIT_MODEL_PATH: {CLIP10_SKILLS_CKPT}
            USE_POV_RELATIVE_MAPPER: true
            USE_POV_FIXED_BILATERAL: false
            USE_POV_FACING_FRAME: true
            EGO_POSE_XY: false
            SAVE_PATH: ./outputs/models/paper_eval/rq3/{TANDEM_TRAIN}
            WANDB_RUN_NAME: sq_paper_tandem_pixel_from_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5_{SPARSE_TAG}_20m
            WANDB_TAGS:
            - suite:{SUITE}
            - rq:3
            - env:seaquest
            - phase:tandem_from_skills
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
            - probe:tandem_from_skills
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
            - 16
            - 32
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
              - 16
              - 32
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
        )
    )


def _tandem_eval_yaml() -> str:
    return _fill_sparse(
        dedent(
            f"""\
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
            WANDB_RUN_NAME: sq_paper_eval_tandem_pixel_from_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5_{SPARSE_TAG}_20m
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
            - probe:tandem_from_skills
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
        )
    )


def _clip_pipeline_stems(tok: str) -> tuple[str, ...]:
    return (_train_stem(tok), _eval_stem(tok)) + tuple(
        _transfer_stem(env, tok) for env in TRANSFER_ENVS
    )


def gpu0_stems() -> tuple[str, ...]:
    return _clip_pipeline_stems("clip0")


def gpu1_stems() -> tuple[str, ...]:
    return _clip_pipeline_stems("noclip")


def gpu2_stems() -> tuple[str, ...]:
    return (TANDEM_TRAIN, TANDEM_EVAL)


def all_stems() -> tuple[str, ...]:
    return gpu0_stems() + gpu1_stems() + gpu2_stems()


def main() -> None:
    for tok, clip_val in CLIP_ARMS:
        _write(_RQ1 / f"{_train_stem(tok)}.yaml", _train_yaml(tok, clip_val))
        _write(_RQ1 / f"{_eval_stem(tok)}.yaml", _eval_yaml(tok, clip_val))
        for env, meta in TRANSFER_ENVS.items():
            _write(
                _RQ2 / f"{_transfer_stem(env, tok)}.yaml",
                _transfer_yaml(env, meta, tok, clip_val),
            )
    _write(_RQ3 / f"{TANDEM_TRAIN}.yaml", _tandem_train_yaml())
    _write(_RQ3 / f"{TANDEM_EVAL}.yaml", _tandem_eval_yaml())
    print(f"\ngpu0 clip0: {len(gpu0_stems())} stems")
    print(f"gpu1 noclip: {len(gpu1_stems())} stems")
    print(f"gpu2 tandem-from-skills: {len(gpu2_stems())} stems")
    print(f"union: {len(all_stems())} stems")


if __name__ == "__main__":
    main()
