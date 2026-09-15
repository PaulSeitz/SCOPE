#!/usr/bin/env python3
"""Materialize sparse C/D/A Seaquest skills at clip50 and clip100 (+ ZS transfers).

Same reward contract as ``sparse_cda_clean`` (COLLECT/AVOID sparse, DESTROY living
tax off; NAVIGATE shaped). Train ~25M + full home mod eval each; ZS transfer
panel matching the clean-story envs.

Suites::

  sparse_cda_clip_sweep
      clip50 train → eval_full → clip100 train → eval_full

  sparse_cda_clip_transfer
      all ZS transfers for clip50 then clip100 (requires train ckpts)

  sparse_cda_clip_transfer_gpu0 / _gpu1
      clip50 ZS (gpu0) vs clip100 ZS (gpu1) — run in parallel

    python scripts/materialize_sparse_cda_clip_sweep.py
    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
        --suite sparse_cda_clip_sweep --wait-deps --skip-rq-gate --video
    # 2-GPU ZS (after train ckpts exist):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
        --suite sparse_cda_clip_transfer_gpu0 --wait-deps --skip-rq-gate
    CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py \\
        --suite sparse_cda_clip_transfer_gpu1 --wait-deps --skip-rq-gate
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "sparse_cda_clip_sweep"
TRANSFER_SUITE = "sparse_cda_clip_transfer"
SPARSE_TAG = "sparse_cda"
TIMESTEPS_25M = 24_985_600
CLIPS = (50, 100)

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


def _spine(clip: int) -> str:
    return f"clip{clip}_facing_psibins8_v2_noblocking_noweapon_surfacesub5"


def _train_stem(clip: int) -> str:
    return f"seaquest_skills_ego_pov_{_spine(clip)}_skillmech_{SPARSE_TAG}"


def _eval_stem(clip: int) -> str:
    return f"{_train_stem(clip)}_eval_full"


def _transfer_stem(env: str, clip: int) -> str:
    return f"{env}_transfer_ego_pov_{_spine(clip)}_skillmech_{SPARSE_TAG}"


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


def _train_yaml(clip: int) -> str:
    stem = _train_stem(clip)
    spine = _spine(clip)
    return _fill_sparse(
        dedent(
            f"""\
            # Seaquest skills-only ~25M — sparse C/D/A, ego clip{clip}.
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
            EGO_DISTANCE_CLIP_MAX_PX: {clip}
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
            - ego:clip{clip}
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
        )
    )


def _eval_yaml(clip: int) -> str:
    stem = _train_stem(clip)
    spine = _spine(clip)
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
            - ego:clip{clip}
            - ablation:{SPARSE_TAG}
            - probe:newseaquest_spawnfix
            SKILL_INPUT_MODE: ego
            HEURISTIC_MODE: env_specific
            SKILL_DEATH_PENALIZE_COLLECT: true
            __SPARSE_TOP__
            USE_EGO_SHOT_FEATURES: false
            EGO_DISTANCE_CLIP_MAX_PX: {clip}
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


def _transfer_yaml(env: str, meta: dict, clip: int) -> str:
    train = _train_stem(clip)
    stem = _transfer_stem(env, clip)
    spine = _spine(clip)
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
            - suite:{TRANSFER_SUITE}
            - rq:2
            - env:{env}
            - phase:zero_shot
            - skill_input:ego
            - mapper:pov_facing
            - ego:phi_bins8
            - ego:noblocking
            - ego:noweapon
            - ego:clip{clip}
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
            EGO_DISTANCE_CLIP_MAX_PX: {clip}
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


def sweep_stems() -> tuple[str, ...]:
    out: list[str] = []
    for clip in CLIPS:
        out.append(_train_stem(clip))
        out.append(_eval_stem(clip))
    return tuple(out)


def transfer_stems() -> tuple[str, ...]:
    out: list[str] = []
    for clip in CLIPS:
        for env in TRANSFER_ENVS:
            out.append(_transfer_stem(env, clip))
    return tuple(out)


def all_stems() -> tuple[str, ...]:
    return sweep_stems() + transfer_stems()


def main() -> None:
    for clip in CLIPS:
        _write(_RQ1 / f"{_train_stem(clip)}.yaml", _train_yaml(clip))
        _write(_RQ1 / f"{_eval_stem(clip)}.yaml", _eval_yaml(clip))
        for env, meta in TRANSFER_ENVS.items():
            _write(
                _RQ2 / f"{_transfer_stem(env, clip)}.yaml",
                _transfer_yaml(env, meta, clip),
            )
    print(f"\n{SUITE}: {len(sweep_stems())} stems")
    print(f"{TRANSFER_SUITE}: {len(transfer_stems())} stems")


if __name__ == "__main__":
    main()
