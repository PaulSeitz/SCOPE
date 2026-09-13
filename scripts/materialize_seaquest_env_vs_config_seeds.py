#!/usr/bin/env python3
"""Materialize 3-seed PQN old_seaquest + clip10 newseaquest OFF_POLICY=true suites.

Writes train/eval YAMLs for seeds 44/45/46 and refreshes suite stem lists are
already wired in ``scripts/run_paper_eval.py``.

  python scripts/materialize_seaquest_env_vs_config_seeds.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RQ1 = ROOT / "configs" / "experiment" / "paper_eval" / "rq1"
SEEDS = (44, 45, 46)


def _pqn_train(seed: int) -> str:
    stem = f"seaquest_pqn_paperhps_oldseaquest_seed{seed}"
    return f"""# Paper-HP PQN on ENV_NAME=old_seaquest (dense diver re-arm), seed={seed}.
# Requires JAXAtari-seaquest-robustness with jax_old_seaquest registered.

PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: train
PAPER_EVAL_KIND: flat_pqn
PAPER_EVAL_MODS: seaquest
NUM_SEEDS: 1

ALG_NAME: flat_pqn
ENV_NAME: old_seaquest
TRAIN_MODE: flat_pqn
OBJECT_CENTRIC: false
PIXEL_RESIZE: true
PIXEL_RESIZE_SHAPE: [84, 84]
PIXEL_GRAYSCALE: true
USE_NATIVE_DOWNSCALING: true
NORM_TYPE: layer_norm
NORM_INPUT: false
LR_LINEAR_DECAY: false
SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}
WANDB_RUN_NAME: sq_paper_rq1_pqn_pixel_paperhps_oldseaquest_seed{seed}
WANDB_TAGS:
- suite:pqn_paperhps_oldseaquest
- rq:1
- env:old_seaquest
- train_mode:flat_pqn
- variant:pqn_pixel_paper_hps
- probe:oldseaquest_paperhps
- seed:{seed}

TOTAL_TIMESTEPS: 24985600
TOTAL_TIMESTEPS_DECAY: 24985600
NUM_ENVS: 128
NUM_STEPS: 32
NUM_EPOCHS: 2
NUM_MINIBATCHES: 32
FRAME_SKIP: 4
FRAME_STACK_SIZE: 4
FIRST_FIRE: true
STICKY_ACTIONS: false
EPISODIC_LIFE: true
CLIP_REWARD: true
GAMMA: 0.99
LAMBDA: 0.65
LR: 0.00025
MAX_GRAD_NORM: 10
HIDDEN_SIZE: 128
NUM_LAYERS: 2
EPS_START: 1.0
EPS_FINISH: 0.001
EPS_DECAY: 0.1
GAME_MODS: null
EVAL_GAME_MODS: null
SEED: {seed}
EVAL_SEED: {seed + 10000}
EVAL_NUM_SEEDS: 3
"""


def _pqn_eval(seed: int) -> str:
    train = f"seaquest_pqn_paperhps_oldseaquest_seed{seed}"
    stem = f"seaquest_pqn_eval_paperhps_oldseaquest_seed{seed}"
    ckpt = (
        f"./outputs/models/paper_eval/rq1/{train}/old_seaquest/"
        f"flat_pqn_old_seaquest_pixel_seed{seed}_vmap0.safetensors"
    )
    return f"""# Mod-eval for {train}. EVAL_SEED≠SEED; EVAL_NUM_SEEDS=3 × 3 episodes.

PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: flat_pqn
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {train}
PAPER_EVAL_FLAT_PQN_CKPT: {ckpt}
PAPER_EVAL_OBJECT_CENTRIC: false
NUM_SEEDS: 1
ALG_NAME: flat_pqn
ENV_NAME: old_seaquest
OBJECT_CENTRIC: false
SAVE_PATH: ./outputs/models/paper_eval/rq1/{train}
WANDB_RUN_NAME: sq_paper_rq1_pqn_eval_paperhps_oldseaquest_seed{seed}
WANDB_TAGS:
- suite:pqn_paperhps_oldseaquest
- rq:1
- env:old_seaquest
- phase:mod_eval
- train_mode:flat_pqn
- probe:oldseaquest_paperhps
- seed:{seed}
SEED: {seed}
EVAL_SEED: {seed + 10000}
EVAL_NUM_SEEDS: 3
"""


def _clip10_train(seed: int) -> str:
    stem = (
        "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
        f"surfacesub5_skillmech_newseaquest_offpolicy_seed{seed}"
    )
    return f"""# clip10 skillmech on spawn-fixed Seaquest, OFF_POLICY=true, seed={seed}.

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
WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_clip10_skillmech_newseaquest_offpolicy_seed{seed}
WANDB_TAGS:
- suite:newseaquest_clip10_offpolicy
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
- ablation:off_policy_true
- probe:newseaquest_spawnfix
- seed:{seed}
SKILL_INPUT_MODE: ego
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
VIDEO_AT_END: false
VIDEO_MAX_FRAMES: 600
VIDEO_FRAME_STRIDE: 4
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
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
SEED: {seed}
EVAL_SEED: {seed + 10000}
EVAL_NUM_SEEDS: 3
SURFACE_SUB_NEAR_RANGE: 5
"""


def _clip10_eval(seed: int) -> str:
    train = (
        "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
        f"surfacesub5_skillmech_newseaquest_offpolicy_seed{seed}"
    )
    stem = (
        "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
        f"surfacesub5_skillmech_eval_full_newseaquest_offpolicy_seed{seed}"
    )
    ckpt = (
        f"./outputs/models/paper_eval/rq1/{train}/seaquest/"
        f"hierarchical_pqn_seaquest_seed{seed}.safetensors"
    )
    return f"""# Mod-eval for {train}. EVAL_SEED≠SEED; EVAL_NUM_SEEDS=3 × 3 episodes.

PAPER_EVAL_RQ: 1
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: skills
PAPER_EVAL_MODS: seaquest
PAPER_EVAL_REQUIRES_RQ1:
- {train}
PAPER_EVAL_SKILLS_CKPT: {ckpt}
PAPER_EVAL_ARCHITECTURE_OF: {train}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: seaquest
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
OFF_POLICY: true
SAVE_PATH: ./outputs/models/paper_eval/rq1/{train}
WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_clip10_skillmech_newseaquest_offpolicy_seed{seed}
WANDB_TAGS:
- suite:newseaquest_clip10_offpolicy
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
- ablation:off_policy_true
- probe:newseaquest_spawnfix
- seed:{seed}
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
SEED: {seed}
EVAL_SEED: {seed + 10000}
EVAL_NUM_SEEDS: 3
SURFACE_SUB_NEAR_RANGE: 5
"""


def main() -> None:
    written: list[Path] = []
    for seed in SEEDS:
        mapping = {
            f"seaquest_pqn_paperhps_oldseaquest_seed{seed}.yaml": _pqn_train(seed),
            f"seaquest_pqn_eval_paperhps_oldseaquest_seed{seed}.yaml": _pqn_eval(seed),
            (
                "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
                f"surfacesub5_skillmech_newseaquest_offpolicy_seed{seed}.yaml"
            ): _clip10_train(seed),
            (
                "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
                f"surfacesub5_skillmech_eval_full_newseaquest_offpolicy_seed{seed}.yaml"
            ): _clip10_eval(seed),
        }
        for name, body in mapping.items():
            path = RQ1 / name
            path.write_text(body)
            written.append(path)
    print(f"Wrote {len(written)} configs under {RQ1.relative_to(ROOT)}")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
