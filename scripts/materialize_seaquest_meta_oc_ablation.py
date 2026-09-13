#!/usr/bin/env python3
"""Materialize Seaquest tandem meta-OC ablation configs (train + home mod eval).

Suite: ``seaquest_meta_oc_ablation``
Spine matches ale_clip10 tandem ego settings (~25M from-scratch).

Modes:
  oc_embed            — legacy full leaves + raw flatten (control)
  oc_simple           — player box + O₂/divers; absolute bbox + label OH
  oc_simple_rel       — same global; relative Δx/Δy/dist/lane + label OH
  oc_simple_role      — oc_simple + role-pool counts/min-dist globals
  oc_simple_rel_role  — rel objects + role-pool globals
  oc_simple_inv       — oc_simple + lives + on-screen active COLLECT count

    python scripts/materialize_seaquest_meta_oc_ablation.py
    python scripts/run_paper_eval.py --suite seaquest_meta_oc_ablation \\
        --gpus 0,1,2 --wait-deps --lean
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

_ROOT = Path(__file__).resolve().parents[1]
_RQ3 = _ROOT / "configs/experiment/paper_eval/rq3"

SUITE = "seaquest_meta_oc_ablation"
SPINE = "clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
TIMESTEPS = 24_985_600

# (meta_input_mode, stem_token, wandb meta tag)
MODES = (
    ("oc_embed", "oc_embed", "oc"),
    ("oc_simple", "oc_simple", "oc_simple"),
    ("oc_simple_rel", "oc_simple_rel", "oc_simple_rel"),
    ("oc_simple_role", "oc_simple_role", "oc_simple_role"),
    ("oc_simple_rel_role", "oc_simple_rel_role", "oc_simple_rel_role"),
    ("oc_simple_inv", "oc_simple_inv", "oc_simple_inv"),
)


def _stem(token: str) -> str:
    return f"seaquest_from_scratch_tandem_{token}_ego_pov_{SPINE}"


def _train_yaml(mode: str, token: str, meta_tag: str) -> str:
    stem = _stem(token)
    return dedent(
        f"""\
        # seaquest tandem from-scratch — {mode} meta + facing ego skills (~25M).
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
        SAVE_PATH: ./outputs/models/paper_eval/rq3/{stem}
        WANDB_RUN_NAME: sq_paper_tandem_{token}_ego_pov_{SPINE}
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:3
        - env:seaquest
        - phase:from_scratch
        - train_mode:tandem
        - meta_input:{meta_tag}
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - probe:meta_oc_ablation
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
        META_INPUT_MODE: {mode}
        SKILL_INPUT_MODE: ego
        HEURISTIC_MODE: env_specific
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
          META_INPUT_MODE: {mode}
          SKILL_INPUT_MODE: ego
          META_PIXEL_CNN_CHANNELS:
          - 16
          - 32
          HEURISTIC_MODE: env_specific
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


def _eval_yaml(mode: str, token: str, meta_tag: str) -> str:
    stem = _stem(token)
    ckpt_dir = f"./outputs/models/paper_eval/rq3/{stem}/seaquest"
    return dedent(
        f"""\
        # Home-env mod eval for {stem} (skills + meta ckpts).
        PAPER_EVAL_RQ: 3
        PAPER_EVAL_PHASE: eval
        PAPER_EVAL_KIND: tandem
        PAPER_EVAL_MODS: seaquest
        PAPER_EVAL_REQUIRES_RQ1:
        - {stem}
        PAPER_EVAL_SKILLS_CKPT: {ckpt_dir}/hierarchical_pqn_seaquest_seed44_skills.safetensors
        PAPER_EVAL_META_CKPT: {ckpt_dir}/hierarchical_pqn_seaquest_seed44_meta.safetensors
        PAPER_EVAL_ARCHITECTURE_OF: {stem}
        NUM_SEEDS: 1
        ALG_NAME: hierarchical_pqn
        TRAIN_MODE: tandem
        META_ARCHITECTURE: per_object_mlp
        ENV_NAME: seaquest
        SAVE_PATH: ./outputs/models/paper_eval/rq3/{stem}
        WANDB_RUN_NAME: sq_paper_eval_tandem_{token}_ego_pov_{SPINE}
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:3
        - env:seaquest
        - phase:mod_eval
        - train_mode:tandem
        - meta_input:{meta_tag}
        - skill_input:ego
        - mapper:pov_facing
        - ego:phi_bins8
        - ego:noblocking
        - ego:noweapon
        - ego:clip10
        - probe:meta_oc_ablation
        USE_POV_RELATIVE_MAPPER: true
        USE_POV_FIXED_BILATERAL: false
        USE_POV_FACING_FRAME: true
        EGO_POSE_XY: false
        USE_LEGACY_EGO_OBS: false
        USE_EGO_SIZE_FEATURES: false
        USE_EGO_VELOCITY_FEATURES: false
        USE_EGO_BLOCKING_FEATURES: false
        USE_EGO_WEAPON_FEATURES: false
        USE_EGO_SHOT_FEATURES: false
        SKILL_INPUT_MODE: ego
        META_INPUT_MODE: {mode}
        HEURISTIC_MODE: env_specific
        SKILL_DEATH_PENALIZE_COLLECT: true
        DESTROY_CONTACT_FAIL: true
        DESTROY_LIVING_PENALTY: true
        EGO_DISTANCE_CLIP_MAX_PX: 10
        EGO_PSI_ANGLE_BINS: 8
        alg:
          META_ARCHITECTURE: per_object_mlp
          SKILL_INPUT_MODE: ego
          META_INPUT_MODE: {mode}
          HEURISTIC_MODE: env_specific
          SKILL_DEATH_PENALIZE_COLLECT: true
          DESTROY_CONTACT_FAIL: true
          DESTROY_LIVING_PENALTY: true
          USE_EGO_SHOT_FEATURES: false
          EGO_DISTANCE_CLIP_MAX_PX: 10
          EGO_PSI_ANGLE_BINS: 8
          MAX_OBJECTS: 35
          NUM_SKILLS: 4
        SEED: 44
        """
    )


def main() -> None:
    _RQ3.mkdir(parents=True, exist_ok=True)
    written = []
    for mode, token, meta_tag in MODES:
        stem = _stem(token)
        train_path = _RQ3 / f"{stem}.yaml"
        eval_path = _RQ3 / f"{stem}_eval.yaml"
        train_path.write_text(_train_yaml(mode, token, meta_tag))
        eval_path.write_text(_eval_yaml(mode, token, meta_tag))
        written.extend([train_path, eval_path])
    print(f"Wrote {len(written)} configs under {_RQ3}")
    for p in written:
        print(f"  {p.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
