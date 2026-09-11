#!/usr/bin/env python3
"""ALE clip10 heuristic smoke: native skills trains + SQ→target ZS transfers.

Locked spine for most envs: facing φ8, clip10, noblocking, noweapon, surfacesub5
(AVOID ranges kept for config parity; only Seaquest uses them).

Asteroids / Time Pilot scratch trains use turn primitives instead (EGO_ACTION_DIM
6, facing off): honest rotate/thrust (AS) and absolute-aim-under-θ_h (TP). SQ→AS/TP
ZS stays on the facing spine so the Seaquest ckpt can load (strafe≠turn fail cell).

Skills trains (heuristic usability):
  phoenix, namethisgame, timepilot, asteroids,  # EVAL_PLAN bases (+ TP/AS smoke)
  pong, breakout, beamrider, spaceinvaders, freeway  # transfer-family scratch

ZS transfers from existing SQ clip10 ckpt (transfer candidates):
  spaceinvaders, beamrider, timepilot, asteroids, pong, breakout,
  freeway, namethisgame, phoenix
  (pong/beamrider/phoenix clip10 ZS already exist — reused)

Full mod grids TBD — every arm is base-mod only for now.

    python scripts/materialize_ale_clip10_heu_smoke.py
    python scripts/run_paper_eval.py --suite ale_clip10_heu_smoke \\
        --gpus 0,1,2 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

_ROOT = Path(__file__).resolve().parents[1]
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"
_MODS = _ROOT / "configs/experiment/paper_eval/mods"

SUITE = "ale_clip10_heu_smoke"
SPINE = "clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
TURN_SPINE = "clip10_turn_psibins8_v2_noblocking_noweapon_surfacesub5"
# Rotate/thrust family: scratch must use turn primitives (not facing strafe).
TURN_SCRATCH_ENVS = frozenset({"asteroids", "timepilot"})
SQ_TRAIN = f"seaquest_skills_ego_pov_{SPINE}"
TIMESTEPS = 19988480  # ~20M (EVAL_PLAN §3)

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
}

# Existing SQ→env clip10 transfers (do not rewrite).
EXISTING_TRANSFERS = {
    "pong",
    "beamrider",
    "phoenix",
}


def _scratch_spine(env: str) -> str:
    return TURN_SPINE if env in TURN_SCRATCH_ENVS else SPINE


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


def _turn_scratch_yaml(env: str, meta: dict) -> str:
    """Asteroids / Time Pilot: 6 turn primitives, POV-relative ψ from θ_h."""
    stem = f"{env}_from_scratch_ego_pov_{TURN_SPINE}"
    heu = meta["heu"]
    step = 22.5 if env == "asteroids" else 45.0
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
        USE_POV_FACING_FRAME: false
        USE_POV_TURN_PRIMITIVES: true
        POV_TURN_STEP_DEG: {step}
        POV_TURN_INITIAL_HEADING_DEG: null
        EGO_POSE_XY: false
        SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
        WANDB_RUN_NAME: {meta['prefix']}_paper_rq2_scratch_ego_pov_{TURN_SPINE}
        WANDB_TAGS:
        - suite:{SUITE}
        - rq:2
        - env:{env}
        - phase:from_scratch
        - skill_input:ego
        - mapper:pov_relative
        - mapper:turn_primitives
        - ego:psi_bins8
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
          EGO_ACTION_DIM: 6
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
          # Turn primitives have no inverse mapper.
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


def materialize(*, dry: bool) -> None:
    for env, meta in ENVS.items():
        mods_path = _MODS / f"{env}.yaml"
        if not mods_path.exists():
            _write(mods_path, _mods_base_only(env), dry=dry)

        if meta.get("scratch"):
            spine = _scratch_spine(env)
            stem = f"{env}_from_scratch_ego_pov_{spine}"
            body = (
                _turn_scratch_yaml(env, meta)
                if env in TURN_SCRATCH_ENVS
                else _facing_scratch_yaml(env, meta)
            )
            _write(_RQ2 / f"{stem}.yaml", body, dry=dry)
            # Drop stale facing scratch YAMLs for turn envs.
            if env in TURN_SCRATCH_ENVS:
                stale = _RQ2 / f"{env}_from_scratch_ego_pov_{SPINE}.yaml"
                if stale.exists() and not dry:
                    stale.unlink()
                    print(f"removed stale {stale.relative_to(_ROOT)}")
                elif stale.exists() and dry:
                    print(f"DRY remove {stale.relative_to(_ROOT)}")

        if meta.get("transfer"):
            stem = f"{env}_transfer_ego_pov_{SPINE}"
            path = _RQ2 / f"{stem}.yaml"
            if env in EXISTING_TRANSFERS and path.exists():
                print(f"OK keep existing transfer {path.relative_to(_ROOT)}")
                continue
            _write(path, _transfer_yaml(env, meta), dry=dry)


def stem_lists() -> tuple[list[str], list[str]]:
    trains = [
        f"{env}_from_scratch_ego_pov_{_scratch_spine(env)}"
        for env, meta in ENVS.items()
        if meta.get("scratch")
    ]
    transfers = [
        f"{env}_transfer_ego_pov_{SPINE}"
        for env, meta in ENVS.items()
        if meta.get("transfer")
    ]
    return trains, transfers


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
        trains, transfers = stem_lists()
        for s in trains + transfers:
            print(s)
        return
    materialize(dry=args.dry)
    trains, transfers = stem_lists()
    print(f"\n{len(trains)} scratch trains + {len(transfers)} ZS transfers")
    print(f"suite stems (trains first): {len(trains) + len(transfers)}")


if __name__ == "__main__":
    main()
