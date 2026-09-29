#!/usr/bin/env python3
"""Materialize Seaquest heu-v2 can_fire A/B + ZS transfers (paper_core mods).

Retrain εB skills under the updated Seaquest heuristic (O₂~10%, no-diver→DESTROY,
divers-only hunt, full-AABB DESTROY gate), then home mod eval + zero-shot transfer
onto the 9 RQ2 envs with each env's paper_core mods.

Arms:
  noweapon  — USE_EGO_WEAPON_FEATURES=false
  canfire   — USE_EGO_WEAPON_FEATURES=true (can_fire channel; facing_enemy off)

    python scripts/materialize_skills_heu_v2_transfer.py
    python scripts/run_paper_eval.py --suite skills_heu_v2_transfer_ab \\
        --gpus 0 --wait-deps --skip-rq-gate --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

from concept_based_learning.paper_eval.mods import load_mod_allowlist

_ROOT = Path(__file__).resolve().parents[1]
_OPTIM = _ROOT / "configs" / "experiment" / "paper_eval" / "optim_scale"

SUITE = "skills_heu_v2_transfer_ab"
SEED = 45
EVAL_SEED = 10045

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

ARMS: dict[str, dict] = {
    "noweapon": {
        "weapon": False,
        "tag": "ego:noweapon",
        "extra_tags": [],
    },
    "canfire": {
        "weapon": True,
        "tag": "ego:weapon",
        "extra_tags": ["ego:can_fire_only", "ego:nofacing_enemy"],
    },
}


def train_stem(arm: str) -> str:
    return f"optim_scale_skills_heu_v2_{arm}_seed{SEED}"


def eval_stem(arm: str) -> str:
    return f"{train_stem(arm)}_eval"


def transfer_stem(env: str, arm: str) -> str:
    return f"{env}_zs_heu_v2_{arm}_seed{SEED}"


def _paper_core_keys(env: str) -> list[str]:
    raw = load_mod_allowlist(env)
    keys = list(raw.get("paper_core") or [])
    if keys:
        return keys
    return [e["key"] for e in (raw.get("eval_mods") or []) if e.get("key")]


def _yaml_list(keys: list[str], *, pad: str = "") -> str:
    return "\n".join(f"{pad}- {k}" for k in keys)


def train_stem_list() -> tuple[str, ...]:
    return tuple(train_stem(arm) for arm in ARMS)


def suite_stems() -> tuple[str, ...]:
    stems: list[str] = []
    for arm in ARMS:
        stems.append(train_stem(arm))
    for arm in ARMS:
        stems.append(eval_stem(arm))
    for arm in ARMS:
        for env in TRANSFER_ENVS:
            stems.append(transfer_stem(env, arm))
    return tuple(stems)


def _train_yaml(arm: str) -> str:
    meta = ARMS[arm]
    stem = train_stem(arm)
    weapon = "true" if meta["weapon"] else "false"
    tags = [
        f"suite:{SUITE}",
        "rq:1",
        "env:seaquest",
        "role:base",
        "ego:noblocking",
        meta["tag"],
        *meta["extra_tags"],
        "heu:v2_o2_10pct_divers_only_aabb",
        "ego:noclip",
        "ablation:sparse_cda",
        "budget:50m_optim",
        "scale:pqn_optim_1to1",
        "num_envs:1024",
        "num_steps:32",
        "num_updates:1525",
        "num_epochs:2",
        "num_minibatches:256",
        "lr:0.00025",
        "lr_decay:linear",
        "max_grad_norm:10",
        "eps:1.0to0.001",
        "phase:skills_only",
        "skill_input:ego",
        "mapper:pov_facing",
        "ego:phi_bins8",
    ]
    return (
        dedent(
            f"""\
            # Seaquest ego skills — heu-v2 A/B ({arm}). εB 1:1 HPs; updated heuristic
            # (O₂~10%, no-diver→DESTROY, divers-only hunt, full-AABB DESTROY gate).
            PAPER_EVAL_RQ: 1
            PAPER_EVAL_PHASE: train
            PAPER_EVAL_KIND: skills
            PAPER_EVAL_MODS: seaquest
            PAPER_EVAL_REQUIRES_RQ1: []
            NUM_SEEDS: 1
            ALG_NAME: hierarchical_pqn
            TRAIN_MODE: skills_only
            ENV_NAME: seaquest
            USE_POV_RELATIVE_MAPPER: true
            USE_POV_FIXED_BILATERAL: false
            USE_POV_FACING_FRAME: true
            EGO_POSE_XY: false
            USE_LEGACY_EGO_OBS: false
            USE_EGO_SIZE_FEATURES: false
            USE_EGO_VELOCITY_FEATURES: false
            USE_EGO_BLOCKING_FEATURES: false
            USE_EGO_WEAPON_FEATURES: {weapon}
            USE_EGO_FACING_ENEMY_FEATURES: false
            USE_EGO_SHOT_FEATURES: false
            AVOID_INTERRUPT_RANGE: 55
            AVOID_INTERRUPT_RANGE_Y: 55
            SURFACE_SUB_AVOID_RANGE_X: 5
            SURFACE_SUB_AVOID_RANGE_Y: 5
            EGO_DISTANCE_CLIP_MAX_PX: null
            EGO_ANGLE_BINS: null
            EGO_PSI_ANGLE_BINS: 8
            EGO_LEGAL_ACTION_MASK: false
            SURFACE_SUB_NEAR_RANGE: 5
            USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
            USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
            USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
            SAVE_PATH: ./outputs/models/paper_eval/optim_scale/{stem}
            WANDB_RUN_NAME: sq_{SUITE}_{arm}_seed{SEED}_50m
            WANDB_TAGS:
            """
        )
        + _yaml_list(tags)
        + "\n"
        + dedent(
            """\
            GAME_MODS: null
            EVAL_GAME_MODS: null
            SKILL_INPUT_MODE: ego
            HEURISTIC_MODE: env_specific
            SKILL_DEATH_PENALIZE_COLLECT: true
            RECORD_VIDEO: false
            VIDEO_AT_END: false
            DESTROY_CONTACT_FAIL: true
            DESTROY_LIVING_PENALTY: false
            COLLECT_SPARSE_ONLY: true
            AVOID_SPARSE_ONLY: true
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
              SKILL_LR: 0.00025
              SKILL_REWARD_LAMBDA: 0.0
              CLIP_REWARD: true
              RESTRICT_FIRE_FOR_NON_DESTROY: false
              FIRST_FIRE: true
              STICKY_ACTIONS: false
              EPISODIC_LIFE: true
              FRAME_STACK_SIZE: 4
              FRAME_SKIP: 4
              TOTAL_TIMESTEPS: 50000000
              TOTAL_TIMESTEPS_DECAY: 50000000
              NUM_ENVS: 1024
              NUM_STEPS: 32
              NUM_EPOCHS: 2
              NUM_MINIBATCHES: 256
              GAMMA: 0.99
              SKILL_GAMMA: 0.99
              LAMBDA: 0.65
              SKILL_LAMBDA: 0.65
              LR: 0.00025
              MAX_GRAD_NORM: 10.0
              LR_LINEAR_DECAY: true
              OFF_POLICY: true
              RELABEL_ALL_OBJECTS: false
              RELABEL_K_OBJECTS: null
              SKILL_BOOTSTRAP: q_lambda
              EPS_START: 1.0
              EPS_FINISH: 0.001
              EPS_DECAY: 0.1
              HEURISTIC_MODE: env_specific
              SKILL_INPUT_MODE: ego
              SKILL_DEATH_PENALIZE_COLLECT: true
              DESTROY_CONTACT_FAIL: true
              DESTROY_LIVING_PENALTY: false
              COLLECT_SPARSE_ONLY: true
              AVOID_SPARSE_ONLY: true
              USE_EGO_SHOT_FEATURES: false
            """
        )
        + f"SEED: {SEED}\n"
    )


def _home_eval_yaml(arm: str) -> str:
    meta = ARMS[arm]
    stem = train_stem(arm)
    weapon = "true" if meta["weapon"] else "false"
    mod_keys = _paper_core_keys("seaquest")
    tags = [
        f"suite:{SUITE}",
        "rq:1",
        "env:seaquest",
        "phase:mod_eval",
        "skill_input:ego",
        "mapper:pov_facing",
        "ego:phi_bins8",
        "budget:50m_optim",
        "scale:pqn_optim_1to1",
        meta["tag"],
        "heu:v2",
    ]
    return (
        dedent(
            f"""\
            # Home mod eval — heu-v2 A/B {arm} seed{SEED} (paper_core incl. role-swap).
            PAPER_EVAL_RQ: 1
            PAPER_EVAL_PHASE: eval
            PAPER_EVAL_KIND: skills
            PAPER_EVAL_MODS: seaquest
            PAPER_EVAL_MOD_KEYS:
            """
        )
        + _yaml_list(mod_keys)
        + "\n"
        + dedent(
            f"""\
            PAPER_EVAL_REQUIRES_RQ1:
            - {stem}
            PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/optim_scale/{stem}/seaquest/hierarchical_pqn_seaquest_seed{SEED}.safetensors
            PAPER_EVAL_ARCHITECTURE_OF: {stem}
            NUM_SEEDS: 1
            ALG_NAME: hierarchical_pqn
            TRAIN_MODE: skills_only
            ENV_NAME: seaquest
            USE_POV_RELATIVE_MAPPER: true
            USE_POV_FIXED_BILATERAL: false
            USE_POV_FACING_FRAME: true
            EGO_POSE_XY: false
            USE_LEGACY_EGO_OBS: false
            USE_EGO_SIZE_FEATURES: false
            USE_EGO_VELOCITY_FEATURES: false
            USE_EGO_BLOCKING_FEATURES: false
            USE_EGO_WEAPON_FEATURES: {weapon}
            USE_EGO_FACING_ENEMY_FEATURES: false
            USE_EGO_SHOT_FEATURES: false
            AVOID_INTERRUPT_RANGE: 55
            AVOID_INTERRUPT_RANGE_Y: 55
            SURFACE_SUB_AVOID_RANGE_X: 5
            SURFACE_SUB_AVOID_RANGE_Y: 5
            EGO_DISTANCE_CLIP_MAX_PX: null
            EGO_ANGLE_BINS: null
            EGO_PSI_ANGLE_BINS: 8
            EGO_LEGAL_ACTION_MASK: false
            SURFACE_SUB_NEAR_RANGE: 5
            USE_LEGACY_SEAQUEST_ACTION_MAPPER: false
            USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false
            USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false
            SAVE_PATH: ./outputs/models/paper_eval/optim_scale/{stem}
            WANDB_RUN_NAME: sq_{SUITE}_eval_{arm}_seed{SEED}_50m
            WANDB_TAGS:
            """
        )
        + _yaml_list(tags)
        + "\n"
        + dedent(
            f"""\
            SKILL_INPUT_MODE: ego
            HEURISTIC_MODE: env_specific
            SKILL_DEATH_PENALIZE_COLLECT: true
            DESTROY_CONTACT_FAIL: true
            DESTROY_LIVING_PENALTY: false
            COLLECT_SPARSE_ONLY: true
            AVOID_SPARSE_ONLY: true
            SEED: {SEED}
            EVAL_SEED: {EVAL_SEED}
            EVAL_NUM_SEEDS: 3
            """
        )
    )


def _transfer_yaml(env: str, arm: str) -> str:
    meta_env = TRANSFER_ENVS[env]
    meta_arm = ARMS[arm]
    stem_train = train_stem(arm)
    stem = transfer_stem(env, arm)
    weapon = "true" if meta_arm["weapon"] else "false"
    mod_keys = _paper_core_keys(env)
    tags = [
        f"suite:{SUITE}",
        "rq:2",
        f"env:{env}",
        "phase:zero_shot",
        "skill_input:ego",
        "mapper:pov_facing",
        "ego:phi_bins8",
        "ego:noblocking",
        meta_arm["tag"],
        "heu:v2",
        "ego:noclip",
        "ablation:sparse_cda",
        "budget:50m_optim",
    ]
    return (
        dedent(
            f"""\
            # ZS transfer — Seaquest heu-v2 {arm} → {env} paper_core mods.
            PAPER_EVAL_RQ: 2
            PAPER_EVAL_PHASE: eval
            PAPER_EVAL_KIND: transfer
            PAPER_EVAL_MODS: {env}
            PAPER_EVAL_MOD_KEYS:
            """
        )
        + _yaml_list(mod_keys)
        + "\n"
        + dedent(
            f"""\
            PAPER_EVAL_REQUIRES_RQ1:
            - {stem_train}
            PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/optim_scale/{stem_train}/seaquest/hierarchical_pqn_seaquest_seed{SEED}.safetensors
            PAPER_EVAL_ARCHITECTURE_OF: {stem_train}
            NUM_SEEDS: 1
            ALG_NAME: hierarchical_pqn
            TRAIN_MODE: skills_only
            ENV_NAME: {env}
            USE_POV_RELATIVE_MAPPER: true
            USE_POV_FIXED_BILATERAL: false
            USE_POV_FACING_FRAME: true
            EGO_POSE_XY: false
            SAVE_PATH: ./outputs/models/paper_eval/optim_scale/{stem}
            WANDB_RUN_NAME: {meta_env['prefix']}_{SUITE}_{arm}_seed{SEED}
            WANDB_TAGS:
            """
        )
        + _yaml_list(tags)
        + "\n"
        + dedent(
            f"""\
            USE_LEGACY_EGO_OBS: false
            USE_EGO_SIZE_FEATURES: false
            USE_EGO_VELOCITY_FEATURES: false
            USE_EGO_BLOCKING_FEATURES: false
            USE_EGO_WEAPON_FEATURES: {weapon}
            USE_EGO_FACING_ENEMY_FEATURES: false
            AVOID_INTERRUPT_RANGE: 55
            AVOID_INTERRUPT_RANGE_Y: 55
            SURFACE_SUB_AVOID_RANGE_X: 5
            SURFACE_SUB_AVOID_RANGE_Y: 5
            USE_EGO_SHOT_FEATURES: false
            EGO_DISTANCE_CLIP_MAX_PX: null
            EGO_PSI_ANGLE_BINS: 8
            EGO_LEGAL_ACTION_MASK: false
            GAME_MODS: null
            EVAL_GAME_MODS: null
            SKILL_INPUT_MODE: ego
            HEURISTIC_MODE: {meta_env['heu']}
            SKILL_DEATH_PENALIZE_COLLECT: true
            RECORD_VIDEO: false
            VIDEO_AT_END: false
            DESTROY_CONTACT_FAIL: true
            DESTROY_LIVING_PENALTY: false
            COLLECT_SPARSE_ONLY: true
            AVOID_SPARSE_ONLY: true
            alg:
              NUM_SKILLS: 4
              SKILL_NAMES:
              - DESTROY
              - COLLECT
              - AVOID
              - NAVIGATE
              MAX_OBJECTS: {meta_env['max_objects']}
              EGO_ACTION_DIM: 18
              HIDDEN_SIZE: 128
              NUM_LAYERS: 3
              SKILL_EMBED_DIM: 256
              FRAME_SKIP: 4
              FRAME_STACK_SIZE: 4
              HEURISTIC_MODE: {meta_env['heu']}
              SKILL_INPUT_MODE: ego
              SKILL_DEATH_PENALIZE_COLLECT: true
              DESTROY_CONTACT_FAIL: true
              DESTROY_LIVING_PENALTY: false
              COLLECT_SPARSE_ONLY: true
              AVOID_SPARSE_ONLY: true
              USE_EGO_SHOT_FEATURES: false
            SEED: {SEED}
            EVAL_SEED: {EVAL_SEED}
            EVAL_NUM_SEEDS: 3
            SURFACE_SUB_NEAR_RANGE: 5
            """
        )
    )


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = text if text.endswith("\n") else text + "\n"
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def materialize(*, dry: bool) -> None:
    for arm in ARMS:
        _write(_OPTIM / f"{train_stem(arm)}.yaml", _train_yaml(arm), dry=dry)
        _write(_OPTIM / f"{eval_stem(arm)}.yaml", _home_eval_yaml(arm), dry=dry)
    for arm in ARMS:
        for env in TRANSFER_ENVS:
            _write(
                _OPTIM / f"{transfer_stem(env, arm)}.yaml",
                _transfer_yaml(env, arm),
                dry=dry,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--print-stems", action="store_true")
    args = parser.parse_args()
    if args.print_stems:
        for s in suite_stems():
            print(s)
        return
    materialize(dry=args.dry)
    print(f"\n{len(suite_stems())} stems ({SUITE})")


if __name__ == "__main__":
    main()
