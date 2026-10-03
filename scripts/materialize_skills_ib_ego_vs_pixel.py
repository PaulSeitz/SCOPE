#!/usr/bin/env python3
"""Materialize Seaquest 50M skills IB ablation: ego vs pixel_embed (+target outline).

Fresh SAVE_PATHs (do **not** reuse old optim_scale heu-v2 ckpts — Seaquest spawn
logic changed). HPs / heuristic / rewards match heu-v2 εB optim-scale 1:1
(50M, 1024 envs, sparse C/D/A, POV facing). Only ``SKILL_INPUT_MODE`` differs:

  ego           — engineered relative features (informational bottleneck)
  pixel_embed   — CNN on 84×84 gray stack with target bbox outline painted

    python scripts/materialize_skills_ib_ego_vs_pixel.py
    python scripts/run_paper_eval.py --suite skills_ib_ego_vs_pixel \\
        --gpus 0 --wait-deps --skip-rq-gate --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

from concept_based_learning.paper_eval.mods import load_mod_allowlist

_ROOT = Path(__file__).resolve().parents[1]
_OPTIM = _ROOT / "configs" / "experiment" / "paper_eval" / "optim_scale"

SUITE = "skills_ib_ego_vs_pixel"
SEED = 45
EVAL_SEED = 10045

EGO_TRAIN = f"skills_ib_ego_heu_v2_seed{SEED}"
EGO_EVAL = f"{EGO_TRAIN}_eval"
PIXEL_TRAIN = f"skills_ib_pixel_outline_heu_v2_seed{SEED}"
PIXEL_EVAL = f"{PIXEL_TRAIN}_eval"


def suite_stems() -> tuple[str, ...]:
    return (EGO_TRAIN, PIXEL_TRAIN, EGO_EVAL, PIXEL_EVAL)


def _paper_core_keys(env: str) -> list[str]:
    raw = load_mod_allowlist(env)
    keys = list(raw.get("paper_core") or [])
    if keys:
        return keys
    return [e["key"] for e in (raw.get("eval_mods") or []) if e.get("key")]


def _yaml_list(keys: list[str]) -> str:
    return "\n".join(f"- {k}" for k in keys)


def _train_yaml(*, stem: str, arm: str, skill_input: str) -> str:
    is_pixel = skill_input == "pixel_embed"
    tags = [
        f"suite:{SUITE}",
        "rq:1",
        "env:seaquest",
        "role:base",
        "ego:noblocking",
        "ego:noweapon",
        "heu:v2_o2_10pct_divers_only_aabb",
        "ego:noclip",
        "ablation:sparse_cda",
        "ablation:ib_ego_vs_pixel",
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
        f"skill_input:{skill_input}",
        *(["pixel:target_outline"] if is_pixel else []),
        "mapper:pov_facing",
        "ego:phi_bins8",
        "seaquest:spawnfix_retrain",
    ]
    pixel_top = ""
    pixel_alg = ""
    if is_pixel:
        pixel_top = dedent(
            """\
            # Soft risk: skill CNN @ 1024 envs may OOM (same caveat as tandem pixel).
            META_PIXEL_DO_RESIZE: true
            META_PIXEL_USE_NATIVE_DOWNSCALING: true
            META_PIXEL_RESIZE_SHAPE:
            - 84
            - 84
            META_PIXEL_GRAYSCALE: true
            SKILL_PIXEL_CNN_CHANNELS:
            - 32
            - 64
            SKILL_PIXEL_PAINT_TARGET_OUTLINE: true
            """
        )
        pixel_alg = (
            "  SKILL_PIXEL_CNN_CHANNELS:\n"
            "  - 32\n"
            "  - 64\n"
            "  SKILL_PIXEL_PAINT_TARGET_OUTLINE: true\n"
        )
    return (
        dedent(
            f"""\
            # Seaquest {arm} skills — fresh heu-v2 εB 50M IB arm (new SAVE_PATH;
            # not the pre-spawn-fix optim_scale heu_v2 ckpt).
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
            USE_EGO_WEAPON_FEATURES: false
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
        + pixel_top
        + dedent(
            f"""\
            GAME_MODS: null
            EVAL_GAME_MODS: null
            SKILL_INPUT_MODE: {skill_input}
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
              SKILL_INPUT_MODE: {skill_input}
            """
        )
        + pixel_alg
        + (
            "  SKILL_DEATH_PENALIZE_COLLECT: true\n"
            "  DESTROY_CONTACT_FAIL: true\n"
            "  DESTROY_LIVING_PENALTY: false\n"
            "  COLLECT_SPARSE_ONLY: true\n"
            "  AVOID_SPARSE_ONLY: true\n"
            "  USE_EGO_SHOT_FEATURES: false\n"
        )
        + f"SEED: {SEED}\n"
    )


def _eval_yaml(*, train_stem: str, arm: str, skill_input: str) -> str:
    is_pixel = skill_input == "pixel_embed"
    mod_keys = _paper_core_keys("seaquest")
    tags = [
        f"suite:{SUITE}",
        "rq:1",
        "env:seaquest",
        "phase:mod_eval",
        f"skill_input:{skill_input}",
        *(["pixel:target_outline"] if is_pixel else []),
        "mapper:pov_facing",
        "ego:phi_bins8",
        "budget:50m_optim",
        "scale:pqn_optim_1to1",
        "ablation:ib_ego_vs_pixel",
        "heu:v2",
        "seaquest:spawnfix_retrain",
    ]
    pixel_block = ""
    if is_pixel:
        pixel_block = dedent(
            """\
            META_PIXEL_DO_RESIZE: true
            META_PIXEL_USE_NATIVE_DOWNSCALING: true
            META_PIXEL_RESIZE_SHAPE:
            - 84
            - 84
            META_PIXEL_GRAYSCALE: true
            SKILL_PIXEL_CNN_CHANNELS:
            - 32
            - 64
            SKILL_PIXEL_PAINT_TARGET_OUTLINE: true
            """
        )
    return (
        dedent(
            f"""\
            # Home mod eval — IB {arm} seed{SEED} (paper_core; fresh spawn-fix ckpt).
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
            - {train_stem}
            PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/optim_scale/{train_stem}/seaquest/hierarchical_pqn_seaquest_seed{SEED}.safetensors
            PAPER_EVAL_ARCHITECTURE_OF: {train_stem}
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
            USE_EGO_WEAPON_FEATURES: false
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
            SAVE_PATH: ./outputs/models/paper_eval/optim_scale/{train_stem}
            WANDB_RUN_NAME: sq_{SUITE}_eval_{arm}_seed{SEED}_50m
            WANDB_TAGS:
            """
        )
        + _yaml_list(tags)
        + "\n"
        + pixel_block
        + dedent(
            f"""\
            SKILL_INPUT_MODE: {skill_input}
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


def materialize(*, force: bool = False) -> list[Path]:
    _OPTIM.mkdir(parents=True, exist_ok=True)
    for stale in (
        f"optim_scale_skills_heu_v2_pixel_outline_seed{SEED}.yaml",
        f"optim_scale_skills_heu_v2_pixel_outline_seed{SEED}_eval.yaml",
    ):
        stale_path = _OPTIM / stale
        if stale_path.is_file():
            stale_path.unlink()
            print(f"removed stale: {stale_path.relative_to(_ROOT)}")

    written: list[Path] = []
    specs = (
        (EGO_TRAIN, _train_yaml(stem=EGO_TRAIN, arm="ego", skill_input="ego")),
        (
            PIXEL_TRAIN,
            _train_yaml(
                stem=PIXEL_TRAIN, arm="pixel_outline", skill_input="pixel_embed"
            ),
        ),
        (
            EGO_EVAL,
            _eval_yaml(train_stem=EGO_TRAIN, arm="ego", skill_input="ego"),
        ),
        (
            PIXEL_EVAL,
            _eval_yaml(
                train_stem=PIXEL_TRAIN,
                arm="pixel_outline",
                skill_input="pixel_embed",
            ),
        ),
    )
    for stem, body in specs:
        path = _OPTIM / f"{stem}.yaml"
        if path.is_file() and not force and path.read_text() == body:
            print(f"unchanged: {path.relative_to(_ROOT)}")
        else:
            path.write_text(body)
            print(f"wrote: {path.relative_to(_ROOT)}")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    materialize(force=bool(args.force))
    print("suite stems:", ", ".join(suite_stems()))


if __name__ == "__main__":
    main()
