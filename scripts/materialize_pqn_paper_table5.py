#!/usr/bin/env python3
"""Materialize PQN paper Table-5 outer loop on Seaquest (+ matched skills).

Paper (Gallici et al., ICLR 2025) Table 5 Atari HPs + Table 9 Seaquest @ 400M
frames. With FRAME_SKIP=4 that is 100M agent steps.

    NUM_ENVS=128, NUM_STEPS=32, NUM_EPOCHS=2, NUM_MINIBATCHES=32
    LR=2.5e-4, LR_LINEAR_DECAY=false, MAX_GRAD_NORM=10
    ε 1→0.001 / decay=0.1, γ=0.99, λ=0.65

Suite ``pqn_paper_t5`` (trains-first for --gpus sharding)::

    flat PQN pixel × seeds 44/45/46  (USE_NATIVE_DOWNSCALING=false)
    skills εB (locked skill HPs) × same seeds + same outer
    then home paper_core evals for each
    (+ paper greedy-8 on flat PQN evals → wandb)

    python scripts/materialize_pqn_paper_table5.py
    # PQN-only re-run (no skills retrain):
    python scripts/run_paper_eval.py --suite pqn_paper_t5_pqn \\
      --gpus 0,1,2 --wait-deps --lean --skip-rq-gate
    # Full suite (skills + PQN):
    python scripts/run_paper_eval.py --suite pqn_paper_t5 \\
      --gpus 0,1,2 --wait-deps --lean --skip-rq-gate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import materialize_final_10m as f10

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "configs/experiment/paper_eval/pqn_paper_t5"

SUITE = "pqn_paper_t5"
BUDGET_TAG = "400m_frames"  # 100M agent × skip4
OUT_MODELS = "./outputs/models/paper_eval/pqn_paper_t5"

SEEDS = (44, 45, 46)
TIMESTEPS = 100_000_000
TIMESTEPS_DECAY = 100_000_000
NUM_ENVS = 128
NUM_STEPS = 32
NUM_UPDATES = TIMESTEPS // (NUM_ENVS * NUM_STEPS)  # 24414
NUM_EPOCHS = 2
NUM_MINIBATCHES = 32
LR = 0.00025
MAX_GRAD_NORM = 10.0
EPS_START = 1.0
EPS_FINISH = 0.001
EPS_DECAY = 0.1
# Skills lock (ε B); outer matches paper Table 5.
EPS_B_START = 0.2
EPS_B_FINISH = 0.01
EPS_B_DECAY = 0.2
SKILL_LR = LR

# Paper greedy-8 (Gallici-style long episodes); not training seeds.
PAPER_GREEDY_SEEDS = (7, 11, 13, 17, 19, 23, 29, 31)
PAPER_GREEDY_MAX_FRAMES = 27_000

MOD_KEYS = f10.MOD_KEYS


def pqn_stem(seed: int) -> str:
    return f"pqn_paper_t5_pixel_seed{seed}"


def skills_stem(seed: int) -> str:
    return f"pqn_paper_t5_skills_epsB_seed{seed}"


def pqn_paper_t5_pqn_stems() -> tuple[str, ...]:
    """PQN pixel trains + home mod evals only (re-run without skills)."""
    trains = tuple(pqn_stem(s) for s in SEEDS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def pqn_paper_t5_stems() -> tuple[str, ...]:
    pqn_trains = tuple(pqn_stem(s) for s in SEEDS)
    skills_trains = tuple(skills_stem(s) for s in SEEDS)
    pqn_evals = tuple(f"{s}_eval" for s in pqn_trains)
    skills_evals = tuple(f"{s}_eval" for s in skills_trains)
    return pqn_trains + skills_trains + pqn_evals + skills_evals


def _pqn_train(seed: int) -> str:
    stem = pqn_stem(seed)
    return f10._lines(
        [
            f"# Flat PQN pixel — Seaquest PQN-paper Table 5 "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; 100M agent / 400M frames; "
            f"LR={LR} no decay; MAX_GRAD_NORM={MAX_GRAD_NORM:g}; "
            f"ε {EPS_START}→{EPS_FINISH}); seed={seed}. "
            f"USE_NATIVE_DOWNSCALING=false (ALE-style 84×84 resize). "
            f"Paper Seaquest ref ≈15899 @ 400M frames.",
            "PAPER_EVAL_RQ: 1",
            "PAPER_EVAL_PHASE: train",
            "PAPER_EVAL_KIND: flat_pqn",
            "PAPER_EVAL_MODS: seaquest",
            "PAPER_EVAL_REQUIRES_RQ1: []",
            "NUM_SEEDS: 1",
            "ALG_NAME: flat_pqn",
            "ENV_NAME: seaquest",
            "TRAIN_MODE: flat_pqn",
            "OBJECT_CENTRIC: false",
            "PIXEL_RESIZE: true",
            "PIXEL_RESIZE_SHAPE:",
            "- 84",
            "- 84",
            "PIXEL_GRAYSCALE: true",
            "USE_NATIVE_DOWNSCALING: false",
            "NORM_TYPE: layer_norm",
            "NORM_INPUT: false",
            "LR_LINEAR_DECAY: false",
            f"SAVE_PATH: {OUT_MODELS}/{stem}",
            f"WANDB_RUN_NAME: sq_{SUITE}_pqn_pixel_seed{seed}_{BUDGET_TAG}",
            "WANDB_TAGS:",
            f"- suite:{SUITE}",
            "- rq:1",
            "- env:seaquest",
            "- role:base",
            "- train_mode:flat_pqn",
            "- variant:pqn_pixel_paper_table5",
            "- native_downscale:false",
            f"- budget:{BUDGET_TAG}",
            "- scale:pqn_paper_t5",
            f"- num_envs:{NUM_ENVS}",
            f"- num_steps:{NUM_STEPS}",
            f"- num_updates:{NUM_UPDATES}",
            f"- num_epochs:{NUM_EPOCHS}",
            f"- num_minibatches:{NUM_MINIBATCHES}",
            f"- lr:{LR}",
            "- lr_decay:none",
            f"- max_grad_norm:{int(MAX_GRAD_NORM)}",
            f"- eps:{EPS_START}to{EPS_FINISH}",
            f"TOTAL_TIMESTEPS: {TIMESTEPS}",
            f"TOTAL_TIMESTEPS_DECAY: {TIMESTEPS_DECAY}",
            f"NUM_ENVS: {NUM_ENVS}",
            f"NUM_STEPS: {NUM_STEPS}",
            f"NUM_EPOCHS: {NUM_EPOCHS}",
            f"NUM_MINIBATCHES: {NUM_MINIBATCHES}",
            "FRAME_SKIP: 4",
            "FRAME_STACK_SIZE: 4",
            "FIRST_FIRE: true",
            "STICKY_ACTIONS: false",
            "EPISODIC_LIFE: true",
            "CLIP_REWARD: true",
            "GAMMA: 0.99",
            "LAMBDA: 0.65",
            f"LR: {LR}",
            f"MAX_GRAD_NORM: {MAX_GRAD_NORM}",
            "HIDDEN_SIZE: 128",
            "NUM_LAYERS: 2",
            f"EPS_START: {EPS_START}",
            f"EPS_FINISH: {EPS_FINISH}",
            f"EPS_DECAY: {EPS_DECAY}",
            "GAME_MODS: null",
            "EVAL_GAME_MODS: null",
            f"SEED: {seed}",
        ]
    )


def _pqn_eval(seed: int) -> str:
    stem = pqn_stem(seed)
    ckpt = (
        f"{OUT_MODELS}/{stem}/seaquest/"
        f"flat_pqn_seaquest_pixel_seed{seed}_vmap0.safetensors"
    )
    lines = [
        f"# Home mod eval + paper greedy-8 — PQN Table 5 pixel seed{seed} "
        f"(native_downscale=false).",
        "PAPER_EVAL_RQ: 1",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: flat_pqn",
        "PAPER_EVAL_MODS: seaquest",
    ]
    lines += f10._mod_keys_yaml(MOD_KEYS)
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_FLAT_PQN_CKPT: {ckpt}",
        "PAPER_EVAL_OBJECT_CENTRIC: false",
        # After CSV mod eval, also run Gallici-style greedy-8 → wandb.
        "PAPER_EVAL_PAPER_GREEDY8: true",
        "PAPER_EVAL_PAPER_GREEDY8_MAX_FRAMES: "
        f"{PAPER_GREEDY_MAX_FRAMES}",
        "PAPER_EVAL_PAPER_GREEDY8_SEEDS:",
        *[f"- {s}" for s in PAPER_GREEDY_SEEDS],
        "NUM_SEEDS: 1",
        "ALG_NAME: flat_pqn",
        "ENV_NAME: seaquest",
        "OBJECT_CENTRIC: false",
        "USE_NATIVE_DOWNSCALING: false",
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: sq_{SUITE}_eval_pqn_pixel_seed{seed}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        "- rq:1",
        "- env:seaquest",
        "- phase:mod_eval",
        "- train_mode:flat_pqn",
        "- variant:pqn_pixel_paper_table5",
        "- native_downscale:false",
        f"- budget:{BUDGET_TAG}",
        "- scale:pqn_paper_t5",
        f"SEED: {seed}",
        f"EVAL_SEED: {seed + 10_000}",
        "EVAL_NUM_SEEDS: 3",
    ]
    return f10._lines(lines)


def _skills_train(seed: int) -> str:
    stem = skills_stem(seed)
    lines = [
        f"# Seaquest ego skills εB — PQN-paper Table 5 outer "
        f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; 100M agent / 400M frames); "
        f"SKILL_LR={SKILL_LR} no decay; ε B; seed={seed}.",
        "PAPER_EVAL_RQ: 1",
        "PAPER_EVAL_PHASE: train",
        "PAPER_EVAL_KIND: skills",
        "PAPER_EVAL_MODS: seaquest",
        "PAPER_EVAL_REQUIRES_RQ1: []",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        "ENV_NAME: seaquest",
    ]
    lines += f10._ego_lines(seaquest_legacy=True)
    lines += [
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: sq_{SUITE}_skills_epsB_seed{seed}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        "- rq:1",
        "- env:seaquest",
        "- role:base",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        "- ablation:sparse_cda",
        f"- budget:{BUDGET_TAG}",
        "- scale:pqn_paper_t5",
        f"- num_envs:{NUM_ENVS}",
        f"- num_steps:{NUM_STEPS}",
        f"- num_updates:{NUM_UPDATES}",
        f"- num_epochs:{NUM_EPOCHS}",
        f"- num_minibatches:{NUM_MINIBATCHES}",
        f"- lr:{SKILL_LR}",
        "- lr_decay:none",
        f"- max_grad_norm:{int(MAX_GRAD_NORM)}",
        "- eps:B",
        "- phase:skills_only",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        "SKILL_INPUT_MODE: ego",
        "HEURISTIC_MODE: env_specific",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
        "RECORD_VIDEO: false",
        "VIDEO_AT_END: false",
        "DESTROY_CONTACT_FAIL: true",
        "DESTROY_LIVING_PENALTY: false",
        "COLLECT_SPARSE_ONLY: true",
        "AVOID_SPARSE_ONLY: true",
        "alg:",
        "  NUM_SKILLS: 4",
        "  SKILL_NAMES:",
        "  - DESTROY",
        "  - COLLECT",
        "  - AVOID",
        "  - NAVIGATE",
        "  MAX_OBJECTS: 35",
        "  EGO_ACTION_DIM: 18",
        "  HIDDEN_SIZE: 128",
        "  NUM_LAYERS: 3",
        "  SKILL_EMBED_DIM: 256",
        f"  SKILL_LR: {SKILL_LR}",
        "  SKILL_REWARD_LAMBDA: 0.0",
        "  CLIP_REWARD: true",
        "  RESTRICT_FIRE_FOR_NON_DESTROY: false",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {TIMESTEPS}",
        f"  TOTAL_TIMESTEPS_DECAY: {TIMESTEPS_DECAY}",
        f"  NUM_ENVS: {NUM_ENVS}",
        f"  NUM_STEPS: {NUM_STEPS}",
        f"  NUM_EPOCHS: {NUM_EPOCHS}",
        f"  NUM_MINIBATCHES: {NUM_MINIBATCHES}",
        "  GAMMA: 0.99",
        "  SKILL_GAMMA: 0.99",
        "  LAMBDA: 0.65",
        "  SKILL_LAMBDA: 0.65",
        f"  LR: {LR}",
        f"  MAX_GRAD_NORM: {MAX_GRAD_NORM}",
        "  LR_LINEAR_DECAY: false",
        "  OFF_POLICY: true",
        "  RELABEL_ALL_OBJECTS: false",
        "  RELABEL_K_OBJECTS: null",
        "  SKILL_BOOTSTRAP: q_lambda",
        f"  EPS_START: {EPS_B_START}",
        f"  EPS_FINISH: {EPS_B_FINISH}",
        f"  EPS_DECAY: {EPS_B_DECAY}",
        "  HEURISTIC_MODE: env_specific",
        "  SKILL_INPUT_MODE: ego",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
        "  DESTROY_CONTACT_FAIL: true",
        "  DESTROY_LIVING_PENALTY: false",
        "  COLLECT_SPARSE_ONLY: true",
        "  AVOID_SPARSE_ONLY: true",
        "  USE_EGO_SHOT_FEATURES: false",
        f"SEED: {seed}",
    ]
    return f10._lines(lines)


def _skills_eval(seed: int) -> str:
    stem = skills_stem(seed)
    ckpt = (
        f"{OUT_MODELS}/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed{seed}.safetensors"
    )
    lines = [
        f"# Home mod eval — PQN paper Table 5 skills εB seed{seed}.",
        "PAPER_EVAL_RQ: 1",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: skills",
        "PAPER_EVAL_MODS: seaquest",
    ]
    lines += f10._mod_keys_yaml(MOD_KEYS)
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {ckpt}",
        "PAPER_EVAL_ARCHITECTURE_OF: " + stem,
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "ENV_NAME: seaquest",
        "TRAIN_MODE: skills_only",
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: sq_{SUITE}_eval_skills_epsB_seed{seed}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        "- rq:1",
        "- env:seaquest",
        "- phase:mod_eval",
        "- train_mode:skills_only",
        "- eps:B",
        f"- budget:{BUDGET_TAG}",
        "- scale:pqn_paper_t5",
        f"SEED: {seed}",
        f"EVAL_SEED: {seed + 10_000}",
        "EVAL_NUM_SEEDS: 3",
    ]
    return f10._lines(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, str]] = []
    for seed in SEEDS:
        pairs.append((_OUT / f"{pqn_stem(seed)}.yaml", _pqn_train(seed)))
        pairs.append((_OUT / f"{pqn_stem(seed)}_eval.yaml", _pqn_eval(seed)))
        pairs.append((_OUT / f"{skills_stem(seed)}.yaml", _skills_train(seed)))
        pairs.append((_OUT / f"{skills_stem(seed)}_eval.yaml", _skills_eval(seed)))
    print(
        f"pqn_paper_t5 → {_OUT.relative_to(_ROOT)}  "
        f"T={TIMESTEPS} envs={NUM_ENVS} steps={NUM_STEPS} updates={NUM_UPDATES} "
        f"seeds={list(SEEDS)}"
    )
    for path, body in pairs:
        if args.dry_run:
            print(f"  would write {path.relative_to(_ROOT)}")
        else:
            path.write_text(body)
            print(f"  wrote {path.relative_to(_ROOT)}")
    print("stems:", list(pqn_paper_t5_stems()))


if __name__ == "__main__":
    main()
