#!/usr/bin/env python3
"""Materialize hyperparameter-sweep suites on the noclip + sparse C/D/A spine.

Stage ``eps`` (done) — exploration sanity; **locked ε = B**
(``0.2 → 0.01 / 0.2``). See suite ``eps_sanity``.

Stage ``skills`` — broader skills HP grid with locked ε B (unless a cell
overrides). Constant ``SKILL_LR`` (no finish). Same mod gate + Pong/Beamrider
base ZS. Each cell = 4 stems (train + SQ eval + 2 ZS).

Stage ``tandem`` — from-scratch pixel-meta tandem ~5M; skill ε locked to B,
``SKILL_LR`` locked to skills-sweep winner ``lr1e3`` (1e-3). Round-2 grid:
meta-ε schedule (incl. PQN-style 1.0), plus LR×ε and switch×ε mixtures.
Home mod eval only (no ZS). Current materializer uses pixel CNN ``[32, 64]``.

Stage ``tandem_ckpt16`` — **eval-only recovery** for checkpoints trained with
pixel CNN ``[16, 32]`` before the baseline rematerialize. Writes frozen
``*_arch_ckpt16.yaml`` + ``*_eval_ckpt16.yaml``; does **not** touch the
``[32, 64]`` train YAMLs. Suite ``hp_sweep_tandem_ckpt16``.

    python scripts/materialize_hp_sweep.py --stage skills
    python scripts/materialize_hp_sweep.py --stage tandem
    python scripts/materialize_hp_sweep.py --stage tandem_ckpt16
    python scripts/materialize_hp_sweep.py --stage all
    python scripts/run_paper_eval.py --suite hp_sweep_skills \\
        --gpus 0,1,2,3 --wait-deps --skip-rq-gate --lean
    python scripts/run_paper_eval.py --suite hp_sweep_tandem \\
        --gpus 0,1,2,3 --wait-deps --skip-rq-gate --lean
    python scripts/run_paper_eval.py --suite hp_sweep_tandem_ckpt16 \\
        --gpus 0,1,2,3 --eval-only --resume --skip-rq-gate --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"
_RQ3 = _ROOT / "configs/experiment/paper_eval/rq3"

SPINE = "noclip_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SPARSE_TAG = "sparse_cda"
# 305 × 128 × 128 — batch-aligned ~5M probe budget.
TIMESTEPS_5M = 4_997_120
BUDGET_TAG = "5m"
# Skills-grid default (hp_sweep_skills baseline / ε-B anchor). Constant LR.
SKILL_LR = 0.0005
# Locked after hp_sweep_skills: lr1e3 won the robustness gate.
LOCKED_SKILL_LR = 0.001

# Locked after eps_sanity (cell B): stable on gate mods.
LOCKED_EPS_START = 0.2
LOCKED_EPS_FINISH = 0.01
LOCKED_EPS_DECAY = 0.2

SUITE_EPS = "eps_sanity"
SUITE_SKILLS = "hp_sweep_skills"
SUITE_TANDEM = "hp_sweep_tandem"
SUITE_TANDEM_CKPT16 = "hp_sweep_tandem_ckpt16"

# Pixel CNN used when the mix-grid trains ran (before baseline rematerialize).
CKPT16_PIXEL_CNN_CHANNELS = (16, 32)

# Home competence / robustness gate (plus base). User ranks the table.
MOD_KEYS: tuple[str, ...] = (
    "base",
    "shift_lanes",
    "lane_scramble",
    "dynamic_lane_drift",
    "change_background_color",
    "no_enemies",
    "gravity",
    "faster_enemies",
)

TRANSFER_ENVS: dict[str, dict] = {
    "pong": {"max_objects": 2, "heu": "nearest", "prefix": "po"},
    "beamrider": {"max_objects": 31, "heu": "env_specific", "prefix": "br"},
}

# tag → (EPS_START, EPS_FINISH, EPS_DECAY)
CELLS_EPS: dict[str, tuple[float, float, float]] = {
    "a_e01": (0.1, 0.01, 0.2),
    "b_e02": (0.2, 0.01, 0.2),
    "c_e05": (0.5, 0.01, 0.2),
    "d_e10_pqn": (1.0, 0.001, 0.1),
    "e_e10": (1.0, 0.01, 0.2),
}

# Skills HP grid — ε locked to B unless a cell overrides EPS_*.
# Explicit lr5e4 duplicates eps_sanity b_e02 so this suite is self-contained.
def _locked_eps(**extra: float | int | str) -> dict:
    d = {
        "EPS_START": LOCKED_EPS_START,
        "EPS_FINISH": LOCKED_EPS_FINISH,
        "EPS_DECAY": LOCKED_EPS_DECAY,
        "SKILL_LR": SKILL_LR,
    }
    d.update(extra)
    return d


CELLS_SKILLS: dict[str, dict] = {
    # --- learning rate (constant) ---
    "lr5e5": _locked_eps(SKILL_LR=5e-5),
    "lr1e4": _locked_eps(SKILL_LR=1e-4),  # NEXUS
    "lr2p5e4": _locked_eps(SKILL_LR=2.5e-4),  # flat PQN
    "lr5e4": _locked_eps(SKILL_LR=5e-4),  # locked B / smoke default
    "lr1e3": _locked_eps(SKILL_LR=1e-3),
    "lr2e3": _locked_eps(SKILL_LR=2e-3),
    # --- ε anneal length (start fixed at 0.2) ---
    "decay01": _locked_eps(EPS_DECAY=0.1),
    "decay05": _locked_eps(EPS_DECAY=0.5),
    "decay08": _locked_eps(EPS_DECAY=0.8),
    # --- ε finish ---
    "finish05": _locked_eps(EPS_FINISH=0.05),  # old code default finish
    # --- grad clip ---
    "clip05": _locked_eps(MAX_GRAD_NORM=0.5),
    "clip10": _locked_eps(MAX_GRAD_NORM=10.0),  # flat PQN
    # --- λ (skill bootstrap) ---
    "lam05": _locked_eps(LAMBDA=0.5, SKILL_LAMBDA=0.5),  # NEXUS options
}

# Tandem round-2 — skills locked (ε B + SKILL_LR=1e-3). Axes are NOT
# one-at-a-time: every cell sets META_LR / META_EPS_* / META_SWITCH_PENALTY
# explicitly so "default" cells are not accidentally duplicated.
#
# Round-1 takeaways (noisy 1-seed): META_LR≈5e-4, switch≈−0.05 preferred;
# −0.1 collapsed; meta ε 0.05–0.1 competitive. Round-2 adds PQN-style high
# meta ε and LR×ε / switch×ε mixes.
def _td(**extra: float | int | str) -> dict:
    d: dict = {
        "META_LR": 5e-4,
        "META_SWITCH_PENALTY": -0.05,
        "META_EPS_START": 0.1,
        "META_EPS_FINISH": 0.01,
        "META_EPS_DECAY": 0.2,
    }
    d.update(extra)
    # Keep EPS_* in sync for train_tandem fallbacks / readability.
    d.setdefault("EPS_START", d["META_EPS_START"])
    d.setdefault("EPS_FINISH", d["META_EPS_FINISH"])
    d.setdefault("EPS_DECAY", d["META_EPS_DECAY"])
    return d


def _td_eps(
    start: float,
    finish: float = 0.01,
    decay: float = 0.2,
    **extra: float | int | str,
) -> dict:
    return _td(
        META_EPS_START=start,
        META_EPS_FINISH=finish,
        META_EPS_DECAY=decay,
        EPS_START=start,
        EPS_FINISH=finish,
        EPS_DECAY=decay,
        **extra,
    )


CELLS_TANDEM: dict[str, dict] = {
    # --- meta ε schedule (META_LR=5e-4, switch=−0.05) ---
    "e005": _td_eps(0.05),
    "e01": _td_eps(0.1),  # round-1-ish anchor
    "e02": _td_eps(0.2),
    "e05": _td_eps(0.5),
    "e10_pqn": _td_eps(1.0, finish=0.001, decay=0.1),  # flat PQN / NEXUS-ish
    "e10": _td_eps(1.0),
    # --- LR × ε (switch=−0.05) ---
    "lr1e3_e005": _td_eps(0.05, META_LR=1e-3),
    "lr1e3_e01": _td_eps(0.1, META_LR=1e-3),
    "lr1e3_e05": _td_eps(0.5, META_LR=1e-3),
    "lr1e3_e10_pqn": _td_eps(1.0, finish=0.001, decay=0.1, META_LR=1e-3),
    # --- free retarget (switch=0) × ε (META_LR=5e-4) ---
    "sw0_e005": _td_eps(0.05, META_SWITCH_PENALTY=0.0),
    "sw0_e01": _td_eps(0.1, META_SWITCH_PENALTY=0.0),
    "sw0_e05": _td_eps(0.5, META_SWITCH_PENALTY=0.0),
    "sw0_e10_pqn": _td_eps(1.0, finish=0.001, decay=0.1, META_SWITCH_PENALTY=0.0),
    # --- triples: higher LR + free switch + strong ε ends ---
    "lr1e3_sw0_e005": _td_eps(0.05, META_LR=1e-3, META_SWITCH_PENALTY=0.0),
    "lr1e3_sw0_e10_pqn": _td_eps(
        1.0, finish=0.001, decay=0.1, META_LR=1e-3, META_SWITCH_PENALTY=0.0
    ),
}


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
    text = body if body.endswith("\n") else body + "\n"
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text)
    print(f"wrote {path.relative_to(_ROOT)}")


def _ego_lines() -> list[str]:
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
        "USE_LEGACY_SEAQUEST_ACTION_MAPPER: false",
        "USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false",
        "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false",
    ]


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


def _mod_keys_yaml() -> list[str]:
    return ["PAPER_EVAL_MOD_KEYS:"] + [f"- {k}" for k in MOD_KEYS]


def train_stem(suite: str, tag: str) -> str:
    return (
        f"seaquest_skills_ego_pov_{SPINE}_{SPARSE_TAG}_{suite}_{tag}_{BUDGET_TAG}"
    )


def tandem_train_stem(tag: str) -> str:
    return (
        f"seaquest_from_scratch_tandem_pixel_ego_pov_{SPINE}_{SPARSE_TAG}"
        f"_{SUITE_TANDEM}_{tag}_{BUDGET_TAG}"
    )


def tandem_eval_stem(tag: str) -> str:
    return f"{tandem_train_stem(tag)}_eval"


def tandem_arch_ckpt16_stem(tag: str) -> str:
    """Frozen architecture snapshot matching already-trained [16, 32] ckpts."""
    return f"{tandem_train_stem(tag)}_arch_ckpt16"


def tandem_eval_ckpt16_stem(tag: str) -> str:
    return f"{tandem_train_stem(tag)}_eval_ckpt16"


def eval_stem(suite: str, tag: str) -> str:
    return f"{train_stem(suite, tag)}_eval"


def transfer_stem(env: str, suite: str, tag: str) -> str:
    return f"{env}_transfer_ego_pov_{SPINE}_{SPARSE_TAG}_{suite}_{tag}_{BUDGET_TAG}"


def _skills_train_yaml(
    *,
    suite: str,
    tag: str,
    eps_start: float,
    eps_finish: float,
    eps_decay: float,
    skill_lr: float = SKILL_LR,
    extra_alg: dict | None = None,
) -> str:
    stem = train_stem(suite, tag)
    lines: list[str] = [
        f"# Seaquest ego skills — {suite} cell {tag} (~5M, constant SKILL_LR).",
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
    lines += _ego_lines()
    lines += [
        f"SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}",
        f"WANDB_RUN_NAME: sq_paper_{suite}_{tag}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{suite}",
        "- rq:1",
        "- env:seaquest",
        "- role:base",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        f"- ablation:{SPARSE_TAG}",
        f"- budget:{BUDGET_TAG}",
        "- phase:skills_only",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        "- lr:constant",
        f"- cell:{tag}",
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        "SKILL_INPUT_MODE: ego",
        "HEURISTIC_MODE: env_specific",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
        "RECORD_VIDEO: false",
        "VIDEO_AT_END: false",
    ]
    extra = extra_alg or {}
    max_grad = float(extra.get("MAX_GRAD_NORM", 0.8))
    lam = float(extra.get("LAMBDA", 0.65))
    skill_lam = float(extra.get("SKILL_LAMBDA", lam))
    lines += _sparse_top()
    lines += [
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
        f"  SKILL_LR: {skill_lr}",
        "  SKILL_REWARD_LAMBDA: 0.0",
        "  CLIP_REWARD: true",
        "  RESTRICT_FIRE_FOR_NON_DESTROY: false",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {TIMESTEPS_5M}",
        "  NUM_ENVS: 128",
        "  NUM_STEPS: 128",
        "  NUM_EPOCHS: 4",
        "  NUM_MINIBATCHES: 64",
        "  GAMMA: 0.99",
        "  SKILL_GAMMA: 0.99",
        f"  LAMBDA: {lam}",
        f"  SKILL_LAMBDA: {skill_lam}",
        "  LR: 0.0001",
        f"  MAX_GRAD_NORM: {max_grad}",
        "  OFF_POLICY: true",
        "  RELABEL_ALL_OBJECTS: false",
        "  RELABEL_K_OBJECTS: null",
        "  SKILL_BOOTSTRAP: q_lambda",
        f"  EPS_START: {eps_start}",
        f"  EPS_FINISH: {eps_finish}",
        f"  EPS_DECAY: {eps_decay}",
        "  HEURISTIC_MODE: env_specific",
        "  SKILL_INPUT_MODE: ego",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_alg()
    skip_extra = {
        "EPS_START",
        "EPS_FINISH",
        "EPS_DECAY",
        "SKILL_LR",
        "MAX_GRAD_NORM",
        "LAMBDA",
        "SKILL_LAMBDA",
    }
    for k, v in extra.items():
        if k in skip_extra:
            continue
        lines.append(f"  {k}: {v}")
    lines += [
        "  USE_EGO_SHOT_FEATURES: false",
        "SEED: 44",
    ]
    return _lines(lines)


def _skills_eval_yaml(*, suite: str, tag: str) -> str:
    stem = train_stem(suite, tag)
    estem = eval_stem(suite, tag)
    ckpt = (
        f"./outputs/models/paper_eval/rq1/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed44.safetensors"
    )
    lines: list[str] = [
        "PAPER_EVAL_RQ: 1",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: skills",
        "PAPER_EVAL_MODS: seaquest",
    ]
    lines += _mod_keys_yaml()
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {ckpt}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        "ENV_NAME: seaquest",
    ]
    lines += _ego_lines()
    lines += [
        f"SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}",
        f"WANDB_RUN_NAME: sq_paper_eval_{suite}_{tag}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{suite}",
        "- rq:1",
        "- env:seaquest",
        "- phase:mod_eval",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        f"- cell:{tag}",
        "SKILL_INPUT_MODE: ego",
        "HEURISTIC_MODE: env_specific",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_top()
    lines += [
        "SEED: 44",
        "EVAL_SEED: 10044",
        "EVAL_NUM_SEEDS: 3",
    ]
    return _lines(lines)


def _transfer_yaml(*, env: str, suite: str, tag: str) -> str:
    meta = TRANSFER_ENVS[env]
    stem = train_stem(suite, tag)
    tstem = transfer_stem(env, suite, tag)
    ckpt = (
        f"./outputs/models/paper_eval/rq1/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed44.safetensors"
    )
    lines: list[str] = [
        "PAPER_EVAL_RQ: 2",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: transfer",
        f"PAPER_EVAL_MODS: {env}",
        "PAPER_EVAL_MOD_KEYS:",
        "- base",
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {ckpt}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        f"ENV_NAME: {env}",
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
        "EGO_PSI_ANGLE_BINS: 8",
        "EGO_LEGAL_ACTION_MASK: false",
        "SURFACE_SUB_NEAR_RANGE: 5",
        f"SAVE_PATH: ./outputs/models/paper_eval/rq2/{tstem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_paper_{suite}_{tag}_zs_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{suite}",
        "- rq:2",
        f"- env:{env}",
        "- phase:zero_shot",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        f"- ablation:{SPARSE_TAG}",
        f"- cell:{tag}",
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
        "  EGO_ACTION_DIM: 18",
        "  HIDDEN_SIZE: 128",
        "  NUM_LAYERS: 3",
        "  SKILL_EMBED_DIM: 256",
        "  FRAME_SKIP: 4",
        "  FRAME_STACK_SIZE: 4",
        f"  HEURISTIC_MODE: {meta['heu']}",
        "  SKILL_INPUT_MODE: ego",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_alg()
    lines += [
        "  USE_EGO_SHOT_FEATURES: false",
        "SEED: 44",
        "EVAL_SEED: 10044",
        "EVAL_NUM_SEEDS: 3",
    ]
    return _lines(lines)


def cell_pipeline_stems(suite: str, tag: str) -> tuple[str, ...]:
    return (
        train_stem(suite, tag),
        eval_stem(suite, tag),
        transfer_stem("pong", suite, tag),
        transfer_stem("beamrider", suite, tag),
    )


def eps_train_stems() -> tuple[str, ...]:
    return tuple(train_stem(SUITE_EPS, t) for t in CELLS_EPS)


def eps_eval_stems() -> tuple[str, ...]:
    out: list[str] = []
    for tag in CELLS_EPS:
        out.extend(cell_pipeline_stems(SUITE_EPS, tag)[1:])
    return tuple(out)


def eps_sanity_stems() -> tuple[str, ...]:
    """Train-first so ``--gpus`` parallelizes the five ~5M trains."""
    return eps_train_stems() + eps_eval_stems()


def skills_sweep_stems() -> tuple[str, ...]:
    trains = tuple(train_stem(SUITE_SKILLS, t) for t in CELLS_SKILLS)
    evals: list[str] = []
    for tag in CELLS_SKILLS:
        evals.extend(cell_pipeline_stems(SUITE_SKILLS, tag)[1:])
    return trains + tuple(evals)


def tandem_sweep_stems() -> tuple[str, ...]:
    trains = tuple(tandem_train_stem(t) for t in CELLS_TANDEM)
    evals = tuple(tandem_eval_stem(t) for t in CELLS_TANDEM)
    return trains + evals


def tandem_ckpt16_eval_stems() -> tuple[str, ...]:
    """Eval-only stems that load existing trains with frozen [16, 32] arch."""
    return tuple(tandem_eval_ckpt16_stem(t) for t in CELLS_TANDEM)


def _materialize_eps(*, dry: bool) -> None:
    for tag, (eps_s, eps_f, eps_d) in CELLS_EPS.items():
        _write(
            _RQ1 / f"{train_stem(SUITE_EPS, tag)}.yaml",
            _skills_train_yaml(
                suite=SUITE_EPS,
                tag=tag,
                eps_start=eps_s,
                eps_finish=eps_f,
                eps_decay=eps_d,
            ),
            dry=dry,
        )
        _write(
            _RQ1 / f"{eval_stem(SUITE_EPS, tag)}.yaml",
            _skills_eval_yaml(suite=SUITE_EPS, tag=tag),
            dry=dry,
        )
        for env in TRANSFER_ENVS:
            _write(
                _RQ2 / f"{transfer_stem(env, SUITE_EPS, tag)}.yaml",
                _transfer_yaml(env=env, suite=SUITE_EPS, tag=tag),
                dry=dry,
            )


def _tandem_train_yaml(
    *,
    tag: str,
    overrides: dict,
    pixel_cnn_channels: tuple[int, int] = (32, 64),
    meta_death_penalty: float = 0.0,
    save_stem: str | None = None,
    header_note: str | None = None,
    suite_tag: str | None = None,
) -> str:
    """Train-shaped tandem YAML. ``save_stem`` controls SAVE_PATH (ckpt dir)."""
    ckpt_stem = save_stem or tandem_train_stem(tag)
    suite = suite_tag or SUITE_TANDEM
    meta_lr = float(overrides.get("META_LR", 5e-4))
    switch = float(overrides.get("META_SWITCH_PENALTY", -0.05))
    eps_start = float(overrides.get("EPS_START", 0.1))
    eps_finish = float(overrides.get("EPS_FINISH", 0.01))
    eps_decay = float(overrides.get("EPS_DECAY", 0.2))
    meta_eps_start = float(overrides.get("META_EPS_START", eps_start))
    meta_eps_finish = float(overrides.get("META_EPS_FINISH", eps_finish))
    meta_eps_decay = float(overrides.get("META_EPS_DECAY", eps_decay))
    c0, c1 = int(pixel_cnn_channels[0]), int(pixel_cnn_channels[1])
    header = header_note or (
        f"# Seaquest tandem — {SUITE_TANDEM} cell {tag} (~5M)."
    )
    lines: list[str] = [
        header,
        "PAPER_EVAL_RQ: 3",
        "PAPER_EVAL_PHASE: train",
        "PAPER_EVAL_KIND: tandem",
        "PAPER_EVAL_MODS: seaquest",
        "PAPER_EVAL_REQUIRES_RQ1: []",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: tandem",
        "META_ARCHITECTURE: per_object_mlp",
        "ENV_NAME: seaquest",
    ]
    lines += _ego_lines()
    lines += [
        f"SAVE_PATH: ./outputs/models/paper_eval/rq3/{ckpt_stem}",
        f"WANDB_RUN_NAME: sq_paper_{suite}_{tag}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{suite}",
        "- rq:3",
        "- env:seaquest",
        "- role:base",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        f"- ablation:{SPARSE_TAG}",
        f"- budget:{BUDGET_TAG}",
        "- phase:from_scratch",
        "- train_mode:tandem",
        "- meta_input:pixel",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        "- probe:tandem_hpsweep",
        f"- cell:{tag}",
        "META_PIXEL_DO_RESIZE: true",
        "META_PIXEL_USE_NATIVE_DOWNSCALING: true",
        "META_PIXEL_RESIZE_SHAPE:",
        "- 84",
        "- 84",
        "META_PIXEL_GRAYSCALE: true",
        "SKILL_PIXEL_CNN_CHANNELS:",
        f"- {c0}",
        f"- {c1}",
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        "META_INPUT_MODE: pixel_embed",
        "SKILL_INPUT_MODE: ego",
        "HEURISTIC_MODE: env_specific",
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
        "  MAX_OBJECTS: 35",
        "  EGO_ACTION_DIM: 18",
        "  HIDDEN_SIZE: 128",
        "  NUM_LAYERS: 3",
        "  META_HIDDEN_SIZE: 256",
        "  META_PER_OBJECT_NUM_LAYERS: 3",
        "  META_FLAT_NUM_LAYERS: 3",
        "  META_EMBED_DIM: 256",
        "  SKILL_EMBED_DIM: 256",
        f"  META_LR: {meta_lr}",
        f"  SKILL_LR: {LOCKED_SKILL_LR}",
        "  META_USE_NOOP_TARGET: true",
        "  META_NOOP_TARGET_IDX: null",
        f"  META_SWITCH_PENALTY: {switch}",
        f"  META_DEATH_PENALTY: {meta_death_penalty}",
        "  META_REWARD_LAMBDA: 1.0",
        "  CLIP_REWARD: true",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {TIMESTEPS_5M}",
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
        f"  EPS_START: {eps_start}",
        f"  EPS_FINISH: {eps_finish}",
        f"  EPS_DECAY: {eps_decay}",
        f"  SKILL_EPS_START: {LOCKED_EPS_START}",
        f"  SKILL_EPS_FINISH: {LOCKED_EPS_FINISH}",
        f"  SKILL_EPS_DECAY: {LOCKED_EPS_DECAY}",
        f"  META_EPS_START: {meta_eps_start}",
        f"  META_EPS_FINISH: {meta_eps_finish}",
        f"  META_EPS_DECAY: {meta_eps_decay}",
        "  META_INPUT_MODE: pixel_embed",
        "  SKILL_INPUT_MODE: ego",
        "  META_PIXEL_CNN_CHANNELS:",
        f"  - {c0}",
        f"  - {c1}",
        "  HEURISTIC_MODE: env_specific",
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


def _tandem_eval_yaml(
    *,
    tag: str,
    architecture_of: str | None = None,
    suite_tag: str | None = None,
    extra_tags: tuple[str, ...] = (),
) -> str:
    stem = tandem_train_stem(tag)
    arch = architecture_of or stem
    suite = suite_tag or SUITE_TANDEM
    skills_ckpt = (
        f"./outputs/models/paper_eval/rq3/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed44_skills.safetensors"
    )
    meta_ckpt = (
        f"./outputs/models/paper_eval/rq3/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed44_meta.safetensors"
    )
    lines: list[str] = [
        "PAPER_EVAL_RQ: 3",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: tandem",
        "PAPER_EVAL_MODS: seaquest",
    ]
    lines += _mod_keys_yaml()
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {skills_ckpt}",
        f"PAPER_EVAL_META_CKPT: {meta_ckpt}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {arch}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: tandem",
        "ENV_NAME: seaquest",
        f"SAVE_PATH: ./outputs/models/paper_eval/rq3/{stem}",
        f"WANDB_RUN_NAME: sq_paper_eval_{suite}_{tag}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{suite}",
        "- rq:3",
        "- env:seaquest",
        "- phase:mod_eval",
        "- train_mode:tandem",
        "- meta_input:pixel",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        f"- cell:{tag}",
    ]
    for t in extra_tags:
        lines.append(f"- {t}")
    lines += [
        "SKILL_INPUT_MODE: ego",
        "META_INPUT_MODE: pixel_embed",
        "HEURISTIC_MODE: env_specific",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_top()
    lines += [
        "EGO_DISTANCE_CLIP_MAX_PX: null",
        "EGO_PSI_ANGLE_BINS: 8",
        "alg:",
        "  SKILL_INPUT_MODE: ego",
        "  META_INPUT_MODE: pixel_embed",
        "  HEURISTIC_MODE: env_specific",
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


def _materialize_skills(*, dry: bool) -> None:
    if not CELLS_SKILLS:
        print(f"(skip {SUITE_SKILLS}: CELLS_SKILLS empty)")
        return
    for tag, overrides in CELLS_SKILLS.items():
        eps_s = float(overrides.get("EPS_START", LOCKED_EPS_START))
        eps_f = float(overrides.get("EPS_FINISH", LOCKED_EPS_FINISH))
        eps_d = float(overrides.get("EPS_DECAY", LOCKED_EPS_DECAY))
        lr = float(overrides.get("SKILL_LR", SKILL_LR))
        _write(
            _RQ1 / f"{train_stem(SUITE_SKILLS, tag)}.yaml",
            _skills_train_yaml(
                suite=SUITE_SKILLS,
                tag=tag,
                eps_start=eps_s,
                eps_finish=eps_f,
                eps_decay=eps_d,
                skill_lr=lr,
                extra_alg=overrides,
            ),
            dry=dry,
        )
        _write(
            _RQ1 / f"{eval_stem(SUITE_SKILLS, tag)}.yaml",
            _skills_eval_yaml(suite=SUITE_SKILLS, tag=tag),
            dry=dry,
        )
        for env in TRANSFER_ENVS:
            _write(
                _RQ2 / f"{transfer_stem(env, SUITE_SKILLS, tag)}.yaml",
                _transfer_yaml(env=env, suite=SUITE_SKILLS, tag=tag),
                dry=dry,
            )


def _materialize_tandem(*, dry: bool) -> None:
    if not CELLS_TANDEM:
        print(f"(skip {SUITE_TANDEM}: CELLS_TANDEM empty)")
        return
    for tag, overrides in CELLS_TANDEM.items():
        _write(
            _RQ3 / f"{tandem_train_stem(tag)}.yaml",
            _tandem_train_yaml(tag=tag, overrides=overrides),
            dry=dry,
        )
        _write(
            _RQ3 / f"{tandem_eval_stem(tag)}.yaml",
            _tandem_eval_yaml(tag=tag),
            dry=dry,
        )


def _materialize_tandem_ckpt16(*, dry: bool) -> None:
    """Frozen arch + eval YAMLs for ckpts trained with pixel CNN [16, 32].

    Does not modify the current [32, 64] train YAMLs. Do not train from
    ``*_arch_ckpt16.yaml`` — architecture reference only.
    """
    if not CELLS_TANDEM:
        print(f"(skip {SUITE_TANDEM_CKPT16}: CELLS_TANDEM empty)")
        return
    for tag, overrides in CELLS_TANDEM.items():
        arch = tandem_arch_ckpt16_stem(tag)
        _write(
            _RQ3 / f"{arch}.yaml",
            _tandem_train_yaml(
                tag=tag,
                overrides=overrides,
                pixel_cnn_channels=CKPT16_PIXEL_CNN_CHANNELS,
                save_stem=tandem_train_stem(tag),
                suite_tag=SUITE_TANDEM_CKPT16,
                header_note=(
                    f"# ARCHITECTURE SNAPSHOT ONLY — do not train.\n"
                    f"# Matches hp_sweep_tandem cell {tag} checkpoints trained "
                    f"with pixel CNN {list(CKPT16_PIXEL_CNN_CHANNELS)} "
                    f"(before baseline rematerialize to [32, 64])."
                ),
            ),
            dry=dry,
        )
        _write(
            _RQ3 / f"{tandem_eval_ckpt16_stem(tag)}.yaml",
            _tandem_eval_yaml(
                tag=tag,
                architecture_of=arch,
                suite_tag=SUITE_TANDEM_CKPT16,
                extra_tags=("recovery:ckpt16_cnn",),
            ),
            dry=dry,
        )


def materialize(*, stage: str, dry: bool) -> None:
    if stage in ("eps", "all"):
        _materialize_eps(dry=dry)
        stems = eps_sanity_stems()
        print(
            f"\n{SUITE_EPS}: {len(CELLS_EPS)} cells, {len(stems)} stems "
            f"(trains-first). MOD_KEYS={list(MOD_KEYS)}"
        )
        for tag, sched in CELLS_EPS.items():
            print(f"  {tag}: EPS {sched[0]} → {sched[1]} / {sched[2]}")
    if stage in ("skills", "all"):
        _materialize_skills(dry=dry)
        stems = skills_sweep_stems()
        print(
            f"\n{SUITE_SKILLS}: {len(CELLS_SKILLS)} cells, {len(stems)} stems "
            f"({len(CELLS_SKILLS)}×4: train+eval+pong+br)"
        )
        for tag in CELLS_SKILLS:
            print(f"  {tag}: {CELLS_SKILLS[tag]}")
    if stage in ("tandem", "all"):
        _materialize_tandem(dry=dry)
        stems = tandem_sweep_stems()
        print(
            f"\n{SUITE_TANDEM}: {len(CELLS_TANDEM)} cells, {len(stems)} stems "
            f"({len(CELLS_TANDEM)}×2: train+home eval; "
            f"skill ε B + LR={LOCKED_SKILL_LR})"
        )
        for tag in CELLS_TANDEM:
            print(f"  {tag}: {CELLS_TANDEM[tag]}")
    if stage in ("tandem_ckpt16", "all"):
        _materialize_tandem_ckpt16(dry=dry)
        stems = tandem_ckpt16_eval_stems()
        print(
            f"\n{SUITE_TANDEM_CKPT16}: {len(stems)} eval-only stems "
            f"(arch frozen to CNN {list(CKPT16_PIXEL_CNN_CHANNELS)}; "
            f"ckpts from {SUITE_TANDEM} trains)"
        )
        for tag in CELLS_TANDEM:
            print(f"  {tag}: arch={tandem_arch_ckpt16_stem(tag)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        choices=("eps", "skills", "tandem", "tandem_ckpt16", "all"),
        default="eps",
        help="Which stage to materialize (default: eps sanity).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    materialize(stage=args.stage, dry=args.dry_run)


if __name__ == "__main__":
    main()
