#!/usr/bin/env python3
"""Materialize the locked Seaquest final-eval configs at a matched 10M budget.

All CBL paper_eval arms live under::

    configs/experiment/paper_eval/final_10m/

NEXUS/HRL overlays (different runner) live under::

    configs/experiment/nexus/alg/*_final_10m.yaml

Budget is batch-aligned ``610 × 128 × 128 = 9_994_240`` for hierarchical
arms (NUM_STEPS=128) and the same agent-step total for flat PQN
(``NUM_STEPS=32`` → 2440 updates).

Locked HPs
----------
* Skills: ε B (0.2→0.01/0.2) + SKILL_LR=1e-3 (from ``hp_sweep_skills``).
* Tandem pixel / OC: e01 lock (meta ε 0.1→0.01/0.2, META_LR=5e-4,
  switch=−0.05; skill ε B + SKILL_LR=1e-3; CNN [32, 64]; death=0).
* Tandem OC meta: ``oc_simple_rel_role`` (strongest abs in clip10 sparse
  meta-OC probe; embed farms ``no_enemies``).
* Flat PQN pixel/OC: Gallici Table-5 HPs, matched 10M steps.
* HRL / NEXUS symbolic: CBL ``ourhps`` schedule at 10M (128 envs).

    python scripts/materialize_final_10m.py
    python scripts/materialize_final_10m.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "configs/experiment/paper_eval/final_10m"
_NEXUS_ALG = _ROOT / "configs/experiment/nexus/alg"

SUITE = "final_10m"
BUDGET_TAG = "10m"
TIMESTEPS_10M = 9_994_240  # 610 × 128 × 128
SPINE = "noclip_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
OUT_MODELS = "./outputs/models/paper_eval/final_10m"

LOCKED_SKILL_LR = 0.001
LOCKED_EPS_START = 0.2
LOCKED_EPS_FINISH = 0.01
LOCKED_EPS_DECAY = 0.2

# Home competence / robustness gate (paper_core).
MOD_KEYS: tuple[str, ...] = (
    "base",
    "shift_lanes",
    "lane_scramble",
    "dynamic_lane_drift",
    "random_spawns",
    "change_background_color",
    "no_enemies",
    "swap_diver_enemy_labels",
    "gravity",
    "faster_enemies",
)

# Stems — globally unique (run_paper_eval resolves by stem via rglob).
STEM_SKILLS = "final_10m_skills_ego"  # seaquest (legacy short stem)
STEM_TANDEM_PIXEL = "final_10m_tandem_pixel_ego"
STEM_TANDEM_OC = "final_10m_tandem_oc_rel_role_ego"
STEM_PQN_PIXEL = "final_10m_pqn_pixel"  # seaquest (legacy short stem)
STEM_PQN_OC = "final_10m_pqn_oc"  # seaquest OC only (for now)

NEXUS_HRL = "cbl_seaquest_hrl_baseline_final_10m"
NEXUS_SYMBOLIC = "cbl_seaquest_nexus_symbolic_final_10m"

# Locked paper bases (EVAL_PLAN / noclip_sparse_smoke). Time Pilot = transfer.
BASE_SKILL_ENVS: dict[str, dict] = {
    "seaquest": {"max_objects": 35, "heu": "env_specific", "prefix": "sq", "rq": 1},
    "phoenix": {"max_objects": 21, "heu": "env_specific", "prefix": "ph", "rq": 2},
    "namethisgame": {"max_objects": 11, "heu": "env_specific", "prefix": "ntg", "rq": 2},
    "asteroids": {"max_objects": 18, "heu": "env_specific", "prefix": "as", "rq": 2},
}
# 2–2 overnight PQN split (chained after nexus_parity / base_skills).
PQN_PIXEL_GPU0_ENVS: tuple[str, ...] = ("seaquest", "phoenix")
PQN_PIXEL_GPU1_ENVS: tuple[str, ...] = ("namethisgame", "asteroids")
_MODS_DIR = _ROOT / "configs/experiment/paper_eval/mods"


def skills_stem(env: str) -> str:
    if env == "seaquest":
        return STEM_SKILLS
    return f"final_10m_skills_ego_{env}"


def pqn_pixel_stem(env: str) -> str:
    if env == "seaquest":
        return STEM_PQN_PIXEL
    return f"final_10m_pqn_pixel_{env}"


def _pqn_pixel_pair_stems(envs: tuple[str, ...]) -> tuple[str, ...]:
    trains = tuple(pqn_pixel_stem(e) for e in envs)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def final_10m_base_skills_stems() -> tuple[str, ...]:
    """Ours-only base ego skills @ 10M: all trains, then home evals."""
    trains = tuple(skills_stem(e) for e in BASE_SKILL_ENVS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def final_10m_pqn_pixel_gpu0_stems() -> tuple[str, ...]:
    return _pqn_pixel_pair_stems(PQN_PIXEL_GPU0_ENVS)


def final_10m_pqn_pixel_gpu1_stems() -> tuple[str, ...]:
    return _pqn_pixel_pair_stems(PQN_PIXEL_GPU1_ENVS)


def final_10m_pqn_pixel_stems() -> tuple[str, ...]:
    """All four base PQN pixel trains then evals (GPU0 envs then GPU1)."""
    return _pqn_pixel_pair_stems(PQN_PIXEL_GPU0_ENVS + PQN_PIXEL_GPU1_ENVS)


def _paper_core_keys(env: str) -> tuple[str, ...]:
    """Load ``paper_core`` from mods/<env>.yaml (fallback: base only)."""
    path = _MODS_DIR / f"{env}.yaml"
    if not path.is_file():
        return ("base",)
    keys: list[str] = []
    in_core = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("paper_core:"):
            in_core = True
            continue
        if in_core:
            if line.startswith("  - "):
                keys.append(line[4:].strip())
            elif line.strip() and not line.startswith(" "):
                break
    return tuple(keys) if keys else ("base",)


def _lines(rows: list[str]) -> str:
    return "\n".join(rows) + "\n"


def _write(path: Path, body: str, *, dry: bool) -> None:
    if dry:
        print(f"  would write {path.relative_to(_ROOT)} ({len(body.splitlines())} lines)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    print(f"  wrote {path.relative_to(_ROOT)}")


def _mod_keys_yaml(keys: tuple[str, ...] | None = None) -> list[str]:
    use = keys if keys is not None else MOD_KEYS
    return ["PAPER_EVAL_MOD_KEYS:"] + [f"- {k}" for k in use]


def _ego_lines(*, seaquest_legacy: bool = False) -> list[str]:
    lines = [
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
    if seaquest_legacy:
        lines += [
            "USE_LEGACY_SEAQUEST_ACTION_MAPPER: false",
            "USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false",
            "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false",
        ]
    return lines


def _mapper_tags(env: str) -> list[str]:
    tags = ["- mapper:pov_facing", "- ego:phi_bins8"]
    if env in {"asteroids", "timepilot"}:
        tags.append("- motor:rotate_aim")
    return tags


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


def _skills_train(env: str = "seaquest") -> str:
    meta = BASE_SKILL_ENVS[env]
    stem = skills_stem(env)
    rq = int(meta["rq"])
    phase = "skills_only" if env == "seaquest" else "from_scratch"
    lines = [
        f"# {env} ego skills — final_10m (locked lr1e3 + ε B, matched 10M).",
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
    lines += _ego_lines(seaquest_legacy=(env == "seaquest"))
    lines += [
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_{SUITE}_skills_ego_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        f"- rq:{rq}",
        f"- env:{env}",
        "- role:base",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        "- ablation:sparse_cda",
        f"- budget:{BUDGET_TAG}",
        f"- phase:{phase}",
        "- skill_input:ego",
    ]
    lines += _mapper_tags(env)
    lines += [
        "- lr:constant",
        "- lock:skills_lr1e3",
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
        f"  SKILL_LR: {LOCKED_SKILL_LR}",
        "  SKILL_REWARD_LAMBDA: 0.0",
        "  CLIP_REWARD: true",
        "  RESTRICT_FIRE_FOR_NON_DESTROY: false",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {TIMESTEPS_10M}",
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
        f"  EPS_START: {LOCKED_EPS_START}",
        f"  EPS_FINISH: {LOCKED_EPS_FINISH}",
        f"  EPS_DECAY: {LOCKED_EPS_DECAY}",
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


def _skills_eval(env: str = "seaquest") -> str:
    meta = BASE_SKILL_ENVS[env]
    stem = skills_stem(env)
    rq = int(meta["rq"])
    mod_keys = MOD_KEYS if env == "seaquest" else _paper_core_keys(env)
    ckpt = (
        f"{OUT_MODELS}/{stem}/{env}/"
        f"hierarchical_pqn_{env}_seed44.safetensors"
    )
    lines = [
        f"# Home mod eval for final_10m {env} skills.",
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: skills",
        f"PAPER_EVAL_MODS: {env}",
    ]
    lines += _mod_keys_yaml(mod_keys)
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {ckpt}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        f"ENV_NAME: {env}",
    ]
    lines += _ego_lines(seaquest_legacy=(env == "seaquest"))
    lines += [
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_{SUITE}_eval_skills_ego_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        f"- rq:{rq}",
        f"- env:{env}",
        "- phase:mod_eval",
        "- skill_input:ego",
    ]
    lines += _mapper_tags(env)
    lines += [
        f"- budget:{BUDGET_TAG}",
        "- lock:skills_lr1e3",
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


def _tandem_train(*, stem: str, meta_mode: str, meta_tag: str) -> str:
    if meta_mode in ("pixel", "pixel_embed"):
        meta_mode = "pixel_embed"
        meta_wandb = "pixel"
        header = (
            "# Seaquest tandem pixel — final_10m locked e01 HPs (matched 10M)."
        )
    else:
        meta_wandb = meta_mode
        header = (
            f"# Seaquest tandem OC ({meta_mode}) — final_10m locked e01 HPs "
            f"(matched 10M)."
        )
    lines = [
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
    lines += _ego_lines(seaquest_legacy=True)
    lines += [
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: sq_{SUITE}_tandem_{meta_tag}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        "- rq:3",
        "- env:seaquest",
        "- role:base",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        "- ablation:sparse_cda",
        f"- budget:{BUDGET_TAG}",
        "- phase:from_scratch",
        "- train_mode:tandem",
        f"- meta_input:{meta_wandb}",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        "- lock:tandem_e01",
        f"- cell:{meta_tag}",
        "META_PIXEL_DO_RESIZE: true",
        "META_PIXEL_USE_NATIVE_DOWNSCALING: true",
        "META_PIXEL_RESIZE_SHAPE:",
        "- 84",
        "- 84",
        "META_PIXEL_GRAYSCALE: true",
        "SKILL_PIXEL_CNN_CHANNELS:",
        "- 32",
        "- 64",
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        f"META_INPUT_MODE: {meta_mode}",
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
        "  META_LR: 0.0005",
        f"  SKILL_LR: {LOCKED_SKILL_LR}",
        "  META_USE_NOOP_TARGET: true",
        "  META_NOOP_TARGET_IDX: null",
        "  META_SWITCH_PENALTY: -0.05",
        "  META_DEATH_PENALTY: 0.0",
        "  META_REWARD_LAMBDA: 1.0",
        "  CLIP_REWARD: true",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {TIMESTEPS_10M}",
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
        f"  SKILL_EPS_START: {LOCKED_EPS_START}",
        f"  SKILL_EPS_FINISH: {LOCKED_EPS_FINISH}",
        f"  SKILL_EPS_DECAY: {LOCKED_EPS_DECAY}",
        "  META_EPS_START: 0.1",
        "  META_EPS_FINISH: 0.01",
        "  META_EPS_DECAY: 0.2",
        f"  META_INPUT_MODE: {meta_mode}",
        "  SKILL_INPUT_MODE: ego",
        "  META_PIXEL_CNN_CHANNELS:",
        "  - 32",
        "  - 64",
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


def _tandem_eval(*, stem: str, meta_mode: str, meta_tag: str) -> str:
    if meta_mode in ("pixel", "pixel_embed"):
        meta_mode = "pixel_embed"
        meta_wandb = "pixel"
    else:
        meta_wandb = meta_mode
    skills_ckpt = (
        f"{OUT_MODELS}/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed44_skills.safetensors"
    )
    meta_ckpt = (
        f"{OUT_MODELS}/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed44_meta.safetensors"
    )
    lines = [
        f"# Home mod eval for final_10m tandem ({meta_tag}).",
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
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: tandem",
        "ENV_NAME: seaquest",
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: sq_{SUITE}_eval_tandem_{meta_tag}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        "- rq:3",
        "- env:seaquest",
        "- phase:mod_eval",
        "- train_mode:tandem",
        f"- meta_input:{meta_wandb}",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        f"- budget:{BUDGET_TAG}",
        "- lock:tandem_e01",
        f"- cell:{meta_tag}",
        "SKILL_INPUT_MODE: ego",
        f"META_INPUT_MODE: {meta_mode}",
        "HEURISTIC_MODE: env_specific",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_top()
    lines += [
        "EGO_DISTANCE_CLIP_MAX_PX: null",
        "EGO_PSI_ANGLE_BINS: 8",
        "alg:",
        "  SKILL_INPUT_MODE: ego",
        f"  META_INPUT_MODE: {meta_mode}",
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


def _pqn_train(*, env: str, object_centric: bool) -> str:
    meta = BASE_SKILL_ENVS[env]
    variant = "oc" if object_centric else "pixel"
    stem = STEM_PQN_OC if object_centric else pqn_pixel_stem(env)
    if object_centric and env != "seaquest":
        raise ValueError("OC PQN final_10m is Seaquest-only for now")
    rq = int(meta["rq"])
    lines = [
        f"# Flat PQN {variant} — {env} final_10m (Table-5 HPs, matched 10M).",
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: train",
        "PAPER_EVAL_KIND: flat_pqn",
        f"PAPER_EVAL_MODS: {env}",
        "PAPER_EVAL_REQUIRES_RQ1: []",
        "NUM_SEEDS: 1",
        "ALG_NAME: flat_pqn",
        f"ENV_NAME: {env}",
        "TRAIN_MODE: flat_pqn",
        f"OBJECT_CENTRIC: {'true' if object_centric else 'false'}",
    ]
    if not object_centric:
        lines += [
            "PIXEL_RESIZE: true",
            "PIXEL_RESIZE_SHAPE:",
            "- 84",
            "- 84",
            "PIXEL_GRAYSCALE: true",
            "USE_NATIVE_DOWNSCALING: true",
        ]
    lines += [
        "NORM_TYPE: layer_norm",
        "NORM_INPUT: false",
        "LR_LINEAR_DECAY: false",
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_{SUITE}_pqn_{variant}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        f"- rq:{rq}",
        f"- env:{env}",
        "- role:base",
        "- train_mode:flat_pqn",
        f"- variant:pqn_{variant}_paper_hps",
        f"- budget:{BUDGET_TAG}",
        f"TOTAL_TIMESTEPS: {TIMESTEPS_10M}",
        f"TOTAL_TIMESTEPS_DECAY: {TIMESTEPS_10M}",
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
    return _lines(lines)


def _pqn_eval(*, env: str, object_centric: bool) -> str:
    meta = BASE_SKILL_ENVS[env]
    variant = "oc" if object_centric else "pixel"
    stem = STEM_PQN_OC if object_centric else pqn_pixel_stem(env)
    if object_centric and env != "seaquest":
        raise ValueError("OC PQN final_10m is Seaquest-only for now")
    rq = int(meta["rq"])
    mod_keys = MOD_KEYS if env == "seaquest" else _paper_core_keys(env)
    ckpt_name = f"flat_pqn_{env}_{variant}_seed44_vmap0.safetensors"
    ckpt = f"{OUT_MODELS}/{stem}/{env}/{ckpt_name}"
    lines = [
        f"# Home mod eval for final_10m flat PQN {variant} ({env}).",
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: flat_pqn",
        f"PAPER_EVAL_MODS: {env}",
    ]
    lines += _mod_keys_yaml(mod_keys)
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_FLAT_PQN_CKPT: {ckpt}",
        f"PAPER_EVAL_OBJECT_CENTRIC: {'true' if object_centric else 'false'}",
        "NUM_SEEDS: 1",
        "ALG_NAME: flat_pqn",
        f"ENV_NAME: {env}",
        f"OBJECT_CENTRIC: {'true' if object_centric else 'false'}",
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_{SUITE}_eval_pqn_{variant}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{SUITE}",
        f"- rq:{rq}",
        f"- env:{env}",
        "- phase:mod_eval",
        "- train_mode:flat_pqn",
        f"- variant:pqn_{variant}",
        f"- budget:{BUDGET_TAG}",
        "SEED: 44",
        "EVAL_SEED: 10044",
        "EVAL_NUM_SEEDS: 3",
    ]
    return _lines(lines)


def _nexus_hrl() -> str:
    return _lines(
        [
            "# HPQN/HRL baseline — final_10m (CBL ourhps schedule, matched 10M).",
            "# Architecture: NEXUS hierarchical PQN, NO_REWARDS=true (no skill rewards).",
            "",
            'NAME_INITIALS: "CBL"',
            'ALG_NAME: "pqn_hier_baseline"',
            f"TOTAL_TIMESTEPS: {TIMESTEPS_10M}",
            f"TOTAL_TIMESTEPS_DECAY: {TIMESTEPS_10M}",
            "NUM_ENVS: 128",
            "NUM_STEPS: 128",
            "EPS_START: 0.2",
            "EPS_FINISH: 0.01",
            "EPS_DECAY: 0.2",
            "META_EPS_START: 0.2",
            "META_EPS_FINISH: 0.01",
            "META_EPS_DECAY: 0.2",
            "NUM_EPOCHS: 4",
            "NUM_MINIBATCHES: 64",
            'NORM_TYPE: "layer_norm"',
            "NORM_INPUT: false",
            "LR: 5.0e-4",
            "MAX_GRAD_NORM: 0.8",
            "LR_LINEAR_DECAY: false",
            "GAMMA: 0.99",
            "LAMBDA: 0.5",
            "META_LAMBDA: 0.9",
            "HIDDEN_SIZE: 128",
            "NUM_LAYERS: 3",
            "STICKY_ACTIONS: false",
            "EPISODIC_LIFE: true",
            "",
            "NO_REWARDS: true",
            "",
            "HIERARCHICAL: true",
            "NUM_AGENTS: 3",
            'META_POLICY: "learned"',
            "META_GREEDY: false",
            "META_SHAPED_REWARD: false",
            "LLM_PRETRAIN: false",
            "RANDOM_PRETRAIN: false",
            "PRETRAIN_LEN: 3700",
            "",
            'ENV_NAME: "Seaquest"',
            "ENV_KWARGS: {}",
            "OBJECT_CENTRIC: true",
            "",
            "TEST_DURING_TRAINING: true",
            "TEST_INTERVAL: 0.1",
            "TEST_NUM_ENVS: 128",
            "TEST_NUM_STEPS: 10000",
            "EPS_TEST: 0.0",
            "RECORD_VIDEO: true",
            "eval_envs:",
            '  - label: "clean"',
            "    noisy_eval: false",
            "    no_enemies: false",
            '  - label: "mod"',
            "    no_enemies: true",
            "    noisy_eval: false",
        ]
    )


def _nexus_symbolic() -> str:
    return _lines(
        [
            "# NEXUS symbolic — final_10m (CBL ourhps schedule, matched 10M).",
            "# Architecture: 3 OC options + LLM meta; skill rewards on.",
            "",
            'NAME_INITIALS: "CBL"',
            'ALG_NAME: "pqn_hier_llm"',
            f"TOTAL_TIMESTEPS: {TIMESTEPS_10M}",
            f"TOTAL_TIMESTEPS_DECAY: {TIMESTEPS_10M}",
            "NUM_ENVS: 128",
            "NUM_STEPS: 128",
            "EPS_START: 0.2",
            "EPS_FINISH: 0.01",
            "EPS_DECAY: 0.2",
            "META_EPS_START: 0.2",
            "META_EPS_FINISH: 0.01",
            "META_EPS_DECAY: 0.2",
            "NUM_EPOCHS: 4",
            "NUM_MINIBATCHES: 64",
            'NORM_TYPE: "layer_norm"',
            "NORM_INPUT: false",
            "LR: 5.0e-4",
            "MAX_GRAD_NORM: 0.8",
            "LR_LINEAR_DECAY: false",
            "GAMMA: 0.99",
            "LAMBDA: 0.65",
            "META_LAMBDA: 0.9",
            "HIDDEN_SIZE: 128",
            "NUM_LAYERS: 3",
            "STICKY_ACTIONS: false",
            "EPISODIC_LIFE: true",
            "",
            "HIERARCHICAL: true",
            "NUM_AGENTS: 3",
            'META_POLICY: "llm"',
            "META_GREEDY: false",
            "META_SHAPED_REWARD: false",
            "LLM_PRETRAIN: false",
            "RANDOM_PRETRAIN: false",
            "PRETRAIN_LEN: 3700",
            "NO_REWARDS: false",
            "",
            'ENV_NAME: "Seaquest"',
            "ENV_KWARGS: {}",
            "OBJECT_CENTRIC: true",
            "",
            "TEST_DURING_TRAINING: true",
            "TEST_INTERVAL: 0.1",
            "TEST_NUM_ENVS: 128",
            "TEST_NUM_STEPS: 10000",
            "EPS_TEST: 0.0",
            "RECORD_VIDEO: true",
            "eval_envs:",
            '  - label: "clean"',
            "    noisy_eval: false",
            "    no_enemies: false",
            '  - label: "mod"',
            "    no_enemies: true",
            "    noisy_eval: false",
        ]
    )


def _manifest() -> str:
    """Not runnable (stem starts with 'manifest'); inventory only."""
    return _lines(
        [
            "# final_10m suite inventory (not a runnable paper_eval config).",
            f"suite: {SUITE}",
            f"budget_tag: {BUDGET_TAG}",
            f"total_timesteps: {TIMESTEPS_10M}",
            "num_envs: 128",
            "spine: " + SPINE,
            "base_envs: [" + ", ".join(BASE_SKILL_ENVS) + "]",
            "pqn_pixel_gpu0_envs: [" + ", ".join(PQN_PIXEL_GPU0_ENVS) + "]",
            "pqn_pixel_gpu1_envs: [" + ", ".join(PQN_PIXEL_GPU1_ENVS) + "]",
            "seaquest_mod_keys:",
            *[f"  - {k}" for k in MOD_KEYS],
            "paper_eval_stems:",
            *[f"  skills_{e}: {skills_stem(e)}" for e in BASE_SKILL_ENVS],
            *[f"  pqn_pixel_{e}: {pqn_pixel_stem(e)}" for e in BASE_SKILL_ENVS],
            f"  tandem_pixel_train: {STEM_TANDEM_PIXEL}",
            f"  tandem_oc_train: {STEM_TANDEM_OC}",
            f"  pqn_oc_train: {STEM_PQN_OC}",
            "suites:",
            "  final_10m_base_skills: ego skills + home evals (all bases)",
            "  final_10m_pqn_pixel_gpu0: SQ+PH PQN pixel + evals",
            "  final_10m_pqn_pixel_gpu1: NTG+AS PQN pixel + evals",
            "  final_10m_overnight_gpu0: nexus_parity then pqn_pixel_gpu0",
            "  final_10m_overnight_gpu1: base_skills then pqn_pixel_gpu1",
            "nexus_algs:",
            f"  hrl: {NEXUS_HRL}",
            f"  nexus_symbolic: {NEXUS_SYMBOLIC}",
            "outputs_root: " + OUT_MODELS,
            "notes:",
            "  - Time Pilot is a transfer (not in base skills / PQN suites)",
            "  - Regenerated by scripts/materialize_final_10m.py",
        ]
    )


def materialize(*, dry: bool) -> None:
    print(f"final_10m → {_OUT.relative_to(_ROOT)}  (timesteps={TIMESTEPS_10M})")
    pairs: list[tuple[Path, str]] = []
    for env in BASE_SKILL_ENVS:
        stem = skills_stem(env)
        pairs.append((_OUT / f"{stem}.yaml", _skills_train(env)))
        pairs.append((_OUT / f"{stem}_eval.yaml", _skills_eval(env)))
    pairs += [
        (
            _OUT / f"{STEM_TANDEM_PIXEL}.yaml",
            _tandem_train(
                stem=STEM_TANDEM_PIXEL, meta_mode="pixel_embed", meta_tag="pixel"
            ),
        ),
        (
            _OUT / f"{STEM_TANDEM_PIXEL}_eval.yaml",
            _tandem_eval(
                stem=STEM_TANDEM_PIXEL, meta_mode="pixel_embed", meta_tag="pixel"
            ),
        ),
        (
            _OUT / f"{STEM_TANDEM_OC}.yaml",
            _tandem_train(
                stem=STEM_TANDEM_OC,
                meta_mode="oc_simple_rel_role",
                meta_tag="oc_rel_role",
            ),
        ),
        (
            _OUT / f"{STEM_TANDEM_OC}_eval.yaml",
            _tandem_eval(
                stem=STEM_TANDEM_OC,
                meta_mode="oc_simple_rel_role",
                meta_tag="oc_rel_role",
            ),
        ),
    ]
    for env in BASE_SKILL_ENVS:
        stem = pqn_pixel_stem(env)
        pairs.append((_OUT / f"{stem}.yaml", _pqn_train(env=env, object_centric=False)))
        pairs.append(
            (_OUT / f"{stem}_eval.yaml", _pqn_eval(env=env, object_centric=False))
        )
    pairs += [
        (
            _OUT / f"{STEM_PQN_OC}.yaml",
            _pqn_train(env="seaquest", object_centric=True),
        ),
        (
            _OUT / f"{STEM_PQN_OC}_eval.yaml",
            _pqn_eval(env="seaquest", object_centric=True),
        ),
        (_OUT / "manifest.yaml", _manifest()),
        (_NEXUS_ALG / f"{NEXUS_HRL}.yaml", _nexus_hrl()),
        (_NEXUS_ALG / f"{NEXUS_SYMBOLIC}.yaml", _nexus_symbolic()),
    ]
    for path, body in pairs:
        _write(path, body, dry=dry)
    print(
        "\nOvernight (2 GPUs):\n"
        "  GPU0: --suite final_10m_overnight_gpu0  "
        "# nexus_parity then SQ+PH PQN\n"
        "  GPU1: --suite final_10m_overnight_gpu1  "
        "# base skills then NTG+AS PQN\n"
        f"\nPQN gpu0: {list(final_10m_pqn_pixel_gpu0_stems())}\n"
        f"PQN gpu1: {list(final_10m_pqn_pixel_gpu1_stems())}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    materialize(dry=bool(args.dry_run))


if __name__ == "__main__":
    main()
