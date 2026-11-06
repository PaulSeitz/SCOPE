#!/usr/bin/env python3
"""Materialize the locked thesis-final 50M sweep (RQ1 / RQ2 / RQ3 + ablations + floors).

One HP set for every learned arm — PQN optimal-scaling 1:1 (1024 envs × 32 steps
× 1525 updates = 50M, 2 epochs × 256 minibatches, LR 2.5e-4 linear decay,
MAX_GRAD_NORM 10, ε 1.0→0.001 over 10%, γ 0.99, λ 0.65). Skills / tandem use the
same values for SKILL_* and META_* heads. Train seeds 44/45/46; every eval runs
EVAL_NUM_SEEDS=3 × 3 episodes per mod (eval-only floors: 9 × 3 so the episode
count matches an arm's 3 train seeds).

Tiers (``TIER`` column in the run log):
  core   — RQ1 (SQ/TP/NTG) baselines, Seaquest ablations, RQ2 Seaquest→9 ZS, RQ3
           tandem, attribution floors
  extras — native ego ceilings on the 7 other transfer targets, in-family ZS
           (TP→Asteroids, NTG→SpaceInvaders), tandem-skill ZS (SQ tandem OC → TP/NTG)

NEXUS (Seaquest only) runs via ``scripts/run_nexus.py``; its commands are emitted
into ``scripts/run_thesis_final_nexus.sh`` and sharded across GPUs after each
GPU's paper_eval queue.

    python scripts/materialize_thesis_final.py               # YAMLs + run log + nexus sh
    python scripts/materialize_thesis_final.py --plan 3      # print 3-GPU schedule
    python scripts/run_paper_eval.py --suite thesis_final_3gpu_g0 --gpus 0 \\
        --wait-deps --wait-deps-timeout-h 72 --skip-rq-gate --lean --resume
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from concept_based_learning.paper_eval.mods import load_mod_allowlist

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "configs" / "experiment" / "paper_eval" / "thesis_final"
_RUNLOG = _ROOT / "docs" / "thesis_final_RUNLOG.md"
_NEXUS_SH = _ROOT / "scripts" / "run_thesis_final_nexus.sh"

SUITE = "thesis_final"
OUT_MODELS = "./outputs/models/paper_eval/thesis_final"
GPU_COUNTS = (2, 3)

SEEDS = (44, 45, 46)
FLOOR_SEED = 44
EVAL_NUM_SEEDS = 3
FLOOR_EVAL_NUM_SEEDS = 9

TIMESTEPS = 50_000_000
NUM_ENVS = 1024
NUM_STEPS = 32
NUM_UPDATES = TIMESTEPS // (NUM_ENVS * NUM_STEPS)
NUM_EPOCHS = 2
NUM_MINIBATCHES = 256
LR = 0.00025
MAX_GRAD_NORM = 10.0
EPS_START = 1.0
EPS_FINISH = 0.001
EPS_DECAY = 0.1
GAMMA = 0.99
LAMBDA = 0.65

# Time Pilot is not a home environment and is not a transfer target.
HOME_ENVS = ("seaquest", "namethisgame")
ENVS: dict[str, dict] = {
    "seaquest": {"max_objects": 35, "heu": "env_specific", "prefix": "sq"},
    "namethisgame": {"max_objects": 11, "heu": "env_specific", "prefix": "ntg"},
    "spaceinvaders": {"max_objects": 41, "heu": "sweep", "prefix": "si"},
    "beamrider": {"max_objects": 31, "heu": "env_specific", "prefix": "br"},
    "asteroids": {"max_objects": 18, "heu": "env_specific", "prefix": "as"},
    "pong": {"max_objects": 2, "heu": "nearest", "prefix": "po"},
    "breakout": {"max_objects": 3, "heu": "nearest", "prefix": "bo"},
    "freeway": {"max_objects": 12, "heu": "env_specific", "prefix": "fw"},
    "phoenix": {"max_objects": 21, "heu": "env_specific", "prefix": "ph"},
}
def transfer_targets(src: str) -> tuple[str, ...]:
    """Every other game in the suite. Time Pilot is not in ``ENVS``."""
    return tuple(e for e in ENVS if e != src)


TRANSFER_TARGETS = transfer_targets("seaquest")
EXTRA_CEILING_ENVS = tuple(e for e in TRANSFER_TARGETS if e not in HOME_ENVS)

# Failure-mode strip (control + semantic + dynamics + layout + visual); drops the
# lane_scramble / dynamic_lane_drift duplicates of shift_lanes / random_spawns.
SEAQUEST_MOD_KEYS = (
    "base",
    "no_enemies",
    "swap_diver_enemy_labels",
    "swap_diver_shark_roles",
    "gravity",
    "faster_enemies",
    "shift_lanes",
    "random_spawns",
    "change_background_color",
)

TANDEM_META_OC = "oc_simple_rel_role"
PQN_OC_MODE = "oc_simple_rel_role"

# RTX A6000 wall-clock hours (wandb `_runtime` of the optim_scale 50M runs where
# measured; ``~`` rows in the run log are estimates). Ada ≈ 0.6–0.75× these.
TRAIN_HOURS: dict[tuple[str, str], float] = {
    ("skills_ego", "seaquest"): 0.28,
    ("skills_ego", "timepilot"): 0.10,
    ("skills_ego", "namethisgame"): 0.20,
    ("skills_ego", "pong"): 0.05,
    ("skills_ego", "spaceinvaders"): 0.08,
    ("skills_ego", "phoenix"): 0.20,
    ("skills_nopov", "seaquest"): 0.28,
    ("skills_oc_flat_rel", "seaquest"): 0.35,
    ("skills_pixel_outline", "seaquest"): 4.02,
    ("pqn_pixel", "seaquest"): 1.28,
    ("pqn_pixel", "timepilot"): 0.52,
    ("pqn_pixel", "namethisgame"): 0.63,
    ("pqn_oc", "seaquest"): 0.3,
    ("tandem_oc", "seaquest"): 0.7,
    ("tandem_pixel", "seaquest"): 1.48,
}
TRAIN_HOURS_DEFAULT = {"skills_ego": 0.15, "pqn_oc": 0.3, "tandem_oc": 0.5}
MEASURED = {
    ("skills_ego", "seaquest"),
    ("skills_ego", "timepilot"),
    ("skills_ego", "namethisgame"),
    ("skills_ego", "pong"),
    ("skills_ego", "spaceinvaders"),
    ("skills_ego", "phoenix"),
    ("skills_pixel_outline", "seaquest"),
    ("pqn_pixel", "seaquest"),
    ("pqn_pixel", "timepilot"),
    ("pqn_pixel", "namethisgame"),
    ("tandem_pixel", "seaquest"),
}
# Parallel mod workers: a full home eval is about one compile plus one rollout
# wave (~3 min), not one serial pass per modification.
EVAL_HOURS = {"seaquest": 0.06, "other": 0.05}
FLOOR_EVAL_SCALE = FLOOR_EVAL_NUM_SEEDS / EVAL_NUM_SEEDS
NEXUS_TRAIN_HOURS = 1.5  # symbolic paper HPs (Seaquest)
NEXUS_HRL_TRAIN_HOURS = 0.7  # optim50m HRL estimate (≈ tandem_oc)
NEXUS_EVAL_HOURS = 0.25
# (arm, alg_stem, env_key) — symbolic stays Seaquest-only; HRL on all homes.
NEXUS_SYMBOLIC = (
    ("nexus_symbolic", "cbl_seaquest_nexus_symbolic_paper", "seaquest"),
)
NEXUS_HRL_OPTIM = (
    ("nexus_hrl", "cbl_seaquest_hrl_baseline_optim50m", "seaquest"),
    ("nexus_hrl", "cbl_namethisgame_hrl_baseline_optim50m", "namethisgame"),
)


# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------


@dataclass
class Run:
    stem: str
    tier: str
    rq: str
    phase: str  # train | eval
    arm: str
    env: str
    seed: int | None
    hours: float
    measured: bool
    doc: dict
    header: str
    deps: list[str] = field(default_factory=list)
    note: str = ""


def _stem(arm: str, env: str, seed: int | None = None, suffix: str = "") -> str:
    s = f"tf_{env}_{arm}"
    if seed is not None:
        s += f"_seed{seed}"
    return s + suffix


def zs_stem(arm: str, src: str, dst: str, seed: int) -> str:
    return f"tf_zs_{arm}_{src}_to_{dst}_seed{seed}"


def floor_stem(kind: str, env: str) -> str:
    return f"tf_floor_{kind}_{env}"


def _save(stem: str) -> str:
    return f"{OUT_MODELS}/{stem}"


def _mod_keys(env: str) -> list[str]:
    if env == "seaquest":
        return list(SEAQUEST_MOD_KEYS)
    raw = load_mod_allowlist(env)
    keys = list(raw.get("paper_core") or [])
    return keys or [e["key"] for e in (raw.get("eval_mods") or []) if e.get("key")]


def _train_hours(arm: str, env: str) -> tuple[float, bool]:
    key = (arm, env)
    if key in TRAIN_HOURS:
        return TRAIN_HOURS[key], key in MEASURED
    return TRAIN_HOURS_DEFAULT.get(arm, 0.5), False


def _eval_hours(env: str, *, floor: bool = False) -> float:
    h = EVAL_HOURS["seaquest" if env == "seaquest" else "other"]
    return h * (FLOOR_EVAL_SCALE if floor else 1.0)


def _tags(*extra: str) -> list[str]:
    return [
        f"suite:{SUITE}",
        *extra,
        "budget:50m_optim",
        "scale:pqn_optim_1to1",
        f"num_envs:{NUM_ENVS}",
        f"num_steps:{NUM_STEPS}",
        f"num_updates:{NUM_UPDATES}",
        f"lr:{LR}",
        "lr_decay:linear",
        f"max_grad_norm:{int(MAX_GRAD_NORM)}",
        f"eps:{EPS_START}to{EPS_FINISH}",
    ]


# ---------------------------------------------------------------------------
# YAML bodies
# ---------------------------------------------------------------------------


def _spine(*, pov: bool = True) -> dict:
    return {
        "USE_POV_RELATIVE_MAPPER": pov,
        "USE_POV_FIXED_BILATERAL": False,
        "USE_POV_FACING_FRAME": pov,
        "EGO_POSE_XY": False,
        "USE_LEGACY_EGO_OBS": False,
        "USE_EGO_SIZE_FEATURES": False,
        "USE_EGO_VELOCITY_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": False,
        "USE_EGO_WEAPON_FEATURES": False,
        "USE_EGO_FACING_ENEMY_FEATURES": False,
        "USE_EGO_SHOT_FEATURES": False,
        "AVOID_INTERRUPT_RANGE": 55,
        "AVOID_INTERRUPT_RANGE_Y": 55,
        "SURFACE_SUB_AVOID_RANGE_X": 5,
        "SURFACE_SUB_AVOID_RANGE_Y": 5,
        "EGO_DISTANCE_CLIP_MAX_PX": None,
        # No-POV: same 8-bin bearing resolution, but world-frame θ instead of ψ.
        "EGO_ANGLE_BINS": None if pov else 8,
        "EGO_PSI_ANGLE_BINS": 8 if pov else None,
        "EGO_LEGAL_ACTION_MASK": False,
        "SURFACE_SUB_NEAR_RANGE": 5,
        "USE_LEGACY_SEAQUEST_ACTION_MAPPER": False,
        "USE_LEGACY_SEAQUEST_ASCENT_BLOCK": False,
        "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD": False,
    }


def _reward_flags(env: str, skill_input: str) -> dict:
    return {
        "GAME_MODS": None,
        "EVAL_GAME_MODS": None,
        "SKILL_INPUT_MODE": skill_input,
        "HEURISTIC_MODE": ENVS[env]["heu"],
        "SKILL_DEATH_PENALIZE_COLLECT": True,
        "RECORD_VIDEO": False,
        "VIDEO_AT_END": False,
        "DESTROY_CONTACT_FAIL": True,
        "DESTROY_LIVING_PENALTY": False,
        "COLLECT_SPARSE_ONLY": True,
        # Reference AVOID is ongoing separation shaping (+0.01, nonterminal).
        # True would pay +1 and cut the return on the first step away.
        "AVOID_SPARSE_ONLY": False,
        # Destruction of the target is not an AVOID failure. Contact and
        # life loss still are, for every label.
        "AVOID_KILLED_EVENT_MODE": "none",
        # Seaquest selector: divers are eligible without the near-surface gate
        # when no DESTROY/AVOID hitbox is on screen. Other games ignore this.
        "COLLECT_EARLY_WHEN_NO_THREATS": env == "seaquest",
    }


def _skill_input_top(skill_input: str) -> dict:
    if skill_input == "pixel_embed":
        return {
            "META_PIXEL_DO_RESIZE": True,
            "META_PIXEL_USE_NATIVE_DOWNSCALING": True,
            "META_PIXEL_RESIZE_SHAPE": [84, 84],
            "META_PIXEL_GRAYSCALE": True,
            "SKILL_PIXEL_CNN_CHANNELS": [32, 64],
            "SKILL_PIXEL_PAINT_TARGET_OUTLINE": True,
        }
    return {}


def _skill_input_alg(skill_input: str) -> dict:
    if skill_input == "pixel_embed":
        return {"SKILL_PIXEL_CNN_CHANNELS": [32, 64], "SKILL_PIXEL_PAINT_TARGET_OUTLINE": True}
    return {}


def _outer_alg() -> dict:
    return {
        "CLIP_REWARD": True,
        "FIRST_FIRE": True,
        "STICKY_ACTIONS": False,
        "EPISODIC_LIFE": True,
        "FRAME_STACK_SIZE": 4,
        "FRAME_SKIP": 4,
        "TOTAL_TIMESTEPS": TIMESTEPS,
        "TOTAL_TIMESTEPS_DECAY": TIMESTEPS,
        "NUM_ENVS": NUM_ENVS,
        "NUM_STEPS": NUM_STEPS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "NUM_MINIBATCHES": NUM_MINIBATCHES,
        "GAMMA": GAMMA,
        "SKILL_GAMMA": GAMMA,
        "LAMBDA": LAMBDA,
        "SKILL_LAMBDA": LAMBDA,
        "LR": LR,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR_LINEAR_DECAY": True,
        "OFF_POLICY": True,
        "RELABEL_ALL_OBJECTS": False,
        "RELABEL_K_OBJECTS": None,
        "SKILL_BOOTSTRAP": "q_lambda",
        "EPS_START": EPS_START,
        "EPS_FINISH": EPS_FINISH,
        "EPS_DECAY": EPS_DECAY,
    }


def _skill_heads_alg(env: str) -> dict:
    return {
        # NAVIGATE is folded onto COLLECT when NUM_SKILLS < 4. Seaquest's
        # surface waypoint is already labeled COLLECT, so the fourth head is
        # unused and is not initialized.
        "NUM_SKILLS": 3,
        "SKILL_NAMES": ["DESTROY", "COLLECT", "AVOID"],
        "MAX_OBJECTS": ENVS[env]["max_objects"],
        "EGO_ACTION_DIM": 18,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 3,
        "SKILL_EMBED_DIM": 256,
        "SKILL_LR": LR,
    }


def _reward_alg(env: str, skill_input: str) -> dict:
    return {
        "HEURISTIC_MODE": ENVS[env]["heu"],
        "SKILL_INPUT_MODE": skill_input,
        "SKILL_DEATH_PENALIZE_COLLECT": True,
        "DESTROY_CONTACT_FAIL": True,
        "DESTROY_LIVING_PENALTY": False,
        "COLLECT_SPARSE_ONLY": True,
        "AVOID_SPARSE_ONLY": False,
        "AVOID_KILLED_EVENT_MODE": "none",
        "COLLECT_EARLY_WHEN_NO_THREATS": env == "seaquest",
        "USE_EGO_SHOT_FEATURES": False,
    }


def _skills_body(
    env: str,
    *,
    stem: str,
    seed: int,
    rq: str,
    skill_input: str = "ego",
    pov: bool = True,
    tags: list[str],
) -> dict:
    # ``screen`` is the clean no-POV arm: world-frame observations and actions.
    # The network is still the ego MLP.
    screen = skill_input == "screen"
    net_input = "ego" if screen else skill_input
    use_pov = False if screen else pov
    alg = {
        **_skill_heads_alg(env),
        "SKILL_REWARD_LAMBDA": 0.0,
        "RESTRICT_FIRE_FOR_NON_DESTROY": False,
        **_outer_alg(),
        **_reward_alg(env, net_input),
        **_skill_input_alg(net_input),
    }
    if screen:
        alg["SKILL_INPUT_MODE"] = "ego"
        alg["EGO_OBSERVATION_FRAME"] = "screen"
        alg["EGO_ACTION_FRAME"] = "screen"
        alg["USE_POV_RELATIVE_MAPPER"] = False
        alg["USE_POV_FACING_FRAME"] = False
    body = {
        "PAPER_EVAL_RQ": int(rq[0]),
        "PAPER_EVAL_PHASE": "train",
        "PAPER_EVAL_KIND": "skills",
        "PAPER_EVAL_MODS": env,
        "PAPER_EVAL_REQUIRES_RQ1": [],
        "NUM_SEEDS": 1,
        "ALG_NAME": "hierarchical_pqn",
        "TRAIN_MODE": "skills_only",
        "ENV_NAME": env,
        **_spine(pov=use_pov),
        "SAVE_PATH": _save(stem),
        "WANDB_RUN_NAME": f"{ENVS[env]['prefix']}_{stem}",
        "WANDB_TAGS": tags,
        **_skill_input_top(net_input),
        **_reward_flags(env, net_input),
        "alg": alg,
        "SEED": seed,
    }
    if screen:
        body["SKILL_INPUT_MODE"] = "ego"
        body["EGO_OBSERVATION_FRAME"] = "screen"
        body["EGO_ACTION_FRAME"] = "screen"
        body["USE_POV_RELATIVE_MAPPER"] = False
        body["USE_POV_FACING_FRAME"] = False
    return body


def _tandem_body(env: str, *, stem: str, seed: int, meta: str, tags: list[str]) -> dict:
    meta_mode = "pixel_embed" if meta == "pixel" else TANDEM_META_OC
    alg = {
        **_skill_heads_alg(env),
        "META_HIDDEN_SIZE": 256,
        "META_PER_OBJECT_NUM_LAYERS": 3,
        "META_FLAT_NUM_LAYERS": 3,
        "META_EMBED_DIM": 256,
        "META_LR": LR,
        "META_USE_NOOP_TARGET": True,
        "META_NOOP_TARGET_IDX": None,
        "META_SWITCH_PENALTY": -0.05,
        "META_DEATH_PENALTY": 0.0,
        "META_REWARD_LAMBDA": 1.0,
        **_outer_alg(),
        "SKILL_EPS_START": EPS_START,
        "SKILL_EPS_FINISH": EPS_FINISH,
        "SKILL_EPS_DECAY": EPS_DECAY,
        "META_EPS_START": EPS_START,
        "META_EPS_FINISH": EPS_FINISH,
        "META_EPS_DECAY": EPS_DECAY,
        "META_INPUT_MODE": meta_mode,
        "META_PIXEL_CNN_CHANNELS": [32, 64],
        **_reward_alg(env, "ego"),
    }
    return {
        "PAPER_EVAL_RQ": 3,
        "PAPER_EVAL_PHASE": "train",
        "PAPER_EVAL_KIND": "tandem",
        "PAPER_EVAL_MODS": env,
        "PAPER_EVAL_REQUIRES_RQ1": [],
        "NUM_SEEDS": 1,
        "ALG_NAME": "hierarchical_pqn",
        "TRAIN_MODE": "tandem",
        "META_ARCHITECTURE": "per_object_mlp",
        "ENV_NAME": env,
        **_spine(),
        "SAVE_PATH": _save(stem),
        "WANDB_RUN_NAME": f"{ENVS[env]['prefix']}_{stem}",
        "WANDB_TAGS": tags,
        "META_PIXEL_DO_RESIZE": True,
        "META_PIXEL_USE_NATIVE_DOWNSCALING": True,
        "META_PIXEL_RESIZE_SHAPE": [84, 84],
        "META_PIXEL_GRAYSCALE": True,
        "META_INPUT_MODE": meta_mode,
        **_reward_flags(env, "ego"),
        "alg": alg,
        "SEED": seed,
    }


def _pqn_body(env: str, *, stem: str, seed: int, variant: str, tags: list[str]) -> dict:
    oc = variant == "oc"
    obs: dict = (
        {
            "OBJECT_CENTRIC": True,
            "OC_INPUT_MODE": PQN_OC_MODE,
            "MAX_OBJECTS": ENVS[env]["max_objects"],
        }
        if oc
        else {
            "OBJECT_CENTRIC": False,
            "PIXEL_RESIZE": True,
            "PIXEL_RESIZE_SHAPE": [84, 84],
            "PIXEL_GRAYSCALE": True,
            # ALE-style resize (Gallici protocol), as in pqn_paper_t5.
            "USE_NATIVE_DOWNSCALING": False,
        }
    )
    return {
        "PAPER_EVAL_RQ": 1,
        "PAPER_EVAL_PHASE": "train",
        "PAPER_EVAL_KIND": "flat_pqn",
        "PAPER_EVAL_MODS": env,
        "PAPER_EVAL_REQUIRES_RQ1": [],
        "NUM_SEEDS": 1,
        "ALG_NAME": "flat_pqn",
        "ENV_NAME": env,
        "TRAIN_MODE": "flat_pqn",
        **obs,
        "NORM_TYPE": "layer_norm",
        "NORM_INPUT": False,
        "LR_LINEAR_DECAY": True,
        "SAVE_PATH": _save(stem),
        "WANDB_RUN_NAME": f"{ENVS[env]['prefix']}_{stem}",
        "WANDB_TAGS": tags,
        "TOTAL_TIMESTEPS": TIMESTEPS,
        "TOTAL_TIMESTEPS_DECAY": TIMESTEPS,
        "NUM_ENVS": NUM_ENVS,
        "NUM_STEPS": NUM_STEPS,
        "NUM_EPOCHS": NUM_EPOCHS,
        "NUM_MINIBATCHES": NUM_MINIBATCHES,
        "FRAME_SKIP": 4,
        "FRAME_STACK_SIZE": 4,
        "FIRST_FIRE": True,
        "STICKY_ACTIONS": False,
        "EPISODIC_LIFE": True,
        "CLIP_REWARD": True,
        "GAMMA": GAMMA,
        "LAMBDA": LAMBDA,
        "LR": LR,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 2,
        "EPS_START": EPS_START,
        "EPS_FINISH": EPS_FINISH,
        "EPS_DECAY": EPS_DECAY,
        "GAME_MODS": None,
        "EVAL_GAME_MODS": None,
        "SEED": seed,
    }


def _eval_head(
    *, rq: int, kind: str, env: str, requires: list[str], arch_of: str | None
) -> dict:
    head = {
        "PAPER_EVAL_RQ": rq,
        "PAPER_EVAL_PHASE": "eval",
        "PAPER_EVAL_KIND": kind,
        "PAPER_EVAL_MODS": env,
        "PAPER_EVAL_MOD_KEYS": _mod_keys(env),
        "PAPER_EVAL_REQUIRES_RQ1": requires,
    }
    if arch_of:
        head["PAPER_EVAL_ARCHITECTURE_OF"] = arch_of
    return head


def _eval_tail(seed: int, *, num_seeds: int = EVAL_NUM_SEEDS) -> dict:
    return {"SEED": seed, "EVAL_SEED": seed + 10_000, "EVAL_NUM_SEEDS": num_seeds}


def _skill_ckpt(train_stem: str, env: str, seed: int, *, tandem: bool = False) -> str:
    suffix = "_skills" if tandem else ""
    return f"{_save(train_stem)}/{env}/hierarchical_pqn_{env}_seed{seed}{suffix}.safetensors"


def _skills_eval_body(
    train: Run, *, eval_env: str, kind: str, stem: str, rq: int, tags: list[str],
    tandem: bool = False, meta: str | None = None,
) -> dict:
    """Home / zero-shot eval of a hierarchical ckpt under the env's heuristic."""
    src_env = train.env
    tdoc = train.doc
    skill_input = tdoc["SKILL_INPUT_MODE"]
    pov = bool(tdoc["USE_POV_RELATIVE_MAPPER"])
    body = _eval_head(
        rq=rq, kind=kind, env=eval_env, requires=[train.stem], arch_of=train.stem
    )
    body["PAPER_EVAL_SKILLS_CKPT"] = _skill_ckpt(train.stem, src_env, train.seed, tandem=tandem)
    if meta is not None:
        body["PAPER_EVAL_META_CKPT"] = (
            f"{_save(train.stem)}/{src_env}/"
            f"hierarchical_pqn_{src_env}_seed{train.seed}_meta.safetensors"
        )
    body.update(
        {
            "NUM_SEEDS": 1,
            "ALG_NAME": "hierarchical_pqn",
            "TRAIN_MODE": "tandem" if meta is not None else "skills_only",
            "ENV_NAME": eval_env,
            **_spine(pov=pov),
            "SAVE_PATH": _save(stem),
            "WANDB_RUN_NAME": f"{ENVS[eval_env]['prefix']}_{stem}",
            "WANDB_TAGS": tags,
            **_skill_input_top(skill_input),
            **_reward_flags(eval_env, skill_input),
        }
    )
    for key in ("EGO_OBSERVATION_FRAME", "EGO_ACTION_FRAME", "USE_POV_RELATIVE_MAPPER", "USE_POV_FACING_FRAME"):
        if key in tdoc:
            body[key] = tdoc[key]
    if meta is not None:
        body["META_INPUT_MODE"] = tdoc["META_INPUT_MODE"]
    body["alg"] = {
        **_skill_heads_alg(eval_env),
        "FRAME_SKIP": 4,
        "FRAME_STACK_SIZE": 4,
        **_reward_alg(eval_env, skill_input),
        **_skill_input_alg(skill_input),
        **({"META_INPUT_MODE": tdoc["META_INPUT_MODE"]} if meta is not None else {}),
    }
    if tdoc.get("EGO_OBSERVATION_FRAME") == "screen":
        body["alg"]["SKILL_INPUT_MODE"] = "ego"
        body["alg"]["EGO_OBSERVATION_FRAME"] = "screen"
        body["alg"]["EGO_ACTION_FRAME"] = "screen"
        body["alg"]["USE_POV_RELATIVE_MAPPER"] = False
        body["alg"]["USE_POV_FACING_FRAME"] = False
    body.update(_eval_tail(train.seed))
    return body


def _pqn_eval_body(train: Run, *, stem: str, tags: list[str]) -> dict:
    from concept_based_learning.paper_eval.checkpoints import resolve_flat_pqn_checkpoint

    tdoc = train.doc
    env = train.env
    oc = bool(tdoc["OBJECT_CENTRIC"])
    ckpt = resolve_flat_pqn_checkpoint(
        object_centric=oc,
        env_name=env,
        seed=train.seed,
        save_root=_save(train.stem),
        vmap_index=0,
        oc_input_mode=tdoc.get("OC_INPUT_MODE"),
    )
    rel_ckpt = "./" + str(Path(ckpt).relative_to(_ROOT))
    body = _eval_head(rq=1, kind="flat_pqn", env=env, requires=[train.stem], arch_of=train.stem)
    body.update(
        {
            "PAPER_EVAL_FLAT_PQN_CKPT": rel_ckpt,
            "PAPER_EVAL_OBJECT_CENTRIC": oc,
            "NUM_SEEDS": 1,
            "ALG_NAME": "flat_pqn",
            "ENV_NAME": env,
            **{
                k: tdoc[k]
                for k in (
                    "OBJECT_CENTRIC",
                    "OC_INPUT_MODE",
                    "MAX_OBJECTS",
                    "PIXEL_RESIZE",
                    "PIXEL_RESIZE_SHAPE",
                    "PIXEL_GRAYSCALE",
                    "USE_NATIVE_DOWNSCALING",
                )
                if k in tdoc
            },
            "SAVE_PATH": _save(train.stem),
            "WANDB_RUN_NAME": f"{ENVS[env]['prefix']}_{stem}",
            "WANDB_TAGS": tags,
        }
    )
    body.update(_eval_tail(train.seed))
    return body


def _floor_body(kind: str, env: str, *, stem: str, tags: list[str]) -> dict:
    if kind == "random_actions":
        body = _eval_head(rq=2, kind="random_actions", env=env, requires=[], arch_of=None)
        body.update(
            {
                "NUM_SEEDS": 1,
                "ALG_NAME": "flat_pqn",
                "ENV_NAME": env,
                "OBJECT_CENTRIC": True,
                "FRAME_SKIP": 4,
                "FRAME_STACK_SIZE": 4,
                "SAVE_PATH": _save(stem),
                "WANDB_RUN_NAME": f"{ENVS[env]['prefix']}_{stem}",
                "WANDB_TAGS": tags,
            }
        )
    else:
        body = _skills_body(env, stem=stem, seed=FLOOR_SEED, rq="1", tags=tags)
        body.pop("SEED")
        head = _eval_head(
            rq=1,
            kind="heuristic_random" if kind == "heuristic_random" else "fixed_skills",
            env=env,
            requires=[],
            arch_of=None,
        )
        for k in ("PAPER_EVAL_RQ", "PAPER_EVAL_PHASE", "PAPER_EVAL_KIND", "PAPER_EVAL_MODS",
                  "PAPER_EVAL_REQUIRES_RQ1"):
            body.pop(k, None)
        body = {**head, **body}
        if kind != "heuristic_random":
            body["SKILL_POLICY_MODE"] = kind
    body.update(_eval_tail(FLOOR_SEED, num_seeds=FLOOR_EVAL_NUM_SEEDS))
    return body


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------


def build_runs() -> list[Run]:
    runs: list[Run] = []
    by_stem: dict[str, Run] = {}

    def add(run: Run) -> Run:
        assert run.stem not in by_stem, run.stem
        runs.append(run)
        by_stem[run.stem] = run
        return run

    def skills_train(arm: str, env: str, seed: int, *, tier: str, rq: str,
                     skill_input: str = "ego", pov: bool = True, note: str = "") -> Run:
        stem = _stem(arm, env, seed)
        hours, measured = _train_hours(arm, env)
        tags = _tags(f"rq:{rq}", f"env:{env}", f"arm:{arm}", f"skill_input:{skill_input}",
                     "mapper:pov_facing" if pov else "mapper:world", "phase:skills_only")
        doc = _skills_body(env, stem=stem, seed=seed, rq=rq, skill_input=skill_input,
                           pov=pov, tags=tags)
        return add(Run(stem, tier, rq, "train", arm, env, seed, hours, measured, doc,
                       f"# Thesis-final {arm} skills — {env} seed{seed} (50M optim 1:1).",
                       note=note))

    def home_eval(train: Run, *, tier: str, kind: str = "skills", tandem: bool = False,
                  meta: str | None = None) -> Run:
        stem = train.stem + "_eval"
        tags = _tags(f"rq:{train.rq}", f"env:{train.env}", f"arm:{train.arm}", "phase:mod_eval")
        if train.doc["PAPER_EVAL_KIND"] == "flat_pqn":
            doc = _pqn_eval_body(train, stem=stem, tags=tags)
        else:
            doc = _skills_eval_body(train, eval_env=train.env, kind=kind, stem=stem,
                                    rq=int(train.rq[0]), tags=tags, tandem=tandem, meta=meta)
        return add(Run(stem, tier, train.rq, "eval", train.arm, train.env, train.seed,
                       _eval_hours(train.env), False, doc,
                       f"# Thesis-final home mod eval — {train.stem}.", deps=[train.stem]))

    def zs_eval(train: Run, dst: str, *, tier: str, arm: str, tandem: bool = False) -> Run:
        stem = zs_stem(arm, train.env, dst, train.seed)
        tags = _tags("rq:2", f"env:{dst}", f"src:{train.env}", f"arm:{arm}", "phase:zero_shot")
        doc = _skills_eval_body(train, eval_env=dst, kind="transfer", stem=stem, rq=2,
                                tags=tags, tandem=tandem)
        return add(Run(stem, tier, "2", "eval", arm, dst, train.seed, _eval_hours(dst),
                       False, doc,
                       f"# Thesis-final ZS — {train.stem} skills + {dst} heuristic.",
                       deps=[train.stem]))

    def floor(kind: str, env: str, *, tier: str) -> Run:
        stem = floor_stem(kind, env)
        tags = _tags("rq:floor", f"env:{env}", f"baseline:{kind}", "phase:mod_eval")
        doc = _floor_body(kind, env, stem=stem, tags=tags)
        return add(Run(stem, tier, "floor", "eval", f"floor_{kind}", env, None,
                       _eval_hours(env, floor=True), False, doc,
                       f"# Thesis-final floor — {kind} on {env} (eval only)."))

    # --- RQ1: ours + monolithic baselines on the three home envs -----------------
    for env in HOME_ENVS:
        for seed in SEEDS:
            t = skills_train("skills_ego", env, seed, tier="core", rq="1")
            home_eval(t, tier="core")
            for variant in ("pixel", "oc"):
                arm = f"pqn_{variant}"
                stem = _stem(arm, env, seed)
                hours, measured = _train_hours(arm, env)
                tags = _tags("rq:1", f"env:{env}", f"arm:{arm}", "train_mode:flat_pqn")
                doc = _pqn_body(env, stem=stem, seed=seed, variant=variant, tags=tags)
                pt = add(Run(stem, "core", "1", "train", arm, env, seed, hours, measured, doc,
                             f"# Thesis-final flat PQN {variant} — {env} seed{seed}."))
                home_eval(pt, tier="core")

    # --- RQ1 ablations (Seaquest, same heuristic) ------------------------------
    # Flat PQN / tandem meta OC stay on oc_simple_rel_role (matrix + role pool).
    # Skill OC ablation uses oc_flat_rel: same relative rows + is_target, flattened
    # into the ego-style MLP (not the mismatched full oc_embed skill path).
    # Seaquest and Name This Game: pixel outline, target-aligned no-POV,
    # screen-frame no-POV, flat OC, and role-aware flat OC. Each transfers
    # to every other suite game.
    ablation_arms = (
        ("skills_pixel_outline", "pixel_embed", True),
        ("skills_nopov", "ego", False),
        ("skills_screen", "screen", False),
        ("skills_oc_flat_rel", "oc_flat_rel", True),
        ("skills_oc_flat_rel_role", "oc_flat_rel_role", True),
    )
    for env in HOME_ENVS:
        for seed in SEEDS:
            for arm, skill_input, pov in ablation_arms:
                t = skills_train(arm, env, seed, tier="core", rq="1abl",
                                 skill_input=skill_input, pov=pov)
                home_eval(t, tier="core")
                for dst in transfer_targets(env):
                    zs_eval(t, dst, tier="core", arm=arm)

    # --- RQ3: OC and pixel tandem meta-policies on both home environments -------
    tandem_oc: dict[tuple[str, int], Run] = {}
    for env in HOME_ENVS:
        metas = ("oc", "pixel")
        for meta in metas:
            for seed in SEEDS:
                arm = f"tandem_{meta}"
                stem = _stem(arm, env, seed)
                hours, measured = _train_hours(arm, env)
                tags = _tags("rq:3", f"env:{env}", f"arm:{arm}", "train_mode:tandem",
                             f"meta_input:{meta}", "skill_input:ego")
                doc = _tandem_body(env, stem=stem, seed=seed, meta=meta, tags=tags)
                t = add(Run(stem, "core", "3", "train", arm, env, seed, hours, measured, doc,
                            f"# Thesis-final tandem {meta} meta — {env} seed{seed}."))
                home_eval(t, tier="core", kind="tandem", tandem=True, meta=meta)
                if meta == "oc":
                    tandem_oc[(env, seed)] = t

    # --- RQ2: every learned Seaquest/NTG skill arm → every other suite game -----
    for seed in SEEDS:
        src = by_stem[_stem("skills_ego", "seaquest", seed)]
        for dst in transfer_targets("seaquest"):
            zs_eval(src, dst, tier="core", arm="skills_ego")
        src = by_stem[_stem("skills_ego", "namethisgame", seed)]
        for dst in transfer_targets("namethisgame"):
            zs_eval(src, dst, tier="core", arm="skills_ego")

    # --- Floors: attribution on homes, transfer floors on every target ---------
    for env in ENVS:
        floor("random_actions", env, tier="core")
        floor("heuristic_random", env, tier="core")
        floor("fixed_ego_rules", env, tier="core")
        if env in HOME_ENVS:
            floor("constant_ego", env, tier="core")

    # --- Extras: native ceilings on the non-home transfer targets ----------------
    for env in EXTRA_CEILING_ENVS:
        for seed in SEEDS:
            t = skills_train("skills_ego", env, seed, tier="extras", rq="2ceil")
            home_eval(t, tier="extras")
            for dst in HOME_ENVS:
                zs_eval(t, dst, tier="extras", arm="skills_ego")

    # Tandem skill heads transfer to every other suite game.
    for (env, _seed), t in tandem_oc.items():
        for dst in transfer_targets(env):
            zs_eval(t, dst, tier="core", arm="tandem_oc_skills", tandem=True)
    for env in HOME_ENVS:
        for seed in SEEDS:
            t = by_stem[_stem("tandem_pixel", env, seed)]
            for dst in transfer_targets(env):
                zs_eval(t, dst, tier="core", arm="tandem_pixel_skills", tandem=True)

    return runs


# ---------------------------------------------------------------------------
# GPU plan (static per-GPU queues; trains first so no worker deadlocks)
# ---------------------------------------------------------------------------


@dataclass
class Slot:
    run_stem: str
    start: float
    end: float


def plan_gpus(runs: list[Run], n: int) -> list[list[Slot]]:
    """LPT for trains, then dependency-aware list scheduling for evals."""
    by_stem = {r.stem: r for r in runs}
    lanes: list[list[Slot]] = [[] for _ in range(n)]
    free = [(0.0, g) for g in range(n)]
    heapq.heapify(free)
    finish: dict[str, float] = {}

    # Per tier: trains (never wait) before evals (wait only on earlier-queued
    # trains), so cross-GPU waits cannot form a cycle.
    for tier in ("core", "extras"):
        trains = sorted(
            (r for r in runs if r.tier == tier and r.phase == "train"),
            key=lambda r: -r.hours,
        )
        for r in trains:
            t, g = heapq.heappop(free)
            lanes[g].append(Slot(r.stem, t, t + r.hours))
            finish[r.stem] = t + r.hours
            heapq.heappush(free, (t + r.hours, g))

        pending = [r for r in runs if r.tier == tier and r.phase == "eval"]
        while pending:
            t, g = heapq.heappop(free)

            def ready(r: Run) -> float:
                return max([finish[d] for d in r.deps if d in by_stem], default=0.0)

            pending.sort(key=lambda r: (max(ready(r), t), -r.hours))
            r = pending.pop(0)
            start = max(t, ready(r))
            lanes[g].append(Slot(r.stem, start, start + r.hours))
            finish[r.stem] = start + r.hours
            heapq.heappush(free, (start + r.hours, g))

    return lanes


def nexus_hours_on_gpu(gpu: int, n: int) -> float:
    total = 0.0
    for k, (arm, _alg, _env, _seed) in enumerate(_nexus_jobs()):
        if k % n != gpu:
            continue
        train_h = NEXUS_TRAIN_HOURS if arm == "nexus_symbolic" else NEXUS_HRL_TRAIN_HOURS
        total += train_h + NEXUS_EVAL_HOURS
    return total


def suite_stems() -> tuple[str, ...]:
    """All runs, trains-first (single-GPU order / --list / --resume)."""
    return tuple(s.run_stem for s in plan_gpus(build_runs(), 1)[0])


def gpu_suite_stems(gpu: int, n: int) -> tuple[str, ...]:
    return tuple(s.run_stem for s in plan_gpus(build_runs(), n)[gpu])


def gpu_suite_name(gpu: int, n: int) -> str:
    return f"{SUITE}_{n}gpu_g{gpu}"


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _dump(run: Run) -> str:
    return run.header + "\n" + yaml.safe_dump(run.doc, sort_keys=False, width=100)


def _nexus_jobs() -> list[tuple[str, str, str, int]]:
    """(arm, alg, env, seed) for symbolic Seaquest + HRL on home envs."""
    jobs: list[tuple[str, str, str, int]] = []
    for arm, alg, env in NEXUS_SYMBOLIC:
        for seed in SEEDS:
            jobs.append((arm, alg, env, seed))
    for arm, alg, env in NEXUS_HRL_OPTIM:
        for seed in SEEDS:
            jobs.append((arm, alg, env, seed))
    return jobs


def _nexus_script() -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# Thesis-final NEXUS: Seaquest symbolic (paper HPs) + home-env HRL (50M optim).",
        "# Generated by scripts/materialize_thesis_final.py.",
        "# usage: GPU=1 NEXUS_SHARD=1/3 bash $0   (job k runs when k % n == i)",
        "set -euo pipefail",
        'cd "$(dirname "$0")/.."',
        'export CUDA_VISIBLE_DEVICES="${GPU:-0}"',
        'SHARD="${NEXUS_SHARD:-0/1}"',
        'I="${SHARD%/*}"',
        'N="${SHARD#*/}"',
        "",
    ]
    for k, (arm, alg, env, seed) in enumerate(_nexus_jobs()):
        stem = f"tf_{env}_{arm}_seed{seed}"
        save = f"{OUT_MODELS}/{stem}"
        out = f"./outputs/models/paper_eval/eval/{stem}_eval_mods.json"
        mods = " ".join(_mod_keys(env))
        lines += [
            f"if [ $(({k} % N)) -eq \"$I\" ] && ! python scripts/run_nexus.py check-eval --path {out} --eval-episodes 3 --eval-num-seeds {EVAL_NUM_SEEDS} --mod-keys {mods}; then",
            f"  if ! compgen -G '{save}/*/*_seed{seed}_vmap0.safetensors' > /dev/null; then",
            f"    python scripts/run_nexus.py train --alg {alg} --seed {seed} "
            f"--save-path {save}",
            "  fi",
            f"  python scripts/run_nexus.py mod-eval --alg {alg} --ckpt {save} "
            f"--seed {seed + 10_000} --eval-num-seeds {EVAL_NUM_SEEDS} "
            f"--eval-episodes 3 --mod-keys {mods} --out-json {out}",
            "fi",
        ]
    return "\n".join(lines) + "\n"


def _launch_lines(n: int) -> list[str]:
    """Per GPU: its paper_eval queue, then its NEXUS shard (``;`` → NEXUS runs
    even if a paper_eval run failed)."""
    out = []
    for g in range(n):
        inner = (
            f"python scripts/run_paper_eval.py --suite {gpu_suite_name(g, n)} "
            f"--gpus {g} --wait-deps --wait-deps-timeout-h 72 --skip-rq-gate --lean --resume; "
            f"GPU={g} NEXUS_SHARD={g}/{n} bash scripts/run_thesis_final_nexus.sh"
        )
        out.append(f"nohup bash -c '{inner}' > logs/thesis_final_g{g}of{n}.log 2>&1 &")
    return out


def _run_table(runs: list[Run], tier: str) -> list[str]:
    lines = [
        "| # | stem | RQ | phase | arm | env | seed | ~h (A6000) | deps |",
        "|---|------|----|-------|-----|-----|------|-----------|------|",
    ]
    i = 0
    for r in runs:
        if r.tier != tier:
            continue
        i += 1
        h = f"{r.hours:.2f}" + ("" if r.measured or r.phase == "eval" else "~")
        seed = "—" if r.seed is None else str(r.seed)
        deps = ", ".join(r.deps) if r.deps else "—"
        lines.append(
            f"| {i} | `{r.stem}` | {r.rq} | {r.phase} | {r.arm} | {r.env} | {seed} | {h} | {deps} |"
        )
    return lines


def _runlog(runs: list[Run]) -> str:
    tot = {t: sum(r.hours for r in runs if r.tier == t) for t in ("core", "extras")}
    nexus_h = (
        len(NEXUS_SYMBOLIC) * len(SEEDS) * (NEXUS_TRAIN_HOURS + NEXUS_EVAL_HOURS)
        + len(NEXUS_HRL_OPTIM) * len(SEEDS) * (NEXUS_HRL_TRAIN_HOURS + NEXUS_EVAL_HOURS)
    )
    lines = [
        "# Thesis-final 50M sweep — run log",
        "",
        "Generated by `scripts/materialize_thesis_final.py` (do not edit by hand).",
        "",
        "## Launch (cluster)",
        "",
        "```bash",
        "# From ConceptBasedLearning root. Point at the Seaquest-robustness jaxtari fork",
        "# (must be first on PYTHONPATH — a site-packages / JAXAtari main install will",
        "# miss no_enemies / spawn_axis_bias / …).",
        "export JAXATARI_ROOT=/path/to/JAXAtari-seaquest-robustness",
        "export PAPER_EVAL_JAXATARI_ROOT=$JAXATARI_ROOT",
        "export PYTHONPATH=$JAXATARI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}",
        "export NEXUS_ROOT=${NEXUS_ROOT:-$PWD/third_party/symbolic_options}",
        "python scripts/materialize_thesis_final.py   # refresh YAMLs + this log",
        "mkdir -p logs",
        "# then the 2-GPU or 3-GPU block below",
        "```",
        "",
        "- Prefer A6000 / RTX 6000 Ada (≥48 GB): pixel skills + pixel PQN + pixel tandem",
        "  at 1024 envs do not fit a 12 GB card.",
        "- `--resume` skips finished stems; wipe `outputs/models/paper_eval/thesis_final/`",
        "  and `outputs/models/paper_eval/eval/tf_*` before a clean full sweep.",
        "- Kill stuck local smokes by **PID** from `nvidia-smi` (RTPT renames the process).",
        "",
        "## Locked recipe",
        "",
        f"- Outer loop: {NUM_ENVS} envs × {NUM_STEPS} steps × {NUM_UPDATES} updates "
        f"(= {TIMESTEPS:,} env steps), {NUM_EPOCHS} epochs × {NUM_MINIBATCHES} minibatches.",
        f"- LR {LR} linear decay (SKILL_LR = META_LR = LR), MAX_GRAD_NORM {MAX_GRAD_NORM:g}, "
        f"ε {EPS_START}→{EPS_FINISH} over {EPS_DECAY:.0%} (skills + meta), γ {GAMMA}, λ {LAMBDA}.",
        "- Ego spine: POV facing frame, ψ 8 bins, no distance clip, no extra channels, "
        "sparse C/D/A, DESTROY contact-fail, no living penalty, surfacesub5.",
        f"- Train seeds {', '.join(map(str, SEEDS))}; eval EVAL_SEED = seed + 10000, "
        f"{EVAL_NUM_SEEDS} eval seeds × 3 episodes per mod (floors: "
        f"{FLOOR_EVAL_NUM_SEEDS} × 3, single config).",
        f"- Seaquest mods: {', '.join(SEAQUEST_MOD_KEYS)}.",
        *[f"- {e} mods: {', '.join(_mod_keys(e))}." for e in ENVS if e != "seaquest"],
        f"- Tandem meta OC = `{TANDEM_META_OC}`; flat PQN OC = `{PQN_OC_MODE}` "
        "(same feature builder). Skill OC ablation = `oc_flat_rel` (same relative "
        "rows + `is_target`, flattened MLP; no role-pool / set encoder).",
        "",
        "## Follow-up (not in this suite)",
        "",
        "- **Robustness-focus probes:** several paper_core mods above are weak "
        "tests of what we care about (layout shift vs task-preserving "
        "simplification). Opt-in allowlists live at "
        "`configs/experiment/paper_eval/mods/<game>_robustness_focus.yaml` "
        "(implemented in the Seaquest-robustness fork’s `robustness_probes.py`). "
        "After the locked thesis_final sweep, re-eval frozen `tf_*` checkpoints "
        "with `PAPER_EVAL_MODS: <game>_robustness_focus`. Details: "
        "`docs/robustness_focus_2026-10-15.md`.",
        "- Transfers: Seaquest ego, Seaquest skill ablations (pixel outline, "
        "no-POV, oc_flat_rel), Name This Game ego, and every scheduled tandem "
        "each go to every other game in the suite. Time Pilot is not included.",
        "",
        "## Budget",
        "",
        f"- core: {sum(r.tier == 'core' for r in runs)} runs ≈ {tot['core']:.1f} A6000-h",
        f"- extras: {sum(r.tier == 'extras' for r in runs)} runs ≈ {tot['extras']:.1f} A6000-h",
        f"- NEXUS (outside paper_eval): "
        f"{len(NEXUS_SYMBOLIC) * len(SEEDS)} Seaquest symbolic (paper HPs) + "
        f"{len(NEXUS_HRL_OPTIM) * len(SEEDS)} home HRL (50M optim, NUM_AGENTS=3) ≈ "
        f"{nexus_h:.1f} h (estimate)",
        "",
    ]
    tier_of = {r.stem: r.tier for r in runs}
    for n in GPU_COUNTS:
        lines += [f"### {n} GPUs", ""]
        for g, lane in enumerate(plan_gpus(runs, n)):
            core_end = max((s.end for s in lane if tier_of[s.run_stem] == "core"), default=0.0)
            end = lane[-1].end if lane else 0.0
            nh = nexus_hours_on_gpu(g, n)
            lines.append(
                f"- GPU {g} (`{gpu_suite_name(g, n)}`, {len(lane)} runs): core done ≈ "
                f"{core_end:.1f} h, all ≈ {end:.1f} h, + NEXUS shard {nh:.1f} h → "
                f"{end + nh:.1f} h"
            )
        lines += ["", "```bash", "mkdir -p logs", *_launch_lines(n), "```", ""]
    lines += ["## Core runs", "", *_run_table(runs, "core"), ""]
    lines += ["## Extras runs", "", *_run_table(runs, "extras"), ""]
    lines += ["## NEXUS runs (`scripts/run_thesis_final_nexus.sh`)", ""]
    for arm, alg, env in (*NEXUS_SYMBOLIC, *NEXUS_HRL_OPTIM):
        for seed in SEEDS:
            lines.append(f"- `tf_{env}_{arm}_seed{seed}` — `{alg}` train + mod eval")
    return "\n".join(lines) + "\n"


def materialize(*, dry: bool) -> list[Run]:
    runs = build_runs()
    if not dry:
        _OUT.mkdir(parents=True, exist_ok=True)
        for old in _OUT.glob("tf_*.yaml"):
            # Probe and avsep files are the live recipe comparison. The
            # generator does not emit them, and a running chain still opens
            # them by path.
            if "probe_" in old.name or "avsep" in old.name:
                continue
            old.unlink()
        for r in runs:
            (_OUT / f"{r.stem}.yaml").write_text(_dump(r), encoding="utf-8")
        _RUNLOG.write_text(_runlog(runs), encoding="utf-8")
        _NEXUS_SH.write_text(_nexus_script(), encoding="utf-8")
        _NEXUS_SH.chmod(0o755)
    print(f"{len(runs)} configs → {_OUT.relative_to(_ROOT)}")
    print(f"run log → {_RUNLOG.relative_to(_ROOT)}; nexus → {_NEXUS_SH.relative_to(_ROOT)}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--print-stems", action="store_true")
    parser.add_argument("--plan", type=int, default=None, help="print N-GPU schedule")
    args = parser.parse_args()
    if args.print_stems:
        print("\n".join(suite_stems()))
        return
    if args.plan:
        for g, lane in enumerate(plan_gpus(build_runs(), args.plan)):
            print(f"\n== GPU {g} ({gpu_suite_name(g, args.plan)}) ==")
            for s in lane:
                print(f"  {s.start:6.2f} → {s.end:6.2f} h  {s.run_stem}")
            print(f"  then NEXUS shard ≈ {nexus_hours_on_gpu(g, args.plan):.1f} h")
        return
    materialize(dry=args.dry)


if __name__ == "__main__":
    main()
