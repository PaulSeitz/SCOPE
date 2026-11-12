"""Resolve archived evaluation configs without importing the historical scheduler."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def flatten_experiment_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    alg = raw.pop("alg", {}) or {}
    return {**raw, **alg}


_EVAL_OVERRIDE_KEYS = (
    # Critical for transfer: architecture YAMLs are Seaquest trains; without these
    # the eval silently keeps ENV_NAME=seaquest (mislabeled Pong/BeamRider scores).
    "ENV_NAME",
    "MAX_OBJECTS",
    "SAVE_PATH",
    "WANDB_RUN_NAME",
    "WANDB_TAGS",
    "HEURISTIC_MODE",
    "NO_TARGET_GAME_ACTION",
    "SKILL_DEATH_PENALIZE_COLLECT",
    "COLLECT_DEATH_PRECEDENCE",
    "AVOID_KILLED_EVENT_MODE",
    "COLLECT_SPARSE_ONLY",
    "COLLECT_LIVING_PENALTY",
    "DESTROY_CONTACT_FAIL",
    "DESTROY_LIVING_PENALTY",
    "DESTROY_Y_ALIGN",
    "DESTROY_STANDOFF",
    "EVAL_GAME_MODS",
    "GAME_MODS",
    "SEAQUEST_SWAP_DIVER_ENEMY_LABELS",
    "SEAQUEST_SWAP_DIVER_SHARK_ROLES",
    "SURFACE_DANGER_CLOSE_RANGE",
    "DESTROY_INTERRUPT_RANGE",
    "AVOID_INTERRUPT_RANGE",
    "AVOID_INTERRUPT_RANGE_Y",
    "SURFACE_SUB_AVOID_RANGE_X",
    "SURFACE_SUB_AVOID_RANGE_Y",
    "SURFACE_SUB_NEAR_RANGE",
    "DIVER_LANE_Y_PX",
    "DIVER_LANE_BLOCK_RANGE",
    "COLLECT_REQUIRES_NO_DESTROY",
    "COLLECT_EDGE_MARGIN_PX",
    "COLLECT_EDGE_HARD",
    "COLLECT_REQUIRE_FULLY_ON_SCREEN",
    "COLLECT_SKIP_WHEN_FULL",
    "COLLECT_NEAREST_IN_RANGE_ONLY",
    "COLLECT_EARLY_WHEN_NO_THREATS",
    "BREAKOUT_AUTO_LAUNCH",
    "COLLECT_NEAR_RANGE",
    "ENEMY_REQUIRE_ON_SCREEN",
    "DESTROY_NEAREST_IN_DANGER_ONLY",
    # Ego layout must follow the transfer YAML / ckpt (scalar snap flags included).
    "USE_LEGACY_EGO_OBS",
    "USE_EGO_SIZE_FEATURES",
    "USE_EGO_VELOCITY_FEATURES",
    "USE_EGO_BLOCKING_FEATURES",
    "USE_EGO_WEAPON_FEATURES",
    "USE_EGO_FACING_ENEMY_FEATURES",
    "EGO_FORCE_CAN_FIRE",
    "USE_EGO_SHOT_FEATURES",
    "USE_EGO_SHOT_BETWEEN",
    "EGO_DISTANCE_CLIP_MAX_PX",
    "EGO_DISTANCE_CLIP_CHOICES_PX",
    "EGO_DISTANCE_MODE",
    "EGO_DISTANCE_DELTA_SIGN_EPS_PX",
    "EGO_ANGLE_BINS",
    "EGO_PSI_ANGLE_BINS",
    "EGO_PSI_ANGLE_SCALAR",
    "EGO_ANGLE_BINS_SCALAR",
    "EGO_ANGLE_SINCOS_SNAP",
    "EGO_DISTANCE_BINS",
    "EGO_DISTANCE_BINS_SCALAR",
    "EGO_DISTANCE_BIN_CENTERS_PX",
    "EGO_DISTANCE_BIN_MAX_PX",
    "EGO_DISTANCE_FLAG_THRESHOLDS_PX",
    "EGO_DISTANCE_HARD_EDGES_PX",
    "EGO_RESTRAINED_POSE",
    "EGO_RESTRAINED_LATERAL_BAND_PX",
    "EGO_RESTRAINED_NEAR_FAR_PX",
    "USE_EGO_HIT_CORRIDOR",
    "EGO_HIT_CORRIDOR_SAT",
    "EGO_AABB_GEOM",
    "EGO_LEGAL_ACTION_MASK",
    "EGO_LEGAL_ACTION_MASK_MODE",
    # Pixel ablation: keep outline on/off from the eval YAML when architecture
    # is pinned to a train config that still paints by default.
    "SKILL_PIXEL_PAINT_TARGET_OUTLINE",
    "USE_POV_RELATIVE_MAPPER",
    "EGO_OBSERVATION_FRAME",
    "EGO_ACTION_FRAME",
    "USE_POV_FIXED_BILATERAL",
    "POV_BILATERAL_AXIS",
    "USE_POV_FACING_FRAME",
    "EGO_POSE_XY",
    "EGO_ACTION_DIM",
    "SKILL_POLICY_MODE",
    "FIXED_EGO_ALIGN_DEG",
    "SEED",
    "EVAL_SEED",
    "EVAL_NUM_SEEDS",
    "OFF_POLICY",
)


def architecture_config(raw: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Load arch YAML referenced by PAPER_EVAL_ARCHITECTURE_OF if present.

    When an architecture ref is used, still apply eval-local overrides from
    ``raw`` (e.g. ``HEURISTIC_MODE: avoid_only`` on an AVOID probe).
    """
    ref = raw.get("PAPER_EVAL_ARCHITECTURE_OF")
    if not ref:
        return flatten_experiment_cfg(copy.deepcopy(raw))
    ref_path = config_dir / "rq1" / f"{str(ref).removesuffix('.yaml')}.yaml"
    if not ref_path.is_file():
        # search
        matches = list(config_dir.rglob(f"{str(ref).removesuffix('.yaml')}.yaml"))
        if not matches:
            raise FileNotFoundError(f"Architecture config not found: {ref}")
        ref_path = matches[0]
    cfg = flatten_experiment_cfg(copy.deepcopy(load_yaml(ref_path)))
    eval_flat = flatten_experiment_cfg(copy.deepcopy(raw))
    for key in _EVAL_OVERRIDE_KEYS:
        if key in eval_flat:
            cfg[key] = eval_flat[key]
    return cfg
