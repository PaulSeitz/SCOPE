"""Merge paper-eval YAML knobs onto a runtime config (visualizer / eval)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Keep in sync with scripts/run_paper_eval.py::_EVAL_OVERRIDE_KEYS.
EVAL_OVERRIDE_KEYS: tuple[str, ...] = (
    "ENV_NAME",
    "MAX_OBJECTS",
    "SAVE_PATH",
    "WANDB_RUN_NAME",
    "WANDB_TAGS",
    "HEURISTIC_MODE",
    "NO_TARGET_GAME_ACTION",
    "SKILL_DEATH_PENALIZE_COLLECT",
    "COLLECT_SPARSE_ONLY",
    "AVOID_SPARSE_ONLY",
    "COLLECT_LIVING_PENALTY",
    "DESTROY_CONTACT_FAIL",
    "DESTROY_LIVING_PENALTY",
    "DESTROY_Y_ALIGN",
    "DESTROY_STANDOFF",
    "EVAL_GAME_MODS",
    "GAME_MODS",
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
    "COLLECT_NEAR_RANGE",
    "ENEMY_REQUIRE_ON_SCREEN",
    "DESTROY_NEAREST_IN_DANGER_ONLY",
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
    "USE_EGO_HIT_CORRIDOR",
    "EGO_HIT_CORRIDOR_SAT",
    "EGO_AABB_GEOM",
    "EGO_LEGAL_ACTION_MASK",
    "EGO_LEGAL_ACTION_MASK_MODE",
    "SKILL_PIXEL_PAINT_TARGET_OUTLINE",
    "USE_POV_RELATIVE_MAPPER",
    "USE_POV_FIXED_BILATERAL",
    "POV_BILATERAL_AXIS",
    "USE_POV_FACING_FRAME",
    "EGO_POSE_XY",
    "EGO_ACTION_DIM",
    "USE_LEGACY_SEAQUEST_ACTION_MAPPER",
    "USE_LEGACY_SEAQUEST_ASCENT_BLOCK",
    "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD",
    "SEAQUEST_SURFACE_LABEL",
    "SEAQUEST_SWAP_DIVER_ENEMY_LABELS",
    "SEAQUEST_SWAP_DIVER_SHARK_ROLES",
    "PONG_BALL_AS_DESTROY",
    "SKILL_INPUT_MODE",
    "SKILL_POLICY_MODE",
    "FIXED_EGO_ALIGN_DEG",
    "FRAME_STACK_SIZE",
)

# Written next to checkpoints; must survive training even if runtime dict is mutated.
EGO_PERSIST_KEYS: tuple[str, ...] = (
    "USE_POV_RELATIVE_MAPPER",
    "USE_POV_FIXED_BILATERAL",
    "POV_BILATERAL_AXIS",
    "USE_POV_FACING_FRAME",
    "EGO_POSE_XY",
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
    "USE_EGO_HIT_CORRIDOR",
    "EGO_HIT_CORRIDOR_SAT",
    "EGO_AABB_GEOM",
    "USE_LEGACY_SEAQUEST_ACTION_MAPPER",
    "USE_LEGACY_SEAQUEST_ASCENT_BLOCK",
    "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD",
    "SEAQUEST_SURFACE_LABEL",
    "SKILL_INPUT_MODE",
    "FRAME_STACK_SIZE",
    "HEURISTIC_MODE",
)


def flatten_experiment_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(raw)
    alg = data.pop("alg", {}) or {}
    if not isinstance(alg, dict):
        alg = {}
    return {**data, **alg}


def _paper_eval_config_dir(project_root: Path) -> Path:
    return project_root / "configs" / "experiment" / "paper_eval"


def resolve_architecture_yaml(ref: str, *, project_root: Path) -> Path:
    stem = str(ref).removesuffix(".yaml")
    root = _paper_eval_config_dir(project_root)
    direct = root / "rq1" / f"{stem}.yaml"
    if direct.is_file():
        return direct
    matches = list(root.rglob(f"{stem}.yaml"))
    if not matches:
        raise FileNotFoundError(f"Architecture config not found: {ref}")
    return matches[0]


def architecture_config(
    raw: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    """Same merge as run_paper_eval._architecture_config (train arch + eval overrides)."""
    ref = raw.get("PAPER_EVAL_ARCHITECTURE_OF")
    if not ref:
        return flatten_experiment_cfg(copy.deepcopy(raw))
    ref_path = resolve_architecture_yaml(str(ref), project_root=project_root)
    with ref_path.open(encoding="utf-8") as f:
        arch_raw = yaml.safe_load(f) or {}
    cfg = flatten_experiment_cfg(copy.deepcopy(arch_raw))
    eval_flat = flatten_experiment_cfg(copy.deepcopy(raw))
    for key in EVAL_OVERRIDE_KEYS:
        if key in eval_flat:
            cfg[key] = eval_flat[key]
    return cfg


def merge_paper_eval_yaml(
    base: dict[str, Any],
    yaml_path: str | Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Apply paper-eval train/eval YAML overrides on top of checkpoint config."""
    root = project_root or Path(__file__).resolve().parents[3]
    path = Path(yaml_path)
    if not path.is_absolute():
        path = root / path
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")
    merged_src = architecture_config(raw, project_root=root)
    out = copy.deepcopy(base)
    for key in EVAL_OVERRIDE_KEYS:
        if key in merged_src:
            out[key] = merged_src[key]
    return out


def snapshot_ego_persist_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: config[k] for k in EGO_PERSIST_KEYS if k in config}


def restore_ego_persist_keys(config: dict[str, Any], snapshot: dict[str, Any]) -> None:
    for key, value in snapshot.items():
        config[key] = value
