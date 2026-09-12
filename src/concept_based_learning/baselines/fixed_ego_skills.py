"""Same-scaffolding fixed ego-action skill rules.

Replaces learned skill Q-heads with hand-coded body-frame rules that only use
the same geometry the ego skill input is allowed to see: facing-relative angle
``φ`` and object distance. Target selection, labels, and the POV action mapper
stay identical to the learned transfer stack.

Modes
-----
``fixed_ego_rules``
    Angle-servo controller:
      COLLECT / NAVIGATE — turn toward target, then FORWARD when aligned
      AVOID              — move away from the threat
      DESTROY            — turn to align, then FIRE when aligned
``constant_ego``
    Degenerate controls: FORWARD / FIRE / BACKWARD / FORWARD by skill.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import jax.numpy as jnp

from concept_based_learning.observations.action_space import EGO_MOVE_DELTAS_DEG
from concept_based_learning.observations.egocentric import (
    _center_y,
    _get_player,
    _uses_bottom_edge_y,
    aabb_geom_distance_px,
)
from concept_based_learning.observations.pov_facing_frame import (
    facing_frame_phi_rad,
    resolve_facing_deg,
    world_delta_to_facing_frame,
)

# Ego action indices (18-dim body / facing-relative space).
EGO_NOOP = 0
EGO_FIRE = 1
EGO_FORWARD = 2
EGO_RIGHT = 3
EGO_LEFT = 4
EGO_BACKWARD = 5
EGO_FORWARD_RIGHT = 6
EGO_FORWARD_LEFT = 7
EGO_BACKWARD_RIGHT = 8
EGO_BACKWARD_LEFT = 9
EGO_FORWARD_FIRE = 10

# ~one half-bin of an 8-way ψ encoding.
DEFAULT_ALIGN_RAD = float(jnp.deg2rad(22.5))

SKILL_POLICY_MODES = ("learned", "fixed_ego_rules", "constant_ego")


def skill_policy_mode_from_config(config: dict | None) -> str:
    mode = str((config or {}).get("SKILL_POLICY_MODE", "learned")).lower().strip()
    if mode not in SKILL_POLICY_MODES:
        raise ValueError(
            f"SKILL_POLICY_MODE must be one of {SKILL_POLICY_MODES}, got {mode!r}"
        )
    return mode


def uses_fixed_skill_policy(config: dict | None) -> bool:
    return skill_policy_mode_from_config(config) in {"fixed_ego_rules", "constant_ego"}


def _skill_index_map(skill_names: Sequence[str] | None, num_skills: int) -> dict[str, int]:
    if skill_names:
        return {str(n).upper(): i for i, n in enumerate(skill_names)}
    defaults = ("DESTROY", "COLLECT", "AVOID", "NAVIGATE", "JUMP_OVER")
    return {defaults[i]: i for i in range(int(num_skills))}


def wrap_pi(rad: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(rad), jnp.cos(rad))


def ego_move_for_bearing(bearing_rad: jnp.ndarray) -> jnp.ndarray:
    """Map a body-frame bearing to the nearest pure-move ego action (2..9)."""
    desired_deg = jnp.asarray(bearing_rad, dtype=jnp.float32) * (180.0 / jnp.pi)
    deltas = EGO_MOVE_DELTAS_DEG[2:10]
    diff = (desired_deg - deltas + 180.0) % 360.0 - 180.0
    return (2 + jnp.argmin(jnp.abs(diff))).astype(jnp.int32)


def target_phi_and_distance(
    raw_obs,
    objects,
    target_idx,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Latest-frame facing-frame ``φ`` and AABB surface-gap distance."""
    player = _get_player(raw_obs)
    y_bot = bool(_uses_bottom_edge_y(raw_obs))
    facing = resolve_facing_deg(raw_obs)
    idx = jnp.asarray(target_idx, dtype=jnp.int32)

    p_x = player.x[-1]
    p_y = player.y[-1]
    p_w = player.width[-1]
    p_h = player.height[-1]
    t_x = objects.x[-1, idx]
    t_y = objects.y[-1, idx]
    t_w = objects.width[-1, idx]
    t_h = objects.height[-1, idx]

    p_cx = p_x + p_w / jnp.float32(2.0)
    p_cy = _center_y(p_y, p_h, y_is_bottom_edge=y_bot)
    t_cx = t_x + t_w / jnp.float32(2.0)
    t_cy = _center_y(t_y, t_h, y_is_bottom_edge=y_bot)
    fwd, lat = world_delta_to_facing_frame(t_cx - p_cx, t_cy - p_cy, facing)
    phi = facing_frame_phi_rad(fwd, lat)
    dist = aabb_geom_distance_px(
        p_x,
        p_y,
        p_w,
        p_h,
        t_x,
        t_y,
        t_w,
        t_h,
        y_is_bottom_edge=y_bot,
        facing_deg=facing,
    )
    return phi.astype(jnp.float32), dist.astype(jnp.float32)


def fixed_ego_action_from_phi(
    skill_idx: jnp.ndarray | int,
    phi: jnp.ndarray,
    distance_px: jnp.ndarray,
    *,
    skill_names: Sequence[str] | None = None,
    num_skills: int = 4,
    align_rad: float = DEFAULT_ALIGN_RAD,
) -> jnp.ndarray:
    """Body-frame angle-servo rules for one skill index."""
    names = _skill_index_map(skill_names, num_skills)
    destroy_i = jnp.int32(names.get("DESTROY", 0))
    collect_i = jnp.int32(names.get("COLLECT", 1))
    avoid_i = jnp.int32(names.get("AVOID", 2))
    navigate_i = jnp.int32(names.get("NAVIGATE", collect_i))

    sk = jnp.asarray(skill_idx, dtype=jnp.int32)
    phi = wrap_pi(jnp.asarray(phi, dtype=jnp.float32))
    _ = distance_px  # reserved for future melee thresholds; align-only for now
    align = jnp.float32(align_rad)
    aligned = jnp.abs(phi) <= align

    approach = jnp.where(aligned, jnp.int32(EGO_FORWARD), ego_move_for_bearing(phi))
    flee = ego_move_for_bearing(wrap_pi(phi + jnp.float32(jnp.pi)))
    destroy = jnp.where(aligned, jnp.int32(EGO_FIRE), ego_move_for_bearing(phi))

    is_collect = jnp.logical_or(sk == collect_i, sk == navigate_i)
    is_avoid = sk == avoid_i
    is_destroy = sk == destroy_i
    return jnp.where(
        is_collect,
        approach,
        jnp.where(is_avoid, flee, jnp.where(is_destroy, destroy, approach)),
    ).astype(jnp.int32)


def constant_ego_action(
    skill_idx: jnp.ndarray | int,
    *,
    skill_names: Sequence[str] | None = None,
    num_skills: int = 4,
) -> jnp.ndarray:
    """Degenerate per-skill constant ego actions."""
    names = _skill_index_map(skill_names, num_skills)
    destroy_i = jnp.int32(names.get("DESTROY", 0))
    collect_i = jnp.int32(names.get("COLLECT", 1))
    avoid_i = jnp.int32(names.get("AVOID", 2))
    navigate_i = jnp.int32(names.get("NAVIGATE", collect_i))
    sk = jnp.asarray(skill_idx, dtype=jnp.int32)
    is_collect = jnp.logical_or(sk == collect_i, sk == navigate_i)
    return jnp.where(
        is_collect,
        jnp.int32(EGO_FORWARD),
        jnp.where(
            sk == avoid_i,
            jnp.int32(EGO_BACKWARD),
            jnp.where(sk == destroy_i, jnp.int32(EGO_FIRE), jnp.int32(EGO_FORWARD)),
        ),
    ).astype(jnp.int32)


def select_fixed_ego_action(
    config: dict[str, Any],
    skill_idx: jnp.ndarray | int,
    raw_obs,
    objects,
    target_idx,
) -> jnp.ndarray:
    """Dispatch ``SKILL_POLICY_MODE`` to a concrete ego action."""
    mode = skill_policy_mode_from_config(config)
    skill_names = config.get("SKILL_NAMES")
    num_skills = int(config.get("NUM_SKILLS", 4))
    if mode == "constant_ego":
        return constant_ego_action(
            skill_idx, skill_names=skill_names, num_skills=num_skills
        )
    # Pure-Python radians: ``float(jnp.deg2rad(...))`` concretizes under JIT.
    align_rad = math.radians(float(config.get("FIXED_EGO_ALIGN_DEG", 22.5)))
    phi, dist = target_phi_and_distance(raw_obs, objects, target_idx)
    return fixed_ego_action_from_phi(
        skill_idx,
        phi,
        dist,
        skill_names=skill_names,
        num_skills=num_skills,
        align_rad=align_rad,
    )
