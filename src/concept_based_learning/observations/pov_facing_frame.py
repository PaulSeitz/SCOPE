"""Facing-locked FPS ego frame: view = player facing; target pose in body axes.

When ``USE_POV_FACING_FRAME`` is enabled (requires ``USE_POV_RELATIVE_MAPPER``):

* **View** — player hull facing (or env synthetic facing), not the target bearing.
* **Mapper** — unchanged POV body tables keyed by sprite/synthetic orientation
  (``USE_POV_FIXED_BILATERAL`` must stay off so the mapper does not flip on target).
* **Pose** — either polar ``(distance, φ)`` with ``φ = atan2(lat, fwd)`` over the
  full circle (target may be behind), or Cartesian ``(fwd, lat)`` when
  ``EGO_POSE_XY`` is on.
* **No sector switch** — flipping L/R only happens when the *player* turns.

Incompatible with fixed bilateral.
"""

from __future__ import annotations

import jax.numpy as jnp

_USE_POV_FACING_FRAME: bool = False
_EGO_POSE_XY: bool = False
_FACING_ENV: str = ""


def using_pov_facing_frame() -> bool:
    return _USE_POV_FACING_FRAME


def using_ego_pose_xy() -> bool:
    return _EGO_POSE_XY


def facing_frame_env_name() -> str:
    return _FACING_ENV


def configure_pov_facing_frame_from_config(config: dict | None) -> None:
    global _USE_POV_FACING_FRAME, _EGO_POSE_XY, _FACING_ENV
    cfg = config or {}
    enabled = bool(cfg.get("USE_POV_FACING_FRAME", False))
    if enabled and not bool(cfg.get("USE_POV_RELATIVE_MAPPER", False)):
        raise ValueError(
            "USE_POV_FACING_FRAME requires USE_POV_RELATIVE_MAPPER: the mapper "
            "must stay body-relative to player facing (not target-selected)."
        )
    if enabled and bool(cfg.get("USE_POV_FIXED_BILATERAL", False)):
        raise ValueError(
            "USE_POV_FACING_FRAME is incompatible with USE_POV_FIXED_BILATERAL: "
            "fixed bilateral selects sectors from target geometry."
        )
    _USE_POV_FACING_FRAME = enabled
    _EGO_POSE_XY = bool(cfg.get("EGO_POSE_XY", False)) if enabled else False
    _FACING_ENV = str(cfg.get("ENV_NAME", "")).lower() if enabled else ""


def resolve_facing_deg(raw_observation, *, env_name: str | None = None) -> jnp.ndarray:
    """Latest-frame facing deg for obs: the env's declared view axis, as the mapper uses."""
    from concept_based_learning.observations.action_space import _default_get_player
    from concept_based_learning.observations.pov_relative import (
        _POV_PROFILES,
        _player_orientation_deg,
    )

    env = (env_name if env_name is not None else _FACING_ENV).lower()
    profile = _POV_PROFILES.get(env, {})
    getter = profile.get("player_getter") or _default_get_player
    return _player_orientation_deg(raw_observation, profile, getter)


def world_delta_to_facing_frame(
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    facing_deg: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate screen ``(dx, dy)`` into body axes: forward along facing, lateral right."""
    rad = jnp.asarray(facing_deg, dtype=jnp.float32) * (jnp.pi / jnp.float32(180.0))
    c = jnp.cos(rad)
    s = jnp.sin(rad)
    fwd = dx * c + dy * s
    lat = -dx * s + dy * c
    return fwd, lat


def facing_frame_phi_rad(fwd: jnp.ndarray, lat: jnp.ndarray) -> jnp.ndarray:
    """Signed bearing of the target in the facing frame (full circle, ±π)."""
    return jnp.arctan2(lat, fwd)
