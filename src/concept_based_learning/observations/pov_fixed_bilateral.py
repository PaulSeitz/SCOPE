"""Fixed bilateral POV: virtual left/right (or up/down) sectors selected by target geometry.

When ``USE_POV_FIXED_BILATERAL`` is enabled (requires ``USE_POV_RELATIVE_MAPPER``):

* **Two sectors** — horizontal games: forward = ±x; vertical games: forward = ±y.
* **Sector selection** — pick the sector whose forward half-plane contains the target
  (sprite orientation is ignored).
* **Obs** — AABB gap distance + angle φ in the active sector's forward hemisphere
  (φ ∈ (−π/2, π/2)); optionally 8 triangular φ bins or sin/cos(φ).
* **Mapper** — same body-relative ego tables as POV, indexed by the selected sector.

Vertical-axis randomisation is intentionally not supported: fire direction is tied to
the virtual forward axis and would require the policy to infer the active frame.
"""

from __future__ import annotations

import jax.numpy as jnp

_USE_POV_FIXED_BILATERAL: bool = False
_BILATERAL_AXIS: str = "horizontal"


def using_pov_fixed_bilateral() -> bool:
    return _USE_POV_FIXED_BILATERAL


def active_bilateral_axis() -> str:
    return _BILATERAL_AXIS


def bilateral_facing_degrees(profile: dict, *, axis: str) -> tuple[float, float]:
    """Mapper table facings for virtual sector 0 / 1 under the active bilateral axis."""
    if axis == "vertical":
        return -90.0, 90.0
    return float(profile.get("facing_deg_bin0", 0.0)), float(
        profile.get("facing_deg_bin1", 180.0)
    )


def configure_pov_fixed_bilateral_from_config(config: dict | None) -> None:
    global _USE_POV_FIXED_BILATERAL, _BILATERAL_AXIS
    cfg = config or {}
    enabled = bool(cfg.get("USE_POV_FIXED_BILATERAL", False))
    if enabled and not bool(cfg.get("USE_POV_RELATIVE_MAPPER", False)):
        raise ValueError(
            "USE_POV_FIXED_BILATERAL requires USE_POV_RELATIVE_MAPPER: the mapper "
            "still uses body-relative facing tables keyed by virtual sector."
        )
    if enabled and bool(cfg.get("USE_POV_TURN_PRIMITIVES", False)):
        raise ValueError(
            "USE_POV_FIXED_BILATERAL is incompatible with USE_POV_TURN_PRIMITIVES: "
            "sector selection is target-based, not heading-integrated."
        )
    if enabled and bool(cfg.get("USE_EGO_VELOCITY_FEATURES", False)):
        raise ValueError(
            "USE_POV_FIXED_BILATERAL does not define a velocity channel; set "
            "USE_EGO_VELOCITY_FEATURES: false."
        )
    _USE_POV_FIXED_BILATERAL = enabled
    if enabled:
        from concept_based_learning.observations.pov_relative import _POV_PROFILES

        env = str(cfg.get("ENV_NAME", "")).lower()
        profile = _POV_PROFILES.get(env, {})
        override = cfg.get("POV_BILATERAL_AXIS")
        if override not in (None, "", False):
            axis = str(override).lower()
            if axis not in ("horizontal", "vertical"):
                raise ValueError(
                    f"POV_BILATERAL_AXIS must be 'horizontal' or 'vertical', got {override!r}"
                )
            _BILATERAL_AXIS = axis
        else:
            _BILATERAL_AXIS = bilateral_axis_from_profile(profile)
    else:
        _BILATERAL_AXIS = "horizontal"


def bilateral_axis_from_profile(profile: dict) -> str:
    """Return ``horizontal`` or ``vertical`` for a POV env profile."""
    return str(profile.get("bilateral_axis", "horizontal"))


def target_sector_idx(
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    *,
    axis: str,
) -> jnp.ndarray:
    """Pick virtual sector so the target lies in the forward half-plane."""
    if axis == "vertical":
        # Sector 0: forward = up (−y screen); sector 1: forward = down (+y).
        return jnp.where(dy <= jnp.float32(0.0), jnp.int32(0), jnp.int32(1))
    # Sector 0: forward = +x; sector 1: forward = −x.
    return jnp.where(dx >= jnp.float32(0.0), jnp.int32(0), jnp.int32(1))


def forward_hemisphere_phi_rad(
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    sector_idx: jnp.ndarray,
    *,
    axis: str,
) -> jnp.ndarray:
    """Signed lateral angle φ in the active sector's forward wedge."""
    if axis == "vertical":
        x_fwd = jnp.where(sector_idx == 0, -dy, dy)
        y_lat = dx
    else:
        x_fwd = jnp.where(sector_idx == 0, dx, -dx)
        y_lat = dy
    return jnp.arctan2(y_lat, x_fwd)


def forward_hemisphere_angle_bins(phi: jnp.ndarray, n_bins: int) -> jnp.ndarray:
    """Triangular bins over φ ∈ [−π/2, π/2] with 50% overlap between neighbours."""
    n = int(n_bins)
    span = jnp.float32(jnp.pi)
    centers = (jnp.arange(n, dtype=jnp.float32) + jnp.float32(0.5)) * span / n - span / jnp.float32(
        2.0
    )
    delta = jnp.asarray(phi, dtype=jnp.float32)[..., None] - centers
    half = span / n
    return jnp.maximum(jnp.float32(0.0), jnp.float32(1.0) - jnp.abs(delta) / half)
