"""Egocentric observation features with runtime-selectable format."""

import jax
import jax.numpy as jnp

from concept_based_learning.observations.line_of_sight import line_of_sight_blocked
from concept_based_learning.observations.skill_ego_features import (
    DISCRETE_SKILL_EGO_FLAGS,
    build_skill_ego_extras_stacked,
    configure_ego_weapon_features,
    skill_ego_extra_channels_per_frame,
    using_ego_weapon_features,
)

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
# Screen diagonal — used to normalise distance so it stays in [0, 1]
MAX_DIST      = 263.0   # sqrt(160² + 210²)
MAX_SPEED     = 32.0    # pixels/frame, for velocity normalisation
SCREEN_WIDTH  = 160.0
SCREEN_HEIGHT = 210.0

TARGET_ALIGNED_EGO_OBS_CHANNELS_PER_FRAME = 7
LEGACY_EGO_OBS_CHANNELS_PER_FRAME = 10

# Backward-compatible constant used by old imports.
EGO_OBS_CHANNELS_PER_FRAME = TARGET_ALIGNED_EGO_OBS_CHANNELS_PER_FRAME

_USE_LEGACY_EGO_OBS = False
_USE_EGO_SIZE_FEATURES = True
_USE_EGO_VELOCITY_FEATURES = True
# Line-of-sight / blocking channel: 1.0 when an occluding object (BLOCKING_LABELS)
# lies on the player→target segment. Shared by all skills. Off by default.
_USE_EGO_BLOCKING_FEATURES = False
# None = normalize by MAX_DIST (full screen diagonal). Otherwise clip Euclidean
# distance in pixels to this cap, then divide by the cap so feature ∈ [0, 1].
_EGO_DISTANCE_CLIP_MAX_PX: float | None = None


def set_ego_obs_format(use_legacy: bool) -> None:
    """Select legacy (10ch) or target-aligned (7ch) ego observation format."""
    global _USE_LEGACY_EGO_OBS
    _USE_LEGACY_EGO_OBS = bool(use_legacy)


def set_ego_size_features(enabled: bool) -> None:
    """Enable/disable size channels while keeping ego feature dimensions unchanged."""
    global _USE_EGO_SIZE_FEATURES
    _USE_EGO_SIZE_FEATURES = bool(enabled)


def set_ego_velocity_features(enabled: bool) -> None:
    """Enable/disable velocity channels while keeping ego feature dimensions unchanged."""
    global _USE_EGO_VELOCITY_FEATURES
    _USE_EGO_VELOCITY_FEATURES = bool(enabled)


def using_ego_size_features() -> bool:
    """Return True when target/player size channels are included in ego features."""
    return _USE_EGO_SIZE_FEATURES


def using_ego_velocity_features() -> bool:
    """Return True when radial/tangential velocity channels are included."""
    return _USE_EGO_VELOCITY_FEATURES


def set_ego_blocking_features(enabled: bool) -> None:
    """Enable/disable the shared line-of-sight blocked channel (adds 1 channel/frame)."""
    global _USE_EGO_BLOCKING_FEATURES
    _USE_EGO_BLOCKING_FEATURES = bool(enabled)


def set_ego_los_features(enabled: bool) -> None:
    """Backward-compatible alias for ``set_ego_blocking_features``."""
    set_ego_blocking_features(enabled)


def using_ego_blocking_features() -> bool:
    """Return True when the shared line-of-sight blocked channel is included."""
    return _USE_EGO_BLOCKING_FEATURES


def using_ego_los_features() -> bool:
    """Backward-compatible alias for ``using_ego_blocking_features``."""
    return using_ego_blocking_features()


def using_legacy_ego_obs() -> bool:
    """Return True when the legacy ego observation format is active."""
    return _USE_LEGACY_EGO_OBS


def set_ego_distance_clip_max_pixels(max_px: float | None) -> None:
    """
    Clip target-aligned Euclidean distance before normalization.

    When set (e.g. 15), distances >= max_px are treated as max_px so the network
    sees a single \"far\" bucket instead of sparse large values. Normalization uses
    max_px, not MAX_DIST. Pass None to disable clipping.
    """
    global _EGO_DISTANCE_CLIP_MAX_PX
    if max_px is None:
        _EGO_DISTANCE_CLIP_MAX_PX = None
        return
    cap = float(max_px)
    if cap <= 0:
        raise ValueError(f"EGO_DISTANCE_CLIP_MAX_PX must be positive, got {max_px}")
    _EGO_DISTANCE_CLIP_MAX_PX = cap


def ego_distance_clip_max_pixels() -> float | None:
    """Return the active distance clip cap in pixels, or None if disabled."""
    return _EGO_DISTANCE_CLIP_MAX_PX


def configure_ego_obs_from_config(config: dict) -> None:
    """Apply all ego observation toggles from a merged experiment config dict."""
    set_ego_obs_format(bool(config.get("USE_LEGACY_EGO_OBS", False)))
    set_ego_size_features(bool(config.get("USE_EGO_SIZE_FEATURES", True)))
    set_ego_velocity_features(bool(config.get("USE_EGO_VELOCITY_FEATURES", True)))
    # Shared blocking/LOS flag (new name preferred; old USE_EGO_LOS_FEATURES still works).
    blocking = config.get(
        "USE_EGO_BLOCKING_FEATURES",
        config.get("USE_EGO_LOS_FEATURES", False),
    )
    set_ego_blocking_features(bool(blocking))
    clip_px = config.get("EGO_DISTANCE_CLIP_MAX_PX", None)
    set_ego_distance_clip_max_pixels(None if clip_px is None else float(clip_px))
    # Weapon/shooting channel (new name preferred; old USE_SKILL_EGO_FEATURES still works).
    # Channel count is fixed (1) when enabled — no separate EXTRA_DIM config.
    weapon = config.get(
        "USE_EGO_WEAPON_FEATURES",
        config.get("USE_SKILL_EGO_FEATURES", False),
    )
    configure_ego_weapon_features(
        enabled=bool(weapon),
        env_name=str(config.get("ENV_NAME", "")),
    )


def distance_feature_from_pixels(distances_px: jnp.ndarray) -> jnp.ndarray:
    """Map raw pixel distances to the normalized distance channel fed to skills."""
    if _EGO_DISTANCE_CLIP_MAX_PX is not None:
        cap = jnp.float32(_EGO_DISTANCE_CLIP_MAX_PX)
        return jnp.minimum(distances_px, cap) / cap
    return distances_px / MAX_DIST


def distance_pixels_from_feature(feature: jnp.ndarray | float) -> float:
    """Invert the distance channel for logging / overlays (clipped view)."""
    if _EGO_DISTANCE_CLIP_MAX_PX is not None:
        return float(feature) * _EGO_DISTANCE_CLIP_MAX_PX
    return float(feature) * MAX_DIST


def ego_obs_channels_per_frame() -> int:
    """Return active per-frame ego feature channel count."""
    los_channels = 1 if _USE_EGO_BLOCKING_FEATURES else 0
    if _USE_LEGACY_EGO_OBS:
        # Legacy channel groups:
        #   always: dx, dy, sin(o), cos(o)             -> 4
        #   velocity (optional): dvx, dvy              -> +2
        #   size (optional): t_w, t_h, p_w, p_h        -> +4
        #   line-of-sight (optional): blocked          -> +1
        return 4 + (2 if _USE_EGO_VELOCITY_FEATURES else 0) + (
            4 if _USE_EGO_SIZE_FEATURES else 0
        ) + los_channels + skill_ego_extra_channels_per_frame()
    # Target-aligned channel groups:
    #   always: distance, sin(psi), cos(psi)          -> 3
    #   velocity (optional): radial, tangential       -> +2
    #   size (optional): target_w, target_h           -> +2
    #   line-of-sight (optional): blocked             -> +1
    return 3 + (2 if _USE_EGO_VELOCITY_FEATURES else 0) + (
        2 if _USE_EGO_SIZE_FEATURES else 0
    ) + los_channels + skill_ego_extra_channels_per_frame()


def ego_base_channels_per_frame() -> int:
    """Per-frame channel count excluding skill extras (includes LOS when enabled)."""
    return ego_obs_channels_per_frame() - skill_ego_extra_channels_per_frame()


def discrete_ego_flag_layout() -> list[tuple[str, int, int, float]]:
    """
    Active discrete flag channels in the latest ego frame.

    Returns ``(name, channel_index_within_frame, n_levels, level_offset)`` in
    digit order used by occupancy histograms (digit 0 = first entry). Shared
    flags (LOS) come first; per-skill extras follow in registry order.
    Binary flags use ``n_levels=2`` (los, can_fire, facing_enemy).
    """
    layout: list[tuple[str, int, int, float]] = []
    extras_width = skill_ego_extra_channels_per_frame()
    base = ego_base_channels_per_frame()
    if _USE_EGO_BLOCKING_FEATURES:
        # LOS is the last base channel (immediately before skill extras).
        layout.append(("los_blocked", base - 1, 2, 0.0))
    if using_ego_weapon_features() and extras_width > 0:
        for flag in DISCRETE_SKILL_EGO_FLAGS:
            idx = base + int(flag.offset)
            if 0 <= flag.offset < extras_width:
                layout.append(
                    (flag.name, idx, int(flag.n_levels), float(flag.level_offset))
                )
    return layout


# ---------------------------------------------------------------------------
# Player entity accessor (supports different field names across games)
# ---------------------------------------------------------------------------

def _get_player(raw_observation):
    """Return the player entity from a raw observation.

    Most games expose ``raw_observation.player``; Freeway uses
    ``raw_observation.chicken``.  Additional aliases can be added here
    without touching the rest of the module.
    """
    for name in ("player", "chicken", "agent", "skier"):
        if hasattr(raw_observation, name):
            return getattr(raw_observation, name)
    raise AttributeError(
        f"Cannot find a player entity in {type(raw_observation).__name__}. "
        "Expected one of: player, chicken, agent, skier."
    )


def _uses_bottom_edge_y(raw_observation) -> bool:
    """
    Return True when this environment stores bbox y as a bottom edge.

    Freeway (``raw_observation.chicken``) is the main case today. Most other games
    expose ``raw_observation.player`` with top-left y semantics.
    """
    return hasattr(raw_observation, "chicken")


def _center_y(y: jnp.ndarray, height: jnp.ndarray, *, y_is_bottom_edge: bool) -> jnp.ndarray:
    """
    Compute bbox centre y under an explicit coordinate convention.

    - top-left convention:    center_y = y + h / 2
    - bottom-edge convention: center_y = y - h / 2
    """
    if y_is_bottom_edge:
        return y - height / 2.0
    return y + height / 2.0


# ---------------------------------------------------------------------------
# Per-frame geometry (vmapped over the frame stack)
# ---------------------------------------------------------------------------

def _box_surface_extent(half_w: jnp.ndarray, half_h: jnp.ndarray, cos_t: jnp.ndarray, sin_t: jnp.ndarray) -> jnp.ndarray:
    """
    Distance from box centre to the surface point hit by ray (cos_t, sin_t).

    Clips the unit ray to the axis-aligned box [−hw, hw] × [−hh, hh], giving the
    exact nearest surface point — a corner when both axes are simultaneously hit,
    an edge midpoint when only one axis is hit.  Fully differentiable; no branches.
    """
    # How far along the ray until each axis boundary is hit.
    # jnp.where guards against division by zero on axis-aligned rays.
    t_x = jnp.where(jnp.abs(cos_t) > 1e-8, half_w / jnp.abs(cos_t), jnp.inf)
    t_y = jnp.where(jnp.abs(sin_t) > 1e-8, half_h / jnp.abs(sin_t), jnp.inf)
    t_exit = jnp.minimum(t_x, t_y)
    # Euclidean length of the clipped ray segment
    return jnp.sqrt((t_exit * cos_t) ** 2 + (t_exit * sin_t) ** 2)


def _frame_geometry_target_aligned(
    p_x,
    p_y,
    p_w,
    p_h,
    p_ori_deg,
    t_x,
    t_y,
    t_w,
    t_h,
    *,
    y_is_bottom_edge: bool,
):
    """
    Compute target-aligned geometry for a single frame in the observation stack.

    All inputs are scalars (a single frame slice); this function is vmapped over
    the stack axis by get_ego_obs.

    Returns
    -------
    distance      : nearest-surface gap between player and target bounding boxes
    theta_target  : angle of player→target vector (centre–centre), screen-Y-down radians
    psi           : player facing minus theta_target, wrapped to [-π, π]
    t_cx, t_cy    : target centre coords (used for cross-frame velocity)
    p_cx, p_cy    : player centre coords (used for cross-frame velocity)
    t_w, t_h      : target bounding-box sizes (passed through)
    """
    # IMPORTANT: centre computation must match the environment's y convention.
    # If this drifts from action mapping/reward geometry, transfer can silently fail.
    p_cx = p_x + p_w / 2.0
    p_cy = _center_y(p_y, p_h, y_is_bottom_edge=y_is_bottom_edge)
    t_cx = t_x + t_w / 2.0
    t_cy = _center_y(t_y, t_h, y_is_bottom_edge=y_is_bottom_edge)

    dx = t_cx - p_cx
    dy = t_cy - p_cy
    center_dist = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-6)

    # θ_target: angle of the centre→centre vector in screen-Y-down coordinates.
    # Kept centre-based so "which direction is the target?" is stable and continuous.
    theta_target = jnp.arctan2(dy, dx)
    cos_t = jnp.cos(theta_target)
    sin_t = jnp.sin(theta_target)

    # Subtract the surface extent of each box along the line of sight so that
    # distance reflects the actual gap between the nearest surfaces / corners.
    p_extent = _box_surface_extent(p_w / 2.0, p_h / 2.0, cos_t, sin_t)
    t_extent = _box_surface_extent(t_w / 2.0, t_h / 2.0, cos_t, sin_t)
    # Clamp to zero: overlapping / touching boxes should report distance = 0.
    distance = jnp.maximum(center_dist - p_extent - t_extent, 0.0)

    # ψ: player physical facing relative to θ_target.
    # Wrap to [-π, π] so sin/cos encoding is smooth and unambiguous.
    p_ori_rad = p_ori_deg * (jnp.pi / 180.0)
    psi = p_ori_rad - theta_target
    psi = jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    return distance, theta_target, psi, t_cx, t_cy, p_cx, p_cy, t_w, t_h


def _los_blocked_stacked(
    objects,
    target_idx,
    p_cxs,
    p_cys,
    t_cxs,
    t_cys,
    *,
    y_is_bottom_edge: bool,
) -> jnp.ndarray:
    """Per-frame line-of-sight blocked flags (float32 [stack]) for the target."""
    return jax.vmap(
        lambda pcx, pcy, tcx, tcy, ox, oy, ow, oh, oact, olab: line_of_sight_blocked(
            pcx,
            pcy,
            tcx,
            tcy,
            target_idx,
            ox,
            oy,
            ow,
            oh,
            oact,
            olab,
            y_is_bottom_edge=y_is_bottom_edge,
        )
    )(
        p_cxs,
        p_cys,
        t_cxs,
        t_cys,
        objects.x,
        objects.y,
        objects.width,
        objects.height,
        objects.active,
        objects.semantic_label,
    ).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Main observation function
# ---------------------------------------------------------------------------

def _velocity_world_to_ego_legacy(vx: jnp.ndarray, vy: jnp.ndarray, p_ori_deg: jnp.ndarray):
    """Rotate world velocity into ego axes using the legacy convention."""
    speed = jnp.sqrt(vx * vx + vy * vy)
    cartesian_angle = jnp.arctan2(vy, vx)
    jaxatari_absolute_angle = cartesian_angle + (jnp.pi / 2.0)
    p_ori_rad = p_ori_deg * (jnp.pi / 180.0)
    relative_angle = jaxatari_absolute_angle - p_ori_rad
    relative_angle = jnp.arctan2(jnp.sin(relative_angle), jnp.cos(relative_angle))
    dvx_ego = speed * jnp.cos(relative_angle)
    dvy_ego = speed * jnp.sin(relative_angle)
    return dvx_ego, dvy_ego


def _single_frame_geom_legacy(
    p_x,
    p_y,
    p_w,
    p_h,
    p_ori,
    t_x,
    t_y,
    t_w,
    t_h,
    *,
    y_is_bottom_edge: bool,
):
    """Legacy per-frame geometry used by pre-refactor checkpoints."""
    p_cx = p_x + (p_w / 2.0)
    p_cy = _center_y(p_y, p_h, y_is_bottom_edge=y_is_bottom_edge)
    t_cx = t_x + (t_w / 2.0)
    t_cy = _center_y(t_y, t_h, y_is_bottom_edge=y_is_bottom_edge)

    dx = t_cx - p_cx
    dy = t_cy - p_cy
    distance = jnp.sqrt(dx ** 2 + dy ** 2)

    cartesian_angle = jnp.arctan2(dy, dx)
    jaxatari_absolute_angle = cartesian_angle + (jnp.pi / 2.0)
    p_ori_rad = p_ori * (jnp.pi / 180.0)

    relative_angle = jaxatari_absolute_angle - p_ori_rad
    relative_angle = jnp.arctan2(jnp.sin(relative_angle), jnp.cos(relative_angle))

    dx_ego = distance * jnp.cos(relative_angle)
    dy_ego = distance * jnp.sin(relative_angle)

    return (
        dx_ego,
        dy_ego,
        p_ori,
        t_w,
        t_h,
        p_w,
        p_h,
        p_cx,
        p_cy,
        t_cx,
        t_cy,
    )


def _get_ego_obs_target_aligned(raw_observation, objects, target_idx):
    """
    Build the target-aligned egocentric observation over the full JAXAtari frame stack.

    Parameters
    ----------
    raw_observation : JAXAtari raw obs PyTree; must expose raw_observation.player.{x,y,width,height,orientation}
    objects         : LabeledObjectObservation PyTree; fields x,y,width,height shaped [stack, max_objects]
    target_idx      : int32 scalar — index into the objects arrays

    Returns
    -------
    flat float32 array of shape (EGO_OBS_CHANNELS_PER_FRAME * stack_size,)
    """
    t_x_hist = objects.x[:, target_idx]
    t_y_hist = objects.y[:, target_idx]
    t_w_hist = objects.width[:, target_idx]
    t_h_hist = objects.height[:, target_idx]

    player = _get_player(raw_observation)
    y_is_bottom_edge = _uses_bottom_edge_y(raw_observation)
    (
        distances,     # nearest surface gap (edge/corner to edge/corner)
        thetas,        # centre→centre angle
        psis,
        t_cxs, t_cys,
        p_cxs, p_cys,
        t_ws,  t_hs,
    ) = jax.vmap(
        lambda px, py, pw, ph, po, tx, ty, tw, th: _frame_geometry_target_aligned(
            px,
            py,
            pw,
            ph,
            po,
            tx,
            ty,
            tw,
            th,
            y_is_bottom_edge=y_is_bottom_edge,
        )
    )(
        player.x,
        player.y,
        player.width,
        player.height,
        player.orientation,
        t_x_hist, t_y_hist, t_w_hist, t_h_hist,
    )

    # -------------------------------------------------------------------
    # Radial and tangential velocities
    # -------------------------------------------------------------------
    # World-space relative position across the stack
    rel_x = t_cxs - p_cxs
    rel_y = t_cys - p_cys

    # Finite-difference velocity; zero-pad the oldest frame (no prior data)
    d_rel_x = jnp.concatenate([jnp.zeros(1, dtype=rel_x.dtype), jnp.diff(rel_x)])
    d_rel_y = jnp.concatenate([jnp.zeros(1, dtype=rel_y.dtype), jnp.diff(rel_y)])

    # Decompose into radial (along θ_target) and tangential (perpendicular) components.
    #
    #   radial unit vector:      ( cos θ,  sin θ)
    #   tangential unit vector:  (-sin θ,  cos θ)   (90° CCW from radial)
    #
    # v_r > 0: relative position growing → target is moving away (or player lagging)
    # v_t > 0: target drifting to the right of the line-of-sight
    cos_t = jnp.cos(thetas)
    sin_t = jnp.sin(thetas)
    v_r =  d_rel_x * cos_t + d_rel_y * sin_t
    v_t = -d_rel_x * sin_t + d_rel_y * cos_t

    # -------------------------------------------------------------------
    # Stack and normalise
    # -------------------------------------------------------------------
    if _USE_EGO_SIZE_FEATURES:
        size_w = t_ws / SCREEN_WIDTH
        size_h = t_hs / SCREEN_HEIGHT
    else:
        size_w = jnp.zeros_like(t_ws, dtype=jnp.float32)
        size_h = jnp.zeros_like(t_hs, dtype=jnp.float32)

    if _USE_EGO_VELOCITY_FEATURES:
        v_r_norm = v_r / MAX_SPEED
        v_t_norm = v_t / MAX_SPEED
    else:
        v_r_norm = jnp.zeros_like(v_r, dtype=jnp.float32)
        v_t_norm = jnp.zeros_like(v_t, dtype=jnp.float32)

    channels = [distance_feature_from_pixels(distances)]
    if _USE_EGO_VELOCITY_FEATURES:
        channels.extend([v_r_norm, v_t_norm])
    if _USE_EGO_SIZE_FEATURES:
        channels.extend([size_w, size_h])
    channels.extend([jnp.sin(psis), jnp.cos(psis)])
    if _USE_EGO_BLOCKING_FEATURES:
        channels.append(
            _los_blocked_stacked(
                objects,
                target_idx,
                p_cxs,
                p_cys,
                t_cxs,
                t_cys,
                y_is_bottom_edge=y_is_bottom_edge,
            )
        )
    skill_extras = build_skill_ego_extras_stacked(
        raw_observation, objects=objects, target_idx=target_idx
    )
    if skill_extras.shape[-1] > 0:
        stacked = jnp.concatenate([jnp.stack(channels, axis=-1), skill_extras], axis=-1)
    else:
        stacked = jnp.stack(channels, axis=-1)
    return stacked.astype(jnp.float32).flatten()


def _get_ego_obs_legacy(raw_observation, objects, target_idx):
    """Legacy 10-channel egocentric observations from pre-refactor training."""
    t_x_history = objects.x[:, target_idx]
    t_y_history = objects.y[:, target_idx]
    t_w_history = objects.width[:, target_idx]
    t_h_history = objects.height[:, target_idx]

    player = _get_player(raw_observation)
    y_is_bottom_edge = _uses_bottom_edge_y(raw_observation)
    (
        dx_ego,
        dy_ego,
        p_ori_deg,
        t_w,
        t_h,
        p_w,
        p_h,
        p_cx,
        p_cy,
        t_cx,
        t_cy,
    ) = jax.vmap(
        lambda px, py, pw, ph, po, tx, ty, tw, th: _single_frame_geom_legacy(
            px,
            py,
            pw,
            ph,
            po,
            tx,
            ty,
            tw,
            th,
            y_is_bottom_edge=y_is_bottom_edge,
        )
    )(
        player.x,
        player.y,
        player.width,
        player.height,
        player.orientation,
        t_x_history,
        t_y_history,
        t_w_history,
        t_h_history,
    )

    rel_x = t_cx - p_cx
    rel_y = t_cy - p_cy
    dvx_world = jnp.concatenate([jnp.zeros(1, dtype=rel_x.dtype), jnp.diff(rel_x)])
    dvy_world = jnp.concatenate([jnp.zeros(1, dtype=rel_y.dtype), jnp.diff(rel_y)])

    dvx_ego, dvy_ego = jax.vmap(_velocity_world_to_ego_legacy)(
        dvx_world, dvy_world, p_ori_deg
    )

    p_ori_rad = p_ori_deg * (jnp.pi / 180.0)
    sin_o = jnp.sin(p_ori_rad)
    cos_o = jnp.cos(p_ori_rad)

    if _USE_EGO_SIZE_FEATURES:
        t_w_norm = t_w / SCREEN_WIDTH
        t_h_norm = t_h / SCREEN_HEIGHT
        p_w_norm = p_w / SCREEN_WIDTH
        p_h_norm = p_h / SCREEN_HEIGHT
    else:
        t_w_norm = jnp.zeros_like(t_w, dtype=jnp.float32)
        t_h_norm = jnp.zeros_like(t_h, dtype=jnp.float32)
        p_w_norm = jnp.zeros_like(p_w, dtype=jnp.float32)
        p_h_norm = jnp.zeros_like(p_h, dtype=jnp.float32)

    if _USE_EGO_VELOCITY_FEATURES:
        dvx_norm = dvx_ego / MAX_SPEED
        dvy_norm = dvy_ego / MAX_SPEED
    else:
        dvx_norm = jnp.zeros_like(dvx_ego, dtype=jnp.float32)
        dvy_norm = jnp.zeros_like(dvy_ego, dtype=jnp.float32)

    channels = [
        dx_ego / SCREEN_WIDTH,
        dy_ego / SCREEN_HEIGHT,
    ]
    if _USE_EGO_VELOCITY_FEATURES:
        channels.extend([dvx_norm, dvy_norm])
    if _USE_EGO_SIZE_FEATURES:
        channels.extend([t_w_norm, t_h_norm, p_w_norm, p_h_norm])
    channels.extend([sin_o, cos_o])
    if _USE_EGO_BLOCKING_FEATURES:
        channels.append(
            _los_blocked_stacked(
                objects,
                target_idx,
                p_cx,
                p_cy,
                t_cx,
                t_cy,
                y_is_bottom_edge=y_is_bottom_edge,
            )
        )
    skill_extras = build_skill_ego_extras_stacked(
        raw_observation, objects=objects, target_idx=target_idx
    )
    if skill_extras.shape[-1] > 0:
        stacked_ego = jnp.concatenate([jnp.stack(channels, axis=-1), skill_extras], axis=-1)
    else:
        stacked_ego = jnp.stack(channels, axis=-1)
    return stacked_ego.astype(jnp.float32).flatten()


def get_ego_obs(raw_observation, objects, target_idx):
    """
    Build egocentric observations using the currently selected format.

    Toggle via `set_ego_obs_format(use_legacy=True/False)`.
    """
    if _USE_LEGACY_EGO_OBS:
        return _get_ego_obs_legacy(raw_observation, objects, target_idx)
    return _get_ego_obs_target_aligned(raw_observation, objects, target_idx)
