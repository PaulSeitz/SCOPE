"""Egocentric observation features with runtime-selectable format."""

import jax
import jax.numpy as jnp

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


def set_ego_obs_format(use_legacy: bool) -> None:
    """Select legacy (10ch) or target-aligned (7ch) ego observation format."""
    global _USE_LEGACY_EGO_OBS
    _USE_LEGACY_EGO_OBS = bool(use_legacy)


def using_legacy_ego_obs() -> bool:
    """Return True when the legacy ego observation format is active."""
    return _USE_LEGACY_EGO_OBS


def ego_obs_channels_per_frame() -> int:
    """Return active per-frame ego feature channel count."""
    if _USE_LEGACY_EGO_OBS:
        return LEGACY_EGO_OBS_CHANNELS_PER_FRAME
    return TARGET_ALIGNED_EGO_OBS_CHANNELS_PER_FRAME


# ---------------------------------------------------------------------------
# Per-frame geometry (vmapped over the frame stack)
# ---------------------------------------------------------------------------

def _frame_geometry_target_aligned(p_x, p_y, p_w, p_h, p_ori_deg, t_x, t_y, t_w, t_h):
    """
    Compute target-aligned geometry for a single frame in the observation stack.

    All inputs are scalars (a single frame slice); this function is vmapped over
    the stack axis by get_ego_obs.

    Returns
    -------
    distance      : Euclidean distance player centre → target centre
    theta_target  : angle of player→target vector, screen-Y-down radians
    psi           : player facing minus theta_target, wrapped to [-π, π]
    t_cx, t_cy    : target centre coords (used for cross-frame velocity)
    p_cx, p_cy    : player centre coords (used for cross-frame velocity)
    t_w, t_h      : target bounding-box sizes (passed through)
    """
    p_cx = p_x + p_w / 2.0
    p_cy = p_y + p_h / 2.0
    t_cx = t_x + t_w / 2.0
    t_cy = t_y + t_h / 2.0

    dx = t_cx - p_cx
    dy = t_cy - p_cy
    # Small epsilon prevents zero-gradient at exactly zero distance
    distance = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-6)

    # θ_target: the angle from the player to the target in screen-Y-down coordinates.
    # (atan2 with Y-down: positive angles are clockwise, e.g. target below → +90°)
    theta_target = jnp.arctan2(dy, dx)

    # ψ: player physical facing relative to θ_target.
    # Wrap to [-π, π] so sin/cos encoding is smooth and unambiguous.
    p_ori_rad = p_ori_deg * (jnp.pi / 180.0)
    psi = p_ori_rad - theta_target
    psi = jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    return distance, theta_target, psi, t_cx, t_cy, p_cx, p_cy, t_w, t_h


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


def _single_frame_geom_legacy(p_x, p_y, p_w, p_h, p_ori, t_x, t_y, t_w, t_h):
    """Legacy per-frame geometry used by pre-refactor checkpoints."""
    p_cx = p_x + (p_w / 2.0)
    p_cy = p_y + (p_h / 2.0)
    t_cx = t_x + (t_w / 2.0)
    t_cy = t_y + (t_h / 2.0)

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

    (
        distances,
        thetas,
        psis,
        t_cxs, t_cys,
        p_cxs, p_cys,
        t_ws,  t_hs,
    ) = jax.vmap(_frame_geometry_target_aligned)(
        raw_observation.player.x,
        raw_observation.player.y,
        raw_observation.player.width,
        raw_observation.player.height,
        raw_observation.player.orientation,
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
    stacked = jnp.stack(
        [
            distances / MAX_DIST,
            v_r       / MAX_SPEED,
            v_t       / MAX_SPEED,
            t_ws      / SCREEN_WIDTH,
            t_hs      / SCREEN_HEIGHT,
            jnp.sin(psis),
            jnp.cos(psis),
        ],
        axis=-1,
    ).astype(jnp.float32)  # shape: [stack_size, 7]

    return stacked.flatten()


def _get_ego_obs_legacy(raw_observation, objects, target_idx):
    """Legacy 10-channel egocentric observations from pre-refactor training."""
    t_x_history = objects.x[:, target_idx]
    t_y_history = objects.y[:, target_idx]
    t_w_history = objects.width[:, target_idx]
    t_h_history = objects.height[:, target_idx]

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
    ) = jax.vmap(_single_frame_geom_legacy)(
        raw_observation.player.x,
        raw_observation.player.y,
        raw_observation.player.width,
        raw_observation.player.height,
        raw_observation.player.orientation,
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

    stacked_ego = jnp.stack(
        [
            dx_ego / SCREEN_WIDTH,
            dy_ego / SCREEN_HEIGHT,
            dvx_ego / MAX_SPEED,
            dvy_ego / MAX_SPEED,
            t_w / SCREEN_WIDTH,
            t_h / SCREEN_HEIGHT,
            p_w / SCREEN_WIDTH,
            p_h / SCREEN_HEIGHT,
            sin_o,
            cos_o,
        ],
        axis=-1,
    ).astype(jnp.float32)

    return stacked_ego.flatten()


def get_ego_obs(raw_observation, objects, target_idx):
    """
    Build egocentric observations using the currently selected format.

    Toggle via `set_ego_obs_format(use_legacy=True/False)`.
    """
    if _USE_LEGACY_EGO_OBS:
        return _get_ego_obs_legacy(raw_observation, objects, target_idx)
    return _get_ego_obs_target_aligned(raw_observation, objects, target_idx)
