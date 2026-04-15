import jax
import jax.numpy as jnp

# Pixels per stack step — used to normalize finite-difference relative velocity.
# JAXAtari shifts the stack by one env step per slot; scale with gameplay if needed.
MAX_SPEED = 32.0
SCREEN_WIDTH = 160.0
SCREEN_HEIGHT = 210.0

# Per-frame **observation** channels fed into the skill MLP (after `jnp.stack` on the time axis).
# This is unrelated to ego **action** space size (6 = bootcamp subset vs 18 = full Atari table);
# see `EGO_ACTION_DIM` / `ego_action_space.py`.
EGO_OBS_CHANNELS_PER_FRAME = 10  # dx_ego, dy_ego, sin(ori), cos(ori) — see `stacked_ego` below.


def _velocity_world_to_ego(vx: jnp.ndarray, vy: jnp.ndarray, p_ori_deg: jnp.ndarray):
    """Rotate world (screen) velocity into egocentric axes; same convention as position."""
    speed = jnp.sqrt(vx * vx + vy * vy)
    cartesian_angle = jnp.arctan2(vy, vx)
    jaxatari_absolute_angle = cartesian_angle + (jnp.pi / 2.0)
    p_ori_rad = p_ori_deg * (jnp.pi / 180.0)
    relative_angle = jaxatari_absolute_angle - p_ori_rad
    relative_angle = jnp.arctan2(jnp.sin(relative_angle), jnp.cos(relative_angle))
    dvx_ego = speed * jnp.cos(relative_angle)
    dvy_ego = speed * jnp.sin(relative_angle)
    return dvx_ego, dvy_ego


def _single_frame_geom(
    p_x, p_y, p_w, p_h, p_ori, t_x, t_y, t_w, t_h
):
    """Centers, egocentric relative position, bbox sizes, and orientation encoding."""
    p_cx = p_x + (p_w / 2.0)
    p_cy = p_y + (p_h / 2.0)
    t_cx = t_x + (t_w / 2.0)
    t_cy = t_y + (t_h / 2.0)

    dx = t_cx - p_cx
    dy = t_cy - p_cy
    distance = jnp.sqrt(dx**2 + dy**2)

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


def get_ego_obs(raw_observation, objects, target_idx):
    """
    Egocentric **observation** features for every frame in the JAXAtari stack; flattened
    for the skill MLP. Total length = ``EGO_OBS_CHANNELS_PER_FRAME * stack_length``.

    Per frame (currently ``EGO_OBS_CHANNELS_PER_FRAME`` floats):
        dx_ego, dy_ego, sin(ori), cos(ori) (extra channels from velocity/bbox are commented
        out in ``stacked_ego``).

    Not to be confused with ego **actions** (``EGO_ACTION_DIM``: 6 vs 18 discrete actions).
    """
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
    ) = jax.vmap(_single_frame_geom)(
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
    dvx_world = jnp.concatenate(
        [jnp.zeros(1, dtype=rel_x.dtype), jnp.diff(rel_x)]
    )
    dvy_world = jnp.concatenate(
        [jnp.zeros(1, dtype=rel_y.dtype), jnp.diff(rel_y)]
    )

    dvx_ego, dvy_ego = jax.vmap(_velocity_world_to_ego)(
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
