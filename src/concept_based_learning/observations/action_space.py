"""Egocentric action space — target-aligned frame.

Ego actions are defined relative to the player→target direction vector (θ_target).
The factory make_ego_action_mapper returns a JAX-compatible, jit/vmap-safe function
that translates integer ego actions into environment-specific game action integers.

Ego action layout (18-dim):
    0   NOOP
    1   FIRE                (no movement, pure fire)
    2   FORWARD             δ=   0°   directly toward target
    3   RIGHT               δ= +90°   right of player→target line
    4   LEFT                δ= -90°   left  of player→target line
    5   BACKWARD            δ= 180°
    6   FORWARD_RIGHT       δ= +45°
    7   FORWARD_LEFT        δ= -45°
    8   BACKWARD_RIGHT      δ=+135°
    9   BACKWARD_LEFT       δ=-135°
    10  FORWARD_FIRE
    11  RIGHT_FIRE
    12  LEFT_FIRE
    13  BACKWARD_FIRE
    14  FORWARD_RIGHT_FIRE
    15  FORWARD_LEFT_FIRE
    16  BACKWARD_RIGHT_FIRE
    17  BACKWARD_LEFT_FIRE

Positive δ is clockwise in screen-Y-down coordinates (right = +x, down = +y).

Direction table convention (used for both move_table and move_fire_table, length 8):
    Index  World direction    Screen-Y-down angle
    -----  ---------------   -------------------
      0    RIGHT              0°
      1    DOWNRIGHT         +45°
      2    DOWN              +90°
      3    DOWNLEFT         +135°
      4    LEFT              180°
      5    UPLEFT           -135°
      6    UP               - 90°
      7    UPRIGHT          - 45°

For environments with no FIRE, pass move_table == move_fire_table.
For 4-directional environments, fill each diagonal entry with the nearest cardinal;
prefer a consistent tie-break (e.g. always prefer horizontal over vertical).

Example — Seaquest (full 18-action Atari layout):
    noop_action     = 0   (NOOP)
    fire_action     = 1   (FIRE)
    move_table      = [3, 8, 5, 9, 4, 7, 2, 6]        # R DR D DL L UL U UR
    move_fire_table = [11, 16, 13, 17, 12, 15, 10, 14] # same with FIRE bit
"""

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Fire action indices (for masking in the training loop)
# ---------------------------------------------------------------------------
FIRE_EGO_ACTION_INDICES = jnp.array(
    [1, 10, 11, 12, 13, 14, 15, 16, 17], dtype=jnp.int32
)

# Backward-compat aliases for configs that use the legacy constant names.
FIRE_EGO_ACTION_INDICES_18 = FIRE_EGO_ACTION_INDICES
FIRE_EGO_ACTION_INDICES_6  = jnp.array([1], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Movement deltas
# ---------------------------------------------------------------------------
_NaN = float("nan")

# Delta angle (degrees, screen-Y-down) added to θ_target to get the desired world angle.
# NaN for NOOP and FIRE-only: these have no movement component and their delta is never used.
EGO_MOVE_DELTAS_DEG = jnp.array(
    [
        _NaN,    # 0  NOOP
        _NaN,    # 1  FIRE
          0.0,   # 2  FORWARD
         90.0,   # 3  RIGHT
        -90.0,   # 4  LEFT
        180.0,   # 5  BACKWARD
         45.0,   # 6  FORWARD_RIGHT
        -45.0,   # 7  FORWARD_LEFT
        135.0,   # 8  BACKWARD_RIGHT
       -135.0,   # 9  BACKWARD_LEFT
          0.0,   # 10 FORWARD_FIRE
         90.0,   # 11 RIGHT_FIRE
        -90.0,   # 12 LEFT_FIRE
        180.0,   # 13 BACKWARD_FIRE
         45.0,   # 14 FORWARD_RIGHT_FIRE
        -45.0,   # 15 FORWARD_LEFT_FIRE
        135.0,   # 16 BACKWARD_RIGHT_FIRE
       -135.0,   # 17 BACKWARD_LEFT_FIRE
    ],
    dtype=jnp.float32,
)

# World direction angles (degrees) for direction indices 0-7.
_WORLD_DIR_ANGLES_DEG = jnp.array(
    [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0],
    dtype=jnp.float32,
)


def _default_get_player(raw_obs):
    return raw_obs.player


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_world_dir(desired_world_angle_deg: jnp.ndarray) -> jnp.ndarray:
    """
    Return the direction index (0-7) whose angle is closest to desired_world_angle_deg.

    Handles 360° wraparound correctly. When desired_world_angle_deg is NaN
    (NOOP/FIRE ego actions), returns 0 — this result is always overridden by the
    downstream jnp.where chain so the value does not matter.
    """
    diff = (desired_world_angle_deg - _WORLD_DIR_ANGLES_DEG + 180.0) % 360.0 - 180.0
    return jnp.argmin(jnp.abs(diff)).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Mapper factory
# ---------------------------------------------------------------------------

def make_ego_action_mapper(
    noop_action: int,
    fire_action: int,
    move_table,
    move_fire_table,
):
    """
    Build a JAX-compatible action mapper for a specific environment.

    The returned function has the signature:
        mapper(ego_action, raw_obs, target_idx, objects) -> int32 game action

    and is safe to use inside jit and vmap.

    Parameters
    ----------
    noop_action     : game action integer for NOOP
    fire_action     : game action integer for stationary FIRE (no movement)
    move_table      : int[8] — game actions for pure movement in each world direction
                      (indexed per _WORLD_DIR_ANGLES_DEG: RIGHT=0 … UPRIGHT=7)
    move_fire_table : int[8] — same, but movement + fire simultaneously
    """
    _noop = jnp.int32(noop_action)
    _fire = jnp.int32(fire_action)
    _move = jnp.array(move_table, dtype=jnp.int32)
    _mfir = jnp.array(move_fire_table, dtype=jnp.int32)

    def mapper(ego_action, raw_obs, target_idx, objects):
        # 1. Compute θ_target from the most recent frame in the stack.
        p_cx = raw_obs.player.x[-1] + raw_obs.player.width[-1] / 2.0
        p_cy = raw_obs.player.y[-1] + raw_obs.player.height[-1] / 2.0
        t_cx = objects.x[-1, target_idx] + objects.width[-1, target_idx] / 2.0
        t_cy = objects.y[-1, target_idx] + objects.height[-1, target_idx] / 2.0

        theta_deg = jnp.arctan2(t_cy - p_cy, t_cx - p_cx) * (180.0 / jnp.pi)

        # 2. Map ego action → desired world angle → nearest Atari direction.
        desired_deg = theta_deg + EGO_MOVE_DELTAS_DEG[ego_action]
        dir_idx = _nearest_world_dir(desired_deg)

        # 3. Select game action based on whether this ego action includes FIRE.
        has_fire     = ego_action >= 10
        is_fire_only = ego_action == 1
        is_noop      = ego_action == 0

        game_action = jnp.where(has_fire, _mfir[dir_idx], _move[dir_idx])
        game_action = jnp.where(is_fire_only, _fire, game_action)
        game_action = jnp.where(is_noop, _noop, game_action)
        return game_action

    return mapper


def make_inverse_ego_action_mapper(
    noop_action: int,
    fire_action: int,
    move_table,
    move_fire_table,
    *,
    player_getter=_default_get_player,
    y_is_bottom_edge: bool = False,
):
    """
    Build a JAX-compatible inverse action mapper for a specific environment.

    The returned function has the signature:
        inverse_mapper(game_action, raw_obs, target_idx, objects) -> int32 ego action
    """
    _noop = jnp.int32(noop_action)
    _fire = jnp.int32(fire_action)
    _move = jnp.array(move_table, dtype=jnp.int32)
    _mfir = jnp.array(move_fire_table, dtype=jnp.int32)
    _move_deltas = EGO_MOVE_DELTAS_DEG[2:10]

    def _center_y(y, height):
        if y_is_bottom_edge:
            return y - height / 2.0
        return y + height / 2.0

    def inverse_mapper(game_action, raw_obs, target_idx, objects):
        is_move = _move == game_action
        is_mfir = _mfir == game_action
        matched_move = jnp.any(is_move)
        matched_fire_move = jnp.any(is_mfir)
        has_fire = matched_fire_move
        dir_idx = jnp.where(has_fire, jnp.argmax(is_mfir), jnp.argmax(is_move))
        world_angle = _WORLD_DIR_ANGLES_DEG[dir_idx]

        player = player_getter(raw_obs)
        p_cx = player.x[-1] + player.width[-1] / 2.0
        p_cy = _center_y(player.y[-1], player.height[-1])
        t_cx = objects.x[-1, target_idx] + objects.width[-1, target_idx] / 2.0
        t_cy = _center_y(objects.y[-1, target_idx], objects.height[-1, target_idx])
        theta_deg = jnp.arctan2(t_cy - p_cy, t_cx - p_cx) * (180.0 / jnp.pi)

        desired_delta = (world_angle - theta_deg + 180.0) % 360.0 - 180.0
        diff = (desired_delta - _move_deltas + 180.0) % 360.0 - 180.0
        ego_move = (2 + jnp.argmin(jnp.abs(diff))).astype(jnp.int32)
        ego_action = jnp.where(has_fire, ego_move + 8, ego_move)

        matched = jnp.logical_or(matched_move, matched_fire_move)
        ego_action = jnp.where(game_action == _fire, 1, ego_action)
        ego_action = jnp.where(
            jnp.logical_or(game_action == _noop, jnp.logical_not(matched)),
            0,
            ego_action,
        )
        return ego_action.astype(jnp.int32)

    return inverse_mapper


# ---------------------------------------------------------------------------
# Convenience: fire ego action index lookup
# ---------------------------------------------------------------------------

def fire_ego_action_indices(ego_action_dim: int) -> jnp.ndarray:
    """Return the fire-capable ego action indices for a given action space size."""
    if ego_action_dim == 6:
        return FIRE_EGO_ACTION_INDICES_6
    return FIRE_EGO_ACTION_INDICES
