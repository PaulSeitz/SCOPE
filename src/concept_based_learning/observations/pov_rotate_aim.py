"""Facing-frame body actions for rotate-to-aim games (Asteroids, Time Pilot).

Same 18-dim ego layout as Seaquest POV, but LEFT/RIGHT **rotate** the ship
instead of strafing. FORWARD advances along the current nose (thrust on
Asteroids; hold heading on Time Pilot). Used for both transfer and
from-scratch so the motor semantics stay identical.

Seaquest→AS/TP zero-shot is still expected to fail (strafe ≠ turn). AS↔TP
homologous transfer can share this motor.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from concept_based_learning.observations.action_space import _default_get_player

# Asteroids Discrete(14) indices (see asteroids_label.py).
# DOWN / DOWNFIRE are hyperspace — allowed on ego BACKWARD*.
_AS_NOOP = 0
_AS_FIRE = 1
_AS_UP = 2
_AS_RIGHT = 3
_AS_LEFT = 4
_AS_DOWN = 5
_AS_UPRIGHT = 6
_AS_UPLEFT = 7
_AS_UPFIRE = 8
_AS_RIGHTFIRE = 9
_AS_LEFTFIRE = 10
_AS_DOWNFIRE = 11
_AS_UPRIGHTFIRE = 12
_AS_UPLEFTFIRE = 13

# Time Pilot Discrete(10) indices.
_TP_NOOP = 0
_TP_FIRE = 1
_TP_UP = 2
_TP_RIGHT = 3
_TP_LEFT = 4
_TP_DOWN = 5
_TP_UPFIRE = 6
_TP_RIGHTFIRE = 7
_TP_LEFTFIRE = 8
_TP_DOWNFIRE = 9

# Ego layout (action_space.py): 0 NOOP, 1 FIRE, 2 FWD, 3 RIGHT, 4 LEFT, 5 BACK,
# 6 FWD_R, 7 FWD_L, 8 BACK_R, 9 BACK_L, 10–17 same + FIRE.


def _asteroids_ego_table() -> jnp.ndarray:
    """Facing-independent rotate/thrust/hyperspace table (length 18)."""
    return jnp.asarray(
        [
            _AS_NOOP,  # 0 NOOP
            _AS_FIRE,  # 1 FIRE
            _AS_UP,  # 2 FORWARD → thrust
            _AS_RIGHT,  # 3 RIGHT → rotate CW
            _AS_LEFT,  # 4 LEFT → rotate CCW
            _AS_DOWN,  # 5 BACKWARD → hyperspace
            _AS_UPRIGHT,  # 6 FORWARD_RIGHT → thrust+rotate CW
            _AS_UPLEFT,  # 7 FORWARD_LEFT → thrust+rotate CCW
            _AS_DOWN,  # 8 BACKWARD_RIGHT → hyperspace
            _AS_DOWN,  # 9 BACKWARD_LEFT → hyperspace
            _AS_UPFIRE,  # 10 FORWARD_FIRE
            _AS_RIGHTFIRE,  # 11 RIGHT_FIRE
            _AS_LEFTFIRE,  # 12 LEFT_FIRE
            _AS_DOWNFIRE,  # 13 BACKWARD_FIRE → hyperspace+fire
            _AS_UPRIGHTFIRE,  # 14 FORWARD_RIGHT_FIRE
            _AS_UPLEFTFIRE,  # 15 FORWARD_LEFT_FIRE
            _AS_DOWNFIRE,  # 16 BACKWARD_RIGHT_FIRE
            _AS_DOWNFIRE,  # 17 BACKWARD_LEFT_FIRE
        ],
        dtype=jnp.int32,
    )


# Facing-bin index → Time Pilot player_rotation (0=up).
# Bins are screen ° (0, 45, 90, 135, 180, -135, -90, -45).
_TIMEPILOT_BIN_TO_ROT = (6, 5, 4, 3, 2, 1, 0, 7)


def _timepilot_button_for_rot_delta(
    rot: jnp.ndarray, *, turn_left: bool, with_fire: bool
) -> jnp.ndarray:
    """Game button that turns one tick LEFT (+1) or RIGHT (−1) from ``rot``.

    Time Pilot has no body-relative turn key. Each Discrete LEFT/RIGHT/UP/DOWN
    requests an *absolute* world heading; the engine then rotates one tick
    toward that heading. So ego LEFT is still “turn left”, but the Discrete
    index we emit depends on the current nose (e.g. facing right, ego LEFT
    emits UP because that is the request that yields +1 rotation).
    """
    # Per current rotation: (left_btn, right_btn) without fire.
    # Derived so (rot+1)%8 follows LEFT button and (rot-1)%8 follows RIGHT.
    left_btns = jnp.asarray(
        [
            _TP_LEFT,  # 0 up → left
            _TP_LEFT,  # 1 UL → left
            _TP_DOWN,  # 2 left → down
            _TP_DOWN,  # 3 DL → down
            _TP_RIGHT,  # 4 down → right
            _TP_RIGHT,  # 5 DR → right
            _TP_UP,  # 6 right → up
            _TP_UP,  # 7 UR → up
        ],
        dtype=jnp.int32,
    )
    right_btns = jnp.asarray(
        [
            _TP_RIGHT,  # 0
            _TP_UP,  # 1
            _TP_UP,  # 2
            _TP_LEFT,  # 3
            _TP_LEFT,  # 4
            _TP_DOWN,  # 5
            _TP_DOWN,  # 6
            _TP_RIGHT,  # 7
        ],
        dtype=jnp.int32,
    )
    btn = jnp.where(turn_left, left_btns[rot], right_btns[rot])
    # FIRE variants: UPFIRE=6 … DOWNFIRE=9 = cardinal + 4.
    return jnp.where(with_fire, btn + jnp.int32(4), btn)


def _timepilot_hold_heading(rot: jnp.ndarray, *, with_fire: bool) -> jnp.ndarray:
    """Hold current cardinal nose; NOOP on diagonal facings (no diagonal button)."""
    cardinal = jnp.logical_not(rot % jnp.int32(2))  # even rot = cardinal
    # rot 0,2,4,6 → UP, LEFT, DOWN, RIGHT
    hold = jnp.asarray(
        [_TP_UP, _TP_NOOP, _TP_LEFT, _TP_NOOP, _TP_DOWN, _TP_NOOP, _TP_RIGHT, _TP_NOOP],
        dtype=jnp.int32,
    )[rot]
    hold = jnp.where(with_fire, jnp.where(hold == _TP_NOOP, _TP_FIRE, hold + jnp.int32(4)), hold)
    return jnp.where(cardinal, hold, jnp.where(with_fire, jnp.int32(_TP_FIRE), jnp.int32(_TP_NOOP)))


def build_timepilot_ego_table_for_rot(rot: int) -> jnp.ndarray:
    """Length-18 ego→game table for a fixed Time Pilot rotation index."""
    r = jnp.int32(rot)
    left = _timepilot_button_for_rot_delta(r, turn_left=True, with_fire=False)
    right = _timepilot_button_for_rot_delta(r, turn_left=False, with_fire=False)
    left_f = _timepilot_button_for_rot_delta(r, turn_left=True, with_fire=True)
    right_f = _timepilot_button_for_rot_delta(r, turn_left=False, with_fire=True)
    fwd = _timepilot_hold_heading(r, with_fire=False)
    fwd_f = _timepilot_hold_heading(r, with_fire=True)
    # Diagonals: treat FORWARD_LEFT/RIGHT as pure turn (no thrust on TP).
    return jnp.asarray(
        [
            _TP_NOOP,  # 0
            _TP_FIRE,  # 1
            fwd,  # 2 FORWARD
            right,  # 3 RIGHT
            left,  # 4 LEFT
            _TP_NOOP,  # 5 BACKWARD
            right,  # 6 FORWARD_RIGHT → turn right
            left,  # 7 FORWARD_LEFT → turn left
            _TP_NOOP,  # 8 BACKWARD_RIGHT
            _TP_NOOP,  # 9 BACKWARD_LEFT
            fwd_f,  # 10 FORWARD_FIRE
            right_f,  # 11 RIGHT_FIRE
            left_f,  # 12 LEFT_FIRE
            _TP_FIRE,  # 13 BACKWARD_FIRE
            right_f,  # 14 FORWARD_RIGHT_FIRE
            left_f,  # 15 FORWARD_LEFT_FIRE
            _TP_FIRE,  # 16 BACKWARD_RIGHT_FIRE
            _TP_FIRE,  # 17 BACKWARD_LEFT_FIRE
        ],
        dtype=jnp.int32,
    )


def make_asteroids_rotate_mappers(
    *,
    player_getter: Callable | None = None,
) -> tuple[Callable, Callable]:
    """Asteroids POV mapper: LEFT/RIGHT rotate, FORWARD thrusts (facing-independent)."""
    table = _asteroids_ego_table()
    _getter = player_getter or _default_get_player
    del _getter  # facing unused for button choice; obs path still uses orientation

    def mapper(ego_action, raw_obs, target_idx, objects):
        del raw_obs, target_idx, objects
        return table[jnp.asarray(ego_action, dtype=jnp.int32)]

    def inverse_mapper(game_action, raw_obs, target_idx, objects):
        del raw_obs, target_idx, objects
        game_action = jnp.asarray(game_action, dtype=jnp.int32)
        matches = table == game_action
        has_match = jnp.any(matches)
        ego_idx = jnp.argmax(matches).astype(jnp.int32)
        ego_idx = jnp.where(game_action == jnp.int32(_AS_NOOP), jnp.int32(0), ego_idx)
        ego_idx = jnp.where(
            jnp.logical_and(game_action == jnp.int32(_AS_FIRE), jnp.logical_not(has_match)),
            jnp.int32(1),
            ego_idx,
        )
        ego_idx = jnp.where(
            jnp.logical_and(jnp.logical_not(has_match), game_action != jnp.int32(_AS_NOOP)),
            jnp.int32(0),
            ego_idx,
        )
        return ego_idx.astype(jnp.int32)

    return mapper, inverse_mapper


def make_timepilot_rotate_mappers(
    *,
    player_getter: Callable | None = None,
    facing_bin_fn: Callable | None = None,
) -> tuple[Callable, Callable]:
    """Time Pilot POV mapper: LEFT/RIGHT = one-tick turn via absolute-aim buttons."""
    from concept_based_learning.observations.pov_relative import (
        _POV_PROFILES,
        _facing_bin_idx,
        facing_bins_deg,
    )

    profile = _POV_PROFILES["timepilot"]
    _getter = player_getter or profile.get("player_getter") or _default_get_player
    bins = facing_bins_deg(profile)
    if len(bins) != len(_TIMEPILOT_BIN_TO_ROT):
        raise ValueError(
            f"Time Pilot facing bins ({len(bins)}) must match "
            f"_TIMEPILOT_BIN_TO_ROT ({len(_TIMEPILOT_BIN_TO_ROT)})"
        )
    # One table per facing bin, ordered like facing_bins_deg.
    tables = jnp.stack(
        [build_timepilot_ego_table_for_rot(_TIMEPILOT_BIN_TO_ROT[i]) for i in range(len(bins))],
        axis=0,
    )
    _n_bins = jnp.int32(len(bins))
    _noop = jnp.int32(_TP_NOOP)
    _fire = jnp.int32(_TP_FIRE)

    def _bin(raw_obs, target_idx, objects):
        if facing_bin_fn is not None:
            return facing_bin_fn(raw_obs, target_idx, objects)
        return _facing_bin_idx(raw_obs, profile, _getter)

    def mapper(ego_action, raw_obs, target_idx, objects):
        bin_idx = jnp.clip(_bin(raw_obs, target_idx, objects), jnp.int32(0), _n_bins - 1)
        return tables[bin_idx, jnp.asarray(ego_action, dtype=jnp.int32)]

    def inverse_mapper(game_action, raw_obs, target_idx, objects):
        bin_idx = jnp.clip(_bin(raw_obs, target_idx, objects), jnp.int32(0), _n_bins - 1)
        table = tables[bin_idx]
        game_action = jnp.asarray(game_action, dtype=jnp.int32)
        matches = table == game_action
        has_match = jnp.any(matches)
        ego_idx = jnp.argmax(matches).astype(jnp.int32)
        ego_idx = jnp.where(game_action == _noop, jnp.int32(0), ego_idx)
        ego_idx = jnp.where(
            jnp.logical_and(game_action == _fire, jnp.logical_not(has_match)),
            jnp.int32(1),
            ego_idx,
        )
        ego_idx = jnp.where(
            jnp.logical_and(jnp.logical_not(has_match), game_action != _noop),
            jnp.int32(0),
            ego_idx,
        )
        return ego_idx.astype(jnp.int32)

    return mapper, inverse_mapper


def is_rotate_aim_env(jax_game_id: str) -> bool:
    return str(jax_game_id).lower() in {"asteroids", "timepilot"}
