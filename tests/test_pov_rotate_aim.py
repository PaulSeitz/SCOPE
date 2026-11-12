"""Tests for Asteroids / Time Pilot rotate-to-aim POV mappers."""

import jax.numpy as jnp
import pytest

from concept_based_learning.observations.pov_relative import make_facing_relative_mappers
from concept_based_learning.observations.pov_rotate_aim import (
    _TIMEPILOT_BIN_TO_ROT,
    build_timepilot_ego_table_for_rot,
)


def _raw_with_sprite_ori(sprite_deg: float):
    class _Player:
        orientation = jnp.array([float(sprite_deg)])

    class _Raw:
        player = _Player()

    return _Raw()


def test_asteroids_left_right_forward_independent_of_facing():
    mapper, _ = make_facing_relative_mappers("asteroids")
    # Sprite 0° = up, 90° = right — rotate/thrust buttons must not flip with facing.
    for sprite in (0.0, 90.0, 180.0, 270.0):
        raw = _raw_with_sprite_ori(sprite)
        assert int(mapper(jnp.int32(2), raw, jnp.int32(0), None)) == 2  # FORWARD→UP
        assert int(mapper(jnp.int32(4), raw, jnp.int32(0), None)) == 4  # LEFT→LEFT
        assert int(mapper(jnp.int32(3), raw, jnp.int32(0), None)) == 3  # RIGHT→RIGHT
        assert int(mapper(jnp.int32(1), raw, jnp.int32(0), None)) == 1  # FIRE
        assert int(mapper(jnp.int32(10), raw, jnp.int32(0), None)) == 8  # FWD_FIRE→UPFIRE
        assert int(mapper(jnp.int32(5), raw, jnp.int32(0), None)) == 5  # BACK→hyperspace
        assert int(mapper(jnp.int32(13), raw, jnp.int32(0), None)) == 11  # BACK_FIRE


def test_asteroids_forward_diagonal_is_thrust_and_turn():
    mapper, _ = make_facing_relative_mappers("asteroids")
    raw = _raw_with_sprite_ori(0.0)
    assert int(mapper(jnp.int32(7), raw, jnp.int32(0), None)) == 7  # FWD_LEFT→UPLEFT
    assert int(mapper(jnp.int32(6), raw, jnp.int32(0), None)) == 6  # FWD_RIGHT→UPRIGHT


def test_asteroids_backward_diagonals_are_hyperspace():
    mapper, _ = make_facing_relative_mappers("asteroids")
    raw = _raw_with_sprite_ori(0.0)
    assert int(mapper(jnp.int32(8), raw, jnp.int32(0), None)) == 5
    assert int(mapper(jnp.int32(9), raw, jnp.int32(0), None)) == 5
    assert int(mapper(jnp.int32(16), raw, jnp.int32(0), None)) == 11
    assert int(mapper(jnp.int32(17), raw, jnp.int32(0), None)) == 11


def test_timepilot_up_facing_left_right_forward():
    """Facing up (sprite 0°, bin −90°) → LEFT/RIGHT rotate, FORWARD holds UP."""
    mapper, _ = make_facing_relative_mappers("timepilot")
    raw = _raw_with_sprite_ori(0.0)  # screen −90° = up
    assert int(mapper(jnp.int32(2), raw, jnp.int32(0), None)) == 2  # FORWARD→UP
    assert int(mapper(jnp.int32(4), raw, jnp.int32(0), None)) == 4  # LEFT→LEFT
    assert int(mapper(jnp.int32(3), raw, jnp.int32(0), None)) == 3  # RIGHT→RIGHT


def test_timepilot_right_facing_uses_absolute_aim_for_turns():
    """Facing right (sprite 90°, screen 0°) → turn LEFT aims UP, turn RIGHT aims DOWN."""
    mapper, _ = make_facing_relative_mappers("timepilot")
    raw = _raw_with_sprite_ori(90.0)
    assert int(mapper(jnp.int32(2), raw, jnp.int32(0), None)) == 3  # FORWARD→RIGHT
    assert int(mapper(jnp.int32(4), raw, jnp.int32(0), None)) == 2  # LEFT→UP (+1 rot)
    assert int(mapper(jnp.int32(3), raw, jnp.int32(0), None)) == 5  # RIGHT→DOWN (−1 rot)


def test_timepilot_tables_cover_all_rotations():
    for bin_idx, rot in enumerate(_TIMEPILOT_BIN_TO_ROT):
        table = build_timepilot_ego_table_for_rot(rot)
        assert table.shape == (18,)
        # LEFT / RIGHT must be real cardinals (not NOOP) so the ship can turn.
        assert int(table[4]) in (2, 3, 4, 5)
        assert int(table[3]) in (2, 3, 4, 5)
        assert int(table[4]) != int(table[3])


def test_seaquest_still_strafes_not_rotate_table():
    """Regression: Seaquest LEFT while facing right is still UP (strafe), not rotate."""
    mapper, _ = make_facing_relative_mappers("seaquest")
    raw = _raw_with_sprite_ori(90.0)  # facing right
    # Facing right: LEFT strafe → world UP = ALE UP = 2 in Seaquest ACTION_SET indexing
    # Seaquest move_table world UP index 6 → action 2.
    assert int(mapper(jnp.int32(4), raw, jnp.int32(0), None)) == 2
