"""Restricted-axis ego mapping + realizable-ego legal masks."""

from types import SimpleNamespace

import jax.numpy as jnp

from concept_based_learning.environments.phoenix.phoenix_label import phoenix_action_mapper
from concept_based_learning.observations.action_space import (
    make_ego_action_mapper,
    make_legal_ego_mask_fn,
    make_realized_ego_mask_fn,
)
from concept_based_learning.observations.pov_relative import make_facing_relative_mappers


def _fake_obs(px, py, pw=8.0, ph=8.0, tx=80.0, ty=40.0, tw=8.0, th=8.0):
    """Minimal stacked raw_obs / objects for the ego action mapper."""
    player = SimpleNamespace(
        x=jnp.array([px, px], dtype=jnp.float32),
        y=jnp.array([py, py], dtype=jnp.float32),
        width=jnp.array([pw, pw], dtype=jnp.float32),
        height=jnp.array([ph, ph], dtype=jnp.float32),
    )
    raw_obs = SimpleNamespace(player=player)
    objects = SimpleNamespace(
        x=jnp.array([[tx], [tx]], dtype=jnp.float32),
        y=jnp.array([[ty], [ty]], dtype=jnp.float32),
        width=jnp.array([[tw], [tw]], dtype=jnp.float32),
        height=jnp.array([[th], [th]], dtype=jnp.float32),
    )
    return raw_obs, objects


def test_phoenix_forward_maps_right_on_upright_diagonal():
    # Player→target ≈ UPRIGHT (−45°); Phoenix table maps UR → RIGHT.
    raw_obs, objects = _fake_obs(px=20.0, py=180.0, tx=100.0, ty=100.0)
    game_a = int(phoenix_action_mapper(jnp.int32(2), raw_obs, jnp.int32(0), objects))
    assert game_a == 2  # RIGHT


def test_phoenix_forward_stays_noop_when_x_aligned():
    raw_obs, objects = _fake_obs(px=76.0, py=180.0, tx=80.0, ty=40.0, pw=8.0, tw=8.0)
    game_a = int(phoenix_action_mapper(jnp.int32(2), raw_obs, jnp.int32(0), objects))
    assert game_a == 0  # UP → NOOP


def test_phoenix_forward_fire_maps_rightfire_on_upright_diagonal():
    raw_obs, objects = _fake_obs(px=20.0, py=180.0, tx=100.0, ty=100.0)
    game_a = int(phoenix_action_mapper(jnp.int32(10), raw_obs, jnp.int32(0), objects))
    assert game_a == 5  # RIGHTFIRE


def test_phoenix_legal_mask_blocks_forward_when_aligned_above():
    raw_obs, objects = _fake_obs(px=76.0, py=180.0, tx=80.0, ty=40.0, pw=8.0, tw=8.0)
    mask_fn = make_realized_ego_mask_fn(phoenix_action_mapper, noop_action=0, fire_action=1)
    legal = mask_fn(raw_obs, jnp.int32(0), objects)
    assert bool(legal[0]) and bool(legal[1])  # NOOP + FIRE always ok
    assert not bool(legal[2])  # FORWARD → NOOP
    assert not bool(legal[10])  # FORWARD_FIRE → FIRE only
    assert bool(legal[3]) or bool(legal[4])  # LEFT/RIGHT still available


def test_phoenix_legal_mask_keeps_forward_when_diagonal_realizes():
    raw_obs, objects = _fake_obs(px=20.0, py=180.0, tx=100.0, ty=100.0)
    mask_fn = make_realized_ego_mask_fn(phoenix_action_mapper, noop_action=0, fire_action=1)
    legal = mask_fn(raw_obs, jnp.int32(0), objects)
    assert bool(legal[2])
    assert bool(legal[10])


def test_full_stick_legal_mask_is_all_true():
    mapper = make_ego_action_mapper(
        noop_action=0,
        fire_action=1,
        move_table=[3, 8, 5, 9, 4, 7, 2, 6],
        move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
    )
    raw_obs, objects = _fake_obs(px=80.0, py=100.0, tx=80.0, ty=40.0)
    mask_fn = make_realized_ego_mask_fn(mapper, noop_action=0, fire_action=1)
    legal = mask_fn(raw_obs, jnp.int32(0), objects)
    assert bool(jnp.all(legal))


def test_pure_cardinal_blocks_diagonals_and_move_fire():
    mapper = make_ego_action_mapper(
        noop_action=0,
        fire_action=1,
        move_table=[3, 8, 5, 9, 4, 7, 2, 6],
        move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
    )
    raw_obs, objects = _fake_obs(px=80.0, py=100.0, tx=80.0, ty=40.0)
    realized = make_realized_ego_mask_fn(mapper, noop_action=0, fire_action=1)
    pure = make_legal_ego_mask_fn(mapper, mode="pure_cardinal", noop_action=0, fire_action=1)
    r = realized(raw_obs, jnp.int32(0), objects)
    p = pure(raw_obs, jnp.int32(0), objects)
    for i in range(6):
        assert bool(r[i]) == bool(p[i])
    for i in range(6, 18):
        assert bool(r[i])
        assert not bool(p[i])


def test_pong_pure_cardinal_only_noop_left_right():
    mapper, _ = make_facing_relative_mappers("pong")
    raw_obs, objects = _fake_obs(px=140.0, py=80.0, tx=80.0, ty=80.0)
    pure = make_legal_ego_mask_fn(mapper, mode="pure_cardinal", noop_action=0, fire_action=1)
    legal = pure(raw_obs, jnp.int32(0), objects)
    assert bool(legal[0])  # NOOP
    assert bool(legal[1])  # FIRE
    assert not bool(legal[2])  # FORWARD → NOOP
    assert bool(legal[3]) or bool(legal[4])  # strafe
    assert not bool(legal[5])  # BACKWARD → NOOP
    for i in range(6, 18):
        assert not bool(legal[i])
