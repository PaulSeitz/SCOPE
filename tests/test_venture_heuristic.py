"""Unit tests for Venture heuristic targeting rules."""

import jax
import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.venture.venture_policy import (
    CLOSE_AVOID_RANGE,
    DESTROY_RANGE,
    NO_TARGET_IDX,
    get_heuristic_target,
)
from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NAVIGATE,
)


@struct.dataclass
class _Player:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


@struct.dataclass
class _Raw:
    player: _Player


@struct.dataclass
class _Objects:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray
    semantic_label: jnp.ndarray


def _stack(val, t=4):
    return jnp.stack([val] * t)


def _make_obs(player_xy, objects):
    px, py = player_xy
    raw = _Raw(
        player=_Player(
            x=_stack(jnp.int32(px)),
            y=_stack(jnp.int32(py)),
            width=_stack(jnp.int32(8)),
            height=_stack(jnp.int32(8)),
        )
    )
    xs = jnp.array([o[0] for o in objects], dtype=jnp.int32)
    ys = jnp.array([o[1] for o in objects], dtype=jnp.int32)
    ws = jnp.array([o[2] for o in objects], dtype=jnp.int32)
    hs = jnp.array([o[3] for o in objects], dtype=jnp.int32)
    act = jnp.array([o[4] for o in objects], dtype=jnp.int32)
    labs = jnp.array([o[5] for o in objects], dtype=jnp.int32)
    objs = _Objects(
        x=_stack(xs),
        y=_stack(ys),
        width=_stack(ws),
        height=_stack(hs),
        active=_stack(act),
        semantic_label=_stack(labs),
    )
    return raw, objs


def test_venture_avoids_close_threat():
    # Slot 0 inactive meta NOOP; close AVOID beats far DESTROY.
    raw, objs = _make_obs(
        (80, 80),
        [
            (0, 0, 1, 1, 0, 0),
            (95, 80, 8, 8, 1, LABEL_AVOID),
            (20, 80, 8, 8, 1, LABEL_DESTROY),
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 1
    assert CLOSE_AVOID_RANGE > 0


def test_venture_destroys_nearby_monster():
    raw, objs = _make_obs(
        (80, 80),
        [
            (0, 0, 1, 1, 0, 0),
            (100, 80, 8, 8, 1, LABEL_DESTROY),
            (20, 20, 8, 8, 1, LABEL_COLLECT),
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 1
    assert DESTROY_RANGE > 0


def test_venture_collects_when_no_destroy():
    raw, objs = _make_obs(
        (80, 80),
        [
            (0, 0, 1, 1, 0, 0),
            (40, 80, 8, 8, 1, LABEL_COLLECT),
            (140, 80, 8, 8, 1, LABEL_NAVIGATE),
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 1


def test_venture_navigates_portal_when_empty_room():
    raw, objs = _make_obs(
        (80, 80),
        [
            (0, 0, 1, 1, 0, 0),
            (140, 80, 8, 8, 1, LABEL_NAVIGATE),
            (40, 80, 8, 8, 0, LABEL_COLLECT),
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 1
