"""Unit tests for Phoenix heuristic targeting rules."""

import jax
import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_DESTROY,
)
from concept_based_learning.environments.phoenix.phoenix_policy import (
    CLOSE_DANGER_RANGE,
    NO_TARGET_IDX,
    get_heuristic_target,
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
    state: jnp.ndarray


def _stack(val, t=4):
    return jnp.stack([val] * t)


def _make_obs(player_xy, objects):
    px, py = player_xy
    raw = _Raw(
        player=_Player(
            x=_stack(jnp.float32(px)),
            y=_stack(jnp.float32(py)),
            width=_stack(jnp.float32(13)),
            height=_stack(jnp.float32(8)),
        )
    )
    xs = jnp.array([o[0] for o in objects], dtype=jnp.float32)
    ys = jnp.array([o[1] for o in objects], dtype=jnp.float32)
    ws = jnp.array([o[2] for o in objects], dtype=jnp.float32)
    hs = jnp.array([o[3] for o in objects], dtype=jnp.float32)
    act = jnp.array([o[4] for o in objects], dtype=jnp.int32)
    labs = jnp.array([o[5] for o in objects], dtype=jnp.int32)
    objs = _Objects(
        x=_stack(xs),
        y=_stack(ys),
        width=_stack(ws),
        height=_stack(hs),
        active=_stack(act),
        semantic_label=_stack(labs),
        state=_stack(jnp.zeros_like(xs, dtype=jnp.int32)),
    )
    return raw, objs


def test_close_danger_range_is_25():
    assert CLOSE_DANGER_RANGE == 25.0


def test_avoids_missile_within_extended_range():
    # Player near bottom; AVOID missile ~20px above (within 25, outside old 15).
    raw, objs = _make_obs(
        (80, 170),
        [
            (80, 40, 8, 8, 1, LABEL_DESTROY),  # far bird
            (80, 145, 2, 4, 1, LABEL_AVOID),  # approaching missile
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 1


def test_ignores_missile_beyond_extended_range():
    raw, objs = _make_obs(
        (80, 170),
        [
            (80, 40, 8, 8, 1, LABEL_DESTROY),
            (80, 120, 2, 4, 1, LABEL_AVOID),  # ~42px gap > 25
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 0
