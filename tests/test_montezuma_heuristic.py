"""Unit tests for Montezuma's Revenge heuristic targeting rules."""

import jax
import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.montezumarevenge.montezuma_policy import (
    CLOSE_AVOID_RANGE,
    NO_TARGET_IDX,
    get_heuristic_target,
)
from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
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


def test_montezuma_avoids_close_enemy():
    raw, objs = _make_obs(
        (80, 80),
        [
            (0, 0, 1, 1, 0, 0),
            (90, 80, 8, 8, 1, LABEL_AVOID),
            (40, 80, 8, 8, 1, LABEL_COLLECT),
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 1
    assert CLOSE_AVOID_RANGE > 0


def test_montezuma_collects_when_no_threat():
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


def test_montezuma_navigates_when_no_collect():
    raw, objs = _make_obs(
        (80, 80),
        [
            (0, 0, 1, 1, 0, 0),
            (140, 40, 4, 38, 1, LABEL_NAVIGATE),
            (20, 80, 8, 8, 0, LABEL_COLLECT),
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 1


def test_montezuma_no_target_when_empty():
    raw, objs = _make_obs(
        (80, 80),
        [
            (0, 0, 1, 1, 0, 0),
            (40, 80, 8, 8, 0, LABEL_COLLECT),
        ],
    )
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == NO_TARGET_IDX
