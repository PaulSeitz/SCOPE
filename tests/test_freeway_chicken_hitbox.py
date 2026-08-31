"""Freeway chicken hitbox must keep bottom-edge y for ego / heuristic geometry."""

import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.freeway.freeway_label import (
    FREEWAY_CHICKEN_HIT_H,
    FREEWAY_CHICKEN_HIT_W,
    FREEWAY_GOAL_LABEL,
    _apply_chicken_hitbox,
)
from concept_based_learning.environments.datatypes import LABEL_COLLECT
from concept_based_learning.observations.egocentric import _center_y


@struct.dataclass
class _Chicken:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


@struct.dataclass
class _Raw:
    chicken: _Chicken


def test_chicken_hitbox_keeps_bottom_edge_y():
    # JAXAtari Freeway: y = bottom edge, sprite 6x8 before insets.
    raw = _Raw(
        chicken=_Chicken(
            x=jnp.array([44], dtype=jnp.int32),
            y=jnp.array([187], dtype=jnp.int32),
            width=jnp.array([6], dtype=jnp.int32),
            height=jnp.array([8], dtype=jnp.int32),
        )
    )
    patched = _apply_chicken_hitbox(raw).chicken
    assert int(patched.y[0]) == 187  # bottom edge unchanged
    assert int(patched.width[0]) == FREEWAY_CHICKEN_HIT_W
    assert int(patched.height[0]) == FREEWAY_CHICKEN_HIT_H
    # Effective top expands 2px upward (inset_y_top=-2).
    assert int(patched.y[0] - patched.height[0]) == 177
    cy = float(_center_y(patched.y[0], patched.height[0], y_is_bottom_edge=True))
    assert cy == 182.0


def test_freeway_top_goal_uses_collect_skill():
    assert FREEWAY_GOAL_LABEL == LABEL_COLLECT
