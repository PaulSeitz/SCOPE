"""Kaboom heuristic: always pick the lowest (largest screen-y) active bomb."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_COLLECT
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

NO_TARGET_IDX = -1


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del prev_target_idx, rng
    distances, _, _, _, _ = player_object_aabb_gaps(raw_observation, objects)
    del distances
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]
    is_collect = labels == LABEL_COLLECT
    mask = jnp.logical_and(active, is_collect)
    # Lowest bomb = largest centre-y (falling toward the buckets).
    o_cy = objects.y[-1] + objects.height[-1] * 0.5
    rank = jnp.where(mask, o_cy, -jnp.inf)
    idx = jnp.argmax(rank).astype(jnp.int32)
    return jnp.where(jnp.any(mask), idx, jnp.int32(NO_TARGET_IDX)).astype(jnp.int32)
