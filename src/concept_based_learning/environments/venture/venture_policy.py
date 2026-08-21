"""Env-specific Venture heuristic meta-policy.

Room explorer / shooter: monsters (DESTROY), chaser/lasers (AVOID), chest
(COLLECT), portals (NAVIGATE). Targeting priority:

  1. AVOID when within CLOSE_AVOID_RANGE (AABB surface gap).
  2. Else nearest DESTROY within DESTROY_RANGE (else any DESTROY).
  3. Else nearest COLLECT.
  4. Else nearest NAVIGATE portal.
  5. Else NO_TARGET.
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NAVIGATE,
)
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

NO_TARGET_IDX = -1
CLOSE_AVOID_RANGE = 28.0
DESTROY_RANGE = 60.0


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del prev_target_idx, rng

    distances, *_ = player_object_aabb_gaps(raw_observation, objects)
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    is_avoid = labels == LABEL_AVOID
    is_destroy = labels == LABEL_DESTROY
    is_collect = labels == LABEL_COLLECT
    is_navigate = labels == LABEL_NAVIGATE

    avoid_threat = jnp.logical_and(
        jnp.logical_and(active, is_avoid),
        distances <= CLOSE_AVOID_RANGE,
    )
    have_avoid = jnp.any(avoid_threat)
    avoid_idx = jnp.argmin(jnp.where(avoid_threat, distances, jnp.inf)).astype(
        jnp.int32
    )

    destroy_near = jnp.logical_and(
        jnp.logical_and(active, is_destroy),
        distances <= DESTROY_RANGE,
    )
    have_destroy_near = jnp.any(destroy_near)
    destroy_near_idx = jnp.argmin(
        jnp.where(destroy_near, distances, jnp.inf)
    ).astype(jnp.int32)

    any_destroy = jnp.logical_and(active, is_destroy)
    have_any_destroy = jnp.any(any_destroy)
    any_destroy_idx = jnp.argmin(jnp.where(any_destroy, distances, jnp.inf)).astype(
        jnp.int32
    )

    collect_mask = jnp.logical_and(active, is_collect)
    have_collect = jnp.any(collect_mask)
    collect_idx = jnp.argmin(jnp.where(collect_mask, distances, jnp.inf)).astype(
        jnp.int32
    )

    navigate_mask = jnp.logical_and(active, is_navigate)
    have_navigate = jnp.any(navigate_mask)
    navigate_idx = jnp.argmin(jnp.where(navigate_mask, distances, jnp.inf)).astype(
        jnp.int32
    )

    target = jnp.array(NO_TARGET_IDX, dtype=jnp.int32)
    target = jnp.where(have_navigate, navigate_idx, target)
    target = jnp.where(have_collect, collect_idx, target)
    target = jnp.where(have_any_destroy, any_destroy_idx, target)
    target = jnp.where(have_destroy_near, destroy_near_idx, target)
    target = jnp.where(have_avoid, avoid_idx, target)
    return target.astype(jnp.int32)
