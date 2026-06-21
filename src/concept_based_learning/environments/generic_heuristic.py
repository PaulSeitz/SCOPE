"""
Generic, environment-agnostic heuristic meta-policy.

Operates purely on the universal `LabeledObjectObservation` interface (semantic
label + kinematics) plus the player entity, so it works for any environment
without per-env code. Intended to bootstrap the meta-policy during warmup.

Targeting rule (priority low -> high):
  1. Default: the nearest COLLECT object (fallback: nearest DESTROY, then
     nearest AVOID) -- the baseline "go do the task" behaviour.
  2. If a DESTROY object is within DESTROY_RANGE, switch to the nearest one --
     handle a nearby threat before it reaches the player.
  3. If an AVOID object is within AVOID_RANGE (< DESTROY_RANGE), switch to the
     nearest one -- dodging an imminent hazard takes top priority.
"""

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)
from concept_based_learning.observations.egocentric import _get_player

# Returned when no object is targetable; downstream executes NOOP.
NO_TARGET_IDX = -1

# Proximity thresholds (pixels, player centre -> object centre). DESTROY reuses
# the Seaquest CLOSE_DANGER_RANGE; AVOID is deliberately tighter so the agent
# only abandons its task for a genuinely imminent hazard. Both are tunable.
DESTROY_RANGE = 25.0
AVOID_RANGE = 15.0


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """Env-agnostic target selection over the labeled-object interface.

    Sticky mission target: once a COLLECT or DESTROY target is locked via
    prev_target_idx it is kept until it disappears, reducing spurious switches
    that would fragment skill trajectories.  Reactive AVOID overrides still
    fire normally when a threat enters AVOID_RANGE.
    """
    del rng

    # Centre convention matches observations.egocentric (x + w/2, y + h/2).
    player = _get_player(raw_observation)
    px = player.x[-1] + player.width[-1] / 2.0
    py = player.y[-1] + player.height[-1] / 2.0

    ox = objects.x[-1] + objects.width[-1] / 2.0
    oy = objects.y[-1] + objects.height[-1] / 2.0
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    # Slot 0 is the reserved (inactive) meta-NOOP placeholder; the `active`
    # mask excludes it automatically.
    distances = jnp.sqrt((px - ox) ** 2 + (py - oy) ** 2)

    collect_mask = jnp.logical_and(active, labels == LABEL_COLLECT)
    destroy_mask = jnp.logical_and(active, labels == LABEL_DESTROY)
    avoid_mask = jnp.logical_and(active, labels == LABEL_AVOID)

    best_collect = jnp.argmin(jnp.where(collect_mask, distances, jnp.inf)).astype(jnp.int32)
    best_destroy = jnp.argmin(jnp.where(destroy_mask, distances, jnp.inf)).astype(jnp.int32)
    best_avoid = jnp.argmin(jnp.where(avoid_mask, distances, jnp.inf)).astype(jnp.int32)

    any_collect = jnp.any(collect_mask)
    any_destroy = jnp.any(destroy_mask)
    any_avoid = jnp.any(avoid_mask)

    # Sticky mission target: keep prev_target_idx when it is still an active
    # COLLECT or DESTROY object; otherwise fall back to nearest fresh target.
    prev_valid = jnp.logical_and(
        prev_target_idx >= 0,
        prev_target_idx < active.shape[0],
    )
    prev_active = jnp.where(prev_valid, active[prev_target_idx], False)
    prev_label = jnp.where(prev_valid, labels[prev_target_idx], -1)
    prev_is_mission = jnp.logical_or(
        prev_label == LABEL_COLLECT, prev_label == LABEL_DESTROY
    )
    keep_prev = jnp.logical_and(prev_active, prev_is_mission)

    # Rule 1 -- fresh default: nearest COLLECT, falling back to DESTROY then AVOID.
    fresh_target = jnp.where(
        any_collect,
        best_collect,
        jnp.where(any_destroy, best_destroy, best_avoid),
    )
    target = jnp.where(keep_prev, prev_target_idx, fresh_target)

    # Rule 2 -- a DESTROY object inside DESTROY_RANGE overrides stickiness.
    destroy_in_range = jnp.any(
        jnp.logical_and(destroy_mask, distances <= DESTROY_RANGE)
    )
    target = jnp.where(destroy_in_range, best_destroy, target)

    # Rule 3 -- an AVOID object inside the tighter AVOID_RANGE overrides all.
    avoid_in_range = jnp.any(jnp.logical_and(avoid_mask, distances <= AVOID_RANGE))
    target = jnp.where(avoid_in_range, best_avoid, target)

    # No targetable object on screen -> signal NOOP downstream.
    any_targetable = jnp.logical_or(any_collect, jnp.logical_or(any_destroy, any_avoid))
    target = jnp.where(
        any_targetable, target, jnp.array(NO_TARGET_IDX, dtype=jnp.int32)
    )
    return target.astype(jnp.int32)
