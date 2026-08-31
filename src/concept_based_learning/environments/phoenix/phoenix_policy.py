import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_DESTROY,
)
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

NO_TARGET_IDX = -1
# AABB surface-gap danger range for AVOID objects.
# Slightly ahead of contact so AVOID (shield/strafe) engages before a salvo lands.
CLOSE_DANGER_RANGE = 25.0
# Added to gap for enemy bullets that are clearly moving away from the player
# (e.g. already past the ship on x), so approaching shots win when ranking threats.
RECEDING_AVOID_PENALTY = 80.0
# Minimum per-frame speed squared (pixels^2) before we trust the velocity-based sign.
MIN_BULLET_SPEED_SQ = 1.0


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Heuristic Phoenix targeting rule with sticky DESTROY lock.

    Distances use AABB surface gap (same metric as skill ego features).
    """
    del rng
    distances, p_cx, p_cy, o_cx, o_cy = player_object_aabb_gaps(
        raw_observation, objects
    )

    # Treat objects in a death-animation state (state >= 4) as inactive.
    state = getattr(objects, "state", None)
    if state is not None:
        last_state = state[-1]
        is_dying = last_state >= 4
        base_active = objects.active[-1] > 0.5
        active_mask = jnp.logical_and(base_active, jnp.logical_not(is_dying))
    else:
        active_mask = objects.active[-1] > 0.5

    # Per-frame velocity from the last two stacked frames (toward player iff dot > 0).
    ox_stack = objects.x
    oy_stack = objects.y
    vx = ox_stack[-1].astype(jnp.float32) - ox_stack[-2].astype(jnp.float32)
    vy = oy_stack[-1].astype(jnp.float32) - oy_stack[-2].astype(jnp.float32)
    dx_f = (p_cx - o_cx).astype(jnp.float32)
    dy_f = (p_cy - o_cy).astype(jnp.float32)
    approach = vx * dx_f + vy * dy_f
    speed_sq = vx * vx + vy * vy
    has_velocity = speed_sq > MIN_BULLET_SPEED_SQ
    is_receding = jnp.logical_and(has_velocity, approach < 0.0)

    labels = objects.semantic_label[-1]

    is_destroy = labels == LABEL_DESTROY
    valid_destroy = jnp.logical_and(active_mask, is_destroy)

    large_distance = jnp.max(distances) + 1.0
    masked_distances = jnp.where(valid_destroy, distances, large_distance)

    has_any_destroy = jnp.any(valid_destroy)
    best_destroy_idx = jnp.argmin(masked_distances).astype(jnp.int32)

    prev_valid = jnp.logical_and(
        prev_target_idx >= 0,
        prev_target_idx < active_mask.shape[0],
    )
    prev_active = jnp.where(prev_valid, active_mask[prev_target_idx], False)
    prev_is_destroy = jnp.where(
        prev_valid, labels[prev_target_idx] == LABEL_DESTROY, False
    )
    keep_prev_destroy = jnp.logical_and(prev_active, prev_is_destroy)

    base_destroy_idx = jnp.where(keep_prev_destroy, prev_target_idx, best_destroy_idx)
    base_destroy_idx = jnp.where(
        has_any_destroy,
        base_destroy_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )

    is_avoid = labels == LABEL_AVOID
    is_close_avoid = jnp.logical_and(is_avoid, distances <= CLOSE_DANGER_RANGE)
    valid_close_avoid = jnp.logical_and(is_close_avoid, active_mask)
    have_valid_close_avoid = jnp.any(valid_close_avoid)

    receding_bullet = jnp.logical_and(is_avoid, is_receding)
    avoid_rank_distance = jnp.where(
        receding_bullet, distances + RECEDING_AVOID_PENALTY, distances
    )
    avoid_distances = jnp.where(valid_close_avoid, avoid_rank_distance, large_distance)
    best_avoid_idx = jnp.argmin(avoid_distances).astype(jnp.int32)

    target_idx = jnp.where(have_valid_close_avoid, best_avoid_idx, base_destroy_idx)

    any_targetable = jnp.logical_or(has_any_destroy, have_valid_close_avoid)
    target_idx = jnp.where(any_targetable, target_idx, jnp.array(NO_TARGET_IDX, dtype=jnp.int32))

    return target_idx.astype(jnp.int32)
