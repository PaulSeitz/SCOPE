import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_DESTROY,
)
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

# When no object is on screen, return this; downstream should execute NOOP.
NO_TARGET_IDX = -1

# AABB surface-gap (px) that counts as "immediate danger" for dodging bullets.
CLOSE_DANGER_RANGE = 20.0

NUM_ENEMIES = 36
ENEMY_COLS = 6
FIRST_ENEMY_SLOT = 1  # slot 0 = reserved meta NOOP


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Space Invaders targeting heuristic with persistent DESTROY selection.

    - When choosing a new DESTROY target, pick uniformly at random among
      enemies on the current bottom formation row.
    - Keep that target until it is destroyed / goes inactive.
    - If a bullet (AVOID) enters CLOSE_DANGER_RANGE (AABB surface gap),
      temporarily override the target to that bullet.  After the dodge,
      stickiness resumes only if ``prev_target_idx`` is still an active
      DESTROY object; otherwise a new random bottom-row enemy is chosen.
    """
    distances, _, _, _, _ = player_object_aabb_gaps(raw_observation, objects)
    active_mask = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]
    n = active_mask.shape[0]
    idxs = jnp.arange(n, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Bottom-row DESTROY candidates (formation enemies only; not UFO)
    # ------------------------------------------------------------------
    enemy_slot_mask = jnp.logical_and(
        idxs >= FIRST_ENEMY_SLOT,
        idxs < FIRST_ENEMY_SLOT + NUM_ENEMIES,
    )
    destroy_enemy_mask = jnp.logical_and(
        jnp.logical_and(active_mask, labels == LABEL_DESTROY),
        enemy_slot_mask,
    )
    any_destroy = jnp.any(destroy_enemy_mask)

    enemy_rows = (idxs - FIRST_ENEMY_SLOT) // ENEMY_COLS
    enemy_rows_for_bottom = jnp.where(destroy_enemy_mask, enemy_rows, -1)
    bottom_row = jnp.max(enemy_rows_for_bottom)
    bottom_mask = jnp.logical_and(destroy_enemy_mask, enemy_rows == bottom_row)

    # Uniform random pick among current bottom-row enemies.
    scores = jnp.where(
        bottom_mask, jax.random.uniform(rng, (n,)), jnp.asarray(-1.0)
    )
    random_bottom_idx = jnp.argmax(scores).astype(jnp.int32)

    # Sticky: keep prev_target_idx if it is still an active DESTROY object.
    prev_valid = jnp.logical_and(
        prev_target_idx >= 0,
        prev_target_idx < n,
    )
    prev_active = jnp.where(prev_valid, active_mask[prev_target_idx], False)
    prev_is_destroy = jnp.where(
        prev_valid, labels[prev_target_idx] == LABEL_DESTROY, False
    )
    keep_prev_destroy = jnp.logical_and(prev_active, prev_is_destroy)

    base_destroy_idx = jnp.where(keep_prev_destroy, prev_target_idx, random_bottom_idx)
    base_destroy_idx = jnp.where(
        any_destroy,
        base_destroy_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )

    # ------------------------------------------------------------------
    # AVOID danger override
    # ------------------------------------------------------------------
    avoid_mask = jnp.logical_and(active_mask, labels == LABEL_AVOID)
    close_avoid_mask = jnp.logical_and(avoid_mask, distances <= CLOSE_DANGER_RANGE)
    have_close_avoid = jnp.any(close_avoid_mask)

    avoid_distances = jnp.where(close_avoid_mask, distances, jnp.inf)
    best_avoid_idx = jnp.argmin(avoid_distances).astype(jnp.int32)

    target_idx = jnp.where(have_close_avoid, best_avoid_idx, base_destroy_idx)

    return target_idx.astype(jnp.int32)
