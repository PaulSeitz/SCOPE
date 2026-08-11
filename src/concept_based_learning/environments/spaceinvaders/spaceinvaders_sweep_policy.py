"""
Bottom-row sweep heuristic for Space Invaders.

Mission pattern:
  1. Find the lowest (bottommost) enemy row that still has active targets.
  2. Start at the leftmost enemy on that row.
  3. After each kill, advance to the next active enemy on the same row (left → right).
  4. When the row is cleared, repeat from the leftmost enemy on the new bottom row.

Reactive dodge:
  - Very close bullets (AVOID) within CLOSE_DANGER_RANGE (AABB surface gap)
    temporarily override the mission.
  - After the dodge, resume the locked sweep mission target (not nearest-in-x).
"""

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_DESTROY
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

NO_TARGET_IDX = -1

# Only dodge when a bullet is extremely close (AABB surface gap).
CLOSE_DANGER_RANGE = 40.0

NUM_ENEMIES = 36
ENEMY_COLS = 6
FIRST_ENEMY_SLOT = 1  # slot 0 = reserved meta NOOP


def _is_enemy_destroy_slot(idxs, labels, slot_idx):
    return jnp.logical_and(
        slot_idx >= FIRST_ENEMY_SLOT,
        jnp.logical_and(
            slot_idx < FIRST_ENEMY_SLOT + NUM_ENEMIES,
            labels[slot_idx] == LABEL_DESTROY,
        ),
    )


def _select_sweep_mission(
    ox_center,
    active_mask,
    labels,
    mission_target_idx,
):
    """Pick the next DESTROY enemy along the current bottom row."""
    n = active_mask.shape[0]
    idxs = jnp.arange(n, dtype=jnp.int32)

    enemy_slot_mask = jnp.logical_and(
        idxs >= FIRST_ENEMY_SLOT,
        idxs < FIRST_ENEMY_SLOT + NUM_ENEMIES,
    )
    destroy_enemy_mask = jnp.logical_and(
        jnp.logical_and(active_mask, labels == LABEL_DESTROY),
        enemy_slot_mask,
    )
    any_enemies = jnp.any(destroy_enemy_mask)

    game_enemy_idx = idxs - FIRST_ENEMY_SLOT
    enemy_rows = game_enemy_idx // ENEMY_COLS
    enemy_rows_for_bottom = jnp.where(destroy_enemy_mask, enemy_rows, -1)
    bottom_row = jnp.max(enemy_rows_for_bottom)

    on_sweep_row = jnp.logical_and(destroy_enemy_mask, enemy_rows == bottom_row)
    sweep_x = jnp.where(on_sweep_row, ox_center, jnp.inf)
    leftmost_idx = jnp.argmin(sweep_x).astype(jnp.int32)

    mission_valid = jnp.logical_and(
        mission_target_idx >= 0,
        mission_target_idx < n,
    )
    mission_is_enemy_destroy = jnp.where(
        mission_valid,
        _is_enemy_destroy_slot(idxs, labels, mission_target_idx),
        False,
    )
    mission_active = jnp.where(
        mission_valid,
        active_mask[mission_target_idx],
        False,
    )
    keep_mission = jnp.logical_and(mission_is_enemy_destroy, mission_active)

    mission_x = ox_center[jnp.maximum(mission_target_idx, 0)]
    next_right_mask = jnp.logical_and(on_sweep_row, ox_center > mission_x + 0.5)
    has_next_right = jnp.any(next_right_mask)
    next_right_idx = jnp.argmin(jnp.where(next_right_mask, sweep_x, jnp.inf)).astype(
        jnp.int32
    )

    advance_after_kill = jnp.logical_and(
        mission_is_enemy_destroy,
        jnp.logical_not(mission_active),
    )

    mission_idx = jnp.where(
        keep_mission,
        mission_target_idx,
        jnp.where(
            advance_after_kill,
            jnp.where(has_next_right, next_right_idx, leftmost_idx),
            leftmost_idx,
        ),
    )
    mission_idx = jnp.where(
        any_enemies,
        mission_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )
    return mission_idx.astype(jnp.int32)


def get_heuristic_target_with_mission(
    raw_observation,
    objects,
    prev_target_idx,
    mission_target_idx,
    rng,
):
    """
    Return (committed_target_idx, mission_target_idx).

    ``mission_target_idx`` is the sticky sweep target and survives AVOID
    interruptions; ``prev_target_idx`` is only used by callers for logging.
    """
    del prev_target_idx, rng

    active_mask = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    distances, _, _, ox_center, _ = player_object_aabb_gaps(raw_observation, objects)

    mission_idx = _select_sweep_mission(
        ox_center, active_mask, labels, mission_target_idx
    )

    avoid_mask = jnp.logical_and(active_mask, labels == LABEL_AVOID)
    close_avoid_mask = jnp.logical_and(avoid_mask, distances <= CLOSE_DANGER_RANGE)
    have_close_avoid = jnp.any(close_avoid_mask)

    avoid_distances = jnp.where(close_avoid_mask, distances, jnp.inf)
    best_avoid_idx = jnp.argmin(avoid_distances).astype(jnp.int32)

    target_idx = jnp.where(have_close_avoid, best_avoid_idx, mission_idx)
    return target_idx.astype(jnp.int32), mission_idx.astype(jnp.int32)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """Backward-compatible wrapper without external mission state."""
    target_idx, _mission_idx = get_heuristic_target_with_mission(
        raw_observation,
        objects,
        prev_target_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
        rng,
    )
    return target_idx
