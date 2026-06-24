"""
Bottom-row sweep heuristic for Space Invaders.

Mission pattern:
  1. Find the lowest (bottommost) enemy row that still has active targets.
  2. Start at the leftmost enemy on that row.
  3. After each kill, advance to the next active enemy on the same row (left → right).
  4. When the row is cleared, repeat from the leftmost enemy on the new bottom row.

Reactive dodge:
  - Very close bullets (AVOID) within CLOSE_DANGER_RANGE temporarily override the mission.
  - After the dodge, resume the sweep mission (closest bottom-row enemy to the player in x).
"""

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_DESTROY

NO_TARGET_IDX = -1

# Only dodge when a bullet is extremely close to the player.
CLOSE_DANGER_RANGE = 40.0

NUM_ENEMIES = 36
ENEMY_COLS = 6
FIRST_ENEMY_SLOT = 1  # slot 0 = reserved meta NOOP


def _select_sweep_mission(
    ox_center,
    active_mask,
    labels,
    prev_target_idx,
    px_center,
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

    prev_valid = jnp.logical_and(
        prev_target_idx >= 0,
        prev_target_idx < n,
    )
    prev_label = jnp.where(prev_valid, labels[prev_target_idx], -1)
    prev_active = jnp.where(prev_valid, active_mask[prev_target_idx], False)
    prev_is_enemy_destroy = jnp.logical_and(
        prev_valid,
        jnp.logical_and(
            prev_label == LABEL_DESTROY,
            jnp.logical_and(
                prev_target_idx >= FIRST_ENEMY_SLOT,
                prev_target_idx < FIRST_ENEMY_SLOT + NUM_ENEMIES,
            ),
        ),
    )
    keep_prev = jnp.logical_and(prev_is_enemy_destroy, prev_active)

    prev_x = ox_center[jnp.maximum(prev_target_idx, 0)]
    next_right_mask = jnp.logical_and(on_sweep_row, ox_center > prev_x + 0.5)
    has_next_right = jnp.any(next_right_mask)
    next_right_idx = jnp.argmin(jnp.where(next_right_mask, sweep_x, jnp.inf)).astype(
        jnp.int32
    )

    advance_after_kill = jnp.logical_and(
        prev_is_enemy_destroy,
        jnp.logical_not(prev_active),
    )

    prev_is_avoid = jnp.logical_and(prev_valid, prev_label == LABEL_AVOID)
    x_dist_to_player = jnp.abs(ox_center - px_center)
    resume_idx = jnp.argmin(
        jnp.where(on_sweep_row, x_dist_to_player, jnp.inf)
    ).astype(jnp.int32)

    mission_idx = jnp.where(
        keep_prev,
        prev_target_idx,
        jnp.where(
            advance_after_kill,
            jnp.where(has_next_right, next_right_idx, leftmost_idx),
            jnp.where(prev_is_avoid, resume_idx, leftmost_idx),
        ),
    )
    mission_idx = jnp.where(
        any_enemies,
        mission_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )
    return mission_idx.astype(jnp.int32)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del rng

    ox = objects.x[-1]
    oy = objects.y[-1]
    active_mask = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    px_center = raw_observation.player.x[-1] + raw_observation.player.width[-1] / 2.0
    py_center = raw_observation.player.y[-1] + raw_observation.player.height[-1] / 2.0
    ox_center = ox + objects.width[-1] / 2.0
    oy_center = oy + objects.height[-1] / 2.0

    dx = px_center - ox_center
    dy = py_center - oy_center
    distances = jnp.sqrt(dx**2 + dy**2)

    mission_idx = _select_sweep_mission(
        ox_center, active_mask, labels, prev_target_idx, px_center
    )

    avoid_mask = jnp.logical_and(active_mask, labels == LABEL_AVOID)
    close_avoid_mask = jnp.logical_and(avoid_mask, distances <= CLOSE_DANGER_RANGE)
    have_close_avoid = jnp.any(close_avoid_mask)

    avoid_distances = jnp.where(close_avoid_mask, distances, jnp.inf)
    best_avoid_idx = jnp.argmin(avoid_distances).astype(jnp.int32)

    target_idx = jnp.where(have_close_avoid, best_avoid_idx, mission_idx)
    return target_idx.astype(jnp.int32)
