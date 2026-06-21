import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)

# When no object is on screen, return this; downstream should execute NOOP.
NO_TARGET_IDX = -1

# Distance (in pixels) that counts as "immediate danger" for dodging bullets.
CLOSE_DANGER_RANGE = 20.0

# Distance (in pixels) below which a bullet is considered past / no longer
# dangerous (behind or beside the player).  Once all close bullets clear this
# threshold the heuristic resumes the previously locked DESTROY target.
DANGER_CLEAR_RANGE = 30.0

X_AXIS_WEIGHT = 1.0


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Space Invaders targeting heuristic with persistent DESTROY selection.

    - Lock onto a DESTROY target (enemy / UFO) and keep it until it
      disappears.  Only switch to a fresher DESTROY target when the locked
      one goes inactive.
    - If a bullet (AVOID) enters CLOSE_DANGER_RANGE, temporarily override
      the target to that bullet.  Once all bullets are outside
      DANGER_CLEAR_RANGE, resume the previously locked DESTROY target so
      the DESTROY skill sees an uninterrupted trajectory.
    - When no DESTROY target has been locked yet, fall back to the topmost
      active enemy.
    """
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    active_mask = objects.active[-1] > 0.5

    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    dx = px_center - ox_center
    dy = py_center - oy_center
    distances = jnp.sqrt(dx**2 + dy**2)
    weighted_distances = jnp.sqrt((X_AXIS_WEIGHT * dx) ** 2 + dy**2)

    labels = objects.semantic_label[-1]

    # ------------------------------------------------------------------
    # DESTROY candidates
    # ------------------------------------------------------------------
    destroy_mask = jnp.logical_and(active_mask, labels == LABEL_DESTROY)
    any_destroy = jnp.any(destroy_mask)

    # Fallback: topmost active enemy (smallest y-center).
    destroy_oy = jnp.where(destroy_mask, oy_center, jnp.inf)
    top_destroy_idx = jnp.argmin(destroy_oy).astype(jnp.int32)

    # Sticky: keep prev_target_idx if it is still an active DESTROY object.
    prev_valid = jnp.logical_and(
        prev_target_idx >= 0,
        prev_target_idx < active_mask.shape[0],
    )
    prev_active = jnp.where(prev_valid, active_mask[prev_target_idx], False)
    prev_is_destroy = jnp.where(
        prev_valid, labels[prev_target_idx] == LABEL_DESTROY, False
    )
    keep_prev_destroy = jnp.logical_and(prev_active, prev_is_destroy)

    base_destroy_idx = jnp.where(keep_prev_destroy, prev_target_idx, top_destroy_idx)
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

    # ------------------------------------------------------------------
    # Final selection: AVOID danger overrides locked DESTROY; once the
    # danger clears we fall back to the locked DESTROY target.
    # ------------------------------------------------------------------
    target_idx = jnp.where(have_close_avoid, best_avoid_idx, base_destroy_idx)

    return target_idx.astype(jnp.int32)
