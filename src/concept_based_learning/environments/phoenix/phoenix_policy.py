import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)

NO_TARGET_IDX = -1
# Danger range for AVOID objects (in pixels / world units).
CLOSE_DANGER_RANGE = 40.0
X_AXIS_WEIGHT = 1.0


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Heuristic Phoenix targeting rule.

    - By default, select the closest active object (any label).
    - If any AVOID object comes within CLOSE_DANGER_RANGE, override and
      select the closest such AVOID object instead.
    """
    # Player position (last frame)
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    # Object positions (last frame)
    ox = objects.x[-1]
    oy = objects.y[-1]

    # Treat objects in a death-animation state (state >= 4) as inactive.
    # This relies on the environment encoding dying enemies via their state.
    state = getattr(objects, "state", None)
    if state is not None:
        last_state = state[-1]
        is_dying = last_state >= 4
        base_active = objects.active[-1] > 0.5
        active_mask = jnp.logical_and(base_active, jnp.logical_not(is_dying))
    else:
        active_mask = objects.active[-1] > 0.5

    # Use centers for distance computation
    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    dx = px_center - ox_center
    dy = py_center - oy_center

    # Slight x-axis weighting kept, but overall we just care about proximity.
    distances = jnp.sqrt((X_AXIS_WEIGHT * dx) ** 2 + dy ** 2)

    labels = objects.semantic_label[-1]

    # ------------------------------------------------------------------
    # Base rule: pick the closest active DESTROY object (boss/enemies).
    # Boss blocks and other NOOP objects are ignored here so that the
    # destroy skill focuses on the actual boss when it is present.
    # ------------------------------------------------------------------
    is_destroy = labels == LABEL_DESTROY
    valid_destroy = jnp.logical_and(active_mask, is_destroy)

    # Mask out non-destroy or inactive objects by giving them a huge distance.
    large_distance = jnp.max(distances) + 1.0
    masked_distances = jnp.where(valid_destroy, distances, large_distance)

    has_any_destroy = jnp.any(valid_destroy)
    best_destroy_idx = jnp.argmin(masked_distances).astype(jnp.int32)

    # ------------------------------------------------------------------
    # Danger override: if any AVOID object is very close, track it instead.
    # ------------------------------------------------------------------
    is_avoid = labels == LABEL_AVOID
    is_close_avoid = jnp.logical_and(is_avoid, distances <= CLOSE_DANGER_RANGE)
    valid_close_avoid = jnp.logical_and(is_close_avoid, active_mask)
    have_valid_close_avoid = jnp.any(valid_close_avoid)

    # Among close AVOID objects, pick the nearest one.
    avoid_distances = jnp.where(valid_close_avoid, distances, large_distance)
    best_avoid_idx = jnp.argmin(avoid_distances).astype(jnp.int32)

    # Apply danger override on top of destroy targeting.
    target_idx = jnp.where(have_valid_close_avoid, best_avoid_idx, best_destroy_idx)

    # Fallback: if there is nothing to destroy and no close AVOID object, emit NO_TARGET_IDX.
    any_targetable = jnp.logical_or(has_any_destroy, have_valid_close_avoid)
    target_idx = jnp.where(any_targetable, target_idx, jnp.array(NO_TARGET_IDX, dtype=jnp.int32))

    return target_idx.astype(jnp.int32)