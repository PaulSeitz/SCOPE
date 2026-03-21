import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)

# When no object is on screen, return this; downstream should execute NOOP.
NO_TARGET_IDX = -1

# Distance (in pixels) that counts as "immediate danger" for dodging bullets
CLOSE_DANGER_RANGE = 20.0

# Set x-axis weight to normal Euclidean distance
X_AXIS_WEIGHT = 1.0


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Simpler Space Invaders targeting rule:

    - By default, always pursue the **closest DESTROY object** (enemy / UFO).
    - If an incoming bullet (AVOID object) is within `CLOSE_DANGER_RANGE`,
      temporarily switch target to that bullet to dodge.
    - If the topmost enemy/UFO is present, prefer it over the nearest enemy.
    """
    # Extract current frame data
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    active_mask = objects.active[-1] > 0.5
    idxs = jnp.arange(active_mask.shape[0])

    # Calculate distances from the centers of the objects
    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    dx = px_center - ox_center
    dy = py_center - oy_center

    # True Euclidean distance for immediate danger checks
    distances = jnp.sqrt(dx**2 + dy**2)

    # Weighted distance (keep standard Euclidean here)
    weighted_distances = jnp.sqrt((X_AXIS_WEIGHT * dx) ** 2 + dy**2)

    labels = objects.semantic_label[-1]

    # ------------------------------------------------------------------
    # Base target: closest DESTROY object (enemy or UFO)
    # ------------------------------------------------------------------
    destroy_mask = jnp.logical_and(active_mask, labels == LABEL_DESTROY)
    any_destroy = jnp.any(destroy_mask)

    # Nearest DESTROY object by distance
    destroy_distances = jnp.where(destroy_mask, weighted_distances, jnp.inf)
    nearest_destroy_idx = jnp.argmin(destroy_distances).astype(jnp.int32)

    # "Top enemy": smallest y-center among DESTROY objects (includes UFO)
    destroy_oy = jnp.where(destroy_mask, oy_center, jnp.inf)
    top_destroy_idx = jnp.argmin(destroy_oy).astype(jnp.int32)

    # Prefer the topmost enemy when any DESTROY object exists.
    # (UFO lives at the very top and will naturally be selected here.)
    base_destroy_idx = jnp.where(any_destroy, top_destroy_idx, nearest_destroy_idx)

    # ------------------------------------------------------------------
    # Danger override: bullets (AVOID) very close to the player
    # ------------------------------------------------------------------
    avoid_mask = jnp.logical_and(active_mask, labels == LABEL_AVOID)
    close_avoid_mask = jnp.logical_and(avoid_mask, distances <= CLOSE_DANGER_RANGE)
    have_close_avoid = jnp.any(close_avoid_mask)

    # Among close bullets, pick the nearest one
    avoid_distances = jnp.where(close_avoid_mask, distances, jnp.inf)
    best_avoid_idx = jnp.argmin(avoid_distances).astype(jnp.int32)

    # ------------------------------------------------------------------
    # Final target selection
    # ------------------------------------------------------------------
    # 1) Default to DESTROY target if any exist, otherwise "no target"
    target_idx = jnp.where(
        any_destroy,
        base_destroy_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )

    # 2) If there is an immediate danger bullet, override with that AVOID object
    target_idx = jnp.where(have_close_avoid, best_avoid_idx, target_idx)

    return target_idx.astype(jnp.int32)
