"""
Heuristic meta-policy for Seaquest.

This module provides a minimal, hard-coded policy that:
- By default, picks a random active object and keeps it as the target
  until it disappears.
- When oxygen is low, prefers the upper-most object,
  unless the surface submarine is nearby, in which case it targets that.
- When six divers have been collected, it targets the special COLLECT
  surface-waypoint object to surface.
"""

import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)

# When no object is on screen, return this; downstream should execute NOOP.
NO_TARGET_IDX = -1

# Index of the custom surface-only waypoint object; it must never be chosen
SURFACE_DUMMY_IDX = 33

# Index of the real surface submarine in the flattened object list
SURFACE_SUB_IDX = 28

# Oxygen threshold (in [0, 1]) below which we consider oxygen "low"
OXYGEN_LOW_THRESHOLD = 0.2

# Distance (in pixels) within which the surface sub is considered "near"
SURFACE_SUB_NEAR_RANGE = 40.0

# Radius (in pixels) around the player within which we apply
# priority-based target selection over semantic labels.
PRIORITY_NEAR_RANGE = 80.0


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Minimal Seaquest targeting rule:
    1. If six divers have been collected, target the surface dummy COLLECT
       object so the agent surfaces.
    2. Otherwise, if oxygen is low:
       - Target the upper-most active object,
       - Unless the surface submarine is near, in which case target it.
    3. Otherwise, pick a random active object and keep it as the target
       until it disappears.
    4. If no active object exists, return NO_TARGET_IDX.
    """
    # Extract current frame data (top-left coordinates)
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    active_mask = objects.active[-1] > 0.5

    # Exclude the surface-only dummy object from all targeting logic
    idxs = jnp.arange(active_mask.shape[0])
    normal_active_mask = jnp.logical_and(
        active_mask,
        idxs != SURFACE_DUMMY_IDX,
    )

    # Calculate distances between centers of player and objects
    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    # Raw deltas
    dx = px_center - ox_center
    dy = py_center - oy_center

    # Euclidean distance (kept for existing logic like surface proximity)
    distances = jnp.sqrt(dx ** 2 + dy ** 2)

    # Weighted distance used for generic target selection.
    # We overweight the x-axis so that objects that are closer horizontally
    # (i.e., coming towards the player) are prioritized over vertical proximity.
    X_AXIS_WEIGHT = 2.0
    weighted_distances = jnp.sqrt((X_AXIS_WEIGHT * dx) ** 2 + dy ** 2)

    num_objects = active_mask.shape[0]
    any_active = jnp.any(normal_active_mask)

    # Upper-most active object (smallest y-center)
    masked_oy_center = jnp.where(normal_active_mask, oy_center, jnp.inf)
    upper_idx = jnp.argmin(masked_oy_center).astype(jnp.int32)

    # Oxygen-based override (from global features)
    oxygen = raw_observation.oxygen_level[-1]
    oxygen_low = oxygen < OXYGEN_LOW_THRESHOLD

    # Number of currently collected divers (0–6); when 6, we want to surface
    collected_divers = raw_observation.collected_divers[-1]
    have_full_divers = collected_divers >= 6

    # Surface submarine proximity check
    has_surface_slot = SURFACE_SUB_IDX < num_objects
    surface_active = jnp.logical_and(
        has_surface_slot,
        active_mask[jnp.minimum(SURFACE_SUB_IDX, num_objects - 1)],
    )
    surface_distance = jnp.where(
        surface_active,
        distances[jnp.minimum(SURFACE_SUB_IDX, num_objects - 1)],
        jnp.inf,
    )
    surface_near = surface_distance <= SURFACE_SUB_NEAR_RANGE

    surface_idx = jnp.array(SURFACE_SUB_IDX, dtype=jnp.int32)
    target_when_low = jnp.where(surface_near, surface_idx, upper_idx)

    # Nearest active object selection (excluding the surface dummy) based on
    # the weighted distance defined above.
    masked_weighted_distances = jnp.where(
        normal_active_mask, weighted_distances, jnp.inf
    )
    base_idx = jnp.argmin(masked_weighted_distances).astype(jnp.int32)

    # ---------------------------------------------------------------------
    # Priority-based selection within a local neighborhood.
    #
    # Among objects that are both active and "near" the player (within
    # PRIORITY_NEAR_RANGE), we prioritize labels as:
    #   1. AVOID
    #   2. DESTROY
    #   3. COLLECT
    # If none of the nearby objects match these labels, we fall back to
    # the generic nearest-object choice above.
    # ---------------------------------------------------------------------
    labels = objects.semantic_label[-1]

    near_mask = distances <= PRIORITY_NEAR_RANGE
    base_priority_mask = jnp.logical_and(normal_active_mask, near_mask)

    avoid_mask = jnp.logical_and(base_priority_mask, labels == LABEL_AVOID)
    destroy_mask = jnp.logical_and(base_priority_mask, labels == LABEL_DESTROY)
    collect_mask = jnp.logical_and(base_priority_mask, labels == LABEL_COLLECT)

    def _pick_closest(mask):
        masked = jnp.where(mask, weighted_distances, jnp.inf)
        return jnp.argmin(masked).astype(jnp.int32)

    have_avoid = jnp.any(avoid_mask)
    have_destroy = jnp.any(destroy_mask)
    have_collect = jnp.any(collect_mask)

    avoid_idx = _pick_closest(avoid_mask)
    destroy_idx = _pick_closest(destroy_mask)
    collect_idx = _pick_closest(collect_mask)

    priority_idx = jnp.where(
        have_avoid,
        avoid_idx,
        jnp.where(
            have_destroy,
            destroy_idx,
            jnp.where(
                have_collect,
                collect_idx,
                base_idx,
            ),
        ),
    )

    # Oxygen-based override
    target_idx = jnp.where(oxygen_low, target_when_low, priority_idx)

    # If absolutely no active object exists, force NO_TARGET_IDX
    target_idx = jnp.where(
        any_active,
        target_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )

    # If we have rescued six divers, override with the surface dummy COLLECT
    # object to force surfacing behavior.
    target_idx = jnp.where(
        have_full_divers,
        jnp.array(SURFACE_DUMMY_IDX, dtype=jnp.int32),
        target_idx,
    )

    return target_idx.astype(jnp.int32)
