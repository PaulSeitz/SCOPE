"""
Heuristic meta-policy for Seaquest.

This module provides a simple, hard-coded policy that selects a target
object index given the raw Seaquest observation and object metadata.
"""

import jax
import jax.numpy as jnp

# When no object is on screen (and not surfacing), return this; downstream should execute NOOP.
NO_TARGET_IDX = -1

# Semantic Labels for readability
LABEL_DESTROY = 1
LABEL_COLLECT = 2
LABEL_AVOID = 3

def get_heuristic_target(raw_observation, objects):
    """
    Heuristic meta-policy:
    1. Default: Target the closest active object.
    2. Surfacing Override: If oxygen < 10 or 6 divers collected, target the surface dummy object.
    3. Surface Keep-Out: If drifting too close to the surface without intent, pull down.
    4. No targets: If no object is on screen and not surfacing, return NO_TARGET_IDX.
    """
    # 1. Extract current frame data (top-left coordinates)
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    active_mask = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    # Calculate Euclidean distance between centers of player and objects
    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    distances = jnp.sqrt((px_center - ox_center)**2 + (py_center - oy_center)**2)

    # Exclude surface dummy (index 33) from normal spatial selection
    SURFACE_DUMMY_IDX = 33
    exclude_surface_from_normal = jnp.arange(distances.shape[0]) != SURFACE_DUMMY_IDX
    normal_active_mask = jnp.logical_and(active_mask, exclude_surface_from_normal)

    # valid_distances masks out inactive and dummy objects
    valid_distances = jnp.where(normal_active_mask, distances, jnp.inf)

    # --- Rule 1: Default closest object ---
    closest_idx = jnp.argmin(valid_distances)
    target_idx = closest_idx

    # --- Rule 2: Surfacing Logic ---
    sub_x = objects.x[-1, 28]
    sub_active = objects.active[-1, 28] > 0.5
    sub_blocking = jnp.logical_and(sub_active, sub_x > px)

    oxygen = raw_observation.oxygen_level[-1]
    divers_collected = raw_observation.collected_divers[-1]

    oxygen_prev = raw_observation.oxygen_level[-2]
    oxygen_increasing = oxygen > oxygen_prev

    SURFACE_Y_THRESHOLD = 40
    at_surface = py <= SURFACE_Y_THRESHOLD
    already_refilling = jnp.logical_or(oxygen_increasing, at_surface)
    needs_air = jnp.logical_and(oxygen < 10, jnp.logical_not(already_refilling))
    full_capacity = divers_collected >= 6

    should_surface = jnp.logical_or(needs_air, full_capacity)
    safe_to_surface = jnp.logical_and(should_surface, jnp.logical_not(sub_blocking))

    # Apply surfacing target
    target_idx = jnp.where(safe_to_surface, SURFACE_DUMMY_IDX, target_idx)

    # --- Rule 3: Surface Keep-Out Zone ---
    # If the agent is above Y=45 and NOT trying to surface, it's in danger of accidental death.
    SURFACE_DANGER_Y = 45.0
    in_surface_danger = jnp.logical_and(py < SURFACE_DANGER_Y, jnp.logical_not(safe_to_surface))

    # Find the lowest active object (highest Y value) to pull the agent back down
    valid_y = jnp.where(normal_active_mask, objects.y[-1], -jnp.inf)
    lowest_obj_idx = jnp.argmax(valid_y)

    # Override with the lowest object if in danger
    target_idx = jnp.where(in_surface_danger, lowest_obj_idx, target_idx)

    # --- Rule 4: No targets fallback ---
    # If no normal targets exist and we aren't surfacing, execute NOOP
    target_idx = jnp.where(
        jnp.logical_and(jnp.logical_not(jnp.any(normal_active_mask)), jnp.logical_not(safe_to_surface)),
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
        target_idx
    )

    return target_idx.astype(jnp.int32)
