"""
Heuristic meta-policy for Seaquest.

This module provides a minimal, hard-coded policy that:
- By default, selects a target purely from the current frame, with **no
  persistence**: the choice is recomputed every step.
- Within a local neighborhood around the player, it prefers objects by
  semantic label priority (AVOID → DESTROY → COLLECT), breaking ties by
  spatial distance to the player.
- When oxygen is low, it prefers the upper-most object, unless the surface
  submarine is nearby, in which case it targets that.
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

# Oxygen threshold below which we consider oxygen "low"
OXYGEN_LOW_THRESHOLD = 15.0

# Oxygen threshold below which we ignore immediate danger and commit to surfacing
CRITICAL_OXYGEN_THRESHOLD = 5.0

# Distance (in pixels) within which the surface sub is considered "near"
SURFACE_SUB_NEAR_RANGE = 40.0

# Distance (in pixels) that counts as "immediate danger". Within this range,
# AVOID and DESTROY targets should take precedence even when we're forcing
# surfacing, unless oxygen is critically low.
CLOSE_DANGER_RANGE = 25.0

# Distance beyond which we safely release a threat we were already targeting (spatial hysteresis).
DANGER_RELEASE_RANGE = 40.0

X_AXIS_WEIGHT = 1.0 # NOTE: deactivated for now


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Continuous utility Seaquest targeting rule.
    """
    # Extract current frame data
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    active_mask = objects.active[-1] > 0.5
    idxs = jnp.arange(active_mask.shape[0])

    # Exclude the surface-only dummy object from normal scoring
    normal_active_mask = jnp.logical_and(active_mask, idxs != SURFACE_DUMMY_IDX)

    # Calculate centers
    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    dx = px_center - ox_center
    dy = py_center - oy_center

    # True Euclidean distance for immediate danger checks
    distances = jnp.sqrt(dx ** 2 + dy ** 2)

    # Weighted distance to prioritize horizontal threats (coming at the player) TODO: make config param
    weighted_distances = jnp.sqrt((X_AXIS_WEIGHT * dx) ** 2 + dy ** 2)

    # ---------------------------------------------------------------------
    # Continuous Utility Scoring
    # ---------------------------------------------------------------------
    labels = objects.semantic_label[-1]

    # Semantic base weights
    avoid_weight = 750.0
    destroy_weight = 500.0

    # Dynamically boost COLLECT weight as we approach 6 divers so that
    # collection cleanly outranks destruction, especially near surfacing.
    collected_divers = raw_observation.collected_divers[-1]
    collected_divers_f = collected_divers.astype(jnp.float32)
    collect_weight = 600.0 + 100.0 * jnp.clip(collected_divers_f / 6.0, 0.0, 1.0)

    weights = jnp.where(
        labels == LABEL_AVOID,
        avoid_weight,
        jnp.where(
            labels == LABEL_DESTROY,
            destroy_weight,
            jnp.where(labels == LABEL_COLLECT, collect_weight, 0.0),
        ),
    )

    # Hysteresis: Force commitment—massive multiplier so we lock on until target is dead/collected or emergency override
    hysteresis_multiplier = jnp.where(idxs == prev_target_idx, 20.0, 1.0)

    # Compute continuous score: (Weight / (Distance + epsilon)) * Hysteresis
    epsilon = 1.0
    scores = (weights / (weighted_distances + epsilon)) * hysteresis_multiplier

    # Mask out inactive objects and the surface dummy
    masked_scores = jnp.where(normal_active_mask, scores, -jnp.inf)
    best_normal_idx = jnp.argmax(masked_scores).astype(jnp.int32)

    # ---------------------------------------------------------------------
    # Surfacing Logic (Oxygen / Divers)
    # ---------------------------------------------------------------------
    oxygen = raw_observation.oxygen_level[-1]
    oxygen_low = oxygen < OXYGEN_LOW_THRESHOLD

    have_full_divers = collected_divers >= 6

    force_surface = jnp.logical_or(oxygen_low, have_full_divers)

    # ---------------------------------------------------------------------
    # Danger Overrides & Surface Deadlock Prevention
    # ---------------------------------------------------------------------
    is_threat = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)

    # Dynamic threshold (spatial hysteresis). If we were already targeting this threat,
    # use the larger release range so we don't flicker; otherwise use the strict close range.
    dynamic_danger_thresholds = jnp.where(
        idxs == prev_target_idx,
        DANGER_RELEASE_RANGE,
        CLOSE_DANGER_RANGE,
    )

    is_close_danger = jnp.logical_and(distances <= dynamic_danger_thresholds, is_threat)
    is_close_danger = jnp.logical_and(is_close_danger, normal_active_mask)

    # If we are actively surfacing, the surface sub is no longer a "danger"
    is_surface_sub = (idxs == SURFACE_SUB_IDX)
    ignore_danger = jnp.logical_and(force_surface, is_surface_sub)

    valid_close_danger = jnp.logical_and(
        is_close_danger, jnp.logical_not(ignore_danger)
    )

    # When oxygen is critically low, ignore immediate dangers and commit to surfacing.
    oxygen_critical = oxygen < CRITICAL_OXYGEN_THRESHOLD
    have_valid_danger = jnp.logical_and(jnp.any(valid_close_danger), jnp.logical_not(oxygen_critical))

    # If there are valid close dangers, pick the one with the highest utility score.
    # scores already includes the hysteresis multiplier, so we stick to the threat we're engaging.
    danger_scores = jnp.where(valid_close_danger, scores, -jnp.inf)
    best_danger_idx = jnp.argmax(danger_scores).astype(jnp.int32)

    # ---------------------------------------------------------------------
    # Final Target Selection
    # ---------------------------------------------------------------------
    
    # 1. Default to the highest scoring normal object
    target_idx = best_normal_idx

    # 2. Override with the surface waypoint if we need to surface
    target_idx = jnp.where(
        force_surface, 
        jnp.array(SURFACE_DUMMY_IDX, dtype=jnp.int32), 
        target_idx
    )

    # 3. Absolute override for immediate, valid danger
    target_idx = jnp.where(have_valid_danger, best_danger_idx, target_idx)

    # 4. Fallback if the screen is empty
    any_active = jnp.any(normal_active_mask)
    any_targetable = jnp.logical_or(any_active, force_surface)

    target_idx = jnp.where(
        any_targetable,
        target_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )

    return target_idx.astype(jnp.int32)