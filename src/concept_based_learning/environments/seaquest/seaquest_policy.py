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

# Distance (in pixels) within which the surface sub is considered "near"
SURFACE_SUB_NEAR_RANGE = 40.0

# Radius (in pixels) around the player within which we apply
# priority-based target selection over semantic labels.
PRIORITY_NEAR_RANGE = 60.0

# Distance (in pixels) that counts as "immediate danger". Within this range,
# AVOID and DESTROY targets should take precedence even when we're forcing
# surfacing.
CLOSE_DANGER_RANGE = 15.0

X_AXIS_WEIGHT = 1.0


def get_heuristic_target_temp(raw_observation, objects, prev_target_idx, rng):
    """
    Minimal Seaquest targeting rule:
    1. If oxygen is low, target the surface-waypoint dummy COLLECT object so the
       agent goes straight up to refill oxygen.
    2. If six divers have been collected, also target the surface dummy COLLECT
       object so the agent surfaces to score them, regardless of oxygen.
    3. Otherwise, pick a target based on local semantic priorities around the
       player (AVOID → DESTROY → COLLECT), falling back to nearest object.
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

    # When oxygen is low, we want to explicitly surface by chasing the
    # surface-waypoint dummy COLLECT object (index SURFACE_DUMMY_IDX). This
    # is required because the action mapper only allows going above the
    # surface-y threshold when this dummy is the current target.
    #
    # NOTE: We no longer use the "upper-most active object" or explicit
    # surface-submarine proximity here, because those targets are blocked
    # from actually surfacing by the action mapper's ascent guard.
    target_when_low = jnp.array(SURFACE_DUMMY_IDX, dtype=jnp.int32)

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

    # "Immediate danger" subset: very close AVOID/DESTROY objects should
    # override even when we are forcing the surface dummy as the target.
    avoid_close_mask = jnp.logical_and(avoid_mask, distances <= CLOSE_DANGER_RANGE)
    destroy_close_mask = jnp.logical_and(destroy_mask, distances <= CLOSE_DANGER_RANGE)
    danger_close_mask = jnp.logical_or(avoid_close_mask, destroy_close_mask)
    have_danger_close = jnp.any(danger_close_mask)
    danger_close_idx = _pick_closest(danger_close_mask)

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

    # Oxygen-based override (but allow immediate danger to take precedence)
    target_idx = jnp.where(
        oxygen_low,
        jnp.where(have_danger_close, danger_close_idx, target_when_low),
        priority_idx,
    )

    # If absolutely no active object exists, force NO_TARGET_IDX
    # Note: during oxygen-low/full-divers situations we may still want to
    # output the surface dummy target even if there are no other active objects.
    any_targetable = jnp.logical_or(any_active, jnp.logical_or(oxygen_low, have_full_divers))
    target_idx = jnp.where(
        any_targetable,
        target_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )

    # If we have rescued six divers, override with the surface dummy COLLECT
    # object to force surfacing behavior.
    target_idx = jnp.where(
        have_full_divers,
        jnp.where(
            have_danger_close,
            danger_close_idx,
            jnp.array(SURFACE_DUMMY_IDX, dtype=jnp.int32),
        ),
        target_idx,
    )

    return target_idx.astype(jnp.int32)



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
    
    # Base semantic weights
    weights = jnp.where(
        labels == LABEL_AVOID, 1000.0,
        jnp.where(
            labels == LABEL_DESTROY, 500.0,
            jnp.where(
                labels == LABEL_COLLECT, 100.0,
                0.0
            )
        )
    )

    # Hysteresis: Apply a 1.5x multiplier if it was the target in the last frame
    hysteresis_multiplier = jnp.where(idxs == prev_target_idx, 1.5, 1.0)

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

    collected_divers = raw_observation.collected_divers[-1]
    have_full_divers = collected_divers >= 6

    force_surface = jnp.logical_or(oxygen_low, have_full_divers)

    # ---------------------------------------------------------------------
    # Danger Overrides & Surface Deadlock Prevention
    # ---------------------------------------------------------------------
    is_threat = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)
    is_close_danger = jnp.logical_and(distances <= CLOSE_DANGER_RANGE, is_threat)
    is_close_danger = jnp.logical_and(is_close_danger, normal_active_mask)

    # If we are actively surfacing, the surface sub is no longer a "danger" (NOTE: might not work properly, we´ll see)
    is_surface_sub = (idxs == SURFACE_SUB_IDX)
    ignore_danger = jnp.logical_and(force_surface, is_surface_sub)
    
    valid_close_danger = jnp.logical_and(is_close_danger, jnp.logical_not(ignore_danger))
    have_valid_danger = jnp.any(valid_close_danger)

    # If there are valid close dangers, pick the one with the highest utility score
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