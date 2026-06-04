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
# Last game-object slot (index 0 is reserved meta NOOP).
SURFACE_DUMMY_IDX = 34

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


def _surface_required_from_raw(raw_observation):
    """True when surfacing is required (low O2 or six divers). Supports batched raw obs."""
    last = jax.tree_util.tree_map(
        lambda x: x[..., -1] if jnp.ndim(x) > 0 else x, raw_observation
    )
    return jnp.logical_or(
        last.oxygen_level < OXYGEN_LOW_THRESHOLD,
        last.collected_divers >= 6,
    )


def apply_meta_mask_overrides(raw_obs, active_masks, config, objects=None):
    del objects
    if active_masks.shape[-1] <= SURFACE_DUMMY_IDX:
        return active_masks

    always_active = bool(config.get("ALWAYS_ACTIVE_SURFACE_TARGET", False))
    force_surface = _surface_required_from_raw(raw_obs)
    if active_masks.ndim == 1:
        force_surface = jnp.reshape(force_surface, ())
        if always_active:
            return active_masks.at[SURFACE_DUMMY_IDX].set(True)
        return active_masks.at[SURFACE_DUMMY_IDX].set(
            jnp.logical_and(active_masks[SURFACE_DUMMY_IDX], force_surface)
        )

    if always_active:
        return active_masks.at[:, SURFACE_DUMMY_IDX].set(True)
    return active_masks.at[:, SURFACE_DUMMY_IDX].set(
        jnp.logical_and(active_masks[:, SURFACE_DUMMY_IDX], force_surface)
    )


def get_forced_target_override(raw_obs, current_targets, config, rng=None):
    if current_targets.ndim == 0:
        return current_targets

    force_surface = _surface_required_from_raw(raw_obs)

    # Optional exploration assist: sample the surface target while forcing is true.
    surface_pick_prob = float(config.get("SURFACE_PICK_PROB", 0.0))
    if rng is not None and surface_pick_prob > 0.0:
        rng_surface = jax.random.split(rng, current_targets.shape[0])
        pick_surface = jax.vmap(lambda r: jax.random.uniform(r, ()) < surface_pick_prob)(
            rng_surface
        )
        pick_surface = jnp.logical_and(force_surface, pick_surface)
        current_targets = jnp.where(
            pick_surface,
            jnp.array(SURFACE_DUMMY_IDX, dtype=current_targets.dtype),
            current_targets,
        )

    force_override = bool(config.get("FORCE_SURFACE_OVERRIDE", False))
    if force_override:
        current_targets = jnp.where(
            force_surface,
            jnp.array(SURFACE_DUMMY_IDX, dtype=current_targets.dtype),
            current_targets,
        )
    return current_targets


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Seaquest targeting rule with local semantic priority.
    """
    del prev_target_idx, rng
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
    # Local semantic-priority target selection
    # ---------------------------------------------------------------------
    labels = objects.semantic_label[-1]

    avoid_mask = jnp.logical_and(normal_active_mask, labels == LABEL_AVOID)
    destroy_mask = jnp.logical_and(normal_active_mask, labels == LABEL_DESTROY)
    collect_mask = jnp.logical_and(normal_active_mask, labels == LABEL_COLLECT)

    avoid_dist = jnp.where(avoid_mask, weighted_distances, jnp.inf)
    destroy_dist = jnp.where(destroy_mask, weighted_distances, jnp.inf)
    collect_dist = jnp.where(collect_mask, weighted_distances, jnp.inf)

    best_avoid_idx = jnp.argmin(avoid_dist).astype(jnp.int32)
    best_destroy_idx = jnp.argmin(destroy_dist).astype(jnp.int32)
    best_collect_idx = jnp.argmin(collect_dist).astype(jnp.int32)

    any_avoid = jnp.any(avoid_mask)
    any_destroy = jnp.any(destroy_mask)
    # Priority: AVOID > DESTROY > COLLECT
    best_normal_idx = jnp.where(
        any_avoid,
        best_avoid_idx,
        jnp.where(any_destroy, best_destroy_idx, best_collect_idx),
    )

    # ---------------------------------------------------------------------
    # Surfacing Logic (Oxygen / Divers)
    # ---------------------------------------------------------------------
    oxygen = raw_observation.oxygen_level[-1]
    oxygen_low = oxygen < OXYGEN_LOW_THRESHOLD

    collected_divers = raw_observation.collected_divers[-1]
    have_full_divers = collected_divers >= 6

    force_surface = jnp.logical_or(oxygen_low, have_full_divers)

    # ---------------------------------------------------------------------
    # Danger override
    # ---------------------------------------------------------------------
    is_threat = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)
    is_close_danger = jnp.logical_and(distances <= CLOSE_DANGER_RANGE, is_threat)
    is_close_danger = jnp.logical_and(is_close_danger, normal_active_mask)

    # If we are actively surfacing, ignore the real surface sub as a danger
    is_surface_sub = (idxs == SURFACE_SUB_IDX)
    ignore_danger = jnp.logical_and(force_surface, is_surface_sub)
    valid_close_danger = jnp.logical_and(
        is_close_danger, jnp.logical_not(ignore_danger)
    )
    have_valid_danger = jnp.any(valid_close_danger)
    danger_dist = jnp.where(valid_close_danger, distances, jnp.inf)
    best_danger_idx = jnp.argmin(danger_dist).astype(jnp.int32)

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

    # 3. Immediate danger override — but never interrupt an active surfacing commit.
    # (Low O2 / full diver rescue must reach y==46; dodging nearby threats in the
    # breathing zone leaves the sub stuck at y=47–52 with no refill.)
    suppress_danger = force_surface
    target_idx = jnp.where(
        jnp.logical_and(have_valid_danger, jnp.logical_not(suppress_danger)),
        best_danger_idx,
        target_idx,
    )

    # 4. Fallback if the screen is empty
    any_active = jnp.any(normal_active_mask)
    any_targetable = jnp.logical_or(any_active, force_surface)

    target_idx = jnp.where(
        any_targetable,
        target_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )

    return target_idx.astype(jnp.int32)