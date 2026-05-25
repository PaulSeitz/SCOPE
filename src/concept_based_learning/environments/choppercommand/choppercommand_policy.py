import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)
# When no object is on screen, return this; downstream should execute NOOP.
NO_TARGET_IDX = -1

# Distance (in pixels) that counts as "immediate danger". Within this range,
# AVOID and DESTROY targets should take precedence even when we're forcing
# surfacing.
CLOSE_DANGER_RANGE = 15.0

# Maximum distance (in pixels) at which we consider enemies as valid targets.
# This helps avoid chasing far-off or effectively off-screen enemies.
MAX_TARGET_DISTANCE = 150.0


X_AXIS_WEIGHT = 1.0  # NOTE: can be used, but with 1.0 it already trained very successfully


def apply_meta_mask_overrides(raw_obs, active_masks, config, objects=None):
    del raw_obs, config, objects
    return active_masks


def get_forced_target_override(raw_obs, current_targets, config, rng=None):
    return current_targets


# NOTE: taken more or less from seaquest_policy.py, e.g. also using hysteresis but without surfacing logic
def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Continuous utility ChopperCommand targeting rule.
    """
    # Extract current frame data
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    labels = objects.semantic_label[-1]
    idxs = jnp.arange(objects.active[-1].shape[0])

    # --- Explicit On-Screen Masking ---
    # Determine what is actually on the screen (0 to 160)
    is_on_screen = (ox + objects.width[-1] > 0) & (ox < 160)
    is_waypoint = (labels == LABEL_COLLECT)
    
    # An object is a valid target if it is on-screen OR if it is our guide waypoint
    valid_spatial_mask = jnp.logical_or(is_on_screen, is_waypoint)
    
    # Combine the base active flag with our spatial restrictions
    active_mask = (objects.active[-1] > 0.5) & valid_spatial_mask
    # ---------------------------------------

    # Sanity check: all per-object arrays must share the same length.
    num_objects = active_mask.shape[0]
    if (
        (ox.shape[0] != num_objects)
        or (oy.shape[0] != num_objects)
        or (objects.width[-1].shape[0] != num_objects)
        or (objects.height[-1].shape[0] != num_objects)
        or (labels.shape[0] != num_objects)
        or (idxs.shape[0] != num_objects)
    ):
        raise ValueError("ChopperCommand heuristic received inconsistent object-axis shapes.")

    # Calculate centers
    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    dx = px_center - ox_center
    dy = py_center - oy_center

    # True Euclidean distance for immediate danger checks
    distances = jnp.sqrt(dx ** 2 + dy ** 2)

    # Weighted distance to prioritize horizontal threats
    weighted_distances = jnp.sqrt((X_AXIS_WEIGHT * dx) ** 2 + dy ** 2)

    # ---------------------------------------------------------------------
    # Continuous Utility Scoring
    # ---------------------------------------------------------------------

    is_enemy = labels == LABEL_DESTROY

    # Restrict normal targets to active enemies that are within a reasonable distance
    in_range = weighted_distances <= MAX_TARGET_DISTANCE
    normal_active_mask = jnp.logical_and(jnp.logical_and(active_mask, is_enemy), in_range)

    # Base semantic weights
    weights = jnp.where(
        labels == LABEL_AVOID, 0.0,
        jnp.where(
            labels == LABEL_DESTROY, 500.0,
            jnp.where(
                labels == LABEL_COLLECT, 100.0,
                0.0
            )
        )
    )

    # Hysteresis
    hysteresis_multiplier = jnp.where(idxs == prev_target_idx, 1.5, 1.0)

    epsilon = 1.0
    scores = (weights / (weighted_distances + epsilon)) * hysteresis_multiplier

    masked_scores = jnp.where(normal_active_mask, scores, -jnp.inf)
    best_normal_idx = jnp.argmax(masked_scores).astype(jnp.int32)

    # ---------------------------------------------------------------------
    # Danger Overrides
    # ---------------------------------------------------------------------
    is_threat = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)
    is_close_danger = jnp.logical_and(distances <= CLOSE_DANGER_RANGE, is_threat)
    is_close_danger = jnp.logical_and(is_close_danger, active_mask)

    have_valid_danger = jnp.any(is_close_danger)

    # For danger overrides, ignore hysteresis and semantic weights and
    # simply pick the closest valid threat inside the danger radius.
    # Use +inf for non-danger objects so argmin selects among dangers only.
    danger_distances = jnp.where(is_close_danger, distances, jnp.inf)
    best_danger_idx = jnp.argmin(danger_distances).astype(jnp.int32)

    # ---------------------------------------------------------------------
    # Final Target Selection
    # ---------------------------------------------------------------------
    
    # 1. Default to the highest scoring normal object
    target_idx = best_normal_idx
    
    # 2. Absolute override for immediate, valid danger
    target_idx = jnp.where(have_valid_danger, best_danger_idx, target_idx)

    # 3. If the screen is empty (no active enemies and no immediate danger),
    #    explicitly return NO_TARGET_IDX to trigger a NOOP downstream.
    no_enemies = jnp.all(normal_active_mask == 0)
    target_idx = jnp.where(
        no_enemies & jnp.logical_not(have_valid_danger),
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
        target_idx,
    )

    return target_idx.astype(jnp.int32)


def debug_heuristic_snapshot(raw_observation, objects, prev_target_idx):
    """
    Helper for inspecting the ChopperCommand heuristic target selection.
    Returns a dictionary with the main intermediate quantities so we can
    check masking / distances / normalization on real frames.
    """
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    labels = objects.semantic_label[-1]
    idxs = jnp.arange(objects.active[-1].shape[0])

    is_on_screen = (ox + objects.width[-1] > 0) & (ox < 160)
    is_waypoint = labels == LABEL_COLLECT
    valid_spatial_mask = jnp.logical_or(is_on_screen, is_waypoint)
    active_mask = (objects.active[-1] > 0.5) & valid_spatial_mask

    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    dx = px_center - ox_center
    dy = py_center - oy_center

    distances = jnp.sqrt(dx**2 + dy**2)
    weighted_distances = jnp.sqrt((X_AXIS_WEIGHT * dx) ** 2 + dy**2)

    is_enemy = labels == LABEL_DESTROY
    in_range = weighted_distances <= MAX_TARGET_DISTANCE
    normal_active_mask = jnp.logical_and(jnp.logical_and(active_mask, is_enemy), in_range)

    weights = jnp.where(
        labels == LABEL_AVOID,
        1000.0,
        jnp.where(
            labels == LABEL_DESTROY,
            500.0,
            jnp.where(labels == LABEL_COLLECT, 100.0, 0.0),
        ),
    )

    hysteresis_multiplier = jnp.where(idxs == prev_target_idx, 1.5, 1.0)
    epsilon = 1.0
    scores = (weights / (weighted_distances + epsilon)) * hysteresis_multiplier

    masked_scores = jnp.where(normal_active_mask, scores, -jnp.inf)
    best_normal_idx = jnp.argmax(masked_scores).astype(jnp.int32)

    is_threat = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)
    is_close_danger = jnp.logical_and(distances <= CLOSE_DANGER_RANGE, is_threat)
    is_close_danger = jnp.logical_and(is_close_danger, active_mask)
    have_valid_danger = jnp.any(is_close_danger)

    # For danger overrides, ignore hysteresis and semantic weights;
    # simply pick the closest valid threat inside the danger radius.
    danger_distances = jnp.where(is_close_danger, distances, jnp.inf)
    best_danger_idx = jnp.argmin(danger_distances).astype(jnp.int32)

    no_enemies = jnp.all(normal_active_mask == 0)
    waypoint_idx = jnp.argmax(is_waypoint).astype(jnp.int32)
    target_idx = best_normal_idx
    target_idx = jnp.where(have_valid_danger, best_danger_idx, target_idx)
    target_idx = jnp.where(
        no_enemies & jnp.logical_not(have_valid_danger),
        waypoint_idx,
        target_idx,
    )

    return {
        "player_center": jnp.array([px_center, py_center]),
        "object_centers": jnp.stack([ox_center, oy_center], axis=-1),
        "labels": labels,
        "active": objects.active[-1],
        "is_on_screen": is_on_screen,
        "valid_spatial_mask": valid_spatial_mask,
        "active_mask": active_mask,
        "distances": distances,
        "weighted_distances": weighted_distances,
        "in_range": in_range,
        "normal_active_mask": normal_active_mask,
        "weights": weights,
        "scores": scores,
        "is_threat": is_threat,
        "is_close_danger": is_close_danger,
        "have_valid_danger": have_valid_danger,
        "best_normal_idx": best_normal_idx,
        "best_danger_idx": best_danger_idx,
        "no_enemies": no_enemies,
        "final_target_idx": target_idx.astype(jnp.int32),
    }