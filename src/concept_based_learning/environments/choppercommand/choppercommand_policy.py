import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_DESTROY,
)
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

# When no object is on screen, return this; downstream should execute NOOP.
NO_TARGET_IDX = -1

# AABB surface-gap (px) that counts as "immediate danger".
CLOSE_DANGER_RANGE = 15.0

# Maximum AABB surface-gap at which we consider enemies as valid DESTROY targets.
MAX_TARGET_DISTANCE = 150.0


def apply_meta_mask_overrides(raw_obs, active_masks, config, objects=None):
    del raw_obs, config, objects
    return active_masks


def get_forced_target_override(raw_obs, current_targets, config, rng=None):
    return current_targets


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Continuous utility ChopperCommand targeting rule.

    Distances use AABB surface gap. Off-screen enemies are snapped to the screen
    edge in the label wrapper and kept as DESTROY (not NAVIGATE/COLLECT) so
    arrival at the edge does not mint a collect/navigate reward.
    """
    del rng
    distances, _, _, _, _ = player_object_aabb_gaps(raw_observation, objects)

    ox = objects.x[-1]
    labels = objects.semantic_label[-1]
    idxs = jnp.arange(objects.active[-1].shape[0])

    # Edge-snapped DESTROY proxies sit at x∈{0,159} and count as on-screen.
    is_on_screen = (ox + objects.width[-1] > 0) & (ox < 160)
    active_mask = (objects.active[-1] > 0.5) & is_on_screen

    num_objects = active_mask.shape[0]
    if (
        (ox.shape[0] != num_objects)
        or (objects.width[-1].shape[0] != num_objects)
        or (objects.height[-1].shape[0] != num_objects)
        or (labels.shape[0] != num_objects)
        or (idxs.shape[0] != num_objects)
    ):
        raise ValueError("ChopperCommand heuristic received inconsistent object-axis shapes.")

    is_enemy = labels == LABEL_DESTROY
    in_range = distances <= MAX_TARGET_DISTANCE
    normal_active_mask = jnp.logical_and(jnp.logical_and(active_mask, is_enemy), in_range)

    weights = jnp.where(labels == LABEL_DESTROY, 500.0, 0.0)
    hysteresis_multiplier = jnp.where(idxs == prev_target_idx, 1.5, 1.0)
    epsilon = 1.0
    scores = (weights / (distances + epsilon)) * hysteresis_multiplier

    masked_scores = jnp.where(normal_active_mask, scores, -jnp.inf)
    best_normal_idx = jnp.argmax(masked_scores).astype(jnp.int32)

    is_threat = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)
    is_close_danger = jnp.logical_and(distances <= CLOSE_DANGER_RANGE, is_threat)
    is_close_danger = jnp.logical_and(is_close_danger, active_mask)
    have_valid_danger = jnp.any(is_close_danger)

    danger_distances = jnp.where(is_close_danger, distances, jnp.inf)
    best_danger_idx = jnp.argmin(danger_distances).astype(jnp.int32)

    target_idx = best_normal_idx
    target_idx = jnp.where(have_valid_danger, best_danger_idx, target_idx)

    no_enemies = jnp.all(normal_active_mask == 0)
    target_idx = jnp.where(
        no_enemies & jnp.logical_not(have_valid_danger),
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
        target_idx,
    )

    return target_idx.astype(jnp.int32)


def debug_heuristic_snapshot(raw_observation, objects, prev_target_idx):
    """Inspect ChopperCommand heuristic intermediates on real frames."""
    distances, p_cx, p_cy, o_cx, o_cy = player_object_aabb_gaps(
        raw_observation, objects
    )

    ox = objects.x[-1]
    labels = objects.semantic_label[-1]
    idxs = jnp.arange(objects.active[-1].shape[0])

    is_on_screen = (ox + objects.width[-1] > 0) & (ox < 160)
    active_mask = (objects.active[-1] > 0.5) & is_on_screen

    is_enemy = labels == LABEL_DESTROY
    in_range = distances <= MAX_TARGET_DISTANCE
    normal_active_mask = jnp.logical_and(jnp.logical_and(active_mask, is_enemy), in_range)

    weights = jnp.where(labels == LABEL_DESTROY, 500.0, 0.0)
    hysteresis_multiplier = jnp.where(idxs == prev_target_idx, 1.5, 1.0)
    scores = (weights / (distances + 1.0)) * hysteresis_multiplier

    masked_scores = jnp.where(normal_active_mask, scores, -jnp.inf)
    best_normal_idx = jnp.argmax(masked_scores).astype(jnp.int32)

    is_threat = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)
    is_close_danger = jnp.logical_and(distances <= CLOSE_DANGER_RANGE, is_threat)
    is_close_danger = jnp.logical_and(is_close_danger, active_mask)
    have_valid_danger = jnp.any(is_close_danger)

    danger_distances = jnp.where(is_close_danger, distances, jnp.inf)
    best_danger_idx = jnp.argmin(danger_distances).astype(jnp.int32)

    no_enemies = jnp.all(normal_active_mask == 0)
    target_idx = best_normal_idx
    target_idx = jnp.where(have_valid_danger, best_danger_idx, target_idx)
    target_idx = jnp.where(
        no_enemies & jnp.logical_not(have_valid_danger),
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
        target_idx,
    )

    return {
        "player_center": jnp.array([p_cx, p_cy]),
        "object_centers": jnp.stack([o_cx, o_cy], axis=-1),
        "labels": labels,
        "active": objects.active[-1],
        "is_on_screen": is_on_screen,
        "active_mask": active_mask,
        "distances": distances,
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
