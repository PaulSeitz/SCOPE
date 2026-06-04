"""Heuristic meta-policy for Freeway."""

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_COLLECT

NO_TARGET_IDX = -1  # unused in practice: the top collect goal is always present

# The collect goal sits at the last slot after meta NOOP + 10 cars.
#   slot 0    = meta NOOP
#   slots 1-10 = cars (AVOID)
#   slot 11   = top collect goal (COLLECT)
COLLECT_GOAL_IDX = 11
# Back-compat alias used by older configs / comments.
COLLECT_WAYPOINT_IDX = COLLECT_GOAL_IDX

FREEWAY_THREAT_X_WEIGHT = 2.0
FREEWAY_THREAT_Y_WEIGHT = 1.0
FREEWAY_THREAT_WEIGHTED_DIST_THRESH = 12.0


def set_freeway_heuristic_params(
    *,
    threat_x_weight: float | None = None,
    threat_y_weight: float | None = None,
    threat_weighted_dist_thresh: float | None = None,
):
    global FREEWAY_THREAT_X_WEIGHT
    global FREEWAY_THREAT_Y_WEIGHT
    global FREEWAY_THREAT_WEIGHTED_DIST_THRESH
    if threat_x_weight is not None:
        FREEWAY_THREAT_X_WEIGHT = max(1e-6, float(threat_x_weight))
    if threat_y_weight is not None:
        FREEWAY_THREAT_Y_WEIGHT = max(1e-6, float(threat_y_weight))
    if threat_weighted_dist_thresh is not None:
        FREEWAY_THREAT_WEIGHTED_DIST_THRESH = float(threat_weighted_dist_thresh)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Freeway targeting rule:
    - default to the COLLECT goal at the top,
    - override to nearest AVOID car by weighted (x, y) distance
      when it is within the configured weighted threshold.
    """
    del prev_target_idx, rng

    # Freeway stores y as bottom edge.
    chicken_x = raw_observation.chicken.x[-1]
    chicken_y  = raw_observation.chicken.y[-1]
    chicken_w  = raw_observation.chicken.width[-1]
    chicken_h  = raw_observation.chicken.height[-1]
    p_cx       = chicken_x + chicken_w / 2.0
    p_cy       = chicken_y - chicken_h / 2.0

    # Object y is bottom edge for cars and goal.
    ox     = objects.x[-1] + objects.width[-1] / 2.0
    oy     = objects.y[-1]
    oh     = objects.height[-1]
    o_cy   = oy - oh / 2.0
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    is_avoid    = labels == LABEL_AVOID
    valid_avoid = jnp.logical_and(active, is_avoid)
    is_collect = labels == LABEL_COLLECT
    valid_collect = jnp.logical_and(active, is_collect)

    dx = jnp.abs(p_cx - ox)
    dy = jnp.abs(p_cy - o_cy)
    weighted_dist = (dx / FREEWAY_THREAT_X_WEIGHT) + (dy / FREEWAY_THREAT_Y_WEIGHT)
    best_avoid_idx = jnp.argmin(
        jnp.where(valid_avoid, weighted_dist, jnp.inf)
    ).astype(jnp.int32)
    best_avoid_dist = jnp.min(jnp.where(valid_avoid, weighted_dist, jnp.inf))
    threat = jnp.logical_and(
        jnp.any(valid_avoid),
        best_avoid_dist <= FREEWAY_THREAT_WEIGHTED_DIST_THRESH,
    )
    collect_idx = jnp.argmin(jnp.where(valid_collect, o_cy, jnp.inf)).astype(jnp.int32)
    collect_idx = jnp.where(jnp.any(valid_collect), collect_idx, COLLECT_GOAL_IDX).astype(
        jnp.int32
    )

    target_idx = jnp.where(threat, best_avoid_idx, collect_idx)

    return target_idx.astype(jnp.int32)
