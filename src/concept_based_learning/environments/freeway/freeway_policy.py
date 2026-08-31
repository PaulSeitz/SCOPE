"""Strict heuristic meta-policy for Freeway.

Coordinate convention (JAXAtari Freeway):
  - y is the BOTTOM edge of the sprite/hitbox.
  - Smaller y = higher on screen. Player scores when chicken_y <= top_border (~15).
  - Cars travel horizontally; the player (chicken) moves vertically upward (decreasing y).
  - car.orientation: 90 = moving right, 270 = moving left.

Targeting priority:
  1. SKIP cars that are fully below the player (already passed — can't hit us).
  2. SKIP cars that are moving away from the player (right-moving car to the right of
     player, or left-moving car to the left of player).
  3. TARGET a car directly ahead (smaller y) if it is imminently dangerous:
       AABB x-overlap / small gap AND vertical surface gap < Y_AHEAD_THRESH
  4. TARGET a car in the same lane if approaching within IN_LANE_TTC_FRAMES.
  5. Otherwise select the fixed top COLLECT goal.
"""

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_NAVIGATE,
)
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

NO_TARGET_IDX = -1  # unused in practice: the top navigate goal is always present
NAVIGATE_GOAL_IDX = 11  # back-compat name: slot 11 is now the top COLLECT goal
COLLECT_GOAL_IDX = NAVIGATE_GOAL_IDX  # back-compat alias

# Ahead car: edge-to-edge vertical gap and horizontal proximity.
Y_AHEAD_THRESH = 10.0   # px: vertical AABB surface gap to trigger ahead rule
X_AHEAD_THRESH = 10.0   # px: |center_x| gap / horizontal proximity

# Same-lane TTC: select a car if it will reach the player's x-band within this
# many frames (at its current speed). Static / very slow cars use a pixel fallback.
IN_LANE_TTC_FRAMES = 16       # frames lookahead for same-lane cars
IN_LANE_Y_THRESH   = 25.0     # px: vertical AABB surface gap to consider "same lane"
IN_LANE_X_GAP      = 6.0      # px: extra clearance added to x-band for overlap test

# Legacy compat shim
FREEWAY_THREAT_DIST_THRESH = 30.0


def set_freeway_heuristic_params(*, threat_dist_thresh: float | None = None):
    global FREEWAY_THREAT_DIST_THRESH
    if threat_dist_thresh is not None:
        FREEWAY_THREAT_DIST_THRESH = float(threat_dist_thresh)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del prev_target_idx, rng

    # Freeway labeled objects / chicken use bottom-edge y in the heuristic path.
    distances, p_cx, p_cy, o_cx, o_cy = player_object_aabb_gaps(
        raw_observation, objects, y_is_bottom_edge=True
    )

    chicken_x = raw_observation.chicken.x[-1]
    chicken_y = raw_observation.chicken.y[-1]          # bottom edge
    chicken_w = raw_observation.chicken.width[-1]
    chicken_h = raw_observation.chicken.height[-1]

    p_top    = chicken_y - chicken_h
    p_left   = chicken_x
    p_right  = chicken_x + chicken_w

    ox  = objects.x[-1]
    oy  = objects.y[-1]                                # bottom edge
    ow  = objects.width[-1]
    oh  = objects.height[-1]

    o_top    = oy - oh
    o_bottom = oy
    o_left   = ox
    o_right  = ox + ow

    raw_dx    = objects.x[-1].astype(jnp.float32) - objects.x[-2].astype(jnp.float32)
    car_vx    = jnp.where(jnp.abs(raw_dx) > 20.0, 0.0, raw_dx)

    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    is_avoid   = labels == LABEL_AVOID
    # LABEL_NAVIGATE remains accepted for old checkpoints/observations, while
    # current Freeway observations label the top waypoint as COLLECT.
    is_navigate = jnp.logical_or(labels == LABEL_NAVIGATE, labels == LABEL_COLLECT)
    valid_avoid   = jnp.logical_and(active, is_avoid)
    valid_navigate = jnp.logical_and(active, is_navigate)

    car_passed = o_top > chicken_y
    valid_avoid = jnp.logical_and(valid_avoid, jnp.logical_not(car_passed))

    car_orientation = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),
        raw_observation.car.orientation[-1],
        jnp.array([0], dtype=jnp.int32),
    ])
    moving_right = car_orientation == 90
    moving_left  = car_orientation == 270
    # Vertical separation via AABB surface gap (already accounts for box sizes).
    vertically_safe = distances >= IN_LANE_Y_THRESH
    car_moving_away = jnp.logical_and(
        vertically_safe,
        jnp.logical_or(
            jnp.logical_and(moving_right, o_cx > p_cx),
            jnp.logical_and(moving_left,  o_cx < p_cx),
        ),
    )
    valid_avoid = jnp.logical_and(valid_avoid, jnp.logical_not(car_moving_away))

    car_is_ahead  = o_cy < p_cy
    y_edge_gap    = p_top - o_bottom
    dx_ahead      = jnp.abs(p_cx - o_cx)
    ahead_threat  = jnp.logical_and(car_is_ahead,
                    jnp.logical_and(distances < Y_AHEAD_THRESH,
                                    dx_ahead < X_AHEAD_THRESH))
    valid_ahead   = jnp.logical_and(valid_avoid, ahead_threat)

    y_separation = jnp.maximum(o_top - chicken_y, p_top - o_bottom)
    in_lane_y    = jnp.logical_or(distances < IN_LANE_Y_THRESH, y_separation < IN_LANE_Y_THRESH)

    gap_approaching = p_left - o_right
    gap_receding    = o_left - p_right
    car_going_right = jnp.logical_or(car_vx > 0.0,
                          jnp.logical_and(car_vx == 0.0, moving_right))
    x_gap = jnp.where(car_going_right, gap_approaching, gap_receding)

    speed      = jnp.abs(car_vx)
    ttc        = jnp.where(speed > 0.0, x_gap / speed, x_gap)
    in_lane_threat = jnp.logical_and(in_lane_y,
                     jnp.logical_and(x_gap > 0.0, ttc < IN_LANE_TTC_FRAMES))
    valid_lane     = jnp.logical_and(valid_avoid, in_lane_threat)

    any_valid_avoid = jnp.logical_or(jnp.any(valid_ahead), jnp.any(valid_lane))

    # Prefer AABB surface gap for ahead urgency; TTC for lane cars.
    urgency = jnp.where(valid_ahead, distances,
              jnp.where(valid_lane,  ttc, jnp.inf))

    best_avoid_idx = jnp.argmin(urgency).astype(jnp.int32)

    navigate_idx = jnp.argmin(jnp.where(valid_navigate, distances, jnp.inf)).astype(
        jnp.int32
    )
    navigate_idx = jnp.where(jnp.any(valid_navigate), navigate_idx, NAVIGATE_GOAL_IDX).astype(
        jnp.int32
    )

    target_idx = jnp.where(any_valid_avoid, best_avoid_idx, navigate_idx)
    return target_idx.astype(jnp.int32)
