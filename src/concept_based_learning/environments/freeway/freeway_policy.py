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
       |dx_center| < X_AHEAD_THRESH  AND  dy_center < Y_AHEAD_THRESH
  4. TARGET a car in the same lane (y overlapping or very close) if it is approaching
     within IN_LANE_DIST and the x bands overlap.
  5. Otherwise select the COLLECT goal.
"""

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_COLLECT

NO_TARGET_IDX = -1  # unused in practice: the top collect goal is always present
COLLECT_GOAL_IDX = 11  # slot 0=NOOP, 1-10=cars, 11=collect goal

# ── Thresholds ───────────────────────────────────────────────────────────────
# Ahead car: edge-to-edge vertical gap and center-to-center horizontal gap.
Y_AHEAD_THRESH = 10.0   # px: player_top - car_bottom gap to trigger ahead rule
X_AHEAD_THRESH = 10.0   # px: center-to-center horizontal distance

# Same-lane TTC: select a car if it will reach the player's x-band within this
# many frames (at its current speed). Static / very slow cars use a pixel fallback.
IN_LANE_TTC_FRAMES = 16       # frames lookahead for same-lane cars
IN_LANE_Y_THRESH   = 25.0     # px: edge-to-edge vertical gap to consider "same lane"
IN_LANE_X_GAP      = 6.0      # px: extra clearance added to x-band for overlap test

# ── Legacy compat shim ───────────────────────────────────────────────────────
FREEWAY_THREAT_DIST_THRESH = 30.0  # kept so old configs importing it don't break


def set_freeway_heuristic_params(*, threat_dist_thresh: float | None = None):
    global FREEWAY_THREAT_DIST_THRESH
    if threat_dist_thresh is not None:
        FREEWAY_THREAT_DIST_THRESH = float(threat_dist_thresh)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del prev_target_idx, rng

    # ── Player geometry (raw_observation uses bottom-edge y convention) ────────
    chicken_x = raw_observation.chicken.x[-1]
    chicken_y = raw_observation.chicken.y[-1]          # bottom edge
    chicken_w = raw_observation.chicken.width[-1]
    chicken_h = raw_observation.chicken.height[-1]

    p_cx     = chicken_x + chicken_w / 2.0
    p_cy     = chicken_y - chicken_h / 2.0             # center y
    p_top    = chicken_y - chicken_h                   # top edge (smallest y = highest on screen)
    p_left   = chicken_x
    p_right  = chicken_x + chicken_w

    # ── Object geometry (objects also uses bottom-edge y convention) ──────────
    ox  = objects.x[-1]                                # left edge
    oy  = objects.y[-1]                                # bottom edge
    ow  = objects.width[-1]
    oh  = objects.height[-1]

    o_cx     = ox + ow / 2.0                           # center x
    o_cy     = oy - oh / 2.0                           # center y
    o_top    = oy - oh                                 # top edge
    o_bottom = oy                                      # bottom edge
    o_left   = ox
    o_right  = ox + ow

    # ── Car speeds from frame stack (finite difference on x) ─────────────────
    # objects.x shape: (T, N). dx > 0 = moving right, dx < 0 = moving left.
    # Use last two frames; clamp to 0 when wrapping causes large jumps (>20px).
    raw_dx    = objects.x[-1].astype(jnp.float32) - objects.x[-2].astype(jnp.float32)
    car_vx    = jnp.where(jnp.abs(raw_dx) > 20.0, 0.0, raw_dx)  # zero out wrap-around noise

    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    is_avoid   = labels == LABEL_AVOID
    is_collect = labels == LABEL_COLLECT
    valid_avoid   = jnp.logical_and(active, is_avoid)
    valid_collect = jnp.logical_and(active, is_collect)

    # ── Rule 1: skip cars already passed (fully below player) ────────────────
    # Car is passed when its top edge is below the player's bottom edge.
    car_passed = o_top > chicken_y
    valid_avoid = jnp.logical_and(valid_avoid, jnp.logical_not(car_passed))

    # ── Rule 2: skip cars moving away from the player ────────────────────────
    # orientation 90 = moving right, 270 = moving left.
    # slots: 0=NOOP, 1-10=cars, 11=collect goal — pad dummies for the non-car slots.
    car_orientation = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),        # slot 0: NOOP
        raw_observation.car.orientation[-1],     # slots 1-10: cars
        jnp.array([0], dtype=jnp.int32),         # slot 11: collect goal
    ])
    moving_right = car_orientation == 90
    moving_left  = car_orientation == 270
    # Only dismiss as moving away when not already in the danger zone vertically.
    y_sep_raw       = jnp.maximum(o_top - chicken_y, p_top - o_bottom)
    vertically_safe = y_sep_raw >= IN_LANE_Y_THRESH
    car_moving_away = jnp.logical_and(
        vertically_safe,
        jnp.logical_or(
            jnp.logical_and(moving_right, o_cx > p_cx),
            jnp.logical_and(moving_left,  o_cx < p_cx),
        ),
    )
    valid_avoid = jnp.logical_and(valid_avoid, jnp.logical_not(car_moving_away))

    # ── Rule 3: imminent ahead car ───────────────────────────────────────────
    # Car is ahead (higher on screen = smaller y). Use edge-to-edge vertical gap:
    # gap = player_top - car_bottom. Negative means overlap (already colliding).
    # Trigger when gap < Y_AHEAD_THRESH and x centers are close.
    car_is_ahead  = o_cy < p_cy
    y_edge_gap    = p_top - o_bottom          # positive = clear gap above player, negative = overlap
    dx_ahead      = jnp.abs(p_cx - o_cx)
    ahead_threat  = jnp.logical_and(car_is_ahead,
                    jnp.logical_and(y_edge_gap < Y_AHEAD_THRESH,
                                    dx_ahead < X_AHEAD_THRESH))
    valid_ahead   = jnp.logical_and(valid_avoid, ahead_threat)

    # ── Rule 4: same-lane approaching car (TTC-based) ────────────────────────
    # Vertically in the same lane, and the car will enter the player's x-band
    # within IN_LANE_TTC_FRAMES frames at its current speed.
    y_separation = jnp.maximum(o_top - chicken_y, p_top - o_bottom)  # >0 gap, <0 overlap
    in_lane_y    = y_separation < IN_LANE_Y_THRESH

    # x-gap: distance the car's leading edge still needs to travel to reach the player.
    # Positive = approaching, zero/negative = already past (moving away).
    # Use car_vx for direction; fall back to orientation when vx=0 (cadence frame).
    gap_approaching = p_left - o_right   # right-moving car approaching from the left
    gap_receding    = o_left - p_right   # left-moving car approaching from the right
    car_going_right = jnp.logical_or(car_vx > 0.0,
                          jnp.logical_and(car_vx == 0.0, moving_right))
    x_gap = jnp.where(car_going_right, gap_approaching, gap_receding)

    # TTC = x_gap / |speed|. Only fire when car is still approaching (x_gap > 0).
    # For cadence frames (speed=0 but x_gap>0) treat as a very slow approach —
    # fire when x_gap itself is within IN_LANE_TTC_FRAMES pixels as a pixel fallback.
    speed      = jnp.abs(car_vx)
    ttc        = jnp.where(speed > 0.0, x_gap / speed, x_gap)  # px fallback when speed=0
    in_lane_threat = jnp.logical_and(in_lane_y,
                     jnp.logical_and(x_gap > 0.0, ttc < IN_LANE_TTC_FRAMES))
    valid_lane     = jnp.logical_and(valid_avoid, in_lane_threat)

    # ── Combine: pick the lowest-TTC threat across both pools ────────────────
    # TTC for ahead cars: use edge-to-edge y gap as a proxy (no horizontal speed).
    # TTC for lane cars: already computed above. Merge into one priority score.
    any_valid_avoid = jnp.logical_or(jnp.any(valid_ahead), jnp.any(valid_lane))

    # Ahead cars have no horizontal speed component — use y edge gap as urgency.
    ahead_urgency = y_edge_gap                           # smaller = more urgent
    lane_urgency  = ttc                                  # smaller = more urgent

    # Unified score: ahead cars get their y-gap score, lane cars get their TTC.
    # Scale y-gap to be comparable to TTC-in-frames: divide by ~1px/frame chicken speed.
    urgency = jnp.where(valid_ahead, ahead_urgency,
              jnp.where(valid_lane,  lane_urgency, jnp.inf))

    best_avoid_idx = jnp.argmin(urgency).astype(jnp.int32)

    # ── Fallback: collect goal ────────────────────────────────────────────────
    collect_idx = jnp.argmin(jnp.where(valid_collect, o_cy, jnp.inf)).astype(jnp.int32)
    collect_idx = jnp.where(jnp.any(valid_collect), collect_idx, COLLECT_GOAL_IDX).astype(
        jnp.int32
    )

    target_idx = jnp.where(any_valid_avoid, best_avoid_idx, collect_idx)
    return target_idx.astype(jnp.int32)
