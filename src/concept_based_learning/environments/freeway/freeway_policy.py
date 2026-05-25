"""
Heuristic meta-policy for Freeway.

Default behaviour: target the fixed COLLECT goal at the top of the screen so
the chicken walks upward.  If any car is very close in y (within DANGER_Y_RANGE
pixels of the chicken's centre), override and target that car instead so the
AVOID skill can steer around it.
"""

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

# Vertical distance (pixels) at which we consider a car "threatening".
DANGER_Y_RANGE = 12.0


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """
    Freeway targeting rule.

    - By default, select the top COLLECT goal so the chicken heads upward.
    - If any AVOID car is within DANGER_Y_RANGE pixels vertically, target the
      closest such car so the AVOID skill kicks in.
    """
    del prev_target_idx, rng

    # Chicken y is stored as the bottom edge; compute centre.
    chicken_y  = raw_observation.chicken.y[-1]
    chicken_h  = raw_observation.chicken.height[-1]
    p_cy       = chicken_y - chicken_h / 2.0

    # Object y is also bottom edge for both cars and the collect goal.
    oy     = objects.y[-1]
    oh     = objects.height[-1]
    o_cy   = oy - oh / 2.0

    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    is_avoid    = labels == LABEL_AVOID
    valid_avoid = jnp.logical_and(active, is_avoid)

    # Vertical distance from chicken to each car.
    dy_avoid = jnp.abs(p_cy - o_cy)
    avoid_dist = jnp.where(valid_avoid, dy_avoid, jnp.inf)

    closest_car_dy  = jnp.min(avoid_dist)
    best_avoid_idx  = jnp.argmin(avoid_dist).astype(jnp.int32)

    have_close_car = closest_car_dy < DANGER_Y_RANGE

    # Default: always aim for the top collect goal.
    target_idx = jnp.array(COLLECT_GOAL_IDX, dtype=jnp.int32)

    # Danger override: target the closest threatening car.
    target_idx = jnp.where(have_close_car, best_avoid_idx, target_idx)

    return target_idx.astype(jnp.int32)
