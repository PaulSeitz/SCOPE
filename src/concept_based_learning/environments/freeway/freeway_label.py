import jax
import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_AVOID,
    LABEL_COLLECT,
)
from concept_based_learning.observations.action_space import EGO_MOVE_DELTAS_DEG

# ---- Freeway game constants (mirror FreewayConstants defaults) ----
FREEWAY_CHICKEN_X = 44
FREEWAY_CHICKEN_W = 6
FREEWAY_CHICKEN_H = 8
FREEWAY_CAR_W = 8
FREEWAY_CAR_H = 10
FREEWAY_TOP_BORDER = 8      # player scores when chicken_y (bottom edge) ≤ this
FREEWAY_BOTTOM_BORDER = 180
FREEWAY_NUM_LANES = 10

# Fixed COLLECT goal: one pixel below the scoring/teleport line (y increases downward).
FREEWAY_COLLECT_GOAL_Y = FREEWAY_TOP_BORDER + 1

# Fixed collect object sits at the last slot after meta NOOP + 10 cars.
COLLECT_GOAL_IDX = FREEWAY_NUM_LANES + 1  # slot 11 in stacked objects

# Normalize distance shaping so a full bottom→top ascent yields ~O(1) dense signal.
_COLLECT_DIST_SCALE = float(FREEWAY_BOTTOM_BORDER - FREEWAY_TOP_BORDER)

# X-zone where a car overlaps the (fixed) chicken column.
# chicken x ∈ [44, 50] (width 6); car width = 8 → overlap when car_x ∈ [36, 50]
_PLAYER_X_LEFT  = float(FREEWAY_CHICKEN_X)                       # 44
_PLAYER_X_RIGHT = float(FREEWAY_CHICKEN_X + FREEWAY_CHICKEN_W)   # 50
_CAR_ZONE_LEFT  = _PLAYER_X_LEFT - FREEWAY_CAR_W                 # 36
_CAR_ZONE_RIGHT = _PLAYER_X_RIGHT                                 # 50

# Scoring teleport: chicken reaches y ≤ top_border then resets to bottom (~188).
_SCORING_MAX_PREV_Y = FREEWAY_TOP_BORDER + FREEWAY_CHICKEN_H + 2  # = 25
_SCORING_MIN_NEXT_Y = 100

# World direction angle table (RIGHT=0°, DOWNRIGHT=45°, …, UPRIGHT=-45°)
_WORLD_DIR_ANGLES = jnp.array(
    [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0], dtype=jnp.float32
)

# Freeway action indices: 0=NOOP, 1=UP, 2=DOWN
_FREEWAY_NOOP = jnp.int32(0)

# move_table[dir_idx] — Freeway only has vertical movement.
#   DIR 0 RIGHT     → NOOP   DIR 1 DOWNRIGHT → DOWN
#   DIR 2 DOWN      → DOWN   DIR 3 DOWNLEFT  → DOWN
#   DIR 4 LEFT      → NOOP   DIR 5 UPLEFT    → UP
#   DIR 6 UP        → UP     DIR 7 UPRIGHT   → UP
_MOVE_TABLE      = jnp.array([0, 2, 2, 2, 0, 1, 1, 1], dtype=jnp.int32)
_MOVE_FIRE_TABLE = jnp.array([0, 2, 2, 2, 0, 1, 1, 1], dtype=jnp.int32)  # no FIRE in Freeway


def freeway_action_mapper(ego_action, raw_obs, target_idx, objects):
    """Translate an egocentric action into a Freeway game action (0/1/2)."""
    # y is stored as the BOTTOM edge in Freeway; use bottom-edge centres so
    # relative direction / distance are self-consistent.
    p_cx = raw_obs.chicken.x[-1] + raw_obs.chicken.width[-1] / 2.0
    p_cy = raw_obs.chicken.y[-1] - raw_obs.chicken.height[-1] / 2.0

    t_cx = objects.x[-1, target_idx] + objects.width[-1, target_idx] / 2.0
    t_cy = objects.y[-1, target_idx] - objects.height[-1, target_idx] / 2.0

    theta_deg  = jnp.arctan2(t_cy - p_cy, t_cx - p_cx) * (180.0 / jnp.pi)
    desired_deg = theta_deg + EGO_MOVE_DELTAS_DEG[ego_action]

    diff    = (desired_deg - _WORLD_DIR_ANGLES + 180.0) % 360.0 - 180.0
    dir_idx = jnp.argmin(jnp.abs(diff)).astype(jnp.int32)

    has_fire = ego_action >= 10
    is_fire  = ego_action == 1
    is_noop  = ego_action == 0

    game_action = jnp.where(has_fire, _MOVE_FIRE_TABLE[dir_idx], _MOVE_TABLE[dir_idx])
    game_action = jnp.where(is_fire, _FREEWAY_NOOP, game_action)
    game_action = jnp.where(is_noop, _FREEWAY_NOOP, game_action)
    return game_action


class FreewaySemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 1 meta NOOP + 10 cars + 1 fixed top collect goal = 12 total slots
        super().__init__(env, max_objects=12)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        """
        Map a single-frame FreewayObservation to a LabeledObjectObservation.

        Fixed collect goal at the top scoring line (y=15): always the same object
        at the screen top, aligned with the chicken column.
        """
        car_labels = jnp.full(FREEWAY_NUM_LANES, LABEL_AVOID, dtype=jnp.int32)

        # ---- Fixed top collect goal ------------------------------------------
        # Bottom-edge convention: bbox bottom at goal line + chicken height.
        goal_x      = jnp.array([FREEWAY_CHICKEN_X], dtype=jnp.int32)
        goal_y      = jnp.array([FREEWAY_COLLECT_GOAL_Y + FREEWAY_CHICKEN_H], dtype=jnp.int32)
        goal_w      = jnp.array([FREEWAY_CHICKEN_W], dtype=jnp.int32)
        goal_h      = jnp.array([FREEWAY_CHICKEN_H], dtype=jnp.int32)
        goal_active = jnp.array([1], dtype=jnp.int32)
        goal_label  = jnp.array([LABEL_COLLECT], dtype=jnp.int32)

        all_x      = jnp.concatenate([obs.car.x,                           goal_x])
        all_y      = jnp.concatenate([obs.car.y,                           goal_y])
        all_w      = jnp.concatenate([obs.car.width,                       goal_w])
        all_h      = jnp.concatenate([obs.car.height,                      goal_h])
        all_active = jnp.concatenate([obs.car.active.astype(jnp.int32),    goal_active])
        all_labels = jnp.concatenate([car_labels,                          goal_label])

        return LabeledObjectObservation(
            x=all_x,
            y=all_y,
            width=all_w,
            height=all_h,
            active=all_active,
            semantic_label=all_labels,
        )


def _chicken_object_overlap(chicken, obj):
    """Bounding-box overlap between the chicken and an object (y is bottom edge)."""
    horiz = jnp.logical_and(
        chicken.x <= obj.x + obj.width,
        chicken.x + chicken.width >= obj.x,
    )
    vert = jnp.logical_and(
        chicken.y - chicken.height <= obj.y,
        chicken.y >= obj.y - obj.height,
    )
    return jnp.logical_and(horiz, vert)


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    """
    Skill reward signals for Freeway.

    All three skills are scored against the *selected object* via hindsight
    relabeling: each skill reward treats the meta-selected object as if it
    carried that skill's label. Every skill can therefore learn off-policy from
    every transition, regardless of the object's true semantic label.

    COLLECT: dense object-relative shaping (prev_dist - next_dist) / scale —
             positive when the chicken closes vertical distance to the selected
             object; +1 sparse bonus when the chicken first reaches it.
    AVOID:   penalty-only — negative reward on collision with the selected
             object, 0 otherwise. A car clearing the column is not a rewardable
             event (cars sweep across regardless of the chicken), so there is
             no +1 for the meta to farm.
    DESTROY: Freeway has no shoot targets; stays zero.
    """
    def last_frame(tree):
        return jax.tree_util.tree_map(lambda x: x[-1] if jnp.ndim(x) > 0 else x, tree)

    prev_chicken = last_frame(prev_raw_obs.chicken)
    next_chicken = last_frame(next_raw_obs.chicken)
    prev_tgt = last_frame(prev_target)
    next_tgt = last_frame(next_target)

    # ---- COLLECT (object-relative) ----
    # Vertical centre-to-centre distance between chicken and the selected object.
    prev_cy     = prev_chicken.y.astype(jnp.float32) - prev_chicken.height.astype(jnp.float32) / 2.0
    next_cy     = next_chicken.y.astype(jnp.float32) - next_chicken.height.astype(jnp.float32) / 2.0
    prev_tgt_cy = prev_tgt.y.astype(jnp.float32) - prev_tgt.height.astype(jnp.float32) / 2.0
    next_tgt_cy = next_tgt.y.astype(jnp.float32) - next_tgt.height.astype(jnp.float32) / 2.0

    prev_dist = jnp.abs(prev_cy - prev_tgt_cy)
    next_dist = jnp.abs(next_cy - next_tgt_cy)

    # After scoring the chicken teleports to the bottom; suppress the shaping
    # discontinuity (large spurious distance jump) on that step.
    teleported = jnp.logical_and(
        prev_chicken.y <= _SCORING_MAX_PREV_Y,
        next_chicken.y >= _SCORING_MIN_NEXT_Y,
    )
    collect_shaping = jnp.where(
        teleported,
        0.0,
        (prev_dist - next_dist) / jnp.float32(_COLLECT_DIST_SCALE),
    )

    # Reached: chicken just entered vertical bounding-box overlap with the object.
    reach_eps = (
        next_chicken.height.astype(jnp.float32)
        + next_tgt.height.astype(jnp.float32)
    ) / 2.0
    collect_success = jnp.logical_and(prev_dist > reach_eps, next_dist <= reach_eps)
    collect_reward  = collect_shaping + jnp.where(collect_success, 1.0, 0.0)
    collect_done    = collect_success

    # ---- AVOID (object-relative, penalty-only) ----
    # Freeway is not lives-based, so collisions cannot be read from life_lost;
    # detect them directly as bounding-box overlap with the selected object.
    collided = jnp.logical_or(
        _chicken_object_overlap(prev_chicken, prev_tgt),
        _chicken_object_overlap(next_chicken, next_tgt),
    )
    # The object clearing the chicken's x-column ends the AVOID encounter but is
    # not itself rewarded: cars sweep across regardless of the chicken, so a +1
    # here would let the meta farm AVOID instead of crossing the road.
    prev_in_zone = jnp.logical_and(prev_tgt.x >= _CAR_ZONE_LEFT, prev_tgt.x <= _CAR_ZONE_RIGHT)
    next_in_zone = jnp.logical_and(next_tgt.x >= _CAR_ZONE_LEFT, next_tgt.x <= _CAR_ZONE_RIGHT)
    object_just_cleared = jnp.logical_and(prev_in_zone, jnp.logical_not(next_in_zone))

    avoid_reward = jnp.where(collided, -2.0, 0.0)
    avoid_done   = jnp.logical_or(object_just_cleared, collided)

    # ---- DESTROY ---- (no destroy targets in Freeway)
    destroy_reward = jnp.float32(0.0)
    destroy_done   = jnp.array(False, dtype=jnp.bool_)

    rewards    = jnp.array([destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32)
    skill_done = jnp.array([destroy_done, collect_done, avoid_done],       dtype=jnp.bool_)
    return rewards, skill_done
