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
FREEWAY_TOP_BORDER = 15     # env's actual top_border (jax_freeway.FreewayConstants):
                            # player scores when chicken_y (bottom edge) ≤ this.
FREEWAY_BOTTOM_BORDER = 180
FREEWAY_NUM_LANES = 10

# Fixed COLLECT goal: at the very top of the screen.
#
# The waypoint position only affects the *shaping gradient* (+1 score reward is
# decoupled from bbox overlap — see `collect_success = teleported` below).
# Placing the waypoint above the scoring line (y=0) makes the dense shaping
# signal monotone in chicken.y all the way up to the scoring threshold, so the
# chicken always benefits from pushing further upward — no "edge at the goal
# centre, never quite score" trap.
FREEWAY_COLLECT_GOAL_Y = 0

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

# AVOID threat-weighted shaping defaults (override via config keys below).
_FREEWAY_AVOID_COLLISION_PENALTY = -2.0
_DEFAULT_FREEWAY_AVOID_PROXIMITY_SIGMA = 30.0   # ~chicken height × 3
_DEFAULT_FREEWAY_AVOID_ALPHA_APPROACH = 0.3
_DEFAULT_FREEWAY_AVOID_BETA_CLEARANCE = 0.05
_DEFAULT_FREEWAY_AVOID_LANE_HALF_HEIGHT = 10.0  # ~car height
_DEFAULT_FREEWAY_AVOID_LATERAL_SCALE = 8.0      # ~car width

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


def extract_all_skills_rewards(
    prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, config=None,
):
    """
    Skill reward signals for Freeway.

    All three skills are scored against the *selected object* via hindsight
    relabeling: each skill reward treats the meta-selected object as if it
    carried that skill's label. Every skill can therefore learn off-policy from
    every transition, regardless of the object's true semantic label.

    COLLECT: dense object-relative shaping (prev_dist - next_dist) / scale —
             positive when the chicken closes vertical distance to the selected
             object; +1 sparse bonus when the chicken first reaches it.
    AVOID:   threat-weighted dense shaping plus collision penalty. Three pieces:
             (1) hard −2 on bbox overlap; (2) approach shaping
             ``−α * (d_prev − d_next) * exp(−d_next / σ)`` so far targets
             contribute near-zero and close approaching targets drive strong
             gradients; (3) optional lateral-clearance bonus when y-aligned
             with the target. Tune via ``FREEWAY_AVOID_*`` config keys. A car
             clearing the chicken's x-column ends the AVOID encounter but is not
             rewarded — cars sweep across regardless of the chicken.
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

    # Reached: tie to the Freeway scoring trigger so the +1 hindsight reward
    # matches the real env score 1-to-1. The env teleports the chicken from
    # near the top (y <= _SCORING_MAX_PREV_Y) to the bottom (y >= _SCORING_MIN_NEXT_Y)
    # on a successful crossing — exactly the `teleported` flag above. This
    # closes a farm where the chicken could enter/exit the goal's bounding
    # box repeatedly to collect +1 without actually crossing the road.
    collect_success = teleported
    collect_reward  = collect_shaping + jnp.where(collect_success, 1.0, 0.0)
    collect_done    = collect_success

    # ---- AVOID (object-relative, threat-weighted shaping) ----
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

    def _cfg(key, default):
        return float(config.get(key, default)) if config is not None else default

    proximity_sigma = _cfg("FREEWAY_AVOID_PROXIMITY_SIGMA", _DEFAULT_FREEWAY_AVOID_PROXIMITY_SIGMA)
    alpha_approach = _cfg("FREEWAY_AVOID_ALPHA_APPROACH", _DEFAULT_FREEWAY_AVOID_ALPHA_APPROACH)
    beta_clearance = _cfg("FREEWAY_AVOID_BETA_CLEARANCE", _DEFAULT_FREEWAY_AVOID_BETA_CLEARANCE)
    lane_half_height = _cfg("FREEWAY_AVOID_LANE_HALF_HEIGHT", _DEFAULT_FREEWAY_AVOID_LANE_HALF_HEIGHT)
    lateral_scale = _cfg("FREEWAY_AVOID_LATERAL_SCALE", _DEFAULT_FREEWAY_AVOID_LATERAL_SCALE)

    prev_cx = prev_chicken.x.astype(jnp.float32) + prev_chicken.width.astype(jnp.float32) / 2.0
    next_cx = next_chicken.x.astype(jnp.float32) + next_chicken.width.astype(jnp.float32) / 2.0
    prev_tgt_cx = prev_tgt.x.astype(jnp.float32) + prev_tgt.width.astype(jnp.float32) / 2.0
    next_tgt_cx = next_tgt.x.astype(jnp.float32) + next_tgt.width.astype(jnp.float32) / 2.0

    dx_prev = prev_cx - prev_tgt_cx
    dy_prev = prev_cy - prev_tgt_cy
    dx_next = next_cx - next_tgt_cx
    dy_next = next_cy - next_tgt_cy

    d_prev = jnp.sqrt(dx_prev ** 2 + dy_prev ** 2)
    d_next = jnp.sqrt(dx_next ** 2 + dy_next ** 2)
    approach = d_prev - d_next  # positive = closing gap

    proximity_weight = jnp.exp(-d_next / jnp.float32(proximity_sigma))
    reward_shaping = -jnp.float32(alpha_approach) * approach * proximity_weight

    in_lane = jnp.abs(dy_next) < jnp.float32(lane_half_height)
    lateral_clearance_norm = jnp.minimum(
        jnp.abs(dx_next) / jnp.float32(lateral_scale), 1.0,
    )
    reward_clearance = (
        jnp.float32(beta_clearance) * in_lane.astype(jnp.float32) * lateral_clearance_norm
    )

    avoid_reward = jnp.where(
        collided,
        jnp.float32(_FREEWAY_AVOID_COLLISION_PENALTY),
        jnp.where(teleported, jnp.float32(0.0), reward_shaping + reward_clearance),
    )
    avoid_done   = jnp.logical_or(object_just_cleared, collided)

    # ---- DESTROY ---- (no destroy targets in Freeway)
    destroy_reward = jnp.float32(0.0)
    destroy_done   = jnp.array(False, dtype=jnp.bool_)

    rewards    = jnp.array([destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32)
    skill_done = jnp.array([destroy_done, collect_done, avoid_done],       dtype=jnp.bool_)
    return rewards, skill_done
