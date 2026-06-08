import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_AVOID,
    LABEL_COLLECT,
)
from concept_based_learning.environments.skill_rewards import extract_skill_rewards
from concept_based_learning.observations.action_space import (
    EGO_MOVE_DELTAS_DEG,
    make_inverse_ego_action_mapper,
)

# ---- Freeway game constants (mirror FreewayConstants defaults) ----
FREEWAY_CHICKEN_X = 44
FREEWAY_NUM_LANES = 10
FREEWAY_TOP_BORDER = 15     # player scores when chicken_y (bottom edge) ≤ this (JAXAtari FreewayConstants.top_border)
# Nudge the fixed COLLECT goal toward the top (smaller y in Freeway coords).
# Must ensure goal band bottom > top_border so the rising-edge fires before the game resets
# the chicken (scoring at chicken_y=top_border immediately spawns chicken at y≈187).
# With top_border=15, goal_y=11, goal_h=10: after 1px inset, band=[12,20]. Rising edge at y=20 → 5px before scoring.
FREEWAY_COLLECT_GOAL_UP_OFFSET_PX = 16

# Effective hitbox sizes derived from JAXAtari insets (jax_freeway.FreewayConstants):
#   chicken: sprite 6×8, inset x=1 each side → effective width=4
#            inset y_top=-2 (expands upward), y_bottom=0 → effective height=10
#   car:     sprite 8×10, inset x=0, y_top=2, y_bottom=0 → effective width=8, height=8
# These are what we store in LabeledObjectObservation so the global AABB check
# fires at exactly the same frame JAXAtari's own collision check fires.
FREEWAY_CHICKEN_HIT_W = 4
FREEWAY_CHICKEN_HIT_H = 10
FREEWAY_CAR_HIT_W = 8
FREEWAY_CAR_HIT_H = 8

# Fixed collect goal sits at the last slot after meta NOOP + 10 cars.
COLLECT_GOAL_IDX = FREEWAY_NUM_LANES + 1  # slot 11 in stacked objects

# World direction angle table (RIGHT=0°, DOWNRIGHT=45°, …, UPRIGHT=-45°)
_WORLD_DIR_ANGLES = jnp.array(
    [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0], dtype=jnp.float32
)

# Freeway action indices: 0=NOOP, 1=UP, 2=DOWN
_FREEWAY_NOOP = jnp.int32(0)

# move_table[dir_idx] — Freeway only has vertical movement.
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


def _freeway_get_player(raw_obs):
    return raw_obs.chicken


freeway_inverse_action_mapper = make_inverse_ego_action_mapper(
    noop_action=int(_FREEWAY_NOOP),
    fire_action=int(_FREEWAY_NOOP),
    move_table=_MOVE_TABLE,
    move_fire_table=_MOVE_FIRE_TABLE,
    player_getter=_freeway_get_player,
    y_is_bottom_edge=True,
)


class FreewaySemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 1 meta NOOP + 10 cars + 1 fixed top collect goal = 12 total slots
        super().__init__(env, max_objects=12)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        """
        Map a single-frame FreewayObservation to a LabeledObjectObservation.

        Object dimensions are effective hitbox sizes (not raw sprite sizes) so the
        global AABB check fires at the same frame as JAXAtari's own collision.

        Cars use bottom-edge y convention from JAXAtari; the effective hitbox
        shrinks the car top by 2px (car_hit_inset_y_top=2), so we shift y up by 2
        and reduce height by 2 to match.

        The chicken's effective hitbox expands 2px upward (chicken_hit_inset_y_top=-2),
        so the collect goal is placed to match that extended reach.
        """
        car_labels = jnp.full(FREEWAY_NUM_LANES, LABEL_AVOID, dtype=jnp.int32)
        car_offscreen = jnp.zeros(FREEWAY_NUM_LANES, dtype=jnp.int32)  # cars wrap, never scroll off

        # Effective car hitbox: shift bottom-edge y up by 2 to account for
        # car_hit_inset_y_top=2, and use effective height.
        car_x = obs.car.x + 0                           # x inset is 0
        car_y = obs.car.y - 2                            # top is inset 2px from sprite top
        car_w = jnp.full(FREEWAY_NUM_LANES, FREEWAY_CAR_HIT_W, dtype=jnp.int32)
        car_h = jnp.full(FREEWAY_NUM_LANES, FREEWAY_CAR_HIT_H, dtype=jnp.int32)

        # Fixed collect goal: placed so the chicken's effective hitbox (which extends
        # 2px above sprite top due to chicken_hit_inset_y_top=-2) overlaps it exactly
        # when the chicken reaches the scoring line.
        # Bottom-edge y convention: goal bottom aligned with scoring-line overlap, shifted up.
        goal_x      = jnp.array([FREEWAY_CHICKEN_X + 1], dtype=jnp.int32)   # match chicken x-inset
        goal_y      = jnp.array(
            [FREEWAY_TOP_BORDER + FREEWAY_CHICKEN_HIT_H - FREEWAY_COLLECT_GOAL_UP_OFFSET_PX],
            dtype=jnp.int32,
        )
        goal_w      = jnp.array([FREEWAY_CHICKEN_HIT_W], dtype=jnp.int32)
        goal_h      = jnp.array([FREEWAY_CHICKEN_HIT_H], dtype=jnp.int32)
        goal_active = jnp.array([1], dtype=jnp.int32)
        goal_label  = jnp.array([LABEL_COLLECT], dtype=jnp.int32)
        goal_visual_id = jnp.array([-1], dtype=jnp.int32)
        goal_offscreen = jnp.array([0], dtype=jnp.int32)  # fixed goal never scrolls

        all_x          = jnp.concatenate([car_x,                           goal_x])
        all_y          = jnp.concatenate([car_y,                           goal_y])
        all_w          = jnp.concatenate([car_w,                           goal_w])
        all_h          = jnp.concatenate([car_h,                           goal_h])
        all_active     = jnp.concatenate([obs.car.active.astype(jnp.int32), goal_active])
        all_labels     = jnp.concatenate([car_labels,                      goal_label])
        all_visual_id  = jnp.concatenate(
            [jnp.arange(FREEWAY_NUM_LANES, dtype=jnp.int32), goal_visual_id]
        )
        all_offscreen  = jnp.concatenate([car_offscreen,                   goal_offscreen])

        return LabeledObjectObservation(
            x=all_x,
            y=all_y,
            width=all_w,
            height=all_h,
            active=all_active,
            semantic_label=all_labels,
            visual_id=all_visual_id,
            can_go_offscreen=all_offscreen,
            was_killed=jnp.zeros_like(all_offscreen),
        )


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info=None, config=None):
    """
    Freeway skill rewards — delegates entirely to the global reward contract.

    Freeway cars wrap horizontally and never scroll off-screen, so no screen-edge
    bounds are needed (can_go_offscreen=0 for all objects).
    """
    del info, config
    return extract_skill_rewards(
        prev_raw_obs,
        next_raw_obs,
        prev_target,
        next_target,
        life_lost,
    )
