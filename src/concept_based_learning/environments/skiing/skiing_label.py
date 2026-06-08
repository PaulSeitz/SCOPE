import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import extract_skill_rewards
from concept_based_learning.observations.action_space import make_ego_action_mapper


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    return extract_skill_rewards(
        prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost
    )


# Skiing: actions NOOP=0, RIGHT=1, LEFT=2.
# Skier at fixed y=46, world scrolls. Pure horizontal movement.
skiing_action_mapper = make_ego_action_mapper(
    noop_action=0,
    fire_action=0,
    move_table=     [1, 1, 0, 2, 2, 2, 0, 1],  # R DR D DL L UL U UR
    move_fire_table=[1, 1, 0, 2, 2, 2, 0, 1],
)

# flag_distance = 32px (space between left and right pole of a gate)
_FLAG_DISTANCE = jnp.int32(32)

# Object layout (game_max_objects=10, +1 NOOP prepended by wrapper):
#   [0-1]  left poles  (AVOID)
#   [2-3]  right poles (AVOID, synthetic)
#   [4-5]  gate centers (COLLECT, synthetic, full 32px wide)
#   [6-9]  trees       (AVOID)
_N_FLAG_PAIRS = 2
_N_TREES      = 4
# Slot ranges used by _stamp_events_on_observation
_POLE_SLOTS   = slice(0, 4)   # left + right poles
_TREE_SLOTS   = slice(6, 10)


class SkiingSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 2 left + 2 right + 2 centers + 4 trees = 10; +1 NOOP = 11
        super().__init__(env, max_objects=11)

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Detect collision rising-edge from game state (more accurate than AABB for Skiing).

        collision_type: 0=none, 1=tree, 2=mogul, 3=flag.
        skier_fell: countdown timer; >0 means in recovery. Rising edge = new collision.
        Returns a 10-element mask over game slots matching the layout above (+1 NOOP later).
        """
        new_collision = jnp.logical_and(
            prev_env_state.skier_fell == 0,
            next_env_state.skier_fell > 0,
        )
        hit_flag = jnp.logical_and(new_collision, next_env_state.collision_type == 3)
        hit_tree = jnp.logical_and(new_collision, next_env_state.collision_type == 1)

        # Stamp only slot 0 (first left pole) for flag hits, slot 0 of trees for tree hits.
        # One representative slot per collision → one penalty per event regardless of active count.
        pole_mask   = jnp.zeros(_N_FLAG_PAIRS * 2, dtype=jnp.int32).at[0].set(hit_flag.astype(jnp.int32))
        center_mask = jnp.zeros(_N_FLAG_PAIRS, dtype=jnp.int32)
        tree_mask   = jnp.zeros(_N_TREES, dtype=jnp.int32).at[0].set(hit_tree.astype(jnp.int32))
        return jnp.concatenate([pole_mask, center_mask, tree_mask])  # (10,)

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        # Prepend NOOP slot (index 0 is meta-NOOP, game objects start at 1).
        was_killed_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), kill_mask.astype(jnp.int32)]
        )  # (11,)
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        flags = obs.flags   # n=2, each is the LEFT pole of one gate
        trees = obs.trees   # n=4

        # Left poles (from observation directly)
        left_x = flags.x
        left_y = flags.y
        left_w = flags.width
        left_h = flags.height
        left_active = flags.active

        # Right poles (synthesised: same y, x shifted by flag_distance)
        right_x = left_x + _FLAG_DISTANCE
        right_y = left_y
        right_w = left_w
        right_h = left_h
        right_active = left_active

        # Gate center waypoints: full 32px wide (the entire gate opening), left edge at left pole x.
        center_x = left_x
        center_y = left_y
        center_w = jnp.full_like(left_w, _FLAG_DISTANCE)
        center_h = left_h
        center_active = left_active

        n_flag_pairs = flags.x.shape[0]   # 2
        n_trees = trees.x.shape[0]        # 4

        pole_labels = jnp.full(n_flag_pairs * 2, LABEL_AVOID, dtype=jnp.int32)
        center_labels = jnp.full(n_flag_pairs, LABEL_COLLECT, dtype=jnp.int32)
        tree_labels = jnp.full(n_trees, LABEL_AVOID, dtype=jnp.int32)

        all_x = jnp.concatenate([left_x, right_x, center_x, trees.x])
        all_y = jnp.concatenate([left_y, right_y, center_y, trees.y])
        all_w = jnp.concatenate([left_w, right_w, center_w, trees.width])
        all_h = jnp.concatenate([left_h, right_h, center_h, trees.height])
        all_active = jnp.concatenate([left_active, right_active, center_active, trees.active])
        all_labels = jnp.concatenate([pole_labels, center_labels, tree_labels])

        n_objects = all_x.shape[0]
        pad_size = self.game_max_objects - n_objects
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)

        # Objects scroll off screen when the world scrolls past; set can_go_offscreen=1.
        all_offscreen = jnp.ones(n_objects, dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([all_x, pad_x]),
            y=jnp.concatenate([all_y, pad_y]),
            width=jnp.concatenate([all_w, pad_w]),
            height=jnp.concatenate([all_h, pad_h]),
            active=jnp.concatenate([all_active, pad_active]),
            semantic_label=jnp.concatenate([all_labels, pad_labels]),
            state=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate([all_offscreen, pad_offscreen]),
            was_killed=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
        )
