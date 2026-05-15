import jax
import jax.numpy as jnp
from typing import Any
from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.observations.action_space import make_ego_action_mapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NOOP,
)

def _bbox_overlap(player, target):
    """True if player and target bounding boxes overlap (e.g. collision)."""
    p_left, p_right = player.x, player.x + player.width
    p_top, p_bottom = player.y, player.y + player.height
    t_left, t_right = target.x, target.x + target.width
    t_top, t_bottom = target.y, target.y + target.height
    horiz = jnp.logical_and(p_left <= t_right, p_right >= t_left)
    vert = jnp.logical_and(p_top <= t_bottom, p_bottom >= t_top)
    return jnp.logical_and(horiz, vert)


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    """
    Returns (rewards, skill_done). skill_done[k] is True only when the transition
    is terminal for skill k because the assigned target caused it (golden rule).
    """
    def last_frame(e):
        return jax.tree_util.tree_map(lambda x: x[-1] if x.ndim > 0 else x, e)

    prev_target = last_frame(prev_target)
    next_target = last_frame(next_target)
    prev_player = last_frame(prev_raw_obs.player)

    was_active = prev_target.active == 1
    is_active = next_target.active == 1
    success = jnp.logical_and(was_active, jnp.logical_not(is_active))

    hit_by_assigned = _bbox_overlap(prev_player, prev_target)

    is_destroy_target = prev_target.semantic_label == LABEL_DESTROY
    destroy_success = jnp.logical_and(
        is_destroy_target,
        jnp.logical_and(success, jnp.logical_not(hit_by_assigned)),
    )
    destroy_failure = (life_lost > 0) & is_destroy_target & hit_by_assigned
    avoid_failure = (life_lost > 0) & (prev_target.semantic_label == LABEL_AVOID) & hit_by_assigned

    destroy_reward = jnp.where(destroy_success & (life_lost == 0), 1.0, 0.0)
    collect_reward = jnp.zeros_like(destroy_reward)
    avoid_reward = jnp.where(avoid_failure, -1.0, 0.0)

    destroy_done = destroy_success | destroy_failure
    collect_done = jnp.zeros_like(destroy_reward, dtype=jnp.bool_)
    avoid_done = avoid_failure

    rewards = jnp.array([destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32)
    skill_done = jnp.array([destroy_done, collect_done, avoid_done], dtype=jnp.bool_)
    return rewards, skill_done


# SpaceInvaders: purely horizontal movement, 6 actions (0=NOOP,1=FIRE,2=RIGHT,3=LEFT,4=RIGHTFIRE,5=LEFTFIRE).
# Direction table maps the 8 world directions to horizontal actions; UP/DOWN collapse to NOOP/FIRE.
spaceinvaders_action_mapper = make_ego_action_mapper(
    noop_action=0,
    fire_action=1,
    move_table=     [2, 2, 0, 3, 3, 3, 0, 2],   # R DR D DL L UL U UR
    move_fire_table=[4, 4, 1, 5, 5, 5, 1, 4],
)


class SpaceInvadersSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 36 Enemies + 3 Enemy Bullets + 1 UFO = 40 active objects
        super().__init__(env, max_objects=40)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        """
        Takes a single Jaxatari SpaceInvaders observation, strips irrelevant entities,
        assigns universal labels, and flattens it into our universal object list.
        """
        enemies = obs.enemies                 # n=36
        enemy_bullets = obs.enemy_bullets     # n=3
        ufo = obs.ufo                         # n=1

        # Note: We skip `player_bullet` because we don't want the agent to target its own bullet

        # Create labels
        enemy_labels = jnp.full(36, LABEL_DESTROY, dtype=jnp.int32)
        enemy_bullet_labels = jnp.full(3, LABEL_AVOID, dtype=jnp.int32)
        ufo_label = jnp.array([LABEL_DESTROY], dtype=jnp.int32)

        # Concatenate features
        def extract_ufo_field(field):
            return jnp.expand_dims(field, axis=0) if field.ndim == 0 else field

        ufo_x = extract_ufo_field(ufo.x)
        ufo_y = extract_ufo_field(ufo.y)
        ufo_w = extract_ufo_field(ufo.width)
        ufo_h = extract_ufo_field(ufo.height)
        ufo_active = extract_ufo_field(ufo.active)

        all_x = jnp.concatenate([enemies.x, enemy_bullets.x, ufo_x])
        all_y = jnp.concatenate([enemies.y, enemy_bullets.y, ufo_y])
        all_w = jnp.concatenate([enemies.width, enemy_bullets.width, ufo_w])
        all_h = jnp.concatenate([enemies.height, enemy_bullets.height, ufo_h])
        all_active = jnp.concatenate([enemies.active, enemy_bullets.active, ufo_active])
        all_labels = jnp.concatenate([enemy_labels, enemy_bullet_labels, ufo_label])

        return LabeledObjectObservation(
            x=all_x,
            y=all_y,
            width=all_w,
            height=all_h,
            active=all_active,
            semantic_label=all_labels
        )
