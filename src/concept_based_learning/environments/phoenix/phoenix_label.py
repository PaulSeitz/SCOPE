import jax
import jax.numpy as jnp
from typing import Any
from jaxatari.environment import JAXAtariAction as Atari

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
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


def phoenix_action_mapper(
    ego_action: jnp.ndarray,
    global_features: Any,
    target_idx: jnp.ndarray,
) -> jnp.ndarray:
    """
    Translates the 18-space Egocentric Action to Phoenix environment indices.
    Because we force the orientation to 90 degrees (Right) to keep the network 
    In-Distribution, the agent's axes are rotated. 
    Forward = Visual Right. Backward = Visual Left. Left = Visual Up. Right = Visual Down.
    
    Phoenix Indices: 
    0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=DOWN(Shield), 5=RIGHTFIRE, 6=LEFTFIRE, 7=DOWNFIRE
    """
    
    action_map = jnp.array([
        0,  # 0: NOOP -> 0 (NOOP)
        1,  # 1: FIRE -> 1 (FIRE)
        2,  # 2: FORWARD -> 2 (RIGHT to align X)
        4,  # 3: RIGHT -> 4 (DOWN / Shield)
        1,  # 4: LEFT -> 1 (Target is ABOVE, so FIRE at it!)
        3,  # 5: BACKWARD -> 3 (LEFT to align X)
        2,  # 6: FORWARD_RIGHT -> 2 (RIGHT)
        5,  # 7: FORWARD_LEFT -> 5 (RIGHTFIRE - aligning and shooting)
        3,  # 8: BACKWARD_RIGHT -> 3 (LEFT)
        6,  # 9: BACKWARD_LEFT -> 6 (LEFTFIRE - aligning and shooting)
        5,  # 10: FORWARD_FIRE -> 5 (RIGHTFIRE)
        7,  # 11: RIGHT_FIRE -> 7 (DOWNFIRE)
        1,  # 12: LEFT_FIRE -> 1 (FIRE)
        6,  # 13: BACKWARD_FIRE -> 6 (LEFTFIRE)
        5,  # 14: FORWARD_RIGHT_FIRE -> 5 (RIGHTFIRE)
        5,  # 15: FORWARD_LEFT_FIRE -> 5 (RIGHTFIRE)
        6,  # 16: BACKWARD_RIGHT_FIRE -> 6 (LEFTFIRE)
        6   # 17: BACKWARD_LEFT_FIRE -> 6 (LEFTFIRE)
    ], dtype=jnp.int32)

    return action_map[ego_action]


class PhoenixSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 8 Enemies + 8 Enemy Bullets + 1 Boss + 204 Boss Blocks = 221 objects
        super().__init__(env, max_objects=221) 

    def step(self, state, action):
        """Override step to trick the network's orientation distribution."""
        obs, env_state, reward, done, info = super().step(state, action)
        
        # Force orientation to 90.0 (Right) so the Seaquest-trained network doesn't panic
        new_player = obs.raw_observation.player.replace(
            orientation=jnp.full_like(obs.raw_observation.player.orientation, 90.0)
        )
        new_raw_obs = obs.raw_observation.replace(player=new_player)
        obs = obs.replace(raw_observation=new_raw_obs)
        
        return obs, env_state, reward, done, info

    def reset(self, key=None):
        """Override reset to trick the network's orientation distribution."""
        obs, env_state = super().reset(key)
        
        new_player = obs.raw_observation.player.replace(
            orientation=jnp.full_like(obs.raw_observation.player.orientation, 90.0)
        )
        new_raw_obs = obs.raw_observation.replace(player=new_player)
        obs = obs.replace(raw_observation=new_raw_obs)
        
        return obs, env_state

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        enemies = obs.enemies                 
        enemy_bullets = obs.enemy_projectiles 
        boss = obs.boss                       
        boss_blocks = obs.boss_blocks         

        # Derive counts from the underlying JAXAtari observation to stay robust
        num_enemies = enemies.x.shape[-1]
        num_enemy_bullets = enemy_bullets.x.shape[-1]
        num_blocks = boss_blocks.x.shape[-1]

        enemy_labels = jnp.full(num_enemies, LABEL_DESTROY, dtype=jnp.int32)
        enemy_bullet_labels = jnp.full(num_enemy_bullets, LABEL_AVOID, dtype=jnp.int32)
        boss_label = jnp.array([LABEL_DESTROY], dtype=jnp.int32)
        # Boss blocks should not be treated as direct DESTROY targets; they act as
        # environmental structure around the boss.
        block_labels = jnp.full(num_blocks, LABEL_NOOP, dtype=jnp.int32)

        def extract_single_field(field):
            return jnp.expand_dims(field, axis=0) if field.ndim == 0 else field

        boss_x = extract_single_field(boss.x)
        boss_y = extract_single_field(boss.y)
        boss_w = extract_single_field(boss.width)
        boss_h = extract_single_field(boss.height)
        boss_active = extract_single_field(boss.active)

        # Carry over per-object state so the policy can treat dying enemies as inactive.
        # Only enemies use the special state encoding; others get a neutral default (0).
        enemy_state = enemies.state
        zeros_enemy_bullets_state = jnp.zeros_like(enemy_bullets.x, dtype=jnp.int32)
        zeros_boss_state = jnp.zeros_like(boss_x, dtype=jnp.int32)
        zeros_blocks_state = jnp.zeros_like(boss_blocks.x, dtype=jnp.int32)

        all_x = jnp.concatenate([enemies.x, enemy_bullets.x, boss_x, boss_blocks.x])
        all_y = jnp.concatenate([enemies.y, enemy_bullets.y, boss_y, boss_blocks.y])
        all_w = jnp.concatenate([enemies.width, enemy_bullets.width, boss_w, boss_blocks.width])
        all_h = jnp.concatenate([enemies.height, enemy_bullets.height, boss_h, boss_blocks.height])
        all_active = jnp.concatenate([enemies.active, enemy_bullets.active, boss_active, boss_blocks.active])
        all_labels = jnp.concatenate([enemy_labels, enemy_bullet_labels, boss_label, block_labels])
        all_state = jnp.concatenate(
            [enemy_state, zeros_enemy_bullets_state, zeros_boss_state, zeros_blocks_state]
        )

        return LabeledObjectObservation(
            x=all_x,
            y=all_y,
            width=all_w,
            height=all_h,
            active=all_active,
            semantic_label=all_labels,
            state=all_state,
        )