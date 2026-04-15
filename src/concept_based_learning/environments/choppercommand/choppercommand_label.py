import jax
import jax.numpy as jnp
from flax import struct
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

    prev_player = last_frame(prev_raw_obs.player)
    next_player = last_frame(next_raw_obs.player)
    prev_target = last_frame(prev_target)
    next_target = last_frame(next_target)

    was_active = prev_target.active == 1
    is_active = next_target.active == 1
    target_destroyed = jnp.logical_and(was_active, jnp.logical_not(is_active))

    hit_by_assigned = _bbox_overlap(prev_player, prev_target)

    t_left = prev_target.x
    t_right = prev_target.x + prev_target.width
    t_top = prev_target.y
    t_bottom = prev_target.y + prev_target.height
    p_left = prev_player.x
    p_right = prev_player.x + prev_player.width
    p_top = prev_player.y
    p_bottom = prev_player.y + prev_player.height
    player_overlap = jnp.logical_and(
        jnp.logical_and(p_left <= t_right, p_right >= t_left),
        jnp.logical_and(p_top <= t_bottom, p_bottom >= t_top)
    )

    is_collect_target = prev_target.semantic_label == LABEL_COLLECT
    collect_success = jnp.logical_and(
        is_collect_target,
        jnp.logical_and(was_active, player_overlap),
    )

    # Destroy: assigned enemy died and we didn't crash into it → we must have shot it.
    is_destroy_target = prev_target.semantic_label == LABEL_DESTROY
    destroy_success = jnp.logical_and(
        is_destroy_target,
        jnp.logical_and(target_destroyed, jnp.logical_not(hit_by_assigned)),
    )

    destroy_failure = (life_lost > 0) & is_destroy_target & hit_by_assigned
    avoid_failure = (life_lost > 0) & (prev_target.semantic_label == LABEL_AVOID) & hit_by_assigned

    collect_reward = jnp.where(collect_success & (life_lost == 0), 1.0, 0.0)
    destroy_reward = jnp.where(
        destroy_success & (life_lost == 0),
        1.0,
        jnp.where(destroy_failure, -1.0, 0.0),
    )
    avoid_reward = jnp.where(avoid_failure, -1.0, 0.0)

    collect_done = collect_success
    destroy_done = destroy_success | destroy_failure
    avoid_done = avoid_failure

    rewards = jnp.array([destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32)
    skill_done = jnp.array([destroy_done, collect_done, avoid_done], dtype=jnp.bool_)
    return rewards, skill_done


def choppercommand_action_mapper(ego_action: jnp.ndarray, global_features: Any, target_idx: jnp.ndarray) -> jnp.ndarray:
    player_ori = global_features.player.orientation[-1] 
    is_facing_right = player_ori < 180.0
    
    map_right = jnp.array([
        Atari.NOOP, Atari.FIRE, Atari.RIGHT, Atari.DOWN, Atari.UP, Atari.LEFT,
        Atari.DOWNRIGHT, Atari.UPRIGHT, Atari.DOWNLEFT, Atari.UPLEFT,
        Atari.RIGHTFIRE, Atari.DOWNFIRE, Atari.UPFIRE, Atari.LEFTFIRE,
        Atari.DOWNRIGHTFIRE, Atari.UPRIGHTFIRE, Atari.DOWNLEFTFIRE, Atari.UPLEFTFIRE
    ], dtype=jnp.int32)
    
    map_left = jnp.array([
        Atari.NOOP, Atari.FIRE, Atari.LEFT, Atari.UP, Atari.DOWN, Atari.RIGHT,
        Atari.UPLEFT, Atari.DOWNLEFT, Atari.UPRIGHT, Atari.DOWNRIGHT,
        Atari.LEFTFIRE, Atari.UPFIRE, Atari.DOWNFIRE, Atari.RIGHTFIRE,
        Atari.UPLEFTFIRE, Atari.DOWNLEFTFIRE, Atari.UPRIGHTFIRE, Atari.DOWNRIGHTFIRE
    ], dtype=jnp.int32)

    chosen_map = jnp.where(is_facing_right, map_right, map_left)
    return chosen_map[ego_action]


class ChopperCommandSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # Reduced from 85 to 84 since we removed the dummy waypoint
        super().__init__(env, max_objects=84) 

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        jets = obs.jets
        choppers = obs.choppers
        trucks = obs.trucks
        projectiles = obs.enemy_missiles

        num_jets = jets.x.shape[0]
        num_choppers = choppers.x.shape[0]
        num_trucks = trucks.x.shape[0]
        num_enemy_missiles = projectiles.x.shape[0]

        enemy_labels = jnp.full(num_jets + num_choppers, LABEL_DESTROY, dtype=jnp.int32)
        truck_labels = jnp.full(num_trucks, LABEL_AVOID, dtype=jnp.int32)
        enemy_missile_labels = jnp.full(num_enemy_missiles, LABEL_AVOID, dtype=jnp.int32)

        final_x_unpadded = jnp.concatenate([jets.x, choppers.x, trucks.x, projectiles.x])
        final_y_unpadded = jnp.concatenate([jets.y, choppers.y, trucks.y, projectiles.y])
        final_w_unpadded = jnp.concatenate([jets.width, choppers.width, trucks.width, projectiles.width])
        final_h_unpadded = jnp.concatenate([jets.height, choppers.height, trucks.height, projectiles.height])
        final_active_unpadded = jnp.concatenate([jets.active, choppers.active, trucks.active, projectiles.active])
        final_labels_unpadded = jnp.concatenate([enemy_labels, truck_labels, enemy_missile_labels])

        is_on_screen = (final_x_unpadded + final_w_unpadded > 0) & (final_x_unpadded < 160)

        # Inject a single virtual waypoint from an off-screen enemy so the
        # meta-policy can navigate toward incoming threats without any trainer logic.
        is_enemy = final_labels_unpadded == LABEL_DESTROY
        is_offscreen_enemy = jnp.logical_and(
            is_enemy,
            jnp.logical_and(final_active_unpadded > 0, jnp.logical_not(is_on_screen)),
        )
        has_virtual_waypoint = jnp.any(is_offscreen_enemy)
        virtual_idx = jnp.argmax(is_offscreen_enemy).astype(jnp.int32)

        enemy_x = final_x_unpadded[virtual_idx]
        enemy_w = final_w_unpadded[virtual_idx]
        enemy_y = final_y_unpadded[virtual_idx]
        offscreen_left = (enemy_x + enemy_w) <= 0
        virtual_x = jnp.where(
            offscreen_left,
            jnp.asarray(0, dtype=final_x_unpadded.dtype),
            jnp.asarray(159, dtype=final_x_unpadded.dtype),
        )
        virtual_y = jnp.clip(
            enemy_y,
            jnp.asarray(0, dtype=final_y_unpadded.dtype),
            jnp.asarray(210, dtype=final_y_unpadded.dtype),
        )

        final_x_unpadded = final_x_unpadded.at[virtual_idx].set(
            jnp.where(has_virtual_waypoint, virtual_x, final_x_unpadded[virtual_idx])
        )
        final_y_unpadded = final_y_unpadded.at[virtual_idx].set(
            jnp.where(has_virtual_waypoint, virtual_y, final_y_unpadded[virtual_idx])
        )
        final_labels_unpadded = final_labels_unpadded.at[virtual_idx].set(
            jnp.where(has_virtual_waypoint, LABEL_COLLECT, final_labels_unpadded[virtual_idx])
        )
        final_active_unpadded = final_active_unpadded.at[virtual_idx].set(
            jnp.where(has_virtual_waypoint, 1, final_active_unpadded[virtual_idx])
        )

        final_active_unpadded = jnp.where(is_on_screen, final_active_unpadded, 0)
        final_active_unpadded = final_active_unpadded.at[virtual_idx].set(
            jnp.where(has_virtual_waypoint, 1, final_active_unpadded[virtual_idx])
        )

        # --- Array Padding to MAX_OBJECTS (84) ---
        pad_size = self.max_objects - final_x_unpadded.shape[0]
        
        pad_x = jnp.zeros(pad_size, dtype=jnp.float32)
        pad_y = jnp.zeros(pad_size, dtype=jnp.float32)
        pad_w = jnp.zeros(pad_size, dtype=jnp.float32)
        pad_h = jnp.zeros(pad_size, dtype=jnp.float32)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)

        final_x = jnp.concatenate([final_x_unpadded, pad_x])
        final_y = jnp.concatenate([final_y_unpadded, pad_y])
        final_w = jnp.concatenate([final_w_unpadded, pad_w])
        final_h = jnp.concatenate([final_h_unpadded, pad_h])
        final_active = jnp.concatenate([final_active_unpadded, pad_active])
        final_labels = jnp.concatenate([final_labels_unpadded, pad_labels])

        return LabeledObjectObservation(
            x=final_x, y=final_y, width=final_w, height=final_h,
            active=final_active, semantic_label=final_labels
        )