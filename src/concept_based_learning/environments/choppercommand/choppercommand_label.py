import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import extract_skill_rewards
from concept_based_learning.observations.action_space import make_ego_action_mapper

# ChopperCommand screen boundaries — all objects scroll off left/right edges.
def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    del next_raw_obs
    return extract_skill_rewards(
        prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost
    )


choppercommand_action_mapper = make_ego_action_mapper(
    noop_action=0,
    fire_action=1,
    move_table=     [3, 8, 5, 9, 4, 7, 2, 6],
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)


class ChopperCommandSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # Reduced from 85 to 84 since we removed the dummy waypoint
        super().__init__(env, max_objects=85) 

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

        # --- Array Padding to game_max_objects (84) ---
        pad_size = self.game_max_objects - final_x_unpadded.shape[0]
        
        pad_x = jnp.zeros(pad_size, dtype=jnp.float32)
        pad_y = jnp.zeros(pad_size, dtype=jnp.float32)
        pad_w = jnp.zeros(pad_size, dtype=jnp.float32)
        pad_h = jnp.zeros(pad_size, dtype=jnp.float32)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)

        # All game objects scroll off the left/right edges.
        all_offscreen = jnp.ones(final_x_unpadded.shape[0], dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        final_x = jnp.concatenate([final_x_unpadded, pad_x])
        final_y = jnp.concatenate([final_y_unpadded, pad_y])
        final_w = jnp.concatenate([final_w_unpadded, pad_w])
        final_h = jnp.concatenate([final_h_unpadded, pad_h])
        final_active = jnp.concatenate([final_active_unpadded, pad_active])
        final_labels = jnp.concatenate([final_labels_unpadded, pad_labels])
        final_offscreen = jnp.concatenate([all_offscreen, pad_offscreen])

        return LabeledObjectObservation(
            x=final_x, y=final_y, width=final_w, height=final_h,
            active=final_active, semantic_label=final_labels,
            can_go_offscreen=final_offscreen,
            was_killed=jnp.zeros_like(final_offscreen),
        )
