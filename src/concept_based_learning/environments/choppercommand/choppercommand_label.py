import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_AVOID,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import (
    avoid_survival_reward_from_config,
    extract_skill_rewards,
    penalize_collect_death_from_config,
)
from concept_based_learning.observations.action_space import make_target_aligned_action_mappers

def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info=None, config=None):
    del info
    return extract_skill_rewards(
        prev_raw_obs,
        next_raw_obs,
        prev_target,
        next_target,
        life_lost,
        avoid_survival_reward=avoid_survival_reward_from_config(config),
        penalize_collect_death=penalize_collect_death_from_config(config),
        config=config,
    )


choppercommand_action_mapper, choppercommand_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [3, 8, 5, 9, 4, 7, 2, 6],
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)


_FRAMES_DEATH_ENEMY = 16  # FRAMES_DEATH_ANIMATION_ENEMY default
# Keep only the 8 on-screen-closest missiles; 48 total slots are mostly inactive.
_MISSILE_SLOTS = 8


class ChopperCommandSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 12 jets + 12 choppers + 8 capped missiles = 32 game objects + 1 NOOP
        # Trucks excluded: they fly on the ground (y≈156) and the player chopper (y≈100)
        # can never physically overlap them, so AVOID/COLLECT would never fire.
        super().__init__(env, max_objects=33)

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Detect jet/chopper kills this substep.

        Kill signal: death_timer transitions from > 16 (lane flag) to <= 16 (dying).
        The game sets timer=16 on hit then immediately decrements to 15 in the same step.
        Returns a mask over game slots [jets(12) | choppers(12) | missiles(8)].
        """
        def _kill_mask(prev_pos, next_pos):
            prev_timer = prev_pos[:, 3]
            next_timer = next_pos[:, 3]
            direction = prev_pos[:, 2]
            # kill_entity sets timer = 16, then update_entity_death immediately
            # decrements to 15 in the same step. So the kill frame has:
            # prev > 16 (alive lane-flag) → next == 15 (first dying frame).
            # Use next <= 16 to be robust against any threshold offset.
            return jnp.logical_and(
                jnp.logical_and(direction != 0, prev_timer > _FRAMES_DEATH_ENEMY),
                jnp.logical_and(next_timer > 0, next_timer <= _FRAMES_DEATH_ENEMY),
            ).astype(jnp.int32)

        jet_killed     = _kill_mask(prev_env_state.jet_positions,    next_env_state.jet_positions)
        chopper_killed = _kill_mask(prev_env_state.chopper_positions, next_env_state.chopper_positions)
        missile_zeros  = jnp.zeros(_MISSILE_SLOTS, dtype=jnp.int32)
        return jnp.concatenate([jet_killed, chopper_killed, missile_zeros])  # (32,)

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        # Layout: [jets(12) | choppers(12) | missiles(8)] = 32 → +1 NOOP = 33
        was_killed_with_noop = self._with_meta_event_padding(kill_mask)
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        jets = obs.jets
        choppers = obs.choppers
        projectiles = obs.enemy_missiles

        num_jets = jets.x.shape[0]
        num_choppers = choppers.x.shape[0]

        # Cap missiles to _MISSILE_SLOTS (8) by keeping the on-screen active ones.
        m_active = projectiles.active.astype(jnp.float32)
        m_on_screen = ((projectiles.x + projectiles.width > 0) & (projectiles.x < 160)).astype(jnp.float32)
        m_score = m_active * m_on_screen
        top_idx = jnp.argsort(-m_score)[:_MISSILE_SLOTS]
        m_x = projectiles.x[top_idx]
        m_y = projectiles.y[top_idx]
        m_w = projectiles.width[top_idx]
        m_h = projectiles.height[top_idx]
        m_active_cap = (projectiles.active[top_idx] * m_on_screen[top_idx].astype(projectiles.active.dtype))

        enemy_labels = jnp.full(num_jets + num_choppers, LABEL_DESTROY, dtype=jnp.int32)
        enemy_missile_labels = jnp.full(_MISSILE_SLOTS, LABEL_AVOID, dtype=jnp.int32)

        final_x_unpadded = jnp.concatenate([jets.x, choppers.x, m_x])
        final_y_unpadded = jnp.concatenate([jets.y, choppers.y, m_y])
        final_w_unpadded = jnp.concatenate([jets.width, choppers.width, m_w])
        final_h_unpadded = jnp.concatenate([jets.height, choppers.height, m_h])
        final_active_unpadded = jnp.concatenate([jets.active, choppers.active, m_active_cap])
        final_labels_unpadded = jnp.concatenate([enemy_labels, enemy_missile_labels])

        is_on_screen = (final_x_unpadded + final_w_unpadded > 0) & (final_x_unpadded < 160)

        # Snap one off-screen DESTROY target to the screen edge so the meta-policy
        # can navigate toward incoming threats (label stays DESTROY — no
        # NAVIGATE/COLLECT reward on arrival).
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
        # Keep DESTROY label so NAVIGATE/COLLECT rewards cannot fire at the
        # screen-edge proxy — only a real kill stamps DESTROY success.
        final_labels_unpadded = final_labels_unpadded.at[virtual_idx].set(
            jnp.where(has_virtual_waypoint, LABEL_DESTROY, final_labels_unpadded[virtual_idx])
        )
        final_active_unpadded = final_active_unpadded.at[virtual_idx].set(
            jnp.where(has_virtual_waypoint, 1, final_active_unpadded[virtual_idx])
        )

        final_active_unpadded = jnp.where(is_on_screen, final_active_unpadded, 0)
        final_active_unpadded = final_active_unpadded.at[virtual_idx].set(
            jnp.where(has_virtual_waypoint, 1, final_active_unpadded[virtual_idx])
        )

        # --- Array Padding to game_max_objects (32) ---
        pad_size = self.game_max_objects - final_x_unpadded.shape[0]
        
        pad_x = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_y = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_w = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_h = jnp.zeros(pad_size, dtype=jnp.int32)
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
            state=jnp.zeros_like(final_offscreen),
            can_go_offscreen=final_offscreen,
            was_killed=jnp.zeros_like(final_offscreen),
        )
