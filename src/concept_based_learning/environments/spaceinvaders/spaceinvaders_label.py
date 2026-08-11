import jax.numpy as jnp
from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.observations.action_space import make_target_aligned_action_mappers
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
    )


# SpaceInvaders: purely horizontal movement, 6 actions (0=NOOP,1=FIRE,2=RIGHT,3=LEFT,4=RIGHTFIRE,5=LEFTFIRE).
# Direction table maps the 8 world directions to horizontal actions; UP/DOWN collapse to NOOP/FIRE.
spaceinvaders_action_mapper, spaceinvaders_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [2, 2, 0, 3, 3, 3, 0, 2],   # R DR D DL L UL U UR
    move_fire_table=[4, 4, 1, 5, 5, 5, 1, 4],
)


class SpaceInvadersSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 36 Enemies + 3 Enemy Bullets + 1 UFO = 40 active objects
        super().__init__(env, max_objects=41)

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Detect enemy kills this substep.

        Returns a 40-element int32 mask (game layout: 36 enemies + 3 bullets + 1 UFO).
        Enemy kill: destroyed goes 0→1 (hit registered).
        UFO kill: ufo_state goes 1→2 (explosion starts).
        """
        # Enemy kills: shape (36,)
        enemy_killed = jnp.logical_and(
            prev_env_state.destroyed == 0,
            next_env_state.destroyed == 1,
        ).astype(jnp.int32)

        # Bullets are never "killed" in a way we reward — zeros
        bullet_killed = jnp.zeros(3, dtype=jnp.int32)

        # UFO kill: state transitions 1→2
        ufo_killed = jnp.logical_and(
            prev_env_state.ufo_state == 1,
            next_env_state.ufo_state == 2,
        ).astype(jnp.int32)
        ufo_killed = jnp.expand_dims(ufo_killed, axis=0)

        return jnp.concatenate([enemy_killed, bullet_killed, ufo_killed])  # (40,)

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs

        # Game layout: [36 enemies | 3 bullets | 1 ufo] = 40 slots.
        # After prepend_meta_noop_slot: shift by 1. Remaining padding stays 0.
        was_killed_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), kill_mask.astype(jnp.int32)]
        )  # (41,)

        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        enemies = obs.enemies                 # n=36
        enemy_bullets = obs.enemy_bullets     # n=3
        ufo = obs.ufo                         # n=1

        # Note: We skip `player_bullet` because we don't want the agent to target its own bullet

        enemy_labels = jnp.full(36, LABEL_DESTROY, dtype=jnp.int32)
        enemy_bullet_labels = jnp.full(3, LABEL_AVOID, dtype=jnp.int32)
        ufo_label = jnp.array([LABEL_DESTROY], dtype=jnp.int32)

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
        all_offscreen = jnp.ones(all_x.shape[0], dtype=jnp.int32)

        pad_size = self.game_max_objects - all_x.shape[0]
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=all_active.dtype)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([all_x, pad_x]),
            y=jnp.concatenate([all_y, pad_y]),
            width=jnp.concatenate([all_w, pad_w]),
            height=jnp.concatenate([all_h, pad_h]),
            active=jnp.concatenate([all_active, pad_active]),
            semantic_label=jnp.concatenate([all_labels, pad_labels]),
            state=jnp.zeros(all_x.shape[0] + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate([all_offscreen, pad_offscreen]),
            was_killed=jnp.zeros(all_x.shape[0] + pad_size, dtype=jnp.int32),
        )
