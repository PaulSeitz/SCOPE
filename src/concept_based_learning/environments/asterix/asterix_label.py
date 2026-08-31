"""Asterix semantic labeling: collectibles = COLLECT, enemies = AVOID."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import (
    avoid_survival_reward_from_config,
    extract_skill_rewards,
    penalize_collect_death_from_config,
)
from concept_based_learning.observations.action_space import make_target_aligned_action_mappers


def extract_all_skills_rewards(
    prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info=None, config=None
):
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


# Asterix Discrete(9) indices into ACTION_SET / move tables:
#   0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN, 5=UPRIGHT, 6=UPLEFT, 7=DOWNRIGHT, 8=DOWNLEFT
# World dirs: R DR D DL L UL U UR
_ASTERIX_MOVE = [2, 7, 4, 8, 3, 6, 1, 5]

asterix_action_mapper, asterix_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=0,  # no FIRE in Asterix
    move_table=_ASTERIX_MOVE,
    move_fire_table=_ASTERIX_MOVE,
)

# Layout (game_max_objects=16, +1 meta NOOP → 17):
#   [0-7]  enemies      (AVOID)
#   [8-15] collectibles (COLLECT)
_N_LANES = 8
_ASTERIX_MAX_OBJECTS = 17  # 16 game + 1 NOOP

# JAXAtari player_direction: 1 = left, 2 = right (sprite index, not screen bearing).
# Map to ObjectObservation.orientation so POV facing tracks turns.
_ASTERIX_ORI_LEFT = 270.0
_ASTERIX_ORI_RIGHT = 90.0


def asterix_sprite_orientation_from_direction(player_direction) -> jnp.ndarray:
    """JAXAtari sprite orientation for POV facing (90=right, 270=left)."""
    direction = jnp.asarray(player_direction, dtype=jnp.int32)
    return jnp.where(
        direction == 2,
        jnp.float32(_ASTERIX_ORI_RIGHT),
        jnp.float32(_ASTERIX_ORI_LEFT),
    )


class AsterixSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_ASTERIX_MAX_OBJECTS)

    def _patch_player_facing_from_direction(self, obs, env_state):
        """Stamp walk direction into player.orientation for facing-frame POV."""
        player = obs.raw_observation.player
        base = self._unwrap_env_state(env_state)
        sprite_ori = asterix_sprite_orientation_from_direction(base.player_direction)
        new_player = player.replace(
            orientation=jnp.full_like(
                jnp.asarray(player.orientation, dtype=jnp.float32),
                sprite_ori,
            )
        )
        return obs.replace(
            raw_observation=obs.raw_observation.replace(player=new_player)
        )

    def step(self, state, action):
        obs, env_state, reward, done, info = super().step(state, action)
        obs = self._patch_player_facing_from_direction(obs, env_state)
        return obs, env_state, reward, done, info

    def reset(self, key=None):
        obs, env_state = super().reset(key)
        obs = self._patch_player_facing_from_direction(obs, env_state)
        return obs, env_state

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Enemy contact → was_killed on the colliding lane(s).

        Scroll-off also clears ``alive``; require a life loss so only hits stamp.
        """
        life_lost = next_env_state.lives < prev_env_state.lives
        enemy_died = jnp.logical_and(
            prev_env_state.enemies.alive, jnp.logical_not(next_env_state.enemies.alive)
        )
        enemy_killed = jnp.logical_and(enemy_died, life_lost).astype(jnp.int32)
        collect_zeros = jnp.zeros(_N_LANES, dtype=jnp.int32)
        return jnp.concatenate([enemy_killed, collect_zeros])  # (16,)

    def _substep_collect_events(self, prev_env_state, next_env_state):
        """
        Collectible pickup → was_collected.

        Scroll-off also clears ``alive``; keep events whose previous x was still
        on-screen (collision kills happen on-playfield, scroll-off at the edges).
        """
        item_died = jnp.logical_and(
            prev_env_state.collectibles.alive,
            jnp.logical_not(next_env_state.collectibles.alive),
        )
        prev_x = prev_env_state.collectibles.x
        on_screen = jnp.logical_and(prev_x > 0.0, prev_x < 152.0)
        collected = jnp.logical_and(item_died, on_screen).astype(jnp.int32)
        enemy_zeros = jnp.zeros(_N_LANES, dtype=jnp.int32)
        return jnp.concatenate([enemy_zeros, collected])  # (16,)

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        was_killed_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), kill_mask.astype(jnp.int32)]
        )
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def _stamp_collect_events_on_observation(self, semantic_obs, collect_mask):
        if collect_mask is None:
            return semantic_obs
        was_collected_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), collect_mask.astype(jnp.int32)]
        )
        objects = semantic_obs.objects
        new_wc = objects.was_collected.at[-1].set(was_collected_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_collected=new_wc))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        enemies = obs.enemies
        collectibles = obs.collectibles

        n_e = enemies.x.shape[0]
        n_c = collectibles.x.shape[0]

        all_x = jnp.concatenate([enemies.x, collectibles.x])
        all_y = jnp.concatenate([enemies.y, collectibles.y])
        all_w = jnp.concatenate([enemies.width, collectibles.width])
        all_h = jnp.concatenate([enemies.height, collectibles.height])
        all_active = jnp.concatenate(
            [enemies.active.astype(jnp.int32), collectibles.active.astype(jnp.int32)]
        )
        all_labels = jnp.concatenate(
            [
                jnp.full(n_e, LABEL_AVOID, dtype=jnp.int32),
                jnp.full(n_c, LABEL_COLLECT, dtype=jnp.int32),
            ]
        )
        # Entities scroll horizontally off the playfield.
        all_offscreen = jnp.ones(n_e + n_c, dtype=jnp.int32)

        n_objects = all_x.shape[0]
        pad_size = self.game_max_objects - n_objects
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)
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
            was_collected=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
        )
