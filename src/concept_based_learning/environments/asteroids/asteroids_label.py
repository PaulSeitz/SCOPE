"""Asteroids semantic labeling: all asteroids are DESTROY."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
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


# ACTION_SET indices (Discrete(14) → ALE ids):
#   0=NOOP 1=FIRE 2=UP 3=RIGHT 4=LEFT 5=DOWN 6=UPRIGHT 7=UPLEFT
#   8=UPFIRE 9=RIGHTFIRE 10=LEFTFIRE 11=DOWNFIRE 12=UPRIGHTFIRE 13=UPLEFTFIRE
# LEFT/RIGHT rotate, UP thrusts. DOWN/DOWNFIRE are hyperspace — never map them.
_ASTEROIDS_MOVE = [3, 6, 2, 7, 4, 7, 2, 6]  # R DR D→UP DL L UL U UR
_ASTEROIDS_MOVE_FIRE = [9, 12, 8, 13, 10, 13, 8, 12]  # no DOWNFIRE (11)

asteroids_action_mapper, asteroids_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=_ASTEROIDS_MOVE,
    move_fire_table=_ASTEROIDS_MOVE_FIRE,
)

# Layout (game_max_objects=17, +1 meta NOOP → 18): asteroids only.
_N_ASTEROIDS = 17
_ASTEROIDS_MAX_OBJECTS = _N_ASTEROIDS + 1
_INACTIVE = 0


def _rotation_to_sprite_deg(rotation) -> jnp.ndarray:
    """Map Asteroids 16-way rotation index → JAXAtari sprite degrees (0=up).

    Game rotation advances counter-clockwise from up; sprite degrees increase
    clockwise (90=right), so negate before scaling.
    """
    return (-jnp.asarray(rotation, dtype=jnp.float32) * jnp.float32(22.5)) % jnp.float32(
        360.0
    )


class AsteroidsSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_ASTEROIDS_MAX_OBJECTS)

    def _patch_player_orientation(self, obs, env_state):
        """Stamp continuous heading into player.orientation (sprite degrees)."""
        base = self._unwrap_env_state(env_state)
        player = obs.raw_observation.player
        ori = _rotation_to_sprite_deg(base.player_rotation)
        cur = jnp.asarray(player.orientation, dtype=jnp.float32)
        if cur.ndim == 0:
            new_ori = ori
        else:
            new_ori = cur.at[-1].set(ori)
        return obs.replace(
            raw_observation=obs.raw_observation.replace(
                player=player.replace(orientation=new_ori)
            )
        )

    def reset(self, key=None):
        obs, env_state = super().reset(key)
        return self._patch_player_orientation(obs, env_state), env_state

    def step(self, state, action):
        obs, env_state, reward, done, info = super().step(state, action)
        obs = self._patch_player_orientation(obs, env_state)
        return obs, env_state, reward, done, info

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """Missile hit / split: asteroid size changes while previously active."""
        prev_size = prev_env_state.asteroid_states[:, 3]
        next_size = next_env_state.asteroid_states[:, 3]
        was_active = prev_size != _INACTIVE
        size_changed = prev_size != next_size
        return jnp.logical_and(was_active, size_changed).astype(jnp.int32)

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        was_killed_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), kill_mask.astype(jnp.int32)]
        )
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        asteroids = obs.asteroids
        n = asteroids.x.shape[0]
        labels = jnp.full(n, LABEL_DESTROY, dtype=jnp.int32)
        active = asteroids.active.astype(jnp.int32)
        # Asteroids wrap around the screen edges.
        offscreen = jnp.ones(n, dtype=jnp.int32)

        pad_size = self.game_max_objects - n
        pad_x = jnp.zeros(pad_size, dtype=asteroids.x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=asteroids.y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=asteroids.width.dtype)
        pad_h = jnp.zeros(pad_size, dtype=asteroids.height.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([asteroids.x, pad_x]),
            y=jnp.concatenate([asteroids.y, pad_y]),
            width=jnp.concatenate([asteroids.width, pad_w]),
            height=jnp.concatenate([asteroids.height, pad_h]),
            active=jnp.concatenate([active, pad_active]),
            semantic_label=jnp.concatenate([labels, pad_labels]),
            state=jnp.zeros(n + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate([offscreen, pad_offscreen]),
            was_killed=jnp.zeros(n + pad_size, dtype=jnp.int32),
            was_collected=jnp.zeros(n + pad_size, dtype=jnp.int32),
        )
