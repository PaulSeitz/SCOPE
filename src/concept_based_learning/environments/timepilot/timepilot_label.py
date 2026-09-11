"""Time Pilot semantic labeling: enemies are DESTROY, enemy missiles are AVOID."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_AVOID,
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


# ACTION_SET indices (Discrete(10)):
#   0=NOOP 1=FIRE 2=UP 3=RIGHT 4=LEFT 5=DOWN
#   6=UPFIRE 7=RIGHTFIRE 8=LEFTFIRE 9=DOWNFIRE
# Diagonals collapse to the nearest cardinal (prefer horizontal on ties).
_TP_MOVE = [3, 3, 5, 4, 4, 4, 2, 3]  # R DR D DL L UL U UR
_TP_MOVE_FIRE = [7, 7, 9, 8, 8, 8, 6, 7]

timepilot_action_mapper, timepilot_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=_TP_MOVE,
    move_fire_table=_TP_MOVE_FIRE,
)

# Layout (game_max_objects=6, +1 meta NOOP → 7):
#   [0-3] enemies          DESTROY
#   [4-5] enemy_missiles   AVOID
_N_ENEMIES = 4
_N_MISSILES = 2
_N_GAME = _N_ENEMIES + _N_MISSILES
_TIMEPILOT_MAX_OBJECTS = _N_GAME + 1
# Policy slots after meta-NOOP:
ENEMY_SLOT0 = 1
ENEMY_SLOT1 = _N_ENEMIES  # 4
MISSILE_SLOT0 = _N_ENEMIES + 1  # 5
MISSILE_SLOT1 = _N_ENEMIES + _N_MISSILES  # 6

# jax_timepilot playfield; obs clips off-screen entities onto the border.
_SCREEN_W = 160
_SCREEN_H = 210
_ENEMY_W = 8
_ENEMY_H = 9
_MISSILE_W = 1
_MISSILE_H = 2


def _rotation_to_sprite_deg(rotation) -> jnp.ndarray:
    """Map Time Pilot 8-way rotation index → JAXAtari sprite degrees (0=up).

    Game rotation advances counter-clockwise from up (0=up, 2=left, 4=down,
    6=right); sprite degrees increase clockwise, so negate before scaling.
    """
    return (-jnp.asarray(rotation, dtype=jnp.float32) * jnp.float32(45.0)) % jnp.float32(
        360.0
    )


class TimePilotSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_TIMEPILOT_MAX_OBJECTS)

    def _patch_player_orientation(self, obs, env_state):
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

    def _onscreen_mask(self, xy, w, h, alive):
        """True when an unclipped AABB intersects the playfield."""
        x = xy[:, 0].astype(jnp.float32)
        y = xy[:, 1].astype(jnp.float32)
        overlaps = jnp.logical_and(
            jnp.logical_and(x + float(w) > 0.0, x < float(_SCREEN_W)),
            jnp.logical_and(y + float(h) > 0.0, y < float(_SCREEN_H)),
        )
        return jnp.logical_and(alive, overlaps)

    def _patch_offscreen_entities(self, obs, env_state):
        """Clear active on enemies/missiles whose true position is off-screen."""
        base = self._unwrap_env_state(env_state)
        enemy_on = self._onscreen_mask(
            base.enemy_states[:, :2],
            _ENEMY_W,
            _ENEMY_H,
            base.enemy_states[:, 3] > 0,
        ).astype(jnp.int32)
        missile_on = self._onscreen_mask(
            base.enemy_missile_states[:, :2],
            _MISSILE_W,
            _MISSILE_H,
            base.enemy_missile_states[:, 3] > 0,
        ).astype(jnp.int32)

        objects = obs.objects
        meta = self.meta_reserved_slots
        e0, m0 = meta, meta + _N_ENEMIES

        def _set(leaf, start, vals):
            stacked = jnp.broadcast_to(vals, (leaf.shape[0], vals.shape[0]))
            return leaf.at[:, start : start + vals.shape[0]].set(
                stacked.astype(leaf.dtype)
            )

        new_active = _set(objects.active, e0, enemy_on)
        new_active = _set(new_active, m0, missile_on)
        return obs.replace(objects=objects.replace(active=new_active))

    def _patch_observation(self, obs, env_state):
        obs = self._patch_player_orientation(obs, env_state)
        return self._patch_offscreen_entities(obs, env_state)

    def reset(self, key=None):
        obs, env_state = super().reset(key)
        return self._patch_observation(obs, env_state), env_state

    def step(self, state, action):
        obs, env_state, reward, done, info = super().step(state, action)
        obs = self._patch_observation(obs, env_state)
        return obs, env_state, reward, done, info

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """Enemy kill: active flag 1→0. Missiles leave no was_killed signal."""
        prev_active = prev_env_state.enemy_states[:, 3] > 0
        next_active = next_env_state.enemy_states[:, 3] > 0
        enemy_killed = jnp.logical_and(
            prev_active, jnp.logical_not(next_active)
        ).astype(jnp.int32)
        missile_zeros = jnp.zeros(_N_MISSILES, dtype=jnp.int32)
        return jnp.concatenate([enemy_killed, missile_zeros])

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
        enemies = obs.enemies
        missiles = obs.enemy_missiles

        all_x = jnp.concatenate([enemies.x, missiles.x])
        all_y = jnp.concatenate([enemies.y, missiles.y])
        all_w = jnp.concatenate([enemies.width, missiles.width])
        all_h = jnp.concatenate([enemies.height, missiles.height])
        all_active = jnp.concatenate(
            [enemies.active.astype(jnp.int32), missiles.active.astype(jnp.int32)]
        )
        all_labels = jnp.concatenate(
            [
                jnp.full(_N_ENEMIES, LABEL_DESTROY, dtype=jnp.int32),
                jnp.full(_N_MISSILES, LABEL_AVOID, dtype=jnp.int32),
            ]
        )
        # Playfield scrolls; entities wrap through the viewport.
        offscreen = jnp.ones(all_x.shape[0], dtype=jnp.int32)

        pad_size = self.game_max_objects - all_x.shape[0]
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
            state=jnp.zeros(all_x.shape[0] + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate([offscreen, pad_offscreen]),
            was_killed=jnp.zeros(all_x.shape[0] + pad_size, dtype=jnp.int32),
            was_collected=jnp.zeros(all_x.shape[0] + pad_size, dtype=jnp.int32),
        )
