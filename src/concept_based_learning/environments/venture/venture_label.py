"""Venture semantic labeling.

Observation entities (jax_venture):
  monsters, portals, chest, lasers, chaser.

Layout (game_max_objects=47, +2 meta NOOP/FIRE → 49):
  [0-32]  monsters  (DESTROY)
  [33]    chaser    (AVOID)
  [34-37] lasers    (AVOID)  — room hazard beams, not player shots
  [38]    chest     (COLLECT)
  [39-46] portals   (NAVIGATE)
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NAVIGATE,
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


# Full 18-action set with FIRE for shooting.
venture_action_mapper, venture_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=[3, 8, 5, 9, 4, 7, 2, 6],  # R DR D DL L UL U UR
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)

_N_MONSTERS = 33
_N_CHASER = 1
_N_LASERS = 4
_N_CHEST = 1
_N_PORTALS = 8
_N_GAME = _N_MONSTERS + _N_CHASER + _N_LASERS + _N_CHEST + _N_PORTALS  # 47
_VENTURE_MAX_OBJECTS = _N_GAME + 2  # + NOOP + FIRE


def _scalar_to_1d(x):
    return jnp.expand_dims(x, axis=0) if x.ndim == 0 else x


class VentureSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_VENTURE_MAX_OBJECTS, meta_reserved_slots=2)

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """Monster shot → was_killed (active True→False without life loss)."""
        life_lost = next_env_state.lives < prev_env_state.lives
        monster_died = jnp.logical_and(
            prev_env_state.monsters.active,
            jnp.logical_not(next_env_state.monsters.active),
        )
        monster_killed = jnp.logical_and(
            monster_died, jnp.logical_not(life_lost)
        ).astype(jnp.int32)
        zeros_after = jnp.zeros(
            _N_CHASER + _N_LASERS + _N_CHEST + _N_PORTALS, dtype=jnp.int32
        )
        return jnp.concatenate([monster_killed, zeros_after])

    def _substep_collect_events(self, prev_env_state, next_env_state):
        """Chest pickup → was_collected (per-room chests_active True→False)."""
        life_lost = next_env_state.lives < prev_env_state.lives
        chest_picked = jnp.logical_and(
            prev_env_state.chests_active,
            jnp.logical_not(next_env_state.chests_active),
        )
        # Obs exposes a single chest slot for the current room; OR any room pickup.
        any_collected = jnp.logical_and(
            jnp.any(chest_picked), jnp.logical_not(life_lost)
        ).astype(jnp.int32)
        zeros_before = jnp.zeros(
            _N_MONSTERS + _N_CHASER + _N_LASERS, dtype=jnp.int32
        )
        zeros_after = jnp.zeros(_N_PORTALS, dtype=jnp.int32)
        return jnp.concatenate(
            [zeros_before, jnp.reshape(any_collected, (1,)), zeros_after]
        )

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        was_killed_with_meta = jnp.concatenate(
            [
                jnp.zeros(self.meta_reserved_slots, dtype=jnp.int32),
                kill_mask.astype(jnp.int32),
            ]
        )
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_meta)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def _stamp_collect_events_on_observation(self, semantic_obs, collect_mask):
        if collect_mask is None:
            return semantic_obs
        was_collected_with_meta = jnp.concatenate(
            [
                jnp.zeros(self.meta_reserved_slots, dtype=jnp.int32),
                collect_mask.astype(jnp.int32),
            ]
        )
        objects = semantic_obs.objects
        new_wc = objects.was_collected.at[-1].set(was_collected_with_meta)
        return semantic_obs.replace(objects=objects.replace(was_collected=new_wc))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        monsters = obs.monsters
        chaser = obs.chaser
        lasers = obs.lasers
        chest = obs.chest
        portals = obs.portals

        ch_x = _scalar_to_1d(chaser.x)
        ch_y = _scalar_to_1d(chaser.y)
        ch_w = _scalar_to_1d(chaser.width)
        ch_h = _scalar_to_1d(chaser.height)
        ch_act = _scalar_to_1d(chaser.active)

        chest_x = _scalar_to_1d(chest.x)
        chest_y = _scalar_to_1d(chest.y)
        chest_w = _scalar_to_1d(chest.width)
        chest_h = _scalar_to_1d(chest.height)
        chest_act = _scalar_to_1d(chest.active)

        all_x = jnp.concatenate([monsters.x, ch_x, lasers.x, chest_x, portals.x])
        all_y = jnp.concatenate([monsters.y, ch_y, lasers.y, chest_y, portals.y])
        all_w = jnp.concatenate(
            [monsters.width, ch_w, lasers.width, chest_w, portals.width]
        )
        all_h = jnp.concatenate(
            [monsters.height, ch_h, lasers.height, chest_h, portals.height]
        )
        all_active = jnp.concatenate(
            [
                monsters.active.astype(jnp.int32),
                ch_act.astype(jnp.int32),
                lasers.active.astype(jnp.int32),
                chest_act.astype(jnp.int32),
                portals.active.astype(jnp.int32),
            ]
        )
        all_labels = jnp.concatenate(
            [
                jnp.full(_N_MONSTERS, LABEL_DESTROY, dtype=jnp.int32),
                jnp.full(_N_CHASER, LABEL_AVOID, dtype=jnp.int32),
                jnp.full(_N_LASERS, LABEL_AVOID, dtype=jnp.int32),
                jnp.full(_N_CHEST, LABEL_COLLECT, dtype=jnp.int32),
                jnp.full(_N_PORTALS, LABEL_NAVIGATE, dtype=jnp.int32),
            ]
        )
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
            can_go_offscreen=jnp.concatenate(
                [jnp.ones(n_objects, dtype=jnp.int32), pad_offscreen]
            ),
            was_killed=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
            was_collected=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
        )
