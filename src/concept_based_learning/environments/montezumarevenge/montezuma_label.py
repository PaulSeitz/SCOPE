"""Montezuma's Revenge semantic labeling.

Observation entities (jax_montezumarevenge):
  enemies, items, doors, ropes, conveyors, platforms.

Lasers exist in env state but are not exposed in the object observation, so
they are omitted here.

Layout (game_max_objects=23, +1 meta NOOP → 24):
  [0-2]   enemies    (AVOID)
  [3-5]   items      (COLLECT)
  [6-7]   doors      (NAVIGATE)
  [8-9]   ropes      (NAVIGATE)
  [10]    conveyors  (BLOCKER)
  [11-22] platforms  (BLOCKER)
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_AVOID,
    LABEL_BLOCKER,
    LABEL_COLLECT,
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


# Full 18-action set (FIRE = jump). Same layout as Kangaroo / Beamrider.
montezuma_action_mapper, montezuma_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=[3, 8, 5, 9, 4, 7, 2, 6],  # R DR D DL L UL U UR
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)

_N_ENEMIES = 3
_N_ITEMS = 3
_N_DOORS = 2
_N_ROPES = 2
_N_CONVEYORS = 1
_N_PLATFORMS = 12
_N_GAME = _N_ENEMIES + _N_ITEMS + _N_DOORS + _N_ROPES + _N_CONVEYORS + _N_PLATFORMS  # 23
_MONTEZUMA_MAX_OBJECTS = _N_GAME + 1  # + meta NOOP


class MontezumaRevengeSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_MONTEZUMA_MAX_OBJECTS)

    def _substep_collect_events(self, prev_env_state, next_env_state):
        """Item pickup → was_collected (items_active 1→0 without life loss)."""
        life_lost = next_env_state.lives < prev_env_state.lives
        item_died = jnp.logical_and(
            prev_env_state.items_active == 1,
            next_env_state.items_active == 0,
        )
        collected = jnp.logical_and(item_died, jnp.logical_not(life_lost)).astype(
            jnp.int32
        )
        # Slot layout: enemies | items | doors | ropes | conveyors | platforms
        zeros_before = jnp.zeros(_N_ENEMIES, dtype=jnp.int32)
        zeros_after = jnp.zeros(
            _N_DOORS + _N_ROPES + _N_CONVEYORS + _N_PLATFORMS, dtype=jnp.int32
        )
        return jnp.concatenate([zeros_before, collected, zeros_after])

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
        items = obs.items
        doors = obs.doors
        ropes = obs.ropes
        conveyors = obs.conveyors
        platforms = obs.platforms

        all_x = jnp.concatenate(
            [enemies.x, items.x, doors.x, ropes.x, conveyors.x, platforms.x]
        )
        all_y = jnp.concatenate(
            [enemies.y, items.y, doors.y, ropes.y, conveyors.y, platforms.y]
        )
        all_w = jnp.concatenate(
            [
                enemies.width,
                items.width,
                doors.width,
                ropes.width,
                conveyors.width,
                platforms.width,
            ]
        )
        all_h = jnp.concatenate(
            [
                enemies.height,
                items.height,
                doors.height,
                ropes.height,
                conveyors.height,
                platforms.height,
            ]
        )
        all_active = jnp.concatenate(
            [
                enemies.active.astype(jnp.int32),
                items.active.astype(jnp.int32),
                doors.active.astype(jnp.int32),
                ropes.active.astype(jnp.int32),
                conveyors.active.astype(jnp.int32),
                platforms.active.astype(jnp.int32),
            ]
        )
        all_labels = jnp.concatenate(
            [
                jnp.full(_N_ENEMIES, LABEL_AVOID, dtype=jnp.int32),
                jnp.full(_N_ITEMS, LABEL_COLLECT, dtype=jnp.int32),
                jnp.full(_N_DOORS, LABEL_NAVIGATE, dtype=jnp.int32),
                jnp.full(_N_ROPES, LABEL_NAVIGATE, dtype=jnp.int32),
                jnp.full(_N_CONVEYORS, LABEL_BLOCKER, dtype=jnp.int32),
                jnp.full(_N_PLATFORMS, LABEL_BLOCKER, dtype=jnp.int32),
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
                [jnp.zeros(n_objects, dtype=jnp.int32), pad_offscreen]
            ),
            was_killed=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
            was_collected=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
        )
