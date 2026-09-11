"""Name This Game semantic labeling.

Entities:
  shark            DESTROY
  tentacle tips    DESTROY  (lowest black square of each growing tentacle)
  oxygen_line      COLLECT  (catcher / treasure field)

Layout (game_max_objects=10, +1 meta NOOP → 11):
  [0]     shark
  [1-8]   tentacle tips
  [9]     oxygen_line

Tentacle tips are injected from env_state (not derived from the jaxatari
observation AABB).  That AABB uses the *top* segment's x, so shrinking it
to the tip height leaves the box stuck at the top column while the tip
swings sideways — the wider the swing, the worse the offset.
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_COLLECT,
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


# ACTION_SET: 0=NOOP 1=FIRE 2=RIGHT 3=LEFT 4=RIGHTFIRE 5=LEFTFIRE
# Horizontal only (same layout as SpaceInvaders / Breakout).
namethisgame_action_mapper, namethisgame_inverse_action_mapper = (
    make_target_aligned_action_mappers(
        noop_action=0,
        fire_action=1,
        move_table=[2, 2, 0, 3, 3, 3, 0, 2],  # R DR D DL L UL U UR
        move_fire_table=[4, 4, 1, 5, 5, 5, 1, 4],
    )
)

_N_SHARK = 1
_N_TENTACLES = 8
_N_OXYGEN = 1
_N_GAME = _N_SHARK + _N_TENTACLES + _N_OXYGEN  # 10
_NTG_MAX_OBJECTS = _N_GAME + 1

# Mirror jax_namethisgame.NameThisGameConstants (tip collision / render).
_TENTACLE_YS = jnp.asarray(
    [64, 71, 78, 85, 92, 99, 106, 113, 120, 127], dtype=jnp.int32
)
_TENTACLE_BASE_X = jnp.asarray(
    [16, 32, 48, 64, 80, 96, 112, 128], dtype=jnp.int32
)
_TENTACLE_COL_WIDTH = 4
_TENTACLE_SQUARE_W = 4
_TENTACLE_SQUARE_H = 6

# Shark lane tops (screen y); bottom two gate COLLECT stickiness in the policy.
SHARK_LANES_Y = (36, 50, 64, 78, 90, 104, 118)
BOTTOM_SHARK_LANES_Y = SHARK_LANES_Y[-2:]

# Slot indices before meta-NOOP (policy slots = these + 1).
SHARK_SLOT0 = 0
TENTACLE_SLOT0 = _N_SHARK  # 1
OXYGEN_SLOT0 = _N_SHARK + _N_TENTACLES  # 9


def _expand_scalar(field):
    return jnp.expand_dims(field, axis=0) if jnp.ndim(field) == 0 else field


def _tentacle_tip_aabbs_from_state(env_state):
    """Lowest black-square AABB per tentacle (same geometry as spear collision)."""
    lens = env_state.tentacle_len.astype(jnp.int32)
    active = (lens > 0).astype(jnp.int32)
    tip_idx = jnp.maximum(lens - 1, 0)
    tip_cols = jnp.take_along_axis(
        env_state.tentacle_cols.astype(jnp.int32), tip_idx[:, None], axis=1
    ).squeeze(1)
    base_x = jnp.asarray(env_state.tentacle_base_x, dtype=jnp.int32)
    if base_x.shape[0] != _N_TENTACLES:
        base_x = _TENTACLE_BASE_X
    tip_x = base_x + tip_cols * jnp.int32(_TENTACLE_COL_WIDTH)
    tip_y = _TENTACLE_YS[tip_idx]
    tip_w = jnp.full((_N_TENTACLES,), _TENTACLE_SQUARE_W, dtype=jnp.int32)
    tip_h = jnp.full((_N_TENTACLES,), _TENTACLE_SQUARE_H, dtype=jnp.int32)
    tip_x = jnp.where(active > 0, tip_x, jnp.zeros_like(tip_x))
    tip_y = jnp.where(active > 0, tip_y, jnp.zeros_like(tip_y))
    tip_w = jnp.where(active > 0, tip_w, jnp.zeros_like(tip_w))
    tip_h = jnp.where(active > 0, tip_h, jnp.zeros_like(tip_h))
    return tip_x, tip_y, tip_w, tip_h, active


class NameThisGameSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_NTG_MAX_OBJECTS)

    def _inject_tentacle_tips(self, semantic_obs, env_state):
        """Overwrite tentacle slots with true tip hitboxes from env_state."""
        tip_x, tip_y, tip_w, tip_h, tip_a = _tentacle_tip_aabbs_from_state(env_state)
        objects = semantic_obs.objects
        meta = self.meta_reserved_slots
        start = meta + TENTACLE_SLOT0
        n = _N_TENTACLES

        def _set_block(leaf, values):
            vals = jnp.asarray(values).astype(leaf.dtype)
            stacked = jnp.broadcast_to(vals, (leaf.shape[0], n))
            return leaf.at[:, start : start + n].set(stacked)

        return semantic_obs.replace(
            objects=objects.replace(
                x=_set_block(objects.x, tip_x),
                y=_set_block(objects.y, tip_y),
                width=_set_block(objects.width, tip_w),
                height=_set_block(objects.height, tip_h),
                active=_set_block(objects.active, tip_a),
            )
        )

    def _patch_observation(self, semantic_obs, env_state):
        unwrapped = self._unwrap_env_state(env_state)
        return self._inject_tentacle_tips(semantic_obs, unwrapped)

    def reset(self, key=None):
        obs, env_state = super().reset(key)
        return self._patch_observation(obs, env_state), env_state

    def step(self, state, action):
        obs, env_state, reward, done, info = super().step(state, action)
        obs = self._patch_observation(obs, env_state)
        return obs, env_state, reward, done, info

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """Shark spear-hit (lane reset) + tentacle tip wipe → was_killed."""
        shark_killed = (
            next_env_state.shark_resets_this_round > prev_env_state.shark_resets_this_round
        ).astype(jnp.int32)
        shark_killed = jnp.expand_dims(shark_killed, axis=0)

        tentacle_killed = jnp.logical_and(
            prev_env_state.tentacle_len > 0,
            next_env_state.tentacle_len == 0,
        ).astype(jnp.int32)

        oxygen_zeros = jnp.zeros(_N_OXYGEN, dtype=jnp.int32)
        return jnp.concatenate([shark_killed, tentacle_killed, oxygen_zeros])

    def _substep_collect_events(self, prev_env_state, next_env_state):
        """Oxygen contact frames (under the drop line) → was_collected."""
        shark_zeros = jnp.zeros(_N_SHARK, dtype=jnp.int32)
        tentacle_zeros = jnp.zeros(_N_TENTACLES, dtype=jnp.int32)
        contacting = jnp.logical_and(
            next_env_state.oxygen_line_active,
            next_env_state.oxygen_contact_counter > prev_env_state.oxygen_contact_counter,
        ).astype(jnp.int32)
        oxygen = jnp.expand_dims(contacting, axis=0)
        return jnp.concatenate([shark_zeros, tentacle_zeros, oxygen])

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
        shark = obs.shark
        tentacles = obs.tentacles
        oxygen = obs.oxygen_line

        shark_x = _expand_scalar(shark.x)
        shark_y = _expand_scalar(shark.y)
        shark_w = _expand_scalar(shark.width)
        shark_h = _expand_scalar(shark.height)
        shark_a = _expand_scalar(shark.active).astype(jnp.int32)

        # Tentacle slots: reserved placeholders; filled from env_state in patch.
        tip_x = jnp.zeros(_N_TENTACLES, dtype=tentacles.x.dtype)
        tip_y = jnp.zeros(_N_TENTACLES, dtype=tentacles.y.dtype)
        tip_w = jnp.zeros(_N_TENTACLES, dtype=tentacles.width.dtype)
        tip_h = jnp.zeros(_N_TENTACLES, dtype=tentacles.height.dtype)
        tip_a = jnp.zeros(_N_TENTACLES, dtype=jnp.int32)

        oxy_x = _expand_scalar(oxygen.x)
        oxy_y = _expand_scalar(oxygen.y)
        oxy_w = _expand_scalar(oxygen.width)
        oxy_h = _expand_scalar(oxygen.height)
        oxy_a = _expand_scalar(oxygen.active).astype(jnp.int32)

        all_x = jnp.concatenate([shark_x, tip_x, oxy_x])
        all_y = jnp.concatenate([shark_y, tip_y, oxy_y])
        all_w = jnp.concatenate([shark_w, tip_w, oxy_w])
        all_h = jnp.concatenate([shark_h, tip_h, oxy_h])
        all_active = jnp.concatenate([shark_a, tip_a, oxy_a])
        all_labels = jnp.concatenate(
            [
                jnp.full(_N_SHARK, LABEL_DESTROY, dtype=jnp.int32),
                jnp.full(_N_TENTACLES, LABEL_DESTROY, dtype=jnp.int32),
                jnp.full(_N_OXYGEN, LABEL_COLLECT, dtype=jnp.int32),
            ]
        )
        all_offscreen = jnp.zeros(all_x.shape[0], dtype=jnp.int32)

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
            can_go_offscreen=jnp.concatenate([all_offscreen, pad_offscreen]),
            was_killed=jnp.zeros(all_x.shape[0] + pad_size, dtype=jnp.int32),
            was_collected=jnp.zeros(all_x.shape[0] + pad_size, dtype=jnp.int32),
        )
