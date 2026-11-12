"""Kaboom semantic labeling: falling bombs are COLLECT.

The jaxatari observation has no ``player`` entity — buckets are the controllable
avatar.  This wrapper injects a synthetic ``player`` proxy from bucket 0 so ego
features / heuristics share the usual ``raw_observation.player`` path.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
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


# ACTION_SET indices: 0=NOOP 1=FIRE 2=RIGHT 3=LEFT (same as Breakout).
kaboom_action_mapper, kaboom_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=[2, 2, 0, 3, 3, 3, 0, 2],  # R DR D DL L UL U UR
    move_fire_table=[2, 2, 1, 3, 3, 3, 1, 2],
)

_N_BOMBS = 15
_KABOOM_MAX_OBJECTS = _N_BOMBS + 1
_DEFAULT_STATE = -1
# bombs_states columns: x, y, fuse, type, explode, explode_bucket, on_bucket, is_active, explodes_in
_COL_BUCKET_EXPLODE = 5
_COL_ACTIVE = 7


@struct.dataclass
class _PlayerProxy:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray
    orientation: jnp.ndarray
    state: jnp.ndarray


@struct.dataclass
class KaboomRawWithPlayer:
    """Kaboom obs plus a synthetic ``player`` (bottom bucket) for ego features."""

    player: _PlayerProxy
    mad_bomber_pos: object
    buckets_pos: object
    bombs: object


def _bucket_player_from_raw(raw):
    """Build a scalar/stacked player from the bottom bucket (shared paddle x)."""
    buckets = raw.buckets_pos
    # Bucket 0 is the lowest (y≈180); all buckets share paddle x.
    x = buckets.x[..., 0] if buckets.x.ndim >= 1 else buckets.x
    y = buckets.y[..., 0] if buckets.y.ndim >= 1 else buckets.y
    w = buckets.width[..., 0] if buckets.width.ndim >= 1 else buckets.width
    h = buckets.height[..., 0] if buckets.height.ndim >= 1 else buckets.height
    a = buckets.active[..., 0] if buckets.active.ndim >= 1 else buckets.active
    # Face upward into the falling bombs (sprite 0° = up).
    ori = jnp.full_like(jnp.asarray(x, dtype=jnp.float32), jnp.float32(0.0))
    st = jnp.zeros_like(jnp.asarray(a, dtype=jnp.int32))
    return _PlayerProxy(
        x=jnp.asarray(x),
        y=jnp.asarray(y),
        width=jnp.asarray(w),
        height=jnp.asarray(h),
        active=jnp.asarray(a),
        orientation=ori,
        state=st,
    )


def _inject_player(raw):
    """Attach ``.player`` as a flax pytree (JIT-safe)."""
    if isinstance(raw, KaboomRawWithPlayer):
        return raw
    return KaboomRawWithPlayer(
        player=_bucket_player_from_raw(raw),
        mad_bomber_pos=raw.mad_bomber_pos,
        buckets_pos=raw.buckets_pos,
        bombs=raw.bombs,
    )


class KaboomSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_KABOOM_MAX_OBJECTS)

    def _patch_raw_observation(self, semantic_obs):
        return semantic_obs.replace(
            raw_observation=_inject_player(semantic_obs.raw_observation)
        )

    def reset(self, key=None):
        obs, env_state = super().reset(key)
        return self._patch_raw_observation(obs), env_state

    def step(self, state, action):
        obs, env_state, reward, done, info = super().step(state, action)
        obs = self._patch_raw_observation(obs)
        return obs, env_state, reward, done, info

    def _substep_collect_events(self, prev_env_state, next_env_state):
        """Bomb caught in a bucket: bucket-explode anim starts (DEFAULT → 0+)."""
        prev_be = prev_env_state.bombs_states[:, _COL_BUCKET_EXPLODE]
        next_be = next_env_state.bombs_states[:, _COL_BUCKET_EXPLODE]
        was_falling = prev_env_state.bombs_states[:, _COL_ACTIVE] == 1
        catch_start = jnp.logical_and(
            prev_be == _DEFAULT_STATE,
            next_be != _DEFAULT_STATE,
        )
        return jnp.logical_and(was_falling, catch_start).astype(jnp.int32)

    def _stamp_collect_events_on_observation(self, semantic_obs, collect_mask):
        if collect_mask is None:
            return semantic_obs
        was_collected_with_noop = self._with_meta_event_padding(collect_mask)
        objects = semantic_obs.objects
        new_wc = objects.was_collected.at[-1].set(was_collected_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_collected=new_wc))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        bombs = obs.bombs
        n = bombs.x.shape[0]
        labels = jnp.full(n, LABEL_COLLECT, dtype=jnp.int32)
        active = bombs.active.astype(jnp.int32)
        offscreen = jnp.zeros(n, dtype=jnp.int32)

        pad_size = self.game_max_objects - n
        pad_x = jnp.zeros(pad_size, dtype=bombs.x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=bombs.y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=bombs.width.dtype)
        pad_h = jnp.zeros(pad_size, dtype=bombs.height.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([bombs.x, pad_x]),
            y=jnp.concatenate([bombs.y, pad_y]),
            width=jnp.concatenate([bombs.width, pad_w]),
            height=jnp.concatenate([bombs.height, pad_h]),
            active=jnp.concatenate([active, pad_active]),
            semantic_label=jnp.concatenate([labels, pad_labels]),
            state=jnp.zeros(n + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate([offscreen, pad_offscreen]),
            was_killed=jnp.zeros(n + pad_size, dtype=jnp.int32),
            was_collected=jnp.zeros(n + pad_size, dtype=jnp.int32),
        )
