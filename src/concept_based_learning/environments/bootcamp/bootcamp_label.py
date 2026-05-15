"""
Semantic wrapper for jax_skillbootcamp (Kinematic Bootcamp).

Physics stays in JAXAtari; this module only flattens entities and assigns labels:
  enemies[0] → DESTROY, enemies[1] → AVOID, divers[0] → COLLECT.
"""

import functools
import jax
import jax.numpy as jnp
from typing import Any

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NOOP,
    SemanticState,
)


def extract_all_skills_rewards(
    prev_raw_obs: Any,
    next_raw_obs: Any,
    prev_target: Any,
    next_target: Any,
    info: Any,
):
    del prev_raw_obs, next_raw_obs, next_target

    def last_frame(e):
        return jax.tree_util.tree_map(lambda x: x[-1] if jnp.ndim(x) > 0 else x, e)

    prev_target = last_frame(prev_target)

    is_destroy = prev_target.semantic_label == LABEL_DESTROY
    is_collect = prev_target.semantic_label == LABEL_COLLECT
    is_avoid = prev_target.semantic_label == LABEL_AVOID

    z_step = jnp.zeros_like(is_destroy, dtype=jnp.int32)
    win_destroy = info.get("step_win_destroy", z_step) > 0
    win_collect = info.get("step_win_collect", z_step) > 0
    win_avoid = info.get("step_win_avoid", z_step) > 0
    fail_destroy = info.get("step_fail_destroy_touch", z_step) > 0
    fail_avoid = info.get("step_fail_avoid", z_step) > 0

    # Bootstrapping fix: only terminate skill episodes on failure.
    failed = jnp.logical_or(fail_destroy, fail_avoid)

    destroy_reward = jnp.where(
        jnp.logical_and(is_destroy, win_destroy),
        1.0,
        jnp.where(jnp.logical_and(is_destroy, fail_destroy), -1.0, 0.0),
    )
    destroy_done = jnp.logical_and(is_destroy, failed)

    collect_reward = jnp.where(jnp.logical_and(is_collect, win_collect), 1.0, 0.0)
    collect_done = jnp.logical_and(is_collect, failed)

    avoid_reward = jnp.where(
        jnp.logical_and(is_avoid, win_avoid),
        1.0,
        jnp.where(jnp.logical_and(is_avoid, fail_avoid), -10.0, 0.0),
    )
    avoid_done = jnp.logical_and(is_avoid, failed)

    rewards = jnp.stack([destroy_reward, collect_reward, avoid_reward]).astype(jnp.float32)
    skill_done = jnp.stack([destroy_done, collect_done, avoid_done]).astype(jnp.bool_)

    return rewards, skill_done


class BootcampSemanticWrapper(JaxatariSemanticBaseWrapper):
    """Flattens BootcampObservation into padded ``LabeledObjectObservation`` stacks."""

    def __init__(self, env):
        super().__init__(env, max_objects=8)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        last_e = obs.enemies
        last_d = obs.divers

        all_x = jnp.concatenate([last_e.x, last_d.x])
        all_y = jnp.concatenate([last_e.y, last_d.y])
        all_w = jnp.concatenate([last_e.width, last_d.width])
        all_h = jnp.concatenate([last_e.height, last_d.height])
        all_active = jnp.concatenate([last_e.active, last_d.active])
        labels = jnp.array(
            [LABEL_DESTROY, LABEL_AVOID, LABEL_COLLECT],
            dtype=jnp.int32,
        )
        n = all_x.shape[0]
        pad = self.max_objects - n
        z = jnp.zeros(pad, dtype=all_x.dtype)
        pad_active = jnp.zeros(pad, dtype=jnp.int32)
        pad_labels = jnp.full(pad, LABEL_NOOP, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([all_x, z]),
            y=jnp.concatenate([all_y, z]),
            width=jnp.concatenate([all_w, z]),
            height=jnp.concatenate([all_h, z]),
            active=jnp.concatenate([all_active, pad_active]),
            semantic_label=jnp.concatenate([labels, pad_labels]),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: SemanticState, action: int):
        """
        Bootcamp-specific step override that ensures reward event signals are always present.

        Bootcamp exposes step outcome events via the underlying env state's sticky fields
        (``info_sticky_*``) to survive wrapper frame-skip. Tools like the reward dashboard
        and the skill trainer expect canonical keys like ``step_win_destroy`` in `info`.
        """
        # Clear sticky event flags before each macro-step so AtariWrapper frame-skip
        # accumulates only events from this outer transition.
        env_state = state.atari_state.env_state
        cleared_env_state = env_state.replace(
            info_sticky_win_collect=jnp.array(0, dtype=jnp.int32),
            info_sticky_win_destroy=jnp.array(0, dtype=jnp.int32),
            info_sticky_fail_avoid=jnp.array(0, dtype=jnp.int32),
            info_sticky_fail_destroy_touch=jnp.array(0, dtype=jnp.int32),
            info_sticky_win_avoid=jnp.array(0, dtype=jnp.int32),
        )
        cleared_atari_state = state.atari_state.replace(env_state=cleared_env_state)
        cleared_state = state.replace(atari_state=cleared_atari_state)
        semantic_obs, new_state, reward, done, info = super().step(cleared_state, action)

        # Prefer sticky aggregated signals if available, otherwise fall back to whatever
        # the base env put into `info`.
        env_state = getattr(new_state.atari_state, "env_state", None)

        def _sticky_or_info(sticky_attr: str, info_key: str):
            if info is not None and hasattr(info, info_key):
                return getattr(info, info_key)
            if isinstance(info, dict) and info_key in info:
                return info.get(info_key)
            if env_state is not None and hasattr(env_state, sticky_attr):
                return getattr(env_state, sticky_attr)
            return jnp.asarray(0, dtype=jnp.int32)

        info_overrides = {
            "step_win_destroy": _sticky_or_info(
                "info_sticky_win_destroy", "step_win_destroy"
            ),
            "step_win_collect": _sticky_or_info(
                "info_sticky_win_collect", "step_win_collect"
            ),
            "step_fail_avoid": _sticky_or_info(
                "info_sticky_fail_avoid", "step_fail_avoid"
            ),
            "step_fail_destroy_touch": _sticky_or_info(
                "info_sticky_fail_destroy_touch", "step_fail_destroy_touch"
            ),
            "step_win_avoid": _sticky_or_info(
                "info_sticky_win_avoid", "step_win_avoid"
            ),
        }
        if isinstance(info, dict):
            info = {**info, **info_overrides}
        else:
            info = info_overrides

        return semantic_obs, new_state, reward, done, info
