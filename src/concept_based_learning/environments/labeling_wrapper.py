from typing import Any, Tuple, Dict

import chex
import functools
import jax
import jax.numpy as jnp
import jaxatari.spaces as spaces
from jaxatari.environment import ObjectObservation
from jaxatari.wrappers import JaxatariWrapper

from concept_based_learning.environments.datatypes import (
    SemanticObservation,
    SemanticState,
    LabeledObjectObservation,
    NUM_LABELS,
)
from concept_based_learning.environments.meta_object_layout import prepend_meta_reserved_slots


class JaxatariSemanticBaseWrapper(JaxatariWrapper):
    """
    Base wrapper that standardizes game-specific observations into SemanticObservations.
    """

    def __init__(self, env, max_objects: int, meta_reserved_slots: int = 1):
        super().__init__(env)
        self.stack_size = self._infer_stack_size()
        self.frame_skip = int(getattr(self._env, "_cbl_frame_skip", 1))
        self.clip_reward = bool(getattr(self._env, "_cbl_clip_reward", False))
        self._single_frame_api = not hasattr(self._env, "frame_stack_size")
        # Total slots: indices 0..meta_reserved_slots-1 are synthetic meta actions.
        self.meta_reserved_slots = int(meta_reserved_slots)
        self.max_objects = max_objects
        self.game_max_objects = max_objects - self.meta_reserved_slots

        # Note: We wrap the original observation space inside a Dict space
        self._observation_space = self._make_observation_space()

    def _infer_stack_size(self) -> int:
        """
        Infer frame stack size across jaxatari versions.

        Older wrappers expose `frame_stack_size`; newer ones only encode it in
        the observation shape.
        """
        explicit_stack = getattr(self._env, "frame_stack_size", None)
        if explicit_stack is not None:
            return int(explicit_stack)

        explicit_stack = getattr(self._env, "_cbl_frame_stack_size", None)
        if explicit_stack is not None:
            return int(explicit_stack)

        shape = getattr(self._env.observation_space(), "shape", ())
        if shape and len(shape) >= 1:
            return int(shape[0])
        return 4

    def _init_obs_stack(self, obs_single):
        return jax.tree.map(
            lambda x: jnp.stack([x] * self.stack_size),
            obs_single,
        )

    def _push_obs_stack(self, obs_stack, obs_single):
        return jax.tree.map(
            lambda stack_leaf, obs_leaf: jnp.concatenate(
                [stack_leaf[1:], jnp.expand_dims(obs_leaf, axis=0)], axis=0
            ),
            obs_stack,
            obs_single,
        )

    def _make_observation_space(self) -> spaces.Dict:
        """Defines the universal Box spaces for the output."""
        original_space = self._env.observation_space()

        # ObjectObservation attributes
        obj_shape = (self.stack_size, self.max_objects)

        return spaces.Dict(
            {
                "global_features": original_space,
                "objects": spaces.Dict(
                    {
                        "x": spaces.Box(0, 255, obj_shape, jnp.int32),
                        "y": spaces.Box(0, 255, obj_shape, jnp.int32),
                        "width": spaces.Box(0, 255, obj_shape, jnp.int32),
                        "height": spaces.Box(0, 255, obj_shape, jnp.int32),
                        "active": spaces.Box(0, 1, obj_shape, jnp.int32),
                        "visual_id": spaces.Box(0, 255, obj_shape, jnp.int32),
                        "state": spaces.Box(0, 255, obj_shape, jnp.int32),
                        "orientation": spaces.Box(-180, 180, obj_shape, jnp.int32),
                    }
                ),
                "object_labels": spaces.Box(
                    0, NUM_LABELS - 1, obj_shape, jnp.int32
                ),
            }
        )

    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    def _extract_frame_objects(
        self, single_frame_obs
    ) -> Tuple[ObjectObservation, chex.Array]:
        """
        ABSTRACT METHOD: Implemented by subclasses (e.g., SeaquestSemanticWrapper).
        Takes a single frame observation and returns:
        (ObjectObservation(size=max_objects))
        """
        raise NotImplementedError(
            "Subclasses must map game entities to generic ObjectObservations."
        )

    def _process_stack(self, obs_stack) -> SemanticObservation:
        """
        Applies the extraction across the frame stack using vmap.
        """
        pixel_stack = None
        raw_stack = obs_stack
        if isinstance(obs_stack, tuple) and len(obs_stack) == 2:
            pixel_stack, raw_stack = obs_stack

        # We vmap over the first axis (the frame stack)
        objects_stack = jax.vmap(self._extract_frame_objects)(raw_stack)
        prepend = lambda objs: prepend_meta_reserved_slots(
            objs, num_reserved=self.meta_reserved_slots
        )
        objects_stack = jax.vmap(prepend)(objects_stack)

        return SemanticObservation(
            raw_observation=raw_stack,  # Pass through the stacked object-centric observation
            objects=objects_stack,
            pixel_observation=pixel_stack,
        )

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Optional per-substep transition events (OR-accumulated across frame_skip).

        Subclasses return an env-specific mask (e.g. confirmed kills); default None.
        """
        del prev_env_state, next_env_state
        return None

    def _substep_collect_events(self, prev_env_state, next_env_state):
        """
        Optional per-substep COLLECT success events (OR-accumulated across frame_skip).

        Subclasses return a game-layout mask for ``was_collected``; default None.
        """
        del prev_env_state, next_env_state
        return None

    def _stamp_events_on_observation(self, semantic_obs, accumulated_events):
        """Apply accumulated substep kill events onto the stacked semantic observation."""
        del accumulated_events
        return semantic_obs

    def _stamp_collect_events_on_observation(self, semantic_obs, accumulated_events):
        """Apply accumulated substep collect events onto the stacked semantic observation."""
        del accumulated_events
        return semantic_obs

    def _step_env_compat(self, state: Any, action: int):
        """
        Normalize jaxatari step output across API versions.
        Returns: (obs, new_state, reward, done, info)
        """
        step_out = self._env.step(state, action)
        if len(step_out) == 5:
            return step_out
        if len(step_out) == 6:
            obs, new_state, reward, terminated, truncated, info = step_out
            done = jnp.logical_or(terminated, truncated)
            return obs, new_state, reward, done, info
        raise ValueError(f"Unexpected step() return length: {len(step_out)}")

    def _unwrap_env_state(self, wrapper_state: Any):
        """
        Recursively unwrap nested wrapper state containers until we reach an env_state.
        Works for AtariState and compound states like PixelAndObjectCentricState.
        """
        cur = wrapper_state
        for _ in range(8):  # defensive cap against accidental cycles
            if hasattr(cur, "env_state"):
                return cur.env_state
            if hasattr(cur, "atari_state"):
                cur = cur.atari_state
                continue
            break
        raise AttributeError(
            "Could not unwrap env_state from wrapper state; "
            f"received type {type(wrapper_state)!r}."
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[SemanticObservation, SemanticState]:
        obs, atari_state = self._env.reset(key)
        if self._single_frame_api:
            obs_stack = self._init_obs_stack(obs)
            state = SemanticState(atari_state=atari_state, obs_stack=obs_stack)
        else:
            obs_stack = obs
            state = SemanticState(atari_state=atari_state)

        semantic_obs = self._process_stack(obs_stack)

        return semantic_obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SemanticState, action: int
    ) -> Tuple[SemanticObservation, SemanticState, float, bool, Dict]:
        accumulated_events = None
        accumulated_collect_events = None
        if self._single_frame_api:
            obs_stack = state.obs_stack
            atari_state = state.atari_state
            total_reward = jnp.array(0.0, dtype=jnp.float32)
            done = jnp.array(False, dtype=jnp.bool_)
            info = {}
            for _ in range(self.frame_skip):
                prev_env_state = atari_state.env_state
                obs_single, atari_state, step_reward, step_done, step_info = self._step_env_compat(
                    atari_state, action
                )
                sub_events = self._substep_env_state_events(
                    prev_env_state, atari_state.env_state
                )
                if sub_events is not None:
                    accumulated_events = (
                        sub_events
                        if accumulated_events is None
                        else jnp.logical_or(accumulated_events, sub_events)
                    )
                sub_collect = self._substep_collect_events(
                    prev_env_state, atari_state.env_state
                )
                if sub_collect is not None:
                    accumulated_collect_events = (
                        sub_collect
                        if accumulated_collect_events is None
                        else jnp.logical_or(accumulated_collect_events, sub_collect)
                    )
                obs_stack = self._push_obs_stack(obs_stack, obs_single)
                total_reward = total_reward + step_reward
                done = jnp.logical_or(done, step_done)
                info = step_info
            reward = jnp.sign(total_reward) if self.clip_reward else total_reward
            new_atari_state = atari_state
            new_state = SemanticState(atari_state=new_atari_state, obs_stack=obs_stack)
        else:
            prev_env_state = self._unwrap_env_state(state.atari_state)
            obs_stack, new_atari_state, reward, done, info = self._step_env_compat(
                state.atari_state, action
            )
            accumulated_events = self._substep_env_state_events(
                prev_env_state, self._unwrap_env_state(new_atari_state)
            )
            accumulated_collect_events = self._substep_collect_events(
                prev_env_state, self._unwrap_env_state(new_atari_state)
            )
            new_state = SemanticState(atari_state=new_atari_state)

        semantic_obs = self._process_stack(obs_stack)
        semantic_obs = self._stamp_events_on_observation(semantic_obs, accumulated_events)
        semantic_obs = self._stamp_collect_events_on_observation(
            semantic_obs, accumulated_collect_events
        )

        return semantic_obs, new_state, reward, done, info

