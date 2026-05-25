from typing import Any, Tuple, Dict

import chex
import functools
import jax
import jax.numpy as jnp
import jaxatari.spaces as spaces
from jaxatari.environment import ObjectObservation
from jaxatari.wrappers import JaxatariWrapper

from .datatypes import (
    SemanticObservation,
    SemanticState,
    LabeledObjectObservation,
    NUM_LABELS,
)
from .meta_object_layout import prepend_meta_noop_slot


class JaxatariSemanticBaseWrapper(JaxatariWrapper):
    """
    Base wrapper that standardizes game-specific observations into SemanticObservations.
    """

    def __init__(self, env, max_objects: int):
        super().__init__(env)
        self.stack_size = self._infer_stack_size()
        self.frame_skip = int(getattr(self._env, "_cbl_frame_skip", 1))
        self.clip_reward = bool(getattr(self._env, "_cbl_clip_reward", False))
        self._single_frame_api = not hasattr(self._env, "frame_stack_size")
        # Total slots in observations: index 0 = meta NOOP, indices 1.. = game objects.
        self.max_objects = max_objects
        self.game_max_objects = max_objects - 1

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
        # We vmap over the first axis (the frame stack)
        objects_stack = jax.vmap(self._extract_frame_objects)(obs_stack)
        objects_stack = jax.vmap(prepend_meta_noop_slot)(objects_stack)

        return SemanticObservation(
            raw_observation=obs_stack,  # Pass through the raw stacked observation
            objects=objects_stack,
        )

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
        if self._single_frame_api:
            obs_stack = state.obs_stack
            atari_state = state.atari_state
            total_reward = jnp.array(0.0, dtype=jnp.float32)
            done = jnp.array(False, dtype=jnp.bool_)
            info = {}
            for _ in range(self.frame_skip):
                obs_single, atari_state, step_reward, step_done, step_info = self._step_env_compat(
                    atari_state, action
                )
                obs_stack = self._push_obs_stack(obs_stack, obs_single)
                total_reward = total_reward + step_reward
                done = jnp.logical_or(done, step_done)
                info = step_info
            reward = jnp.sign(total_reward) if self.clip_reward else total_reward
            new_atari_state = atari_state
            new_state = SemanticState(atari_state=new_atari_state, obs_stack=obs_stack)
        else:
            obs_stack, new_atari_state, reward, done, info = self._step_env_compat(
                state.atari_state, action
            )
            new_state = SemanticState(atari_state=new_atari_state)

        semantic_obs = self._process_stack(obs_stack)

        return semantic_obs, new_state, reward, done, info

