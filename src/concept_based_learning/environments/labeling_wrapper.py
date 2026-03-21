from typing import Tuple, Dict

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


class JaxatariSemanticBaseWrapper(JaxatariWrapper):
    """
    Base wrapper that standardizes game-specific observations into SemanticObservations.
    """

    def __init__(self, env, max_objects: int):
        super().__init__(env)
        self.stack_size = self._env.frame_stack_size
        self.max_objects = max_objects

        # Note: We wrap the original observation space inside a Dict space
        self._observation_space = self._make_observation_space()

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

        return SemanticObservation(
            raw_observation=obs_stack,  # Pass through the raw stacked observation
            objects=objects_stack,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[SemanticObservation, SemanticState]:
        obs_stack, atari_state = self._env.reset(key)

        semantic_obs = self._process_stack(obs_stack)
        state = SemanticState(atari_state=atari_state)

        return semantic_obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SemanticState, action: int
    ) -> Tuple[SemanticObservation, SemanticState, float, bool, Dict]:
        obs_stack, new_atari_state, reward, done, info = self._env.step(
            state.atari_state, action
        )

        semantic_obs = self._process_stack(obs_stack)
        new_state = SemanticState(atari_state=new_atari_state)

        return semantic_obs, new_state, reward, done, info

