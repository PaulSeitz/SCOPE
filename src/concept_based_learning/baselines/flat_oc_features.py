"""Flat PQN observations built from tandem meta OC input modes.

Maps the locked tandem OC feature builders onto a single MLP vector so flat PQN
can be compared on the same inputs as ``META_INPUT_MODE`` candidates::

    oc_simple_rel_role, oc_simple_role, oc_embed

Architecture still differs (game-action MLP vs per-object meta head); only the
feature construction is shared.
"""

from __future__ import annotations

import functools
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jaxtari.spaces as spaces
from jaxtari.wrappers import JaxatariWrapper

from concept_based_learning.networks.meta_network import (
    _full_object_feature_matrix,
    flatten_raw_observation,
)
from concept_based_learning.observations.meta_oc_simple import (
    build_meta_oc_simple_features,
    normalize_meta_oc_simple_mode,
)

# Locked tandem OC candidates (see materialize_hp_sweep.LOCKED_META_OC_MODES).
FLAT_OC_BUILD_MODES: tuple[str, ...] = (
    "oc_simple_rel_role",
    "oc_simple_role",
    "oc_embed",
)

_MODE_ALIASES = {
    "oc_simple_rel_role": "oc_simple_rel_role",
    "oc_rel_role": "oc_simple_rel_role",
    "rel_role": "oc_simple_rel_role",
    "oc_simple_role": "oc_simple_role",
    "oc_role": "oc_simple_role",
    "role": "oc_simple_role",
    "oc_embed": "oc_embed",
    "object_embed": "oc_embed",
    "object_embedding": "oc_embed",
    "embed": "oc_embed",
}


def normalize_flat_oc_build_mode(mode: str) -> str:
    key = str(mode or "").lower().strip()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"Unknown flat OC build mode={mode!r}. "
            f"Expected one of: {', '.join(FLAT_OC_BUILD_MODES)}"
        )
    return _MODE_ALIASES[key]


def flat_oc_mode_token(mode: str) -> str:
    """Short stem token for suite / checkpoint naming."""
    canon = normalize_flat_oc_build_mode(mode)
    return {
        "oc_simple_rel_role": "rel_role",
        "oc_simple_role": "role",
        "oc_embed": "embed",
    }[canon]


def build_flat_oc_feature_vector(
    raw_obs: Any,
    objects: Any,
    *,
    mode: str,
    max_objects: int,
) -> jnp.ndarray:
    """Flatten tandem-style OC features into a 1D MLP observation."""
    canon = normalize_flat_oc_build_mode(mode)
    simple = normalize_meta_oc_simple_mode(canon)
    if simple is not None:
        global_f, obj_m = build_meta_oc_simple_features(
            raw_obs,
            objects,
            max_objects=max_objects,
            mode=simple,
        )
        return jnp.concatenate(
            [global_f.reshape(-1), obj_m.reshape(-1)], axis=0
        ).astype(jnp.float32)

    # oc_embed: same streams MetaQNetwork uses before the per-object MLP.
    global_f = flatten_raw_observation(raw_obs)
    obj_m = _full_object_feature_matrix(objects, max_objects)
    return jnp.concatenate(
        [global_f.reshape(-1), obj_m.reshape(-1)], axis=0
    ).astype(jnp.float32)


def _semantic_to_flat(
    semantic_obs: Any,
    *,
    mode: str,
    max_objects: int,
) -> jnp.ndarray:
    return build_flat_oc_feature_vector(
        semantic_obs.raw_observation,
        semantic_obs.objects,
        mode=mode,
        max_objects=max_objects,
    )


class FlatOcFeatureWrapper(JaxatariWrapper):
    """Semantic env → flat OC feature vector (Box) for flat PQN."""

    def __init__(self, env, *, mode: str, max_objects: int, feature_dim: int):
        super().__init__(env)
        self.mode = normalize_flat_oc_build_mode(mode)
        self.max_objects = int(max_objects)
        self.feature_dim = int(feature_dim)
        self._observation_space = spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.feature_dim,),
            dtype=jnp.float32,
        )

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, state = self._env.reset(key)
        flat = _semantic_to_flat(
            obs, mode=self.mode, max_objects=self.max_objects
        )
        return flat, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        step_out = self._env.step(state, action)
        if len(step_out) == 6:
            obs, state, reward, terminated, truncated, info = step_out
            flat = _semantic_to_flat(
                obs, mode=self.mode, max_objects=self.max_objects
            )
            return flat, state, reward, terminated, truncated, info
        if len(step_out) == 5:
            obs, state, reward, done, info = step_out
            flat = _semantic_to_flat(
                obs, mode=self.mode, max_objects=self.max_objects
            )
            # LogWrapper / flat PQN expect gymnasium-style 6-tuple.
            terminated = jnp.asarray(done)
            truncated = jnp.zeros_like(terminated)
            return flat, state, reward, terminated, truncated, info
        raise ValueError(f"Unexpected step() return length: {len(step_out)}")


def _semantic_env_config(
    config: dict | None,
    *,
    env_name: str,
    max_objects: int,
    train: bool,
    game_mods: list | None = None,
) -> dict:
    cfg = dict(config or {})
    cfg["ENV_NAME"] = env_name
    cfg.setdefault("FRAME_STACK_SIZE", 4)
    cfg.setdefault("FRAME_SKIP", 4)
    cfg.setdefault("FIRST_FIRE", True)
    cfg.setdefault("STICKY_ACTIONS", False)
    cfg["MAX_OBJECTS"] = int(max_objects)
    # Semantic path only — no pixel meta wrapper.
    cfg["META_INPUT_MODE"] = "flat"
    cfg["SKILL_INPUT_MODE"] = "ego"
    if game_mods is not None:
        cfg["GAME_MODS"] = list(game_mods) if game_mods else None
    if train:
        cfg.setdefault("CLIP_REWARD", True)
        cfg.setdefault("EPISODIC_LIFE", True)
    else:
        cfg["CLIP_REWARD"] = False
        cfg["EPISODIC_LIFE"] = False
    return cfg


def make_flat_oc_build_env(
    *,
    mode: str,
    max_objects: int,
    env_name: str = "seaquest",
    config: dict | None = None,
    train: bool = True,
    game_mods: list | None = None,
):
    """Build LogWrapper(FlatOcFeatureWrapper(semantic_env))."""
    from jaxtari.wrappers import LogWrapper

    from concept_based_learning.environments.env_manager import EnvManager

    canon = normalize_flat_oc_build_mode(mode)
    cfg = _semantic_env_config(
        config,
        env_name=env_name,
        max_objects=max_objects,
        train=train,
        game_mods=game_mods,
    )
    mgr = EnvManager()
    env, _ = mgr.make_env(env_name, cfg, train=train)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    feat_dim = int(
        _semantic_to_flat(obs, mode=canon, max_objects=int(max_objects)).shape[-1]
    )
    env = FlatOcFeatureWrapper(
        env, mode=canon, max_objects=int(max_objects), feature_dim=feat_dim
    )
    return LogWrapper(env)


def probe_flat_oc_feature_dim(
    *,
    mode: str,
    max_objects: int,
    env_name: str = "seaquest",
    config: dict | None = None,
) -> int:
    """Reset once and measure feature width (used to size the Box space)."""
    env = make_flat_oc_build_env(
        mode=mode,
        max_objects=max_objects,
        env_name=env_name,
        config=config,
        train=True,
    )
    obs, _ = env.reset(jax.random.PRNGKey(0))
    return int(jnp.asarray(obs).shape[-1])
