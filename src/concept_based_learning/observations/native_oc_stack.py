"""Four-frame native OC vector matching the monolithic PQN baseline.

The baseline stack is ``ObjectCentricWrapper`` → ``NormalizeObservationWrapper``
→ ``FlattenObservationWrapper``:

* one flattened native frame per environment decision (the last frame of the
  skip), not the four skipped substeps;
* reset repeats that frame across the stack;
* each field is normalized to ``[0, 1]`` in float32 using the observation-space
  bounds, then clipped. Float16 overflows Seaquest's score bound (999999);
* the stack is flattened in frame-major order.

``flatten_raw_observation`` is unchanged. It still keeps only the latest frame
and does not normalize. Aligned models use this module and distinct stems.

Skill conditioning is a one-hot over the selector slot axis (``MAX_OBJECTS``),
which includes synthetic targets such as the Seaquest surface waypoint. That
index is not a position inside the native vector.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import flatten_util

import jaxtari.spaces as spaces


def native_frame_bounds(observation_space) -> tuple[np.ndarray, np.ndarray]:
    """Single-frame ravel bounds, in the same leaf order as ``ObjectCentricWrapper``."""
    lows, highs = [], []
    for leaf_space in jax.tree.leaves(observation_space):
        if isinstance(leaf_space, spaces.Box):
            lows.append(np.broadcast_to(leaf_space.low, leaf_space.shape).flatten())
            highs.append(np.broadcast_to(leaf_space.high, leaf_space.shape).flatten())
        elif isinstance(leaf_space, spaces.Discrete):
            lows.append(np.array([0], dtype=leaf_space.dtype))
            highs.append(np.array([leaf_space.n - 1], dtype=leaf_space.dtype))
        else:
            raise TypeError(f"Unsupported space type for native OC: {type(leaf_space)}")
    if not lows:
        raise ValueError("Native OC observation space has no leaves.")
    return np.concatenate(lows), np.concatenate(highs)


def ravel_native_frame(obs_single) -> jnp.ndarray:
    """Flatten one native frame the way ``ObjectCentricWrapper`` does."""
    return flatten_util.ravel_pytree(obs_single)[0].astype(jnp.float32)


def init_native_frame_stack(frame: jnp.ndarray, stack_size: int) -> jnp.ndarray:
    return jnp.stack([frame] * int(stack_size))


def push_native_frame_stack(stack: jnp.ndarray, frame: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([stack[1:], jnp.expand_dims(frame, 0)], axis=0)


def normalize_native_stack(flat_stack: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray) -> jnp.ndarray:
    """Normalize ``(T, F)`` in float32 with the observation-space bounds and flatten."""
    low32 = jnp.asarray(low, dtype=jnp.float32)
    high32 = jnp.asarray(high, dtype=jnp.float32)
    values = jnp.asarray(flat_stack, dtype=jnp.float32)
    span = high32 - low32
    scale = 1.0 / jnp.where(span > jnp.float32(1e-8), span, jnp.float32(1.0))
    norm = (values - low32) * scale
    norm = jnp.clip(norm, jnp.float32(0.0), jnp.float32(1.0))
    return norm.reshape(-1)


def oc_native_skill_features(native_vector: jnp.ndarray, target_idx, max_objects: int) -> jnp.ndarray:
    """Native vector plus a one-hot of the selected selector slot."""
    slot = jnp.asarray(target_idx, dtype=jnp.int32)
    one_hot = jax.nn.one_hot(slot, int(max_objects), dtype=jnp.float32)
    return jnp.concatenate([jnp.asarray(native_vector, dtype=jnp.float32).reshape(-1), one_hot])


def stack_pqn_pixel_channels(images: jnp.ndarray) -> jnp.ndarray:
    """``[B,T,H,W,C]`` → ``[B,H,W,T*C]`` so the first convolution sees every frame."""
    if images.ndim != 5:
        raise ValueError(f"pixel PQN expected [B,T,H,W,C], got shape {images.shape}")
    batch, frames, height, width, channels = images.shape
    moved = jnp.transpose(images, (0, 2, 3, 1, 4))
    return moved.reshape((batch, height, width, frames * channels))
