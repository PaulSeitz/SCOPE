"""Memory-friendly skill-network forwards for full-rollout bootstrap paths."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

# Peak CNN batch ≈ chunk_steps * NUM_ENVS. 1024 frames is ~16× smaller than a
# 128×128 rollout while still a healthy GPU batch.
PIXEL_CNN_TARGET_FRAMES = 1024


def _pad_time_axis(x: jnp.ndarray, pad_steps: int) -> jnp.ndarray:
    if pad_steps <= 0:
        return x
    pad = jnp.repeat(x[-1:], pad_steps, axis=0)
    return jnp.concatenate([x, pad], axis=0)


def _pad_time_tree(tree, pad_steps: int):
    if tree is None:
        return None
    return jax.tree_util.tree_map(lambda x: _pad_time_axis(x, pad_steps), tree)


def resolve_skill_chunk_steps(
    *,
    num_steps: int,
    num_envs: int,
    configured: int,
    use_pixel: bool,
) -> int:
    """Time-chunk size for skill Q bootstrap.

    Ego/OC keep the historical default (full ``NUM_STEPS`` when unset).
    Pixel skills default to a ~1024-frame CNN batch so a 128×128 rollout is
    not pushed through the convs in one shot (and not 4× via ``vmap``).
    """
    num_steps = max(1, int(num_steps))
    configured = int(configured)
    if configured > 0:
        return max(1, min(configured, num_steps))
    if not use_pixel:
        return num_steps
    per = max(1, PIXEL_CNN_TARGET_FRAMES // max(1, int(num_envs)))
    return max(1, min(num_steps, per))


def map_over_skills(fn, *xs, sequential: bool):
    """``vmap`` over skill heads, or ``scan`` them one-at-a-time.

    Pixel skill CNNs must not be ``vmap``'d: that broadcasts the same image
    batch through 4 independent convs and peaks at ~4× activations. Scan is
    the same math (independent params) with a single CNN live at a time.
    """
    if sequential:
        def _body(_carry, args):
            return None, fn(*args)

        _, out = jax.lax.scan(_body, None, xs)
        return out
    return jax.vmap(fn)(*xs)


def skill_network_q_max(
    apply_fn: Callable[..., jnp.ndarray],
    params: Any,
    batch_stats: Any,
    ego_obs: jnp.ndarray,
    *,
    target_idx: jnp.ndarray,
    pixel_obs: jnp.ndarray | None,
    objects=None,
    chunk_steps: int,
    train: bool = False,
    chunk: bool = False,
) -> jnp.ndarray:
    """Max-Q over actions for a ``[T, ...]`` rollout, optionally time-chunked."""
    if chunk:
        return chunked_skill_network_q_max(
            apply_fn,
            params,
            batch_stats,
            ego_obs,
            target_idx=target_idx,
            pixel_obs=pixel_obs,
            objects=objects,
            chunk_steps=chunk_steps,
            train=train,
        )
    q = apply_fn(
        {"params": params, "batch_stats": batch_stats},
        ego_obs,
        train=train,
        target_idx=target_idx,
        pixel_obs=pixel_obs,
        objects=objects,
    )
    return jnp.max(q, axis=-1)


def chunked_skill_network_q_max(
    apply_fn: Callable[..., jnp.ndarray],
    params: Any,
    batch_stats: Any,
    ego_obs: jnp.ndarray,
    *,
    target_idx: jnp.ndarray,
    pixel_obs: jnp.ndarray | None,
    objects=None,
    chunk_steps: int,
    train: bool = False,
) -> jnp.ndarray:
    """
    Evaluate max Q over actions for a ``[T, ...]`` rollout in fixed time chunks.

    Limits peak CNN batch size to ``chunk_steps * NUM_ENVS`` instead of
    ``T * NUM_ENVS`` when ``T`` is the full rollout horizon.
    """
    num_steps = int(ego_obs.shape[0])
    chunk_steps = max(1, min(int(chunk_steps), num_steps))

    def _apply(ego, tgt, pix, objs):
        return apply_fn(
            {"params": params, "batch_stats": batch_stats},
            ego,
            train=train,
            target_idx=tgt,
            pixel_obs=pix,
            objects=objs,
        )

    if chunk_steps >= num_steps:
        q = _apply(ego_obs, target_idx, pixel_obs, objects)
        return jnp.max(q, axis=-1)

    padded_steps = ((num_steps + chunk_steps - 1) // chunk_steps) * chunk_steps
    pad = padded_steps - num_steps
    ego_padded = _pad_time_axis(ego_obs, pad)
    target_padded = _pad_time_axis(target_idx, pad)
    pixel_padded = _pad_time_axis(pixel_obs, pad) if pixel_obs is not None else None
    objects_padded = _pad_time_tree(objects, pad)

    num_chunks = padded_steps // chunk_steps
    chunks_ego = ego_padded.reshape(num_chunks, chunk_steps, *ego_padded.shape[1:])
    chunks_target = target_padded.reshape(num_chunks, chunk_steps, *target_padded.shape[1:])
    chunks_pixel = (
        pixel_padded.reshape(num_chunks, chunk_steps, *pixel_padded.shape[1:])
        if pixel_padded is not None
        else None
    )
    chunks_objects = (
        jax.tree_util.tree_map(
            lambda x: x.reshape(num_chunks, chunk_steps, *x.shape[1:]),
            objects_padded,
        )
        if objects_padded is not None
        else None
    )

    def _chunk_fn(_carry, i):
        ego_c = chunks_ego[i]
        target_c = chunks_target[i]
        pixel_c = None if chunks_pixel is None else chunks_pixel[i]
        objects_c = (
            None
            if chunks_objects is None
            else jax.tree_util.tree_map(lambda x: x[i], chunks_objects)
        )
        q = _apply(ego_c, target_c, pixel_c, objects_c)
        return None, jnp.max(q, axis=-1)

    _, q_chunks = jax.lax.scan(_chunk_fn, None, jnp.arange(num_chunks))
    q_max = q_chunks.reshape(padded_steps, *q_chunks.shape[2:])
    return q_max[:num_steps]
