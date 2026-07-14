"""Memory-friendly skill-network forwards for full-rollout bootstrap paths."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def _pad_time_axis(x: jnp.ndarray, pad_steps: int) -> jnp.ndarray:
    if pad_steps <= 0:
        return x
    pad = jnp.repeat(x[-1:], pad_steps, axis=0)
    return jnp.concatenate([x, pad], axis=0)


def chunked_skill_network_q_max(
    apply_fn: Callable[..., jnp.ndarray],
    params: Any,
    batch_stats: Any,
    ego_obs: jnp.ndarray,
    *,
    target_idx: jnp.ndarray,
    pixel_obs: jnp.ndarray | None,
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
    if chunk_steps >= num_steps:
        q = apply_fn(
            {"params": params, "batch_stats": batch_stats},
            ego_obs,
            train=train,
            target_idx=target_idx,
            pixel_obs=pixel_obs,
        )
        return jnp.max(q, axis=-1)

    padded_steps = ((num_steps + chunk_steps - 1) // chunk_steps) * chunk_steps
    pad = padded_steps - num_steps
    ego_padded = _pad_time_axis(ego_obs, pad)
    target_padded = _pad_time_axis(target_idx, pad)
    pixel_padded = _pad_time_axis(pixel_obs, pad) if pixel_obs is not None else None

    num_chunks = padded_steps // chunk_steps
    chunks_ego = ego_padded.reshape(num_chunks, chunk_steps, *ego_padded.shape[1:])
    chunks_target = target_padded.reshape(num_chunks, chunk_steps, *target_padded.shape[1:])

    if pixel_padded is not None:
        chunks_pixel = pixel_padded.reshape(
            num_chunks, chunk_steps, *pixel_padded.shape[1:]
        )

        def _chunk_fn(_carry, xs):
            ego_c, target_c, pixel_c = xs
            q = apply_fn(
                {"params": params, "batch_stats": batch_stats},
                ego_c,
                train=train,
                target_idx=target_c,
                pixel_obs=pixel_c,
            )
            return None, jnp.max(q, axis=-1)

        scan_xs = (chunks_ego, chunks_target, chunks_pixel)
    else:

        def _chunk_fn(_carry, xs):
            ego_c, target_c = xs
            q = apply_fn(
                {"params": params, "batch_stats": batch_stats},
                ego_c,
                train=train,
                target_idx=target_c,
                pixel_obs=None,
            )
            return None, jnp.max(q, axis=-1)

        scan_xs = (chunks_ego, chunks_target)

    _, q_chunks = jax.lax.scan(_chunk_fn, None, scan_xs)
    q_max = q_chunks.reshape(padded_steps, *q_chunks.shape[2:])
    return q_max[:num_steps]
