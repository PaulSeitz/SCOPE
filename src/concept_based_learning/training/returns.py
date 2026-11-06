"""PQN lambda-return targets (Gallici et al., Eq. 18--19).

All inputs use time first. ``next_values[t]`` must be V(s_{t+1}) for
transition t, including when observations are conditioned on a target.
Targets are fixed before minibatch optimization (semi-gradient Q-learning).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def lambda_return_targets(rewards, dones, next_values, gamma, trace_lambda):
    """Return [T, ...] targets, with independent trailing batch dimensions.

    At a terminal transition the target is exactly its reward. At the end
    of a nonterminal rollout the return bootstraps from next_values[-1].
    Caller-supplied skill outcome / target-switch terminals are respected.
    """
    rewards = jnp.asarray(rewards)
    dones = jnp.asarray(dones, dtype=jnp.bool_)
    next_values = jnp.asarray(next_values)
    if rewards.shape != dones.shape or rewards.shape != next_values.shape:
        raise ValueError("rewards, dones and next_values must have identical [T, ...] shapes")
    if rewards.ndim < 1 or rewards.shape[0] < 1:
        raise ValueError("lambda returns require a nonempty time axis")

    def backward(next_return, transition):
        reward, done, next_value = transition
        continuation = (1.0 - trace_lambda) * next_value + trace_lambda * next_return
        target = reward + gamma * jnp.where(done, 0.0, continuation)
        return target, target

    _, targets = jax.lax.scan(
        backward, next_values[-1], (rewards, dones, next_values), reverse=True
    )
    return jax.lax.stop_gradient(targets)


def object_lambda_return_targets(
    rewards, dones, next_values, object_ids, valid, gamma, trace_lambda
):
    """Lambda returns for independently sampled object rows [T, E, K].

    A row position is not a target identity: K rows may be permuted or
    resampled at the next decision. Match the global object-slot IDs.
    When an eligible next row for the same target is unavailable, end the
    sampled multi-step trace with V(s_{t+1}, target), retaining bootstrapping.
    This is a sampling boundary, unlike caller-supplied task terminals.
    """
    rewards, next_values = jnp.asarray(rewards), jnp.asarray(next_values)
    dones, valid = jnp.asarray(dones, bool), jnp.asarray(valid, bool)
    object_ids = jnp.asarray(object_ids)
    shape = rewards.shape
    if rewards.ndim != 3 or not shape[0] or not shape[-1]:
        raise ValueError("object returns require nonempty [T, E, K] inputs")
    if any(x.shape != shape for x in (dones, next_values, object_ids, valid)):
        raise ValueError("all object-return inputs must have identical [T, E, K] shapes")

    def backward(carry, transition):
        next_returns, next_ids, next_valid = carry
        reward, done, value, ids, eligible = transition
        matches = (ids[..., :, None] == next_ids[..., None, :]) & next_valid[..., None, :]
        found = jnp.any(matches, axis=-1)
        row = jnp.argmax(matches, axis=-1)
        matched_return = jnp.take_along_axis(next_returns, row, axis=-1)
        continuation = jnp.where(found, matched_return, value)
        target = reward + gamma * jnp.where(
            done, 0.0, (1.0 - trace_lambda) * value + trace_lambda * continuation
        )
        return (target, ids, eligible), target

    # No sampled future transition at the rollout boundary: bootstrap.
    initial = (next_values[-1], object_ids[-1], jnp.zeros_like(valid[-1]))
    _, targets = jax.lax.scan(
        backward, initial, (rewards, dones, next_values, object_ids, valid), reverse=True
    )
    return jax.lax.stop_gradient(targets)
