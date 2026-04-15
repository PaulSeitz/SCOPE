"""Shared egocentric action-space helpers (18-dim full vs 6-dim primitive subset)."""

import jax.numpy as jnp

# Full Atari-style egocentric layout (NOOP, FIRE, 8 directions, 8 direction+FIRE).
FIRE_EGO_ACTION_INDICES_18 = jnp.array(
    [1, 10, 11, 12, 13, 14, 15, 16, 17], dtype=jnp.int32
)
# Minimal layout: NOOP, FIRE, FORWARD, BACKWARD, STRAFE_LEFT, STRAFE_RIGHT → FIRE is 1.
FIRE_EGO_ACTION_INDICES_6 = jnp.array([1], dtype=jnp.int32)


def fire_ego_action_indices(ego_action_dim: int) -> jnp.ndarray:
    if ego_action_dim == 6:
        return FIRE_EGO_ACTION_INDICES_6
    return FIRE_EGO_ACTION_INDICES_18
