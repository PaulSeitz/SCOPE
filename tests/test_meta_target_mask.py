import jax.numpy as jnp

from concept_based_learning.environments.meta_target_mask import (
    PHOENIX_DYING_STATE_THRESHOLD,
    phoenix_meta_mask_hook,
)


class _FakeObjects:
    def __init__(self, state):
        self.state = state


def test_phoenix_meta_mask_hook_batched_latest_frame():
  num_envs, stack_len, max_objects = 8, 4, 21
  state = jnp.zeros((num_envs, stack_len, max_objects), dtype=jnp.int32)
  state = state.at[:, -1, 3].set(PHOENIX_DYING_STATE_THRESHOLD)
  active_masks = jnp.ones((num_envs, max_objects), dtype=bool)

  masked = phoenix_meta_mask_hook(
      None, active_masks, {}, objects=_FakeObjects(state)
  )

  assert masked.shape == (num_envs, max_objects)
  assert bool(masked[0, 3]) is False
  assert bool(masked[0, 5]) is True


def test_phoenix_meta_mask_hook_single_env_latest_frame():
  stack_len, max_objects = 4, 21
  state = jnp.zeros((stack_len, max_objects), dtype=jnp.int32)
  state = state.at[-1, 2].set(PHOENIX_DYING_STATE_THRESHOLD)
  active_masks = jnp.ones((max_objects,), dtype=bool)

  masked = phoenix_meta_mask_hook(
      None, active_masks, {}, objects=_FakeObjects(state)
  )

  assert masked.shape == (max_objects,)
  assert bool(masked[2]) is False
  assert bool(masked[4]) is True
