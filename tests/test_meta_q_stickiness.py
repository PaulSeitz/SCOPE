import jax.numpy as jnp

from concept_based_learning.environments.meta_target_persistence import (
    apply_meta_q_stickiness,
    commit_meta_targets,
)


def _objects(active_last: float):
  class Objects:
    active = jnp.array([[[active_last]]], dtype=jnp.float32)
    was_killed = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    was_collected = jnp.zeros((1, 1, 1), dtype=jnp.float32)

  return Objects()


def test_q_stickiness_keeps_target_above_threshold():
    config = {
        "META_Q_STICKINESS_ENABLED": True,
        "META_Q_STICKINESS_THRESHOLD": 0.5,
        "MAX_OBJECTS": 3,
    }
    proposed = jnp.array([2], dtype=jnp.int32)
    last = jnp.array([1], dtype=jnp.int32)
    q_vals = jnp.array([[0.1, 0.9, 0.2]], dtype=jnp.float32)
    active_masks = jnp.array([[False, True, True]], dtype=bool)

    committed = apply_meta_q_stickiness(
        proposed,
        last,
        _objects(1.0),
        q_vals,
        active_masks,
        config,
        no_target_idx=-1,
        max_objects=3,
    )
    assert int(committed[0]) == 1


def test_q_stickiness_releases_target_below_threshold():
    config = {
        "META_Q_STICKINESS_ENABLED": True,
        "META_Q_STICKINESS_THRESHOLD": 0.5,
        "MAX_OBJECTS": 3,
    }
    proposed = jnp.array([2], dtype=jnp.int32)
    last = jnp.array([1], dtype=jnp.int32)
    q_vals = jnp.array([[0.1, 0.2, 0.8]], dtype=jnp.float32)
    active_masks = jnp.array([[False, True, True]], dtype=bool)

    committed = apply_meta_q_stickiness(
        proposed,
        last,
        _objects(1.0),
        q_vals,
        active_masks,
        config,
        no_target_idx=-1,
        max_objects=3,
    )
    assert int(committed[0]) == 2


def test_commit_meta_targets_applies_q_stickiness_before_persistence():
    config = {
        "META_Q_STICKINESS_ENABLED": True,
        "META_Q_STICKINESS_THRESHOLD": 0.5,
        "META_TARGET_PERSIST_MODE": None,
        "MAX_OBJECTS": 3,
    }
    proposed = jnp.array([2], dtype=jnp.int32)
    last = jnp.array([1], dtype=jnp.int32)
    q_vals = jnp.array([[0.1, 0.9, 0.2]], dtype=jnp.float32)
    active_masks = jnp.array([[False, True, True]], dtype=bool)
    steps = jnp.array([0], dtype=jnp.int32)
    suspended = jnp.array([-1], dtype=jnp.int32)

    committed, new_steps, new_suspended = commit_meta_targets(
        proposed,
        last,
        _objects(1.0),
        q_vals,
        active_masks,
        config,
        no_target_idx=-1,
        max_objects=3,
        persist_steps_remaining=steps,
        suspended_mission_idxs=suspended,
    )
    assert int(committed[0]) == 1
    assert int(new_steps[0]) == 0
    assert int(new_suspended[0]) == -1
