import jax.numpy as jnp
import pytest

from concept_based_learning.training.episode_stats import (
    episodic_log_metrics,
    init_per_env_episode_stats,
    update_per_env_episode_stats,
)


def test_ppo_style_smoothed_mean():
    returned, returned_len, ep_len = init_per_env_episode_stats(3)
    running = jnp.zeros(3, dtype=jnp.float32)

    # Env 0 completes with return 100; others keep stale 0.
    running, returned, returned_len, ep_len = update_per_env_episode_stats(
        running,
        returned,
        returned_len,
        ep_len,
        jnp.array([10.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([False, False, False]),
    )
    running, returned, returned_len, ep_len = update_per_env_episode_stats(
        running,
        returned,
        returned_len,
        ep_len,
        jnp.array([90.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([True, False, False]),
    )

    metrics = episodic_log_metrics(
        returned,
        returned_len,
        jnp.array(100.0),
        jnp.array(1.0),
    )
    assert float(metrics["returned_episode_returns"]) == pytest.approx(100.0 / 3.0)
    assert float(metrics["env/episodic_return_completed_batch_mean"]) == 100.0
