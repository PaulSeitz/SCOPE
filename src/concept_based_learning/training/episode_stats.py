"""Per-env episodic statistics aligned with CleanRL envpool PPO logging."""

from __future__ import annotations

import jax.numpy as jnp


def init_per_env_episode_stats(num_envs: int):
    """Return (returned_episode_returns, returned_episode_lengths, episode_lengths)."""
    zeros_f = jnp.zeros((num_envs,), dtype=jnp.float32)
    zeros_i = jnp.zeros((num_envs,), dtype=jnp.int32)
    return zeros_f, zeros_f, zeros_i


def update_per_env_episode_stats(
    running_returns,
    returned_episode_returns,
    returned_episode_lengths,
    episode_lengths,
    step_reward,
    done,
):
    """
    Update per-env return accumulators after one env step.

    Mirrors ``step_env_wrappeed`` in CleanRL's envpool PPO script:
    - ``returned_episode_*`` hold the last **completed** life per env slot.
    - Incomplete lives keep their previous completed values (smoothed logging mean).
    """
    episode_return = running_returns + step_reward
    done_f = done.astype(jnp.float32)
    new_episode_length = episode_lengths + 1

    returned_episode_returns = jnp.where(done, episode_return, returned_episode_returns)
    returned_episode_lengths = jnp.where(
        done, new_episode_length.astype(jnp.float32), returned_episode_lengths
    )
    running_returns = episode_return * (1.0 - done_f)
    episode_lengths = (new_episode_length * (1.0 - done_f)).astype(jnp.int32)

    return (
        running_returns,
        returned_episode_returns,
        returned_episode_lengths,
        episode_lengths,
    )


def episodic_log_metrics(
    returned_episode_returns,
    returned_episode_lengths,
    completed_returns_sum,
    num_completed,
):
    """Metrics dict: PPO-style mean over env slots + legacy batch-completed mean."""
    ppo_mean_return = jnp.mean(returned_episode_returns)
    ppo_mean_length = jnp.mean(returned_episode_lengths)
    batch_completed_mean = completed_returns_sum / jnp.maximum(1.0, num_completed)
    return {
        "returned_episode_returns": ppo_mean_return,
        "charts/avg_episodic_return": ppo_mean_return,
        "env/episodic_return": ppo_mean_return,
        "charts/avg_episodic_length": ppo_mean_length,
        "env/episodic_return_completed_batch_mean": batch_completed_mean,
        "env/num_episodes_completed": num_completed,
    }
