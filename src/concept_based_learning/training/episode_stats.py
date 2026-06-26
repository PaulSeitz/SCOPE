"""Per-env episodic statistics aligned with CleanRL envpool PPO logging."""

from __future__ import annotations

import jax.numpy as jnp


def init_per_env_episode_stats(num_envs: int):
    """Return (returned_episode_returns, returned_episode_lengths, episode_lengths)."""
    zeros_f = jnp.zeros((num_envs,), dtype=jnp.float32)
    zeros_i = jnp.zeros((num_envs,), dtype=jnp.int32)
    return zeros_f, zeros_f, zeros_i


def init_full_game_episode_stats(num_envs: int):
    """Return (full_game_running, returned_game_returns, returned_game_lengths, game_lengths)."""
    zeros_f = jnp.zeros((num_envs,), dtype=jnp.float32)
    zeros_i = jnp.zeros((num_envs,), dtype=jnp.int32)
    return zeros_f, zeros_f, zeros_f, zeros_i


def remaining_lives_from_semantic_state(state):
    """Remaining lives after a step (1 when the env has no life counter)."""
    env_state = state.atari_state.env_state
    if hasattr(env_state, "lives"):
        lives = env_state.lives
    elif hasattr(env_state, "player_lives"):
        lives = env_state.player_lives
    else:
        lives = jnp.array(1, dtype=jnp.int32)
    return lives.astype(jnp.int32)


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
    - ``returned_episode_*`` hold the last **completed life** per env slot when
      ``EPISODIC_LIFE`` is enabled (Atari wrapper ``done`` on each death).
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


def update_full_game_episode_stats(
    full_game_running,
    returned_game_returns,
    returned_game_lengths,
    game_lengths,
    step_reward,
    done,
    remaining_lives,
):
    """
    Track score across all lives until game over.

    Matches eval / recorded videos (``episodic_life=False``): the on-screen score
    accumulated until the last life is lost.
    """
    full_game_running = full_game_running + step_reward
    game_lengths = game_lengths + 1
    game_over = jnp.logical_and(done, remaining_lives <= 0)
    game_over_f = game_over.astype(jnp.float32)

    returned_game_returns = jnp.where(game_over, full_game_running, returned_game_returns)
    returned_game_lengths = jnp.where(
        game_over, game_lengths.astype(jnp.float32), returned_game_lengths
    )
    full_game_running = full_game_running * (1.0 - game_over_f)
    game_lengths = (game_lengths * (1.0 - game_over.astype(jnp.int32))).astype(jnp.int32)

    return (
        full_game_running,
        returned_game_returns,
        returned_game_lengths,
        game_lengths,
    )


def episodic_log_metrics(
    returned_episode_returns,
    returned_episode_lengths,
    completed_returns_sum,
    num_completed,
    *,
    returned_game_returns=None,
    returned_game_lengths=None,
):
    """
    Metrics dict: PPO-style per-life mean, full-game mean, legacy batch-completed mean.

    ``returned_episode_returns`` is comparable to CleanRL PPO with ``episodic_life=True``.
    ``returned_game_returns`` matches eval videos / ``episodic_life=False`` rollouts.
    """
    ppo_mean_return = jnp.mean(returned_episode_returns)
    ppo_mean_length = jnp.mean(returned_episode_lengths)
    batch_completed_mean = completed_returns_sum / jnp.maximum(1.0, num_completed)
    metrics = {
        # Per-life (training wrapper with EPISODIC_LIFE=true).
        "returned_episode_returns": ppo_mean_return,
        "charts/avg_episodic_return": ppo_mean_return,
        "env/episodic_return": ppo_mean_return,
        "charts/avg_episodic_length": ppo_mean_length,
        "env/per_life_return": ppo_mean_return,
        # Full game until last life lost (matches recorded eval videos).
        "env/episodic_return_completed_batch_mean": batch_completed_mean,
        "env/num_episodes_completed": num_completed,
    }
    if returned_game_returns is not None:
        game_mean_return = jnp.mean(returned_game_returns)
        metrics["returned_game_returns"] = game_mean_return
        metrics["charts/avg_game_return"] = game_mean_return
        metrics["env/game_return"] = game_mean_return
    if returned_game_lengths is not None:
        metrics["charts/avg_game_length"] = jnp.mean(returned_game_lengths)
    return metrics
