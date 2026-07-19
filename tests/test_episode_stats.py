import jax.numpy as jnp
import pytest

from concept_based_learning.training.episode_stats import (
    episodic_log_metrics,
    init_full_game_episode_stats,
    init_per_env_episode_stats,
    is_game_over,
    update_full_game_episode_stats,
    update_per_env_episode_stats,
)


def test_ppo_style_per_life_mean():
    returned, returned_len, ep_len = init_per_env_episode_stats(3)
    running = jnp.zeros(3, dtype=jnp.float32)

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

    metrics = episodic_log_metrics(returned, returned_len)
    assert float(metrics["returned_episode_returns"]) == pytest.approx(100.0 / 3.0)
    assert float(metrics["env/per_life_return"]) == pytest.approx(100.0 / 3.0)


def test_full_game_accumulates_across_lives():
    full_run, returned_game, returned_game_len, game_len = init_full_game_episode_stats(1)
    returned_life, returned_life_len, life_len = init_per_env_episode_stats(1)
    running = jnp.zeros(1, dtype=jnp.float32)

    # Life 1: score 200, one life remains -> per-life done, game continues.
    running, returned_life, returned_life_len, life_len = update_per_env_episode_stats(
        running,
        returned_life,
        returned_life_len,
        life_len,
        jnp.array([200.0]),
        jnp.array([True]),
    )
    mid_game = is_game_over(
        jnp.array([True]),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([2], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    )
    full_run, returned_game, returned_game_len, game_len = update_full_game_episode_stats(
        full_run,
        returned_game,
        returned_game_len,
        game_len,
        jnp.array([200.0]),
        mid_game,
    )
    assert float(returned_life[0]) == 200.0
    assert float(returned_game[0]) == 0.0
    assert float(full_run[0]) == 200.0

    # Life 2: score 400 more, game over.
    running, returned_life, returned_life_len, life_len = update_per_env_episode_stats(
        running,
        returned_life,
        returned_life_len,
        life_len,
        jnp.array([400.0]),
        jnp.array([True]),
    )
    game_over = is_game_over(
        jnp.array([True]),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )
    full_run, returned_game, returned_game_len, game_len = update_full_game_episode_stats(
        full_run,
        returned_game,
        returned_game_len,
        game_len,
        jnp.array([400.0]),
        game_over,
    )
    assert float(returned_life[0]) == 400.0
    assert float(returned_game[0]) == 600.0

    metrics = episodic_log_metrics(
        returned_life,
        returned_life_len,
        returned_game_returns=returned_game,
        returned_game_lengths=returned_game_len,
    )
    assert float(metrics["returned_game_returns"]) == 600.0
    assert float(metrics["returned_episode_returns"]) == 400.0


def test_full_game_commits_on_time_limit_without_life_loss():
    """Time-limited envs (e.g. Freeway) end with done=True, life_lost=0."""
    full_run, returned_game, returned_game_len, game_len = init_full_game_episode_stats(1)

    game_over = is_game_over(
        jnp.array([True]),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    )
    assert bool(game_over[0])

    full_run, returned_game, returned_game_len, game_len = update_full_game_episode_stats(
        full_run,
        returned_game,
        returned_game_len,
        game_len,
        jnp.array([35.0]),
        game_over,
    )
    assert float(returned_game[0]) == 35.0

    metrics = episodic_log_metrics(
        jnp.zeros(1),
        jnp.zeros(1),
        returned_game_returns=returned_game,
    )
    assert float(metrics["env/game_return"]) == 35.0


def test_mid_life_death_is_not_game_over():
    game_over = is_game_over(
        jnp.array([True]),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([3], dtype=jnp.int32),
        jnp.array([2], dtype=jnp.int32),
    )
    assert not bool(game_over[0])
