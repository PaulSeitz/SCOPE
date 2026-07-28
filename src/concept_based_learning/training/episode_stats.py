"""Per-env episodic statistics aligned with CleanRL envpool PPO logging."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def logging_env_reward(info: Any, fallback_reward: jnp.ndarray) -> jnp.ndarray:
    """
    Unclipped env score increment for episode / full-game return logging.

    JAXAtari stores the pre-clip reward in ``info["env_reward"]`` (see
    ``AtariWrapper`` / ``LogWrapper``). Prefer that so ``CLIP_REWARD`` only
    affects TD targets, not ``charts/avg_game_return`` / video-comparable scores.
    """
    if isinstance(info, dict) and "env_reward" in info:
        return info["env_reward"]
    env_reward = getattr(info, "env_reward", None)
    if env_reward is not None:
        return env_reward
    return fallback_reward


def unwrap_base_env_state(wrapper_state):
    """
    Recursively unwrap nested wrapper states to the base JAXAtari env_state.

    Handles SemanticState -> PixelAndObjectCentricState -> AtariState chains
    produced when pixel-based input modes add PixelAndObjectObsWrapper.
    """
    cur = wrapper_state
    for _ in range(8):
        if hasattr(cur, "env_state"):
            return cur.env_state
        if hasattr(cur, "atari_state"):
            cur = cur.atari_state
            continue
        break
    raise AttributeError(
        "Could not unwrap env_state from wrapper state; "
        f"received type {type(wrapper_state)!r}."
    )


# Backwards-compatible alias used internally in this module.
_unwrap_env_state = unwrap_base_env_state


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
    env_state = _unwrap_env_state(state.atari_state)
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


def is_game_over(done, life_lost, prev_lives, lives_after_step):
    """
    True when the full game ends, not a single episodic-life death.

    Covers last-life loss, time-limit truncation, and envs with no life counter
    (e.g. Freeway), all of which hard-reset the env slot via ``reset_mask``.

    Uses pre-reset env state from the step output; post-reset merged state still has
    full lives and must not be used here.
    """
    last_life_lost = jnp.logical_and(life_lost > 0, prev_lives <= life_lost)
    no_lives_left = lives_after_step <= 0
    return jnp.logical_and(
        done,
        jnp.logical_or(
            life_lost <= 0,
            jnp.logical_or(no_lives_left, last_life_lost),
        ),
    )


def update_full_game_episode_stats(
    full_game_running,
    returned_game_returns,
    returned_game_lengths,
    game_lengths,
    step_reward,
    game_over,
):
    """
    Track score across all lives until game over.

    Matches eval / recorded videos (``episodic_life=False``): the on-screen score
    accumulated until the last life is lost.
    """
    full_game_running = full_game_running + step_reward
    game_lengths = game_lengths + 1
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
    *,
    returned_game_returns=None,
    returned_game_lengths=None,
):
    """
    Metrics dict: PPO-style per-life mean and full-game mean.

    ``returned_episode_returns`` is comparable to CleanRL PPO with ``episodic_life=True``.
    ``returned_game_returns`` matches eval videos / ``episodic_life=False`` rollouts
    when fed unclipped ``info["env_reward"]`` (not sign-clipped step rewards).
    """
    ppo_mean_return = jnp.mean(returned_episode_returns)
    ppo_mean_length = jnp.mean(returned_episode_lengths)
    metrics = {
        # Per-life (training wrapper with EPISODIC_LIFE=true).
        "returned_episode_returns": ppo_mean_return,
        "charts/avg_episodic_return": ppo_mean_return,
        "env/episodic_return": ppo_mean_return,
        "charts/avg_episodic_length": ppo_mean_length,
        "env/per_life_return": ppo_mean_return,
    }
    if returned_game_returns is not None:
        game_mean_return = jnp.mean(returned_game_returns)
        metrics["returned_game_returns"] = game_mean_return
        metrics["charts/avg_game_return"] = game_mean_return
        metrics["env/game_return"] = game_mean_return
    if returned_game_lengths is not None:
        metrics["charts/avg_game_length"] = jnp.mean(returned_game_lengths)
    return metrics
