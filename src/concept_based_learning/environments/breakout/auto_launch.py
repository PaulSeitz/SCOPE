"""Shared Breakout launch adapter for training and evaluation."""

import jax.numpy as jnp
from jaxtari.wrappers import JaxtariWrapper


class BreakoutAutoLaunchWrapper(JaxtariWrapper):
    """Launch a waiting ball with one native-frame FIRE pulse.

    This wrapper belongs below AtariWrapper, so it also works after life loss
    and during frame skipping. Once play starts, the requested action is used.
    The launch flag stays inside the adapter and is never added to policy input.
    """

    @staticmethod
    def _visible_ball(obs, state):
        # The native renderer hides the ball before launch, but its OC record
        # incorrectly remains active. Keep target availability consistent.
        return obs.replace(ball=obs.ball.replace(
            active=jnp.asarray(state.game_started & (state.lives > 0),
                               dtype=obs.ball.active.dtype)))

    def reset(self, key=None):
        obs, state = self._env.reset(key)
        return self._visible_ball(obs, state), state

    def step(self, state, action):
        waiting = ~jnp.asarray(state.game_started, dtype=jnp.bool_) & (state.lives > 0)
        effective_action = jnp.where(waiting, jnp.asarray(1, dtype=jnp.int32), action)
        obs, state, reward, done, info = self._env.step(state, effective_action)
        return self._visible_ball(obs, state), state, reward, done, info


def maybe_wrap_breakout_auto_launch(env, env_name, config):
    """Enabled by default for Breakout; disable to reproduce legacy rollouts."""
    if str(env_name).lower() != "breakout" or not config.get("BREAKOUT_AUTO_LAUNCH", True):
        return env
    if isinstance(env, BreakoutAutoLaunchWrapper):
        return env
    return BreakoutAutoLaunchWrapper(env)
