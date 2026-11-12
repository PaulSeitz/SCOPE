"""Launch once per waiting state, with unchanged control during live rallies."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtari.games.jax_breakout import JaxBreakout
from jaxtari.wrappers import AtariWrapper, ObjectCentricWrapper

from concept_based_learning.environments.breakout.auto_launch import (
    BreakoutAutoLaunchWrapper, maybe_wrap_breakout_auto_launch,
)
from concept_based_learning.environments.breakout.breakout_label import BreakoutSemanticWrapper


def assert_same(actual, expected):
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, e)


def test_waiting_launches_once_then_requested_action_is_preserved():
    native = JaxBreakout()
    env = BreakoutAutoLaunchWrapper(native)
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert not bool(obs.ball.active)
    # Deliberately request RIGHT. The first native frame must be FIRE instead.
    step = jax.jit(env.step)
    result = step(state, jnp.int32(2))
    expected = native.step(state, jnp.int32(1))
    assert_same(result[1:], expected[1:])
    assert bool(result[1].game_started)
    assert bool(result[0].ball.active)
    for action in (0, 1, 2, 3):
        assert_same(step(result[1], jnp.int32(action)),
                    native.step(result[1], jnp.int32(action)))


def test_launch_after_actual_life_loss_and_no_launch_after_game_over():
    native = JaxBreakout()
    env = BreakoutAutoLaunchWrapper(native)
    _, state = native.reset(jax.random.PRNGKey(0))
    # Force an actual lost-ball transition through the game implementation.
    state = state.replace(game_started=jnp.bool_(True),
                          ball_y=jnp.int32(250), ball_vel_y=jnp.int32(1))
    obs, lost, _, _, _ = env.step(state, jnp.int32(0))
    assert int(lost.lives) == int(state.lives) - 1
    assert not bool(lost.game_started)
    assert not bool(obs.ball.active)
    result = env.step(lost, jnp.int32(3))
    assert bool(result[1].game_started)
    assert int(result[1].lives) == int(lost.lives)
    assert_same(result[1:], native.step(lost, jnp.int32(1))[1:])
    dead = lost.replace(lives=jnp.int32(0))
    assert_same(env.step(dead, jnp.int32(0))[1:], native.step(dead, jnp.int32(0))[1:])


def test_launch_during_frame_skip_does_not_repeat_fire():
    native = JaxBreakout()
    env = AtariWrapper(BreakoutAutoLaunchWrapper(native), first_fire=False,
                       noop_max=0, sticky_actions=0.0,
                       episodic_life=False)
    env = ObjectCentricWrapper(env, frame_skip=4)
    obs, state = env.reset(jax.random.PRNGKey(0))
    result = env.step(state, jnp.int32(2))
    # Compare with exactly FIRE, RIGHT, RIGHT, RIGHT at native resolution.
    _, expected = native.reset(jax.random.PRNGKey(0))
    for action in (1, 2, 2, 2):
        _, expected, _, _, _ = native.step(expected, jnp.int32(action))
    from concept_based_learning.training.episode_stats import unwrap_base_env_state
    assert_same(unwrap_base_env_state(result[1]), expected)


def test_configuration_scope_and_ball_visibility():
    native = JaxBreakout()
    assert maybe_wrap_breakout_auto_launch(native, "pong", {}) is native
    assert maybe_wrap_breakout_auto_launch(native, "breakout", {"BREAKOUT_AUTO_LAUNCH": False}) is native
    env = maybe_wrap_breakout_auto_launch(native, "Breakout", {})
    assert isinstance(env, BreakoutAutoLaunchWrapper)
    assert maybe_wrap_breakout_auto_launch(env, "breakout", {}) is env
    obs, _ = env.reset(jax.random.PRNGKey(0))
    semantic = BreakoutSemanticWrapper(env)
    assert not bool(semantic._extract_frame_objects(obs).active[-1])
