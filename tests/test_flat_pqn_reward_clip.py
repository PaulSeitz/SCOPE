"""Regression: flat PQN Atari wrap stack must sign-clip the frame-skip sum for TD.

Matches purejaxql / EnvPool ``reward_clip=True`` (sign into {-1,0,1}).
``info['env_reward']`` and LogWrapper episode returns stay unclipped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtari
import pytest
from jaxtari.wrappers import AtariWrapper, LogWrapper, PixelObsWrapper


def _pixel_stack(*, clip_reward: bool):
    base = jaxtari.make("seaquest")
    wrapped = AtariWrapper(
        base,
        sticky_actions=0.0,
        episodic_life=True,
        first_fire=True,
        noop_max=0,
        full_action_space=False,
    )
    wrapped = PixelObsWrapper(
        wrapped,
        do_pixel_resize=True,
        pixel_resize_shape=(84, 84),
        grayscale=True,
        use_native_downscaling=False,
        frame_stack_size=4,
        frame_skip=4,
        max_pooling=True,
        clip_reward=clip_reward,
    )
    return LogWrapper(wrapped)


def _force_base_reward(env, value: float = 20.0) -> None:
    """Patch the raw game so every Atari frame returns a fixed reward."""
    # LogWrapper -> PixelObsWrapper -> AtariWrapper -> game
    pixel = env._env
    atari = pixel._env
    game = atari._env
    orig = game.step

    def fake_step(state, action):
        obs, st, rew, done, info = orig(state, action)
        return obs, st, jnp.asarray(value, dtype=rew.dtype), done, info

    game.step = fake_step


@pytest.mark.parametrize("clip,expected_td,expected_env", [(True, 1.0, 80.0), (False, 80.0, 80.0)])
def test_pixel_obs_frame_skip_reward_clip(clip, expected_td, expected_env):
    env = _pixel_stack(clip_reward=clip)
    assert bool(env._env.clip_reward) is clip
    _force_base_reward(env, 20.0)

    _, state = env.reset(jax.random.PRNGKey(0))
    _, new_state, reward, _, _, info = env.step(state, 0)

    assert float(reward) == pytest.approx(expected_td)
    assert float(info["env_reward"]) == pytest.approx(expected_env)
    # LogWrapper accumulates unclipped env_reward into episode_returns.
    assert float(new_state.episode_returns) == pytest.approx(expected_env)


def test_flat_pqn_train_apply_wrappers_wires_clip():
    """PixelObsWrapper must receive CLIP_REWARD from the flat PQN config path."""
    from concept_based_learning.baselines.flat_pqn import normalize_flat_pqn_config

    ncfg = normalize_flat_pqn_config(
        {
            "ENV_NAME": "seaquest",
            "CLIP_REWARD": True,
            "OBJECT_CENTRIC": False,
            "TOTAL_TIMESTEPS": 128,
            "NUM_ENVS": 2,
            "NUM_STEPS": 4,
            "NUM_EPOCHS": 1,
            "NUM_MINIBATCHES": 1,
        }
    )
    assert bool(ncfg.get("CLIP_REWARD", True)) is True

    env = _pixel_stack(clip_reward=bool(ncfg.get("CLIP_REWARD", True)))
    assert env._env.clip_reward is True
    _force_base_reward(env, 20.0)
    _, state = env.reset(jax.random.PRNGKey(1))
    _, _, reward, _, _, info = env.step(state, 0)
    assert float(reward) == pytest.approx(1.0)
    assert float(info["env_reward"]) == pytest.approx(80.0)