"""Native OC stack matches the monolithic baseline; pixel frames become channels."""

import jax
import jax.numpy as jnp
import jaxtari
from jaxtari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
)

from concept_based_learning.environments.env_manager import EnvManager
from concept_based_learning.observations.native_oc_stack import stack_pqn_pixel_channels
from concept_based_learning.baselines.flat_pqn import QNetwork


def _baseline_env(game, *, noop_max=30, first_fire=True):
    env = jaxtari.make(game)
    env = AtariWrapper(
        env,
        sticky_actions=0.0,
        episodic_life=False,
        first_fire=first_fire,
        noop_max=noop_max,
        full_action_space=False,
    )
    env = ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=False)
    env = NormalizeObservationWrapper(env)
    env = FlattenObservationWrapper(env)
    return env


def _semantic_env(game):
    mgr = EnvManager()
    env, _ = mgr.make_env(
        game,
        {
            "FRAME_STACK_SIZE": 4,
            "FRAME_SKIP": 4,
            "FIRST_FIRE": True,
            "STICKY_ACTIONS": False,
            "EPISODIC_LIFE": False,
            "CLIP_REWARD": False,
            "OC_NATIVE_STACK": True,
            "META_INPUT_MODE": "flat",
            "SKILL_INPUT_MODE": "ego",
        },
        train=False,
    )
    return env


def test_pixel_channel_stack_keeps_frame_order_and_kernel():
    images = jnp.arange(2 * 4 * 6 * 6 * 1, dtype=jnp.float32).reshape(2, 4, 6, 6, 1)
    stacked = stack_pqn_pixel_channels(images)
    assert stacked.shape == (2, 6, 6, 4)
    assert jnp.allclose(stacked[1, 3, 2], images[1, :, 3, 2, 0])

    net = QNetwork(action_dim=6, object_centric=False, channel_stack=True, num_layers=1)
    variables = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 4, 84, 84, 1)), train=False)
    kernel = variables["params"]["CNN_0"]["Conv_0"]["kernel"]
    assert kernel.shape == (8, 8, 4, 32)
    dense = variables["params"]["CNN_0"]["Dense_0"]["kernel"]
    assert dense.shape[0] == 3136


def test_native_oc_stack_matches_baseline_reset_and_step():
    for game in ("seaquest", "namethisgame"):
        key = jax.random.PRNGKey(0)
        base = _baseline_env(game)
        sem = _semantic_env(game)
        base_obs, base_state = base.reset(key)
        sem_obs, sem_state = sem.reset(key)
        assert sem_obs.native_oc_vector.shape == base_obs.shape
        assert jnp.allclose(sem_obs.native_oc_vector, base_obs, atol=0, rtol=0)

        for _ in range(3):
            base_obs, base_state, _, _, _, _ = base.step(base_state, jnp.int32(1))
            sem_obs, sem_state, _, _, _ = sem.step(sem_state, jnp.int32(1))
            assert jnp.allclose(sem_obs.native_oc_vector, base_obs, atol=0, rtol=0)

        # An earlier frame is inside the vector: shifting the stack changes it.
        assert not jnp.allclose(
            sem_obs.native_oc_vector, sem.reset(key)[0].native_oc_vector
        )
