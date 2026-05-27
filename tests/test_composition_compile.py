"""Smoke compile test for K=1 and K>1 composition paths.

Constructs a real Seaquest env via EnvManager, runs ``make_train`` with a
tiny config (small NUM_ENVS/NUM_STEPS so JIT compile is fast), and lowers the
training function for both K=1 and K=3 to surface any shape/typing bugs in
the composition wiring.

Does not actually execute training — purely a structural verification.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from concept_based_learning.environments.env_manager import EnvManager
from concept_based_learning.training.train_skills import make_train
from concept_based_learning.training.train_tandem import make_tandem_train


def _base_config() -> dict:
    """Minimal config sufficient to build make_train for Seaquest."""
    return {
        "ALG_NAME": "hierarchical_pqn",
        "ENV_NAME": "seaquest",
        "TRAIN_MODE": "skills_only",
        "USE_LEGACY_EGO_OBS": False,
        "USE_LEGACY_SEAQUEST_ACTION_MAPPER": False,

        # Tiny scales — only need JIT compile, not actual training.
        "TOTAL_TIMESTEPS": 8_000,
        "NUM_ENVS": 4,
        "NUM_STEPS": 8,
        "NUM_EPOCHS": 1,
        "NUM_MINIBATCHES": 4,

        # Architecture.
        "NUM_SKILLS": 3,
        "SKILL_NAMES": ["DESTROY", "COLLECT", "AVOID"],
        "MAX_OBJECTS": 35,
        "EGO_ACTION_DIM": 18,
        "HIDDEN_SIZE": 32,
        "NUM_LAYERS": 2,

        # PQN hyperparams.
        "GAMMA": 0.99,
        "LAMBDA": 0.65,
        "LR": 1e-4,
        "MAX_GRAD_NORM": 0.8,
        "OFF_POLICY": True,
        "EPS_START": 0.2,
        "EPS_FINISH": 0.01,
        "EPS_DECAY": 0.8,

        # Env wrappers.
        "FIRST_FIRE": True,
        "STICKY_ACTIONS": False,
        "EPISODIC_LIFE": True,
        "FRAME_STACK_SIZE": 4,

        # Misc.
        "RESTRICT_FIRE_FOR_NON_DESTROY": False,
        "SEED": 0,
    }


def _build_train(config: dict, components, env, test_env):
    return make_train(
        config,
        env,
        test_env,
        renderer=None,
        frozen_meta_params=None,
        get_heuristic_target=components.get_heuristic_target,
        no_target_idx=components.no_target_idx,
        action_mapper=components.action_mapper,
        extract_all_skills_rewards=components.reward_extractor,
        extract_life_lost=components.extract_life_lost,
        fixed_target_idx_from_config=components.fixed_target_idx_from_config,
        meta_mask_hook=components.meta_mask_hook,
        forced_target_hook=components.forced_target_hook,
    )


@pytest.fixture(scope="module")
def env_components():
    manager = EnvManager()
    components = manager.get_components("seaquest")
    env, _ = manager.make_env("seaquest", _base_config(), train=True)
    test_env, _ = manager.make_env("seaquest", _base_config(), train=False)
    return components, env, test_env


def test_train_factory_builds_k1(env_components):
    """K=1 path: must construct the train function without error."""
    components, env, test_env = env_components
    config = _base_config()
    config["COMPOSITION_K"] = 1
    train_fn, _eval_fn = _build_train(config, components, env, test_env)
    assert callable(train_fn)


def test_train_factory_builds_k3_heuristic_slots(env_components):
    """K>1 with heuristic-only (no frozen meta): use slots_from_heuristic_target."""
    components, env, test_env = env_components
    config = _base_config()
    config["COMPOSITION_K"] = 3
    train_fn, _eval_fn = _build_train(config, components, env, test_env)
    assert callable(train_fn)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 1)
    lowered = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None))).lower(
        rngs, None, None
    )
    assert lowered is not None


def test_train_factory_builds_k3_with_frozen_meta(env_components):
    """K>1 with a frozen meta policy: must build the composition path."""
    components, env, test_env = env_components
    config = _base_config()
    config["COMPOSITION_K"] = 3

    # Build dummy frozen meta params by initializing the meta network on a
    # probe observation. We only need parameter shapes; the meta is never
    # trained in this skills factory.
    from concept_based_learning.networks.meta_factory import create_meta_network

    meta_network = create_meta_network(config)
    rng = jax.random.PRNGKey(0)

    # Reset env once to get a sample observation.
    obs, _ = env.reset(rng)
    raw_obs = obs.raw_observation
    objs = obs.objects
    meta_variables = meta_network.init(rng, raw_obs, objs, train=False)
    frozen_meta_params = meta_variables["params"]

    train_fn, _eval_fn = make_train(
        config,
        env,
        test_env,
        renderer=None,
        frozen_meta_params=frozen_meta_params,
        get_heuristic_target=components.get_heuristic_target,
        no_target_idx=components.no_target_idx,
        action_mapper=components.action_mapper,
        extract_all_skills_rewards=components.reward_extractor,
        extract_life_lost=components.extract_life_lost,
        fixed_target_idx_from_config=components.fixed_target_idx_from_config,
        meta_mask_hook=components.meta_mask_hook,
        forced_target_hook=components.forced_target_hook,
    )
    assert callable(train_fn)

    # Lower the train function to JIT — this is the real shape/type check.
    rngs = jax.random.split(rng, 1)
    lowered = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None))).lower(
        rngs, None, None
    )
    # If lowering succeeds without error, the K>1 path is JAX-valid.
    assert lowered is not None


def _build_tandem(config: dict, components, env, test_env):
    return make_tandem_train(
        config,
        env,
        test_env,
        renderer=None,
        no_target_idx=components.no_target_idx,
        get_heuristic_target=components.get_heuristic_target,
        action_mapper=components.action_mapper,
        extract_all_skills_rewards=components.reward_extractor,
        extract_life_lost=components.extract_life_lost,
        meta_mask_hook=components.meta_mask_hook,
        forced_target_hook=components.forced_target_hook,
    )


def test_tandem_factory_builds_k1(env_components):
    """Tandem K=1: factory must construct without error."""
    components, env, test_env = env_components
    config = _base_config()
    config["COMPOSITION_K"] = 1
    train_fn = _build_tandem(config, components, env, test_env)
    assert callable(train_fn)


def test_tandem_factory_builds_k3(env_components):
    """Tandem K=3: factory must construct without error and JIT-lower."""
    components, env, test_env = env_components
    config = _base_config()
    config["COMPOSITION_K"] = 3
    train_fn = _build_tandem(config, components, env, test_env)
    assert callable(train_fn)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 1)
    lowered = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None))).lower(
        rngs, None, None
    )
    assert lowered is not None


def test_tandem_factory_builds_k3_soft_q(env_components):
    """Tandem K=3 with soft-Q meta loss: factory must build and JIT-lower."""
    components, env, test_env = env_components
    config = _base_config()
    config["COMPOSITION_K"] = 3
    config["COMPOSITION_META_LOSS"] = "soft_q"
    config["COMPOSITION_TAU"] = 1.0
    train_fn = _build_tandem(config, components, env, test_env)
    assert callable(train_fn)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 1)
    lowered = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None))).lower(
        rngs, None, None
    )
    assert lowered is not None
