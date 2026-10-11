"""Generic NO_REWARDS HRL env factory for thesis home envs.

Used when NEXUS ``main.build_setup`` has no upstream branch (TimePilot /
NameThisGame). Seaquest HRL can keep the existing ``reward_funcs=`` path;
symbolic NEXUS is untouched.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

# NEXUS ENV_NAME (TitleCase) → jaxtari.make key / paper_eval env id
NEXUS_ENV_TO_KEY: dict[str, str] = {
    "Seaquest": "seaquest",
    "TimePilot": "timepilot",
    "NameThisGame": "namethisgame",
}

# Home envs that use this factory when missing from upstream build_setup.
HRL_FACTORY_ENVS: frozenset[str] = frozenset({"TimePilot", "NameThisGame"})


def nexus_env_to_key(env_name: str) -> str:
    name = str(env_name)
    if name in NEXUS_ENV_TO_KEY:
        return NEXUS_ENV_TO_KEY[name]
    return name.lower().replace(" ", "")


def learned_meta_policy(network, meta_train_state, last_obs, env_state):
    """Shared learned-meta scorer (obs MLP only; env_state unused)."""
    del env_state
    return network.apply(
        {
            "params": meta_train_state.params,
            "batch_stats": meta_train_state.batch_stats,
        },
        last_obs,
        train=False,
    )


def _game_and_renderer(env_key: str):
    if env_key == "timepilot":
        from jaxtari.games.jax_timepilot import JaxTimePilot, TimePilotRenderer

        return JaxTimePilot(), TimePilotRenderer()
    if env_key == "namethisgame":
        from jaxtari.games.jax_namethisgame import JaxNameThisGame, Renderer_NameThisGame

        return JaxNameThisGame(), Renderer_NameThisGame()
    if env_key == "seaquest":
        from jaxtari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer

        return JaxSeaquest(), SeaquestRenderer()
    raise ValueError(f"Unsupported HRL factory env_key={env_key!r}")


class _MultiRewardBeforeAtari:
    """Like ``MultiRewardWrapper`` but tolerates 5- or 6-tuple base steps."""

    def __init__(self, env, reward_funcs: list[Callable]):
        self._env = env
        self._reward_funcs = list(reward_funcs)
        # Expose for hierarchical_pqn_jaxtari NUM_AGENTS fallback.
        self.reward_funcs = tuple(reward_funcs)

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def reset(self, key):
        return self._env.reset(key)

    def step(self, state, action):
        out = self._env.step(state, action)
        if len(out) == 6:
            obs, new_state, reward, terminated, truncated, info = out
            done = jnp.logical_or(terminated, truncated)
        else:
            obs, new_state, reward, done, info = out
        if hasattr(info, "_asdict"):
            info = dict(info._asdict())
        elif hasattr(info, "__dataclass_fields__"):
            info = {k: getattr(info, k) for k in info.__dataclass_fields__}
        elif not isinstance(info, dict):
            info = {}
        else:
            info = dict(info)
        info["all_rewards"] = jnp.array(
            [rf(state, new_state) for rf in self._reward_funcs]
        )
        return obs, new_state, reward, done, info


def build_hrl_no_rewards_setup(config: dict[str, Any]) -> dict[str, Any]:
    """Build NEXUS ``build_setup`` fields for NO_REWARDS hierarchical PQN."""
    from jaxtari.wrappers import (
        AtariWrapper,
        FlattenObservationWrapper,
        MultiRewardLogWrapper,
        ObjectCentricWrapper,
    )

    env_name = str(config.get("ENV_NAME", ""))
    env_key = nexus_env_to_key(env_name)
    num_agents = int(config.get("NUM_AGENTS", 3))
    sticky = bool(config.get("STICKY_ACTIONS", False))
    episodic_life = bool(config.get("EPISODIC_LIFE", False))
    object_centric = bool(config.get("OBJECT_CENTRIC", True))
    if not object_centric:
        raise ValueError("CBL HRL factory requires OBJECT_CENTRIC=true")

    _, renderer = _game_and_renderer(env_key)

    def create_env(train: bool = False, **_kwargs):
        # Fresh game instance; env_reward closes over that same instance.
        game, _ = _game_and_renderer(env_key)

        def env_reward(prev_state, state):
            return game._get_reward(prev_state, state)

        reward_funcs_local = [env_reward] * num_agents
        env = _MultiRewardBeforeAtari(game, reward_funcs_local)
        if train:
            try:
                env = AtariWrapper(
                    env, sticky_actions=sticky, episodic_life=episodic_life
                )
            except TypeError:
                env = AtariWrapper(
                    env,
                    sticky_actions=0.0 if not sticky else 0.25,
                    episodic_life=episodic_life,
                )
        else:
            try:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            except TypeError:
                env = AtariWrapper(env, sticky_actions=0.0, episodic_life=False)
        env = ObjectCentricWrapper(env)
        env = FlattenObservationWrapper(env)
        env = MultiRewardLogWrapper(env)
        return env

    env = create_env(train=True)
    # Placeholder list (length = NUM_AGENTS); live rewards come from the wrapper.
    game_for_rf, _ = _game_and_renderer(env_key)

    def _rf(prev_state, state, _g=game_for_rf):
        return _g._get_reward(prev_state, state)

    reward_funcs = [_rf] * num_agents
    eval_envs_list = []
    for entry in config.get("eval_envs", []) or [{"label": "clean"}]:
        label = entry.get("label", "default")
        kwargs = {k: v for k, v in entry.items() if k != "label"}
        eval_envs_list.append((create_env(train=False, **kwargs), label))
    if not eval_envs_list:
        eval_envs_list.append((create_env(train=False), "clean"))

    # JAX traces all lax.switch branches in compute_meta_q_vals, so every
    # meta callable must be a real function with the learned-meta signature.
    return {
        "env": env,
        "eval_envs": tuple(eval_envs_list),
        "create_env": create_env,
        "renderer": renderer,
        "reward_funcs": reward_funcs,
        "meta_policy": learned_meta_policy,
        "llm_meta_policy": learned_meta_policy,
        "learned_meta_policy": learned_meta_policy,
        "conditional_meta_policy": learned_meta_policy,
        "combined_meta_policy": learned_meta_policy,
        "test_env": None,
        "test_env_modif": None,
        "env_params": None,
        "basic_env": None,
    }


def try_build_hrl_no_rewards_setup(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return setup dict for supported factory envs, else None."""
    env_name = str(config.get("ENV_NAME", ""))
    if env_name not in HRL_FACTORY_ENVS:
        return None
    if not bool(config.get("NO_REWARDS", False)):
        raise NotImplementedError(
            f"Env {env_name} is only wired for NO_REWARDS HRL in CBL "
            "(symbolic / skill-reward NEXUS is Seaquest-only)."
        )
    if not bool(config.get("HIERARCHICAL", False)):
        raise NotImplementedError(f"Env {env_name} CBL path requires HIERARCHICAL=true")
    return build_hrl_no_rewards_setup(config)
