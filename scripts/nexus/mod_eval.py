"""Evaluate a NEXUS / HRL checkpoint on the CBL paper_eval mod allowlist.

Uses the same jaxatari mod API as paper_eval (``jaxtari.make(..., mods=...)``)
and writes ``*_mods.json`` / CSV next to other RQ1 results. Env is taken from
the alg config (Seaquest symbolic NEXUS or multi-home HRL).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from nexus.hrl_env import learned_meta_policy as shared_learned_meta_policy
from nexus.hrl_env import nexus_env_to_key


def run_nexus_mod_eval(
    *,
    nexus_root: Path,
    jaxatari_root: Path,
    alg: str,
    ckpt_dir: Path,
    seed: int = 10044,
    num_episodes: int = 3,
    num_seeds: int = 1,
    max_frames: int = 27000,
    trial: bool = False,
    mod_keys: list[str] | None = None,
    output_json: Path,
    use_vendored_seaquest: bool = False,
) -> dict[str, Any]:
    scripts_dir = Path(__file__).resolve().parents[1]
    project_root = scripts_dir.parent
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(project_root / "src"))

    from nexus.compat import prepare_nexus_runtime

    prepare_nexus_runtime(
        nexus_root=nexus_root,
        jaxatari_root=jaxatari_root,
        use_vendored_seaquest=use_vendored_seaquest,
    )

    from concept_based_learning.paper_eval.csv_export import write_mod_eval_csvs
    from concept_based_learning.paper_eval.mods import mod_specs_for_env
    from omegaconf import OmegaConf
    from symbolic_options.purejaxql.save_load import load_params

    alg_path = nexus_root / "src" / "symbolic_options" / "config" / "alg" / f"{alg}.yaml"
    root_cfg = OmegaConf.load(nexus_root / "src" / "symbolic_options" / "config" / "config.yaml")
    alg_cfg = OmegaConf.load(alg_path)
    config = OmegaConf.to_container(OmegaConf.merge(root_cfg, {"alg": alg_cfg}), resolve=True)
    assert isinstance(config, dict)
    config = {**config, **config["alg"]}

    alg_name = str(config["ALG_NAME"])
    env_name = str(config["ENV_NAME"])
    env_key = nexus_env_to_key(env_name)
    params_path = _find_ckpt(ckpt_dir, alg_name, env_name)
    print(f"Loading NEXUS params from {params_path} (env={env_key})")
    params = load_params(params_path)
    bs_path = params_path.with_name(params_path.stem + "_bs.safetensors")
    batch_stats = load_params(bs_path) if bs_path.is_file() else None

    # Prefer explicit meta ckpt for learned/combined meta policies.
    meta_params = None
    meta_bs = None
    if config.get("META_POLICY") in {"learned", "combined"}:
        meta_path = params_path.with_name(params_path.stem + "_meta.safetensors")
        if meta_path.is_file():
            meta_params = load_params(meta_path)
            meta_bs_path = meta_path.with_name(meta_path.stem + "_bs.safetensors")
            if not meta_bs_path.is_file():
                # NEXUS saves as ..._meta_bs.safetensors
                meta_bs_path = params_path.with_name(
                    params_path.stem + "_meta_bs.safetensors"
                )
            if meta_bs_path.is_file():
                meta_bs = load_params(meta_bs_path)

    specs = mod_specs_for_env(env_key, trial=trial, keys=mod_keys)
    results: dict[str, Any] = {}
    for spec in specs:
        print(f"\n=== NEXUS mod-eval {env_key}/{spec.key} mods={list(spec.mods) or 'none'} ===")
        try:
            stats = _rollout_mod(
                config=config,
                env_key=env_key,
                params=params,
                batch_stats=batch_stats,
                meta_params=meta_params,
                meta_batch_stats=meta_bs,
                game_mods=list(spec.mods),
                num_episodes=num_episodes,
                num_seeds=num_seeds,
                max_frames=max_frames,
                base_seed=seed,
            )
        except Exception as e:
            print(f"SKIP {spec.key}: {e}")
            results[spec.key] = {
                "error": str(e),
                "mods": list(spec.mods),
                "category": spec.category,
                "mean_return": None,
                "mean": None,
                "n_episodes": 0,
            }
            continue
        stats["mods"] = list(spec.mods)
        stats["category"] = spec.category
        results[spec.key] = stats
        print(
            f"  mean_return={stats['mean_return']:.2f} "
            f"± {stats.get('std_return', float('nan')):.2f} "
            f"(n={stats['n_episodes']})"
        )

    _attach_drop_from_base(results)
    payload = {
        "agent_kind": "nexus",
        "alg": alg,
        "env": env_key,
        "ckpt": str(params_path),
        "base_seed": seed,
        "num_episodes": num_episodes,
        "num_seeds": num_seeds,
        "max_frames": max_frames,
        "results": results,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {output_json}")
    write_mod_eval_csvs(
        payload,
        output_stem=output_json.with_suffix(""),
        extra_meta={"alg": alg, "agent_kind": "nexus"},
        master_dir=output_json.parent,
    )
    return payload


def _find_ckpt(ckpt_dir: Path, alg_name: str, env_name: str) -> Path:
    for base in (ckpt_dir / env_name, ckpt_dir):
        if not base.is_dir():
            continue
        hits = sorted(
            p
            for p in base.glob(f"{alg_name}_*.safetensors")
            if "_bs" not in p.name and "_meta" not in p.name
        )
        if hits:
            return hits[0]
        hits = sorted(
            p
            for p in base.glob("*.safetensors")
            if "_bs" not in p.name and "_meta" not in p.name
        )
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No NEXUS checkpoint under {ckpt_dir}")


def _attach_drop_from_base(results: dict[str, Any]) -> None:
    base_ret = (results.get("base") or {}).get("mean_return")
    if base_ret is None:
        return
    for key, stats in results.items():
        if key == "base" or stats.get("mean_return") is None:
            continue
        stats["drop_from_base"] = float(stats["mean_return"]) - float(base_ret)
        if abs(float(base_ret)) > 1e-8:
            stats["retention"] = float(stats["mean_return"]) / float(base_ret)


def _normalize_game_mods(game_mods: list[str]) -> list[str] | None:
    """Map paper_eval keys onto jaxatari mod registry names."""
    out: list[str] = []
    for m in game_mods:
        # Alias kept for allowlist compatibility; same plugin as no_enemies.
        if m == "disable_enemies":
            out.append("no_enemies")
        else:
            out.append(m)
    # de-dupe preserving order
    seen: set[str] = set()
    uniq = []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq or None


def _make_nexus_eval_env(env_key: str, game_mods: list[str]):
    """Build env via jaxtari.make (paper_eval style) + NEXUS OC stack."""
    import jaxtari
    from jaxtari.wrappers import (
        AtariWrapper,
        FlattenObservationWrapper,
        LogWrapper,
        ObjectCentricWrapper,
    )

    mods = _normalize_game_mods(game_mods)
    env = jaxtari.make(env_key, mods=mods)

    # Match NEXUS eval wrappers: no sticky / no episodic life.
    # AtariWrapper signature differs across forks — try common kwargs.
    try:
        env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
    except TypeError:
        env = AtariWrapper(env, sticky_actions=0.0, episodic_life=False)

    # Prefer fork ObjectCentricWrapper; fall back if signature differs.
    try:
        env = ObjectCentricWrapper(env)
    except TypeError:
        env = ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=False)
    try:
        env = FlattenObservationWrapper(env)
    except TypeError:
        pass
    env = LogWrapper(env)
    return env


def _meta_callables(env_key: str):
    """LLM/combined meta stay Seaquest-only; learned meta is shared."""
    learned_meta_policy = shared_learned_meta_policy
    llm_meta_policy = None
    combined_meta_policy = None
    if env_key == "seaquest":
        from symbolic_options.reward_functions.seaquest import (
            combined_meta_policy as _comb,
        )
        from symbolic_options.reward_functions.seaquest import (
            llm_meta_policy as _llm,
        )

        llm_meta_policy = _llm
        combined_meta_policy = _comb
    return llm_meta_policy, learned_meta_policy, combined_meta_policy


def _rollout_mod(
    *,
    config: dict[str, Any],
    env_key: str,
    params,
    batch_stats,
    meta_params,
    meta_batch_stats,
    game_mods: list[str],
    num_episodes: int,
    num_seeds: int,
    max_frames: int,
    base_seed: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import optax
    from symbolic_options.hierarchical_pqn_jaxtari import QNetwork, CustomTrainState

    llm_meta_policy, learned_meta_policy, combined_meta_policy = _meta_callables(env_key)

    env = _make_nexus_eval_env(env_key, game_mods)
    num_actions = int(env.action_space().n)
    obs_shape = env.observation_space().shape
    hierarchical = bool(config.get("HIERARCHICAL", False))
    num_agents = int(config.get("NUM_AGENTS", 3)) if hierarchical else 1
    meta_mode = str(config.get("META_POLICY", "llm")).lower()

    network = QNetwork(
        action_dim=num_actions,
        hidden_size=config.get("HIDDEN_SIZE", 64),
        num_layers=config.get("NUM_LAYERS", 3),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )
    meta_network = QNetwork(
        action_dim=num_agents,
        hidden_size=config.get("HIDDEN_SIZE", 64),
        num_layers=config.get("NUM_LAYERS", 3),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )

    rng = jax.random.PRNGKey(base_seed)
    obs_len = int(np.prod(obs_shape))
    init_x = jnp.zeros(obs_len)
    rng, r1, r2 = jax.random.split(rng, 3)
    net_vars = network.init(r1, init_x, train=False)
    meta_vars = meta_network.init(r2, init_x, train=False)
    tx = optax.radam(1e-4)

    def one_state(p, bs, apply_fn):
        return CustomTrainState.create(
            apply_fn=apply_fn,
            params=p,
            batch_stats=bs if bs is not None else net_vars["batch_stats"],
            tx=tx,
        )

    leaf0 = jax.tree_util.tree_leaves(params)[0]
    if hierarchical and hasattr(leaf0, "shape") and leaf0.shape[0] == num_agents:
        train_states = jax.vmap(
            lambda p, b: one_state(p, b, network.apply),
            in_axes=(0, 0 if batch_stats is not None else None),
        )(params, batch_stats)
    else:
        st = one_state(params, batch_stats, network.apply)
        train_states = (
            jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_agents), st)
            if hierarchical
            else st
        )

    meta_state = None
    if meta_params is not None:
        meta_state = one_state(
            meta_params,
            meta_batch_stats if meta_batch_stats is not None else meta_vars["batch_stats"],
            meta_network.apply,
        )

    returns: list[float] = []
    for seed_i in range(num_seeds):
        for ep_i in range(num_episodes):
            rng, erng = jax.random.split(rng)
            ep_key = jax.random.fold_in(erng, seed_i * 10_000 + ep_i * 997)
            reset_out = env.reset(ep_key)
            if len(reset_out) == 2:
                obs, env_state = reset_out
            else:
                obs, env_state = reset_out[0], reset_out[1]
            ep_ret = 0.0
            for _t in range(max_frames):
                if hierarchical:

                    def _q(ts):
                        return network.apply(
                            {"params": ts.params, "batch_stats": ts.batch_stats},
                            obs.reshape(1, -1) if obs.ndim > 1 else obs[None, ...],
                            train=False,
                        )[0]

                    # Flatten obs for MLP if needed
                    flat_obs = obs.reshape(-1) if hasattr(obs, "reshape") else obs
                    flat_obs_b = flat_obs[None, ...]

                    def _q2(ts):
                        return network.apply(
                            {"params": ts.params, "batch_stats": ts.batch_stats},
                            flat_obs_b,
                            train=False,
                        )[0]

                    all_q = jax.vmap(_q2)(train_states)
                    all_actions = jnp.argmax(all_q, axis=-1)

                    if meta_mode == "llm":
                        if llm_meta_policy is None:
                            raise ValueError(
                                f"META_POLICY=llm is only supported for seaquest "
                                f"(got env={env_key})"
                            )
                        meta_q = llm_meta_policy(None, None, flat_obs_b, env_state)
                    elif meta_mode == "learned" and meta_state is not None:
                        meta_q = learned_meta_policy(
                            meta_network, meta_state, flat_obs_b, env_state
                        )
                    elif meta_mode == "combined" and meta_state is not None:
                        if combined_meta_policy is None:
                            raise ValueError(
                                f"META_POLICY=combined is only supported for seaquest "
                                f"(got env={env_key})"
                            )
                        meta_q = combined_meta_policy(
                            meta_network, meta_state, flat_obs_b, env_state
                        )
                    elif meta_mode == "learned":
                        raise ValueError("learned meta requires a *_meta.safetensors ckpt")
                    else:
                        raise ValueError(
                            f"Unsupported META_POLICY={meta_mode!r} for env={env_key}"
                        )
                    if meta_q.ndim == 1:
                        meta_q = meta_q[None, ...]
                    active = int(jnp.argmax(meta_q, axis=-1)[0])
                    action = int(all_actions[active])
                else:
                    flat_obs = obs.reshape(-1)
                    q = network.apply(
                        {
                            "params": train_states.params,
                            "batch_stats": train_states.batch_stats,
                        },
                        flat_obs[None, ...],
                        train=False,
                    )[0]
                    action = int(jnp.argmax(q))

                step_out = env.step(env_state, action)
                if len(step_out) == 6:
                    obs, env_state, reward, terminated, truncated, info = step_out
                    done = bool(terminated) or bool(truncated)
                else:
                    obs, env_state, reward, done, info = step_out
                    done = bool(done)

                ep_ret += float(reward)
                if isinstance(info, dict):
                    if info.get("returned_episode", False):
                        ep_ret = float(
                            info.get(
                                "returned_episode_returns",
                                info.get("returned_episode_env_returns", ep_ret),
                            )
                        )
                        break
                if done:
                    break
            returns.append(ep_ret)

    arr = np.asarray(returns, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float("nan")
    return {
        "mean_return": mean,
        "mean": mean,
        "std_return": float(arr.std()) if arr.size > 1 else 0.0,
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "returns": returns,
        "n_episodes": int(arr.size),
        "num_episodes": int(arr.size),
        "num_seeds": int(num_seeds),
        "max_frames_per_episode": int(max_frames),
    }
