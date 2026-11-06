"""Evaluate a NEXUS / HRL checkpoint on the CBL paper_eval mod allowlist.

Uses the same jaxatari mod API as paper_eval (``jaxtari.make(..., mods=...)``)
and writes ``*_mods.json`` / CSV next to other RQ1 results. Env is taken from
the alg config (Seaquest symbolic NEXUS or multi-home HRL).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from nexus.hrl_env import learned_meta_policy as shared_learned_meta_policy
from nexus.hrl_env import nexus_env_to_key


def _max_parallel_mods() -> int:
    raw = os.environ.get("CBL_EVAL_MAX_PARALLEL_MODS", "2")
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(1, n)


def _run_nexus_mods_in_subprocesses(
    *,
    specs: list[Any],
    nexus_root: Path,
    jaxatari_root: Path,
    alg: str,
    ckpt_dir: Path,
    seed: int,
    num_episodes: int,
    num_seeds: int,
    max_frames: int,
    trial: bool,
    output_json: Path,
    use_vendored_seaquest: bool,
    env_key: str,
) -> dict[str, Any]:
    """One process per mod. Each child runs the serial one-mod path."""
    import pickle
    import subprocess
    import tempfile

    from concept_based_learning.paper_eval.csv_export import write_mod_eval_csvs

    width = min(_max_parallel_mods(), len(specs))
    frac = f"{min(0.22, 0.70 / width):.3f}"
    print(
        f"  Parallel mods: {len(specs)} total, {width} at a time, "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={frac} (preallocate off)",
        flush=True,
    )
    tmp = Path(tempfile.mkdtemp(prefix="cbl_nexus_eval_mods_"))
    procs: list[tuple[Any, subprocess.Popen, Path]] = []
    for start in range(0, len(specs), width):
        batch = specs[start : start + width]
        running: list[tuple[Any, subprocess.Popen, Path]] = []
        for spec in batch:
            out_path = tmp / f"{spec.key}.json"
            job = {
                "nexus_root": str(nexus_root),
                "jaxatari_root": str(jaxatari_root),
                "alg": alg,
                "ckpt_dir": str(ckpt_dir),
                "seed": int(seed),
                "num_episodes": int(num_episodes),
                "num_seeds": int(num_seeds),
                "max_frames": int(max_frames),
                "trial": bool(trial),
                "mod_key": spec.key,
                "out": str(out_path),
                "use_vendored_seaquest": bool(use_vendored_seaquest),
            }
            job_path = tmp / f"{spec.key}.pkl"
            job_path.write_bytes(pickle.dumps(job))
            env = os.environ.copy()
            env["CBL_EVAL_MOD_WORKER"] = "1"
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = frac
            env.setdefault("SDL_VIDEODRIVER", "dummy")
            env.setdefault("WANDB_MODE", "disabled")
            proc = subprocess.Popen(
                [sys.executable, "-m", "nexus.mod_eval_worker", str(job_path)],
                env=env,
            )
            running.append((spec, proc, out_path))
        for spec, proc, out_path in running:
            proc.wait()
            procs.append((spec, proc, out_path))

    results: dict[str, Any] = {}
    failed: list[str] = []
    for spec, proc, out_path in procs:
        rc = proc.wait()
        stats = None
        if out_path.is_file():
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            stats = (payload.get("results") or {}).get(spec.key)
        if rc != 0 or not isinstance(stats, dict):
            failed.append(spec.key)
            stats = stats if isinstance(stats, dict) else {
                "error": f"worker exit {rc}",
                "mods": list(spec.mods),
                "category": spec.category,
                "mean_return": None,
                "mean": None,
                "n_episodes": 0,
            }
        stats["mods"] = list(spec.mods)
        stats["category"] = spec.category
        results[spec.key] = stats
        if stats.get("error"):
            print(f"FAILED {spec.key}: {stats['error']}", flush=True)
        else:
            print(
                f"  {spec.key} mean_return={stats['mean_return']:.2f} "
                f"(n={stats['n_episodes']})",
                flush=True,
            )

    _attach_drop_from_base(results)
    payload = {
        "status": "failed" if failed else "complete",
        "failed_mods": failed,
        "agent_kind": "nexus",
        "alg": alg,
        "env": env_key,
        "ckpt": str(ckpt_dir),
        "base_seed": seed,
        "num_episodes": num_episodes,
        "num_seeds": num_seeds,
        "max_frames": max_frames,
        "results": results,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {output_json}")
    write_mod_eval_csvs(
        payload,
        output_stem=output_json.with_suffix(""),
        extra_meta={"alg": alg, "agent_kind": "nexus"},
        master_dir=output_json.parent,
    )
    if failed:
        raise RuntimeError(
            f"NEXUS evaluation failed for {failed}; diagnostics: {output_json}"
        )
    return payload


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

    if min(num_episodes, num_seeds, max_frames) <= 0:
        raise ValueError("episodes, seeds and rollout limit must be positive")
    specs = mod_specs_for_env(env_key, trial=trial, keys=mod_keys)
    if not specs:
        raise ValueError("No evaluation mods selected")
    if mod_keys and set(mod_keys) != {spec.key for spec in specs}:
        raise ValueError(f"Unknown or disabled mod keys: {set(mod_keys) - {spec.key for spec in specs}}")
    if os.environ.get("CBL_EVAL_MOD_WORKER") != "1" and len(specs) > 1:
        return _run_nexus_mods_in_subprocesses(
            specs=specs,
            nexus_root=nexus_root,
            jaxatari_root=jaxatari_root,
            alg=alg,
            ckpt_dir=ckpt_dir,
            seed=seed,
            num_episodes=num_episodes,
            num_seeds=num_seeds,
            max_frames=max_frames,
            trial=trial,
            output_json=output_json,
            use_vendored_seaquest=use_vendored_seaquest,
            env_key=env_key,
        )
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
            traceback.print_exc()
            print(f"FAILED {spec.key}: {e}")
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
    failed = [key for key, stats in results.items() if stats.get("error")]
    payload = {
        "status": "failed" if failed else "complete",
        "failed_mods": failed,
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
    if failed:
        raise RuntimeError(f"NEXUS evaluation failed for {failed}; diagnostics: {output_json}")
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


def _make_nexus_eval_env(env_key: str, game_mods: list[str], *, object_centric: bool = True):
    """Build env via jaxtari.make. OC flattens the object stack; pixel keeps images."""
    import jaxtari
    from jaxtari.wrappers import (
        AtariWrapper,
        FlattenObservationWrapper,
        LogWrapper,
        ObjectCentricWrapper,
        PixelObsWrapper,
    )

    mods = _normalize_game_mods(game_mods)
    env = jaxtari.make(env_key, mods=mods)

    # Match NEXUS eval wrappers: no sticky / no episodic life.
    # AtariWrapper signature differs across forks — try common kwargs.
    try:
        env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
    except TypeError:
        env = AtariWrapper(env, sticky_actions=0.0, episodic_life=False)

    if object_centric:
        # Prefer fork ObjectCentricWrapper; fall back if signature differs.
        try:
            env = ObjectCentricWrapper(env)
        except TypeError:
            env = ObjectCentricWrapper(env, frame_stack_size=4, frame_skip=4, clip_reward=False)
        try:
            env = FlattenObservationWrapper(env)
        except TypeError:
            pass
    else:
        env = PixelObsWrapper(
            env,
            do_pixel_resize=True,
            pixel_resize_shape=(84, 84),
            grayscale=True,
            frame_stack_size=4,
            frame_skip=4,
            max_pooling=True,
            clip_reward=False,
        )
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
    from symbolic_options.hierarchical_pqn_jaxtari import (
        CustomTrainState,
        QNetwork,
        optional_batch_stats,
    )

    llm_meta_policy, learned_meta_policy, combined_meta_policy = _meta_callables(env_key)

    object_centric = bool(config.get("OBJECT_CENTRIC", True))
    env = _make_nexus_eval_env(env_key, game_mods, object_centric=object_centric)
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
        object_centric=object_centric,
    )
    meta_network = QNetwork(
        action_dim=num_agents,
        hidden_size=config.get("HIDDEN_SIZE", 64),
        num_layers=config.get("NUM_LAYERS", 3),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        object_centric=object_centric,
    )

    rng = jax.random.PRNGKey(base_seed)
    if object_centric:
        init_x = jnp.zeros((int(np.prod(obs_shape)),))
    else:
        init_x = jnp.zeros((1, *tuple(obs_shape)))
    rng, r1, r2 = jax.random.split(rng, 3)
    net_vars = network.init(r1, init_x, train=False)
    meta_vars = meta_network.init(r2, init_x, train=False)
    tx = optax.radam(1e-4)

    def one_state(p, bs, apply_fn):
        return CustomTrainState.create(
            apply_fn=apply_fn,
            params=p,
            batch_stats=bs if bs is not None else optional_batch_stats(net_vars),
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
            meta_batch_stats if meta_batch_stats is not None else optional_batch_stats(meta_vars),
            meta_network.apply,
        )

    @jax.jit
    def policy_action(obs, env_state):
        if object_centric:
            model_obs = jnp.reshape(obs, (1, -1))
        elif obs.ndim == 4:
            model_obs = obs[None, ...]
        elif obs.ndim == 5:
            model_obs = obs
        else:
            raise ValueError(f"Unexpected NEXUS eval observation shape {obs.shape}")
        if not hierarchical:
            q = network.apply(
                {"params": train_states.params, "batch_stats": train_states.batch_stats},
                model_obs, train=False,
            )
            return jnp.argmax(q[0])

        def option_q(ts):
            return network.apply(
                {"params": ts.params, "batch_stats": ts.batch_stats},
                model_obs, train=False,
            )[0]

        all_actions = jnp.argmax(jax.vmap(option_q)(train_states), axis=-1)
        # Upstream selectors consume (N, ...) state leaves, just like training.
        batched_state = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], env_state)
        if meta_mode == "llm" and llm_meta_policy is not None:
            meta_q = llm_meta_policy(None, None, model_obs, batched_state)
        elif meta_mode == "learned" and meta_state is not None:
            meta_q = learned_meta_policy(meta_network, meta_state, model_obs, batched_state)
        elif meta_mode == "combined" and meta_state is not None and combined_meta_policy is not None:
            meta_q = combined_meta_policy(meta_network, meta_state, model_obs, batched_state)
        else:
            raise ValueError(f"Unsupported/missing meta-policy checkpoint: {meta_mode!r} ({env_key})")
        active = jnp.argmax(meta_q, axis=-1).reshape(-1)[0]
        return all_actions[active]

    @jax.jit
    def episode(ep_key):
        obs, state = env.reset(ep_key)

        def keep_running(carry):
            _, _, _, done, length = carry
            return (~done) & (length < max_frames)

        def step(carry):
            obs, state, total, _, length = carry
            action = policy_action(obs, state)
            out = env.step(state, action)
            if len(out) == 6:
                obs, state, reward, terminated, truncated, info = out
                done = terminated | truncated
            else:
                obs, state, reward, done, info = out
            total = total + info.get("env_reward", reward)
            return obs, state, total, done, length + 1

        _, _, total, done, length = jax.lax.while_loop(
            keep_running, step,
            (obs, state, jnp.float32(0), jnp.bool_(False), jnp.int32(0)),
        )
        return total, done, length

    returns: list[float] = []
    lengths: list[int] = []
    cutoff: list[bool] = []
    for seed_i in range(num_seeds):
        seed_key = jax.random.PRNGKey(base_seed + seed_i)
        for ep_i in range(num_episodes):
            ep_key = jax.random.fold_in(seed_key, ep_i)
            total, done, length = jax.device_get(episode(ep_key))
            if not np.isfinite(total):
                raise ValueError("Non-finite evaluation return")
            returns.append(float(total))
            lengths.append(int(length))
            cutoff.append(not bool(done))

    arr = np.asarray(returns, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float("nan")
    return {
        "mean_return": mean,
        "mean": mean,
        "std_return": float(arr.std()) if arr.size > 1 else 0.0,
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "returns": returns,
        "episode_lengths": lengths,
        "cutoff": cutoff,
        "n_cutoff": sum(cutoff),
        "rollout_limit_unit": "agent_steps (4 game frames each)",
        "n_episodes": int(arr.size),
        "num_episodes": int(arr.size),
        "num_seeds": int(num_seeds),
        "max_frames_per_episode": int(max_frames),
    }
