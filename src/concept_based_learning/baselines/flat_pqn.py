"""
Flat PQN baseline (pixel + object-centric) for evaluating JAXAtari benchmark checkpoints.

Architecture matches ``JAXAtari/scripts/benchmarks/pqn_agent.py`` so weights saved by
that training script load without conversion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from omegaconf import OmegaConf
from safetensors.flax import load_file

import jaxtari
from jaxtari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)

from concept_based_learning.environments.breakout.auto_launch import maybe_wrap_breakout_auto_launch
from concept_based_learning.observations.native_oc_stack import stack_pqn_pixel_channels
from concept_based_learning.pqn_visualizer import resolve_config_path
from concept_based_learning.paper_eval.env_metrics import (
    aggregate_episode_metrics,
    init_episode_metrics,
    make_metric_fns,
    metrics_to_python,
)

from concept_based_learning.jaxtari_mods import register_modifications

register_modifications()


class CNN(nn.Module):
    """Pixel backbone — class name must match ``pqn_agent.CNN`` for checkpoint keys."""

    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda t: nn.LayerNorm()(t)
        elif self.norm_type == "batch_norm":
            normalize = lambda t: nn.BatchNorm(use_running_average=not train)(t)
        else:
            normalize = lambda t: t

        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    """Q-head — class name must match ``pqn_agent.QNetwork`` for checkpoint keys."""

    action_dim: int
    hidden_size: int = 128
    num_layers: int = 2
    norm_type: str = "layer_norm"
    norm_input: bool = False
    object_centric: bool = True
    channel_stack: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # Dummy batch-norm path kept for checkpoint compatibility with pqn_agent.
            _ = nn.BatchNorm(use_running_average=not train)(x)

        if self.object_centric:
            if self.norm_type == "layer_norm":
                normalize = lambda t: nn.LayerNorm()(t)
            elif self.norm_type == "batch_norm":
                normalize = lambda t: nn.BatchNorm(use_running_average=not train)(t)
            else:
                normalize = lambda t: t

            for _ in range(self.num_layers):
                x = nn.Dense(self.hidden_size)(x)
                x = normalize(x)
                x = nn.relu(x)
        else:
            # Old checkpoints leave frames as a batch axis (in_channels=1).
            # Corrected runs set channel_stack so [B,T,H,W,C] becomes [B,H,W,T*C].
            if self.channel_stack:
                x = stack_pqn_pixel_channels(x)
            x = x / 255.0
            x = CNN(norm_type=self.norm_type)(x, train)

        x = nn.Dense(self.action_dim)(x)
        return x


def load_flat_pqn_params(filename: str | os.PathLike) -> dict:
    """Load a flat PQN params pytree from safetensors (pqn_agent / train_utils format)."""
    flattened_dict = load_file(str(filename))
    return unflatten_dict(flattened_dict, sep=",")


def _unwrap_wandb_config_value(raw: Any) -> Any:
    """Unwrap wandb export nodes like ``{"value": 42}`` / nested ``alg.value``."""
    if isinstance(raw, dict):
        if set(raw.keys()) == {"value"}:
            return _unwrap_wandb_config_value(raw["value"])
        return {k: _unwrap_wandb_config_value(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_unwrap_wandb_config_value(v) for v in raw]
    return raw


def normalize_flat_pqn_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten pqn_agent Hydra configs (top-level + ``alg``) like ``single_run`` does.

    Also accepts wandb JSON exports where every leaf is wrapped in ``{"value": ...}``.
    """
    if not config:
        return {}

    unwrapped = _unwrap_wandb_config_value(config)
    if not isinstance(unwrapped, dict):
        return {}

    out = {k: v for k, v in unwrapped.items() if k != "_wandb"}
    alg = out.pop("alg", None) or out.pop("ALG", None)
    if isinstance(alg, dict):
        out = {**out, **alg}
    return out


def infer_flat_pqn_num_layers(params: dict, *, action_dim: int, object_centric: bool) -> int:
    """
    Infer MLP depth from checkpoint weights (matches pqn_agent when config omits NUM_LAYERS).
    """
    if not object_centric:
        return 2

    flat = flatten_dict(params, sep="/")
    hidden_layers = 0
    for key, val in flat.items():
        if not key.endswith("kernel") or "Dense" not in key:
            continue
        if getattr(val, "ndim", 0) != 2:
            continue
        out_dim = int(val.shape[1])
        if out_dim != int(action_dim):
            hidden_layers += 1
    return max(1, hidden_layers) if hidden_layers else 2


def infer_flat_pqn_action_dim(params: dict) -> int | None:
    """Read Q-head output width from checkpoint (last Dense ``kernel``)."""
    flat = flatten_dict(params, sep="/")
    best = None
    for key, val in flat.items():
        if not key.endswith("kernel") or "Dense" not in key:
            continue
        if getattr(val, "ndim", 0) != 2:
            continue
        # Prefer the top-level Q head (not CNN Dense_0).
        if key.startswith("CNN"):
            continue
        best = int(val.shape[1])
    return best


def flat_pqn_network_kwargs(
    config: dict[str, Any],
    params: dict,
    *,
    action_dim: int,
    object_centric: bool,
) -> dict[str, Any]:
    """Build QNetwork kwargs aligned with a pqn_agent checkpoint."""
    cfg = normalize_flat_pqn_config(config)
    num_layers = cfg.get("NUM_LAYERS")
    if num_layers is None:
        num_layers = infer_flat_pqn_num_layers(
            params, action_dim=action_dim, object_centric=object_centric
        )
    return {
        "action_dim": action_dim,
        "hidden_size": int(cfg.get("HIDDEN_SIZE", 128)),
        "num_layers": int(num_layers),
        "norm_type": str(cfg.get("NORM_TYPE", "layer_norm")),
        "norm_input": bool(cfg.get("NORM_INPUT", False)),
        "object_centric": object_centric,
        "channel_stack": bool(cfg.get("PIXEL_CHANNEL_STACK", False)),
    }


def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_flat_pqn_training_config(model_path: str) -> dict[str, Any]:
    """Load ``*_config.yaml`` saved next to a pqn_agent checkpoint."""
    resolved = resolve_config_path(model_path)
    model_dir = Path(resolved).parent
    model_stem = Path(resolved).stem.lower()

    def _matches_checkpoint(candidate: Path) -> bool:
        name = candidate.name.lower()
        if "_pixel_" in model_stem:
            return "_pixel_" in name
        if "_oc_" in model_stem:
            return "_oc_" in name
        return True

    candidates = sorted(model_dir.glob("*_config.yaml"))
    matched = [c for c in candidates if _matches_checkpoint(c)]
    if not matched:
        matched = candidates

    for candidate in matched:
        try:
            raw = OmegaConf.to_container(OmegaConf.load(candidate), resolve=True)
        except Exception:
            raw = _load_yaml_config(candidate)
        return normalize_flat_pqn_config(raw)
    return {}


def infer_flat_pqn_object_centric(model_path: str, config: dict[str, Any]) -> bool:
    cfg = normalize_flat_pqn_config(config)
    if "OBJECT_CENTRIC" in cfg:
        return bool(cfg["OBJECT_CENTRIC"])
    name = Path(resolve_config_path(model_path)).name.lower()
    if "_pixel_" in name:
        return False
    if "_oc_" in name:
        return True
    return True


def _prepare_pixel_policy_obs(obs: jnp.ndarray) -> jnp.ndarray:
    """
    Match ``pqn_agent`` video eval: if the env returns RGB stacks but the policy
    was trained with ``PIXEL_GRAYSCALE=true``, reduce the channel axis.
    """
    if obs.ndim >= 3 and int(obs.shape[-1]) == 3:
        weights = jnp.array([0.2989, 0.5870, 0.1140], dtype=jnp.float32)
        return jnp.tensordot(obs, weights, axes=([-1], [0]))[..., None]
    return obs


def make_flat_pqn_eval_env(
    *,
    object_centric: bool,
    game_mods: list[str],
    config: dict[str, Any],
    env_name: str | None = None,
    full_action_space: bool | None = None,
    required_action_dim: int | None = None,
):
    cfg = normalize_flat_pqn_config(config)
    mods = list(game_mods) if game_mods else None
    game = str(env_name or cfg.get("ENV_NAME", "seaquest")).lower()
    oc_input_mode = str(cfg.get("OC_INPUT_MODE") or "").strip()

    def _build(full: bool):
        if oc_input_mode:
            from concept_based_learning.baselines.flat_oc_features import (
                make_flat_oc_build_env,
            )

            # full_action_space is unused for Seaquest home evals; OC builds
            # always use the EnvManager Atari defaults.
            del full
            return make_flat_oc_build_env(
                mode=oc_input_mode,
                max_objects=int(cfg.get("MAX_OBJECTS", 35)),
                env_name=game,
                config=cfg,
                train=False,
                game_mods=mods,
            )

        env = jaxtari.make(game, mods=mods)
        env = maybe_wrap_breakout_auto_launch(env, game, cfg)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=False,
            first_fire=True,
            noop_max=30,
            full_action_space=bool(full),
        )
        if object_centric:
            env = ObjectCentricWrapper(
                env,
                frame_stack_size=int(cfg.get("FRAME_STACK_SIZE", 4)),
                frame_skip=int(cfg.get("FRAME_SKIP", 4)),
                clip_reward=False,
            )
            env = NormalizeObservationWrapper(env)
            env = FlattenObservationWrapper(env)
        else:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=bool(cfg.get("PIXEL_RESIZE", True)),
                pixel_resize_shape=tuple(cfg.get("PIXEL_RESIZE_SHAPE", [84, 84])),
                grayscale=bool(cfg.get("PIXEL_GRAYSCALE", True)),
                use_native_downscaling=bool(cfg.get("USE_NATIVE_DOWNSCALING", True)),
                frame_stack_size=int(cfg.get("FRAME_STACK_SIZE", 4)),
                frame_skip=int(cfg.get("FRAME_SKIP", 4)),
                max_pooling=True,
                clip_reward=False,
            )
            # Pixel PQN trains on raw uint8 stacks inside PixelObsWrapper; the CNN
            # applies ``x / 255.0`` internally. Do NOT add NormalizeObservationWrapper.
        return LogWrapper(env)

    if full_action_space is not None:
        return _build(bool(full_action_space))

    # Prefer the training default (minimal set). If a checkpoint expects 18
    # actions and this game only exposes e.g. 6, fall back to full ALE set so
    # Seaquest-trained weights can zero-shot on other envs.
    env = _build(False)
    if required_action_dim is not None and int(env.action_space().n) != int(
        required_action_dim
    ):
        env = _build(True)
    return env


def _make_jitted_flat_pqn_episode_runner(
    env,
    network: QNetwork,
    params,
    batch_stats,
    *,
    max_frames: int,
    env_name: str,
    random_action_dim: int | None = None,
):
    """``random_action_dim`` set → uniform random actions; ``network`` is unused."""
    update_metrics, _finalize_unused = make_metric_fns(
        env_name, has_semantic_obs=False
    )

    @jax.jit
    def run_episode(rng):
        obs, state = env.reset(rng)
        action_rng = jax.random.fold_in(rng, 1)

        def cond(carry):
            _obs, _state, _total, done, step, _metrics = carry
            return jnp.logical_and(step < max_frames, jnp.logical_not(done))

        def body(carry):
            obs_c, state_c, total_c, done, step, metrics = carry
            if random_action_dim is not None:
                action = jax.random.randint(
                    jax.random.fold_in(action_rng, step), (), 0, int(random_action_dim)
                )
            else:
                policy_obs = obs_c
                if not network.object_centric:
                    policy_obs = _prepare_pixel_policy_obs(policy_obs)
                q_vals = network.apply(
                    {"params": params, "batch_stats": batch_stats},
                    policy_obs[None, ...],
                    train=False,
                )
                action = jnp.argmax(q_vals, axis=-1)[0]
            obs_n, state_n, reward, terminated, truncated, _info = env.step(
                state_c, action
            )
            step_done = jnp.logical_or(terminated, truncated)

            def _reward_only():
                # Terminal OC/pixel wrappers autoreset; state deltas vs reset are junk.
                pos = jnp.where(
                    jnp.asarray(reward, dtype=jnp.float32) > 0,
                    jnp.float32(1.0),
                    jnp.float32(0.0),
                )
                return {
                    **metrics,
                    "env_positive_reward_steps": metrics["env_positive_reward_steps"]
                    + pos,
                    "env_steps": metrics["env_steps"] + jnp.float32(1.0),
                }

            metrics = jax.lax.cond(
                done,
                lambda: metrics,
                lambda: jax.lax.cond(
                    step_done,
                    _reward_only,
                    lambda: update_metrics(metrics, state_c, state_n, obs_n, reward),
                ),
            )
            new_done = jnp.logical_or(done, step_done)
            return (
                obs_n,
                state_n,
                total_c + reward,
                new_done,
                step + 1,
                metrics,
            )

        init = (
            obs,
            state,
            jnp.float32(0.0),
            jnp.bool_(False),
            jnp.int32(0),
            init_episode_metrics(),
        )
        final = jax.lax.while_loop(cond, body, init)
        # Prefer last pre-done metrics; finalize may see autoreset state.
        metrics = final[5]
        return final[2], metrics

    return run_episode


def run_flat_pqn_greedy_eval(
    *,
    model_path: str,
    game_mods: list[str],
    num_episodes: int,
    num_seeds: int,
    base_seed: int,
    max_frames: int,
    object_centric: bool | None = None,
    env_name: str | None = None,
    progress: bool = True,
    on_episode_done: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config = load_flat_pqn_training_config(model_path)
    if object_centric is None:
        object_centric = infer_flat_pqn_object_centric(model_path, config)

    resolved = resolve_config_path(model_path)
    params = load_flat_pqn_params(resolved)
    bs_path = resolved.replace(".safetensors", "_bs.safetensors")
    batch_stats = load_flat_pqn_params(bs_path) if Path(bs_path).is_file() else {}

    train_env = str(config.get("ENV_NAME", "seaquest")).lower()
    eval_env = str(env_name or train_env).lower()
    ckpt_action_dim = infer_flat_pqn_action_dim(params)
    cross_env = eval_env != train_env
    env = make_flat_pqn_eval_env(
        object_centric=object_centric,
        game_mods=game_mods,
        config=config,
        env_name=eval_env,
        full_action_space=True if cross_env else None,
        required_action_dim=ckpt_action_dim,
    )
    action_dim = env.action_space().n
    if ckpt_action_dim is not None and int(action_dim) != int(ckpt_action_dim):
        raise ValueError(
            f"Flat PQN checkpoint action_dim={ckpt_action_dim} but "
            f"env={eval_env!r} exposes n={action_dim}. Cannot zero-shot load."
        )
    network = QNetwork(
        **flat_pqn_network_kwargs(
            config,
            params,
            action_dim=action_dim,
            object_centric=object_centric,
        )
    )

    run_episode = _make_jitted_flat_pqn_episode_runner(
        env,
        network,
        params,
        batch_stats,
        max_frames=max_frames,
        env_name=eval_env,
    )
    if progress:
        label = "oc" if object_centric else "pixel"
        xfer = "" if eval_env == train_env else f"  zero-shot {train_env}→{eval_env}"
        print(f"  Compiling flat PQN ({label}) greedy eval...{xfer}", flush=True)
    return {
        **_collect_episode_stats(
            run_episode,
            num_episodes=num_episodes,
            num_seeds=num_seeds,
            base_seed=base_seed,
            max_frames=max_frames,
            label="Flat PQN",
            progress=progress,
            on_episode_done=on_episode_done,
        ),
        "object_centric": bool(object_centric),
        "train_env": train_env,
        "eval_env": eval_env,
    }


def _collect_episode_stats(
    run_episode,
    *,
    num_episodes: int,
    num_seeds: int,
    base_seed: int,
    max_frames: int,
    label: str,
    progress: bool,
    on_episode_done: Callable[[], None] | None,
) -> dict[str, Any]:
    _ = jax.block_until_ready(run_episode(jax.random.PRNGKey(base_seed + 999))[0])

    returns: list[float] = []
    episode_metrics: list[dict[str, float]] = []
    total_runs = num_seeds * num_episodes
    run_idx = 0
    for seed_idx in range(num_seeds):
        for ep_idx in range(num_episodes):
            run_idx += 1
            if progress:
                print(
                    f"  {label} episode {run_idx}/{total_runs} "
                    f"(seed {seed_idx + 1}/{num_seeds}, ep {ep_idx + 1}/{num_episodes})...",
                    flush=True,
                )
            ep_seed = base_seed + seed_idx * 10_000 + ep_idx * 997
            ep_return_j, ep_metrics_j = run_episode(jax.random.PRNGKey(ep_seed))
            ep_return = float(jax.device_get(ep_return_j))
            ep_metrics = metrics_to_python(jax.device_get(ep_metrics_j))
            returns.append(ep_return)
            episode_metrics.append(ep_metrics)
            if on_episode_done is not None:
                on_episode_done()
            if progress:
                print(f"    return={ep_return:.2f}", flush=True)

    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "returns": returns,
        "episode_metrics": episode_metrics,
        "num_episodes": int(arr.size),
        "num_seeds": int(num_seeds),
        "episodes_per_seed": int(num_episodes),
        "max_frames_per_episode": int(max_frames),
        **aggregate_episode_metrics(episode_metrics),
    }


def run_random_action_eval(
    *,
    config: dict[str, Any],
    game_mods: list[str],
    num_episodes: int,
    num_seeds: int,
    base_seed: int,
    max_frames: int,
    env_name: str,
    progress: bool = True,
    on_episode_done: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Uniform random actions over the game's minimal action set (Atari floor)."""
    eval_env = str(env_name).lower()
    env = make_flat_pqn_eval_env(
        object_centric=True,
        game_mods=game_mods,
        config={**config, "OC_INPUT_MODE": None},
        env_name=eval_env,
        full_action_space=False,
    )
    action_dim = int(env.action_space().n)
    run_episode = _make_jitted_flat_pqn_episode_runner(
        env,
        None,
        None,
        None,
        max_frames=max_frames,
        env_name=eval_env,
        random_action_dim=action_dim,
    )
    if progress:
        print(f"  Compiling random-action eval ({eval_env}, n={action_dim})...", flush=True)
    return {
        **_collect_episode_stats(
            run_episode,
            num_episodes=num_episodes,
            num_seeds=num_seeds,
            base_seed=base_seed,
            max_frames=max_frames,
            label="Random",
            progress=progress,
            on_episode_done=on_episode_done,
        ),
        "action_dim": action_dim,
        "eval_env": eval_env,
    }


# Backwards-compatible aliases (Flax param paths depend on class names).
FlatPQNCNN = CNN
FlatPQNQNetwork = QNetwork
