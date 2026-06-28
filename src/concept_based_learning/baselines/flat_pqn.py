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

import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)

from concept_based_learning.pqn_visualizer import resolve_config_path


class FlatPQNCNN(nn.Module):
    """Pixel backbone — must stay in sync with pqn_agent.CNN."""

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


class FlatPQNQNetwork(nn.Module):
    """Q-head — must stay in sync with pqn_agent.QNetwork."""

    action_dim: int
    hidden_size: int = 128
    num_layers: int = 2
    norm_type: str = "layer_norm"
    norm_input: bool = False
    object_centric: bool = True

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
            x = x / 255.0
            x = FlatPQNCNN(norm_type=self.norm_type)(x, train)

        x = nn.Dense(self.action_dim)(x)
        return x


def load_flat_pqn_params(filename: str | os.PathLike) -> dict:
    """Load a flat PQN params pytree from safetensors (pqn_agent / train_utils format)."""
    flattened_dict = load_file(str(filename))
    return unflatten_dict(flattened_dict, sep=",")


def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_flat_pqn_training_config(model_path: str) -> dict[str, Any]:
    """Load ``*_config.yaml`` saved next to a pqn_agent checkpoint."""
    resolved = resolve_config_path(model_path)
    model_dir = Path(resolved).parent
    for candidate in sorted(model_dir.glob("*_config.yaml")):
        try:
            return OmegaConf.to_container(OmegaConf.load(candidate), resolve=True)
        except Exception:
            return _load_yaml_config(candidate)
    return {}


def infer_flat_pqn_object_centric(model_path: str, config: dict[str, Any]) -> bool:
    if "OBJECT_CENTRIC" in config:
        return bool(config["OBJECT_CENTRIC"])
    name = Path(resolve_config_path(model_path)).name.lower()
    if "_pixel_" in name:
        return False
    if "_oc_" in name:
        return True
    return True


def make_flat_pqn_eval_env(
    *,
    object_centric: bool,
    game_mods: list[str],
    config: dict[str, Any],
):
    mods = list(game_mods) if game_mods else None
    env = jaxatari.make("seaquest", mods=mods)
    env = AtariWrapper(
        env,
        sticky_actions=0.0,
        episodic_life=False,
        first_fire=True,
        noop_max=30,
        full_action_space=False,
    )
    if object_centric:
        env = ObjectCentricWrapper(
            env,
            frame_stack_size=int(config.get("FRAME_STACK_SIZE", 4)),
            frame_skip=int(config.get("FRAME_SKIP", 4)),
            clip_reward=False,
        )
        env = NormalizeObservationWrapper(env)
        env = FlattenObservationWrapper(env)
    else:
        env = PixelObsWrapper(
            env,
            do_pixel_resize=bool(config.get("PIXEL_RESIZE", True)),
            pixel_resize_shape=tuple(config.get("PIXEL_RESIZE_SHAPE", [84, 84])),
            grayscale=bool(config.get("PIXEL_GRAYSCALE", True)),
            use_native_downscaling=bool(config.get("USE_NATIVE_DOWNSCALING", True)),
            frame_stack_size=int(config.get("FRAME_STACK_SIZE", 4)),
            frame_skip=int(config.get("FRAME_SKIP", 4)),
            max_pooling=True,
            clip_reward=False,
        )
        env = NormalizeObservationWrapper(env)
    return LogWrapper(env)


def _make_jitted_flat_pqn_episode_runner(
    env,
    network: FlatPQNQNetwork,
    params,
    batch_stats,
    *,
    max_frames: int,
):
    @jax.jit
    def run_episode(rng):
        obs, state = env.reset(rng)
        total_return = jnp.float32(0.0)

        def body(carry, _):
            obs_c, state_c, total_c = carry
            q_vals = network.apply(
                {"params": params, "batch_stats": batch_stats},
                obs_c[None, ...],
                train=False,
            )
            action = jnp.argmax(q_vals, axis=-1)[0]
            obs_n, state_n, reward, terminated, truncated, _info = env.step(state_c, action)
            done = jnp.logical_or(terminated, truncated)
            total_n = total_c + reward
            return (obs_n, state_n, total_n), done

        (_, _, total_return), _dones = jax.lax.scan(
            body,
            (obs, state, total_return),
            None,
            length=max_frames,
        )
        return total_return

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

    env = make_flat_pqn_eval_env(
        object_centric=object_centric,
        game_mods=game_mods,
        config=config,
    )
    network = FlatPQNQNetwork(
        action_dim=env.action_space().n,
        hidden_size=int(config.get("HIDDEN_SIZE", 128)),
        num_layers=int(config.get("NUM_LAYERS", 3)),
        norm_type=str(config.get("NORM_TYPE", "layer_norm")),
        norm_input=bool(config.get("NORM_INPUT", False)),
        object_centric=object_centric,
    )

    run_episode = _make_jitted_flat_pqn_episode_runner(
        env,
        network,
        params,
        batch_stats,
        max_frames=max_frames,
    )
    if progress:
        label = "oc" if object_centric else "pixel"
        print(f"  Compiling flat PQN ({label}) greedy eval...", flush=True)
    _ = jax.block_until_ready(run_episode(jax.random.PRNGKey(base_seed + 999)))

    returns: list[float] = []
    total_runs = num_seeds * num_episodes
    run_idx = 0
    for seed_idx in range(num_seeds):
        for ep_idx in range(num_episodes):
            run_idx += 1
            if progress:
                print(
                    f"  Flat PQN episode {run_idx}/{total_runs} "
                    f"(seed {seed_idx + 1}/{num_seeds}, ep {ep_idx + 1}/{num_episodes})...",
                    flush=True,
                )
            ep_seed = base_seed + seed_idx * 10_000 + ep_idx * 997
            ep_return = float(jax.device_get(run_episode(jax.random.PRNGKey(ep_seed))))
            returns.append(ep_return)
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
        "num_episodes": int(arr.size),
        "num_seeds": int(num_seeds),
        "episodes_per_seed": int(num_episodes),
        "max_frames_per_episode": int(max_frames),
        "object_centric": bool(object_centric),
    }
