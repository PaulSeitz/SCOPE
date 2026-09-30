# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqn_atari_envpoolpy
"""CleanRL PQN Atari training loop (exact copy of jaxatari/baselines_new pqn_atari.py)
with a drop-in JAXAtari Seaquest vector env instead of ALE ``AtariVectorEnv``.

Source W&B run: https://wandb.ai/jaxatari/baselines_new/runs/f163gauq
Original script kept at: scripts/benchmarks/baselines/pqn_atari_ale_original.py

Default HPs match that run (seed=2, 10M, 128×32, …). Architecture / optimizer /
Q(λ) / ε schedule / logging are unchanged — only the env backend differs.

Obs layout matches ALE: uint8 ``(N, 4, 84, 84)`` NCHW for the CleanRL Conv2d net.
Wrap stack mirrors AtariVectorEnv defaults used in f163gauq:
  frameskip=4, stack=4, 84×84 gray, maxpool, noop=30, fire-reset, sticky=0,
  episodic_life=True, reward_clipping=False here + ClipRewardsWrapper after
  RecordEpisodeStatistics (unclipped charts, clipped TD targets).

JAXAtari path uses ``USE_NATIVE_DOWNSCALING=false`` (ALE-style bilinear resize).

Example::

    # dual-GPU: jaxatari env on GPU0, torch PQN on GPU1
    CUDA_VISIBLE_DEVICES=0,1 python scripts/benchmarks/baselines/pqn_atari_jaxatari.py \\
      --env-id seaquest --seed 2 --env-gpu 0 --train-gpu 1 \\
      --jaxatari-root /path/to/JAXAtari

    # original ALE backend (bit-identical code path to the W&B script)
    python scripts/benchmarks/baselines/pqn_atari_jaxatari.py \\
      --backend ale --env-id seaquest --seed 2
"""
from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Limit to n threads/processes
n = "8"

os.environ["OMP_NUM_THREADS"] = n  # OpenMP
os.environ["MKL_NUM_THREADS"] = n  # Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = n  # OpenBLAS
os.environ["VECLIB_MAXIMUM_THREADS"] = n  # Apple Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = n  # NumExpr

# IMPORTANT: do NOT import torch before JAX when --backend jaxatari.
# Torch CUDA init first often breaks XLA/cuDNN ("dnn_support != nullptr") on
# shared-driver clusters. Torch is imported in main() after JAX warms up.

import gym
import numpy as np
import tyro


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 2
    """seed of the experiment (f163gauq used 2)"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "baselines_new"
    """the wandb's project name"""
    wandb_entity: str = "jaxatari"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "seaquest"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.001
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""

    # --- only additions vs the ALE W&B script ---
    backend: str = "jaxatari"
    """env backend: 'jaxatari' (default) or 'ale' (AtariVectorEnv, original script)"""
    jaxatari_root: Optional[str] = None
    """optional path to a JAXAtari checkout (adds <root>/src to sys.path)"""
    env_gpu: int = 0
    """CUDA index (within CUDA_VISIBLE_DEVICES) for jaxatari env stepping"""
    train_gpu: int = 1
    """CUDA index (within CUDA_VISIBLE_DEVICES) for the torch Q-network.
    Use a different GPU than --env-gpu so 128 JAX envs do not share with torch.
    Ignored if CUDA is unavailable (falls back to CPU)."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, info = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, terminations, truncations, infos = super().step(action)
        dones = np.logical_or(terminations, truncations)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


# Implement own RewardClipWrapper, since recording unclipped rewards in ALE-vecenv is not supported: https://github.com/Farama-Foundation/Arcade-Learning-Environment/issues/687
# Apply after RecordWrapper to record unclipped rewards, but learn on clipped rewards.
class ClipRewardsWrapper(gym.Wrapper):
    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        clipped_rewards = np.clip(rewards, -1, 1)
        return observations, clipped_rewards, dones, infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    import torch

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_q_network(env):
    """Build CleanRL PQN CNN after torch has been imported."""
    import torch.nn as nn

    class QNetwork(nn.Module):
        def __init__(self, env_):
            super().__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.LayerNorm([32, 20, 20]),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.LayerNorm([64, 9, 9]),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.LayerNorm([64, 7, 7]),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(3136, 512)),
                nn.LayerNorm(512),
                nn.ReLU(),
                layer_init(nn.Linear(512, env_.single_action_space.n)),
            )

        def forward(self, x):
            return self.network(x / 255.0)

    return QNetwork(env)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# ---------------------------------------------------------------------------
# JAXAtari vector env — same surface as ale_py.vector_env.AtariVectorEnv for
# this training loop (reset → (obs, info), step → 5-tuple, NCHW stacks).
# ---------------------------------------------------------------------------


def _maybe_add_jaxatari_path(jaxatari_root: Optional[str]) -> None:
    candidates: list[Path] = []
    if jaxatari_root:
        candidates.append(Path(jaxatari_root))
    env_root = os.environ.get("JAXATARI_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    # Common local checkouts used by this repo.
    candidates.extend(
        [
            Path("/home/paul/Documents/JAXAtari-seaquest-robustness"),
            Path(__file__).resolve().parents[3] / "third_party" / "symbolic_options" / "JAXAtari",
        ]
    )
    for root in candidates:
        src = root / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
            return


def _to_nchw_uint8(obs: np.ndarray) -> np.ndarray:
    """PixelObsWrapper stacks as (N, 4, H, W, C) with C=1 when grayscale → (N, 4, H, W)."""
    arr = np.asarray(obs)
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 4 or arr.shape[1] != 4:
        raise RuntimeError(
            f"Expected batched NCHW stack (N, 4, H, W); got shape {arr.shape}"
        )
    # JAX device_get arrays are often read-only; torch warns / UB without a copy.
    return np.array(arr, dtype=np.uint8, copy=True)


def _resolve_jax_device(env_gpu: int):
    import jax

    gpus = jax.devices("gpu")
    if not gpus:
        cpu = jax.devices("cpu")[0]
        print(f"[jaxatari] no GPU for JAX — using {cpu}")
        return cpu
    if env_gpu < 0 or env_gpu >= len(gpus):
        raise SystemExit(
            f"--env-gpu={env_gpu} out of range for {len(gpus)} visible JAX GPU(s): {gpus}"
        )
    dev = gpus[env_gpu]
    print(f"[jaxatari] env on JAX device {dev} (--env-gpu={env_gpu})")
    return dev


def _resolve_torch_device(*, cuda: bool, train_gpu: int):
    import torch

    if not (cuda and torch.cuda.is_available()):
        print("[torch] CUDA unavailable — training on CPU")
        return torch.device("cpu")
    n_dev = torch.cuda.device_count()
    if train_gpu < 0 or train_gpu >= n_dev:
        raise SystemExit(
            f"--train-gpu={train_gpu} out of range for {n_dev} visible torch GPU(s)"
        )
    dev = torch.device(f"cuda:{train_gpu}")
    print(
        f"[torch] train on {dev} ({torch.cuda.get_device_name(train_gpu)}) "
        f"(--train-gpu={train_gpu})"
    )
    return dev


def _warmup_jax_before_torch(args: Args) -> None:
    """Initialize JAX/XLA (+jaxatari import) before torch touches the driver.

    Avoids the common cluster failure:
      jax.errors.JaxRuntimeError: ... dnn_support != nullptr
    which shows up when torch CUDA init runs first, then jaxatari import
    evaluates ``jnp.array(...)`` defaults during class body execution.
    """
    if args.backend != "jaxatari":
        return
    _maybe_add_jaxatari_path(args.jaxatari_root)
    import jax
    import jax.numpy as jnp

    dev = _resolve_jax_device(args.env_gpu)
    try:
        with jax.default_device(dev):
            _ = jnp.asarray([0, 0])
            import jaxatari  # noqa: F401  — force the fragile import here

        print(f"[jax] warmup ok on {dev} (jaxatari import succeeded)")
    except Exception as exc:
        raise SystemExit(
            "JAX GPU/cuDNN failed before torch import.\n"
            f"  device={dev}\n"
            f"  error={type(exc).__name__}: {exc}\n"
            "Check: `python -c 'import jax; import jax.numpy as jnp; "
            "print(jax.devices()); print(jnp.array([0,0]))'` WITHOUT torch.\n"
            "If that fails, fix the JAX CUDA stack "
            '(e.g. `uv pip install -U \"jax[cuda12]\" nvidia-cudnn-cu12`).\n'
            "If that works alone but this script fails, CUDA_VISIBLE_DEVICES / "
            "multi-GPU pinning is wrong."
        ) from exc


class JaxAtariVectorEnv:
    """Minimal ALE AtariVectorEnv stand-in over jaxatari AtariWrapper+PixelObsWrapper."""

    def __init__(
        self,
        game: str,
        num_envs: int,
        *,
        episodic_life: bool = True,
        seed: int = 0,
        frame_skip: int = 4,
        frame_stack: int = 4,
        noop_max: int = 30,
        sticky_prob: float = 0.0,
        use_native_downscaling: bool = False,
        env_gpu: int = 0,
    ):
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        import jaxatari
        from gymnasium import spaces as gym_spaces
        from jaxatari.wrappers import AtariWrapper, PixelObsWrapper

        self._jax = jax
        self._jnp = jnp
        self._jrandom = jrandom
        self._jax_device = _resolve_jax_device(env_gpu)

        base = jaxatari.make(game.lower())
        wrapped = AtariWrapper(
            base,
            sticky_actions=float(sticky_prob),
            episodic_life=bool(episodic_life),
            first_fire=True,
            noop_max=int(noop_max),
            full_action_space=False,
        )
        # clip_reward=False so RecordEpisodeStatistics sees unclipped rewards;
        # ClipRewardsWrapper (outer) matches the ALE script's learning clip.
        self._env = PixelObsWrapper(
            wrapped,
            do_pixel_resize=True,
            pixel_resize_shape=(84, 84),
            grayscale=True,
            use_native_downscaling=bool(use_native_downscaling),
            smooth_image=False,
            frame_stack_size=int(frame_stack),
            frame_skip=int(frame_skip),
            max_pooling=True,
            clip_reward=False,
        )
        self.num_envs = int(num_envs)
        n_actions = int(self._env.action_space().n)
        self.single_action_space = gym_spaces.Discrete(n_actions)
        self.single_observation_space = gym_spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        # gym.Wrapper expects these attributes on the base env.
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

        with jax.default_device(self._jax_device):
            self._reset_fn = jax.jit(jax.vmap(self._env.reset))
            self._step_fn = jax.jit(jax.vmap(self._env.step))
            self._key = jrandom.PRNGKey(int(seed))
        self._state = None

    def reset(self, seed: Optional[int] = None, **kwargs):
        del kwargs
        with self._jax.default_device(self._jax_device):
            if seed is not None:
                self._key = self._jrandom.PRNGKey(int(seed))
            keys = self._jrandom.split(self._key, self.num_envs + 1)
            self._key = keys[0]
            obs, self._state = self._reset_fn(keys[1:])
        obs = _to_nchw_uint8(self._jax.device_get(obs))
        return obs, {}

    def step(self, actions):
        with self._jax.default_device(self._jax_device):
            actions = self._jnp.asarray(actions, dtype=self._jnp.int32)
            obs, self._state, reward, terminated, truncated, info = self._step_fn(
                self._state, actions
            )
        obs = _to_nchw_uint8(self._jax.device_get(obs))
        reward = np.asarray(self._jax.device_get(reward), dtype=np.float32)
        terminations = np.asarray(self._jax.device_get(terminated), dtype=bool)
        truncations = np.asarray(self._jax.device_get(truncated), dtype=bool)
        # info leaves are batched; RecordEpisodeStatistics overwrites r/l.
        out_info: dict = {}
        if isinstance(info, dict):
            for k, v in info.items():
                try:
                    out_info[k] = np.asarray(self._jax.device_get(v))
                except Exception:
                    continue
        return obs, reward, terminations, truncations, out_info

    def close(self):
        return None


def make_envs(args: Args):
    game_name = args.env_id.split("-")[0].lower() if "-" in args.env_id else args.env_id.lower()
    if args.backend == "ale":
        from ale_py.vector_env import AtariVectorEnv

        envs = AtariVectorEnv(
            game=game_name,
            num_envs=args.num_envs,
            episodic_life=True,
            num_threads=int(n),
            reward_clipping=False,  # we implement our own reward clipping wrapper below
        )
    elif args.backend == "jaxatari":
        _maybe_add_jaxatari_path(args.jaxatari_root)
        if args.env_gpu == args.train_gpu:
            print(
                f"WARNING: --env-gpu and --train-gpu both {args.env_gpu}; "
                "JAX env and torch will share one GPU (likely OOM / thrash)."
            )
        envs = JaxAtariVectorEnv(
            game=game_name,
            num_envs=args.num_envs,
            episodic_life=True,
            seed=args.seed,
            use_native_downscaling=False,
            env_gpu=args.env_gpu,
        )
    else:
        raise SystemExit(f"Unknown --backend {args.backend!r} (use 'jaxatari' or 'ale')")
    envs = RecordEpisodeStatistics(envs)
    envs = ClipRewardsWrapper(envs)
    return envs


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # JAX must claim the driver/cuDNN before torch when using jaxatari.
    _warmup_jax_before_torch(args)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    run_name = f"{args.env_id}__{args.exp_name}__{args.backend}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = _resolve_torch_device(cuda=args.cuda, train_gpu=args.train_gpu)

    # env setup
    print(
        f"Backend: {args.backend}  env_id={args.env_id}  num_envs={args.num_envs}  "
        f"env_gpu={args.env_gpu}  train_gpu={args.train_gpu}"
    )
    envs = make_envs(args)
    # assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = make_q_network(envs).to(device)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(envs.reset(), device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(
                args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step
            )

            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,)).to(device)
            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            explore = torch.rand((args.num_envs,)).to(device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(next_obs, device=device)
            next_done = torch.tensor(next_done, device=device)

            for idx, d in enumerate(next_done):
                if d:
                    # print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma
                        * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value)
                        * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        # Optimizing the Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                old_val = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                loss = F.mse_loss(b_returns[mb_inds], old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("losses/td_loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
