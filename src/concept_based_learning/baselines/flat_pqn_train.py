"""
Flat PQN training matched to JAXAtari ``pqn_agent.py`` (architecture + save layout).

Used by ``scripts/run_paper_eval.py`` for RQ1 flat baselines. Frame skip / stack /
rollout sizes come from config (unlike upstream hardcoding).
"""

from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)

from concept_based_learning.baselines.flat_pqn import QNetwork, normalize_flat_pqn_config
from concept_based_learning.pqn_visualizer import resolve_config_path
from concept_based_learning.paper_eval.env_metrics import append_training_metrics_csv
from concept_based_learning.training.rtpt_progress import (
    is_rtpt_enabled,
    rtpt_step_callback,
)


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def _save_params(params: Any, filename: str | os.PathLike) -> None:
    flat = flatten_dict(params, sep=",")
    save_file(flat, str(filename))


def make_train(config: dict[str, Any]):
    """Build a single-seed PQN train fn (vmapped externally over NUM_SEEDS)."""
    cfg = normalize_flat_pqn_config(config)
    total = int(cfg["TOTAL_TIMESTEPS"])
    num_steps = int(cfg["NUM_STEPS"])
    num_envs = int(cfg["NUM_ENVS"])
    num_minibatches = int(cfg["NUM_MINIBATCHES"])
    num_epochs = int(cfg["NUM_EPOCHS"])
    frame_skip = int(cfg.get("FRAME_SKIP", 4))
    frame_stack = int(cfg.get("FRAME_STACK_SIZE", 4))
    object_centric = bool(cfg.get("OBJECT_CENTRIC", True))

    cfg["NUM_UPDATES"] = total // num_steps // num_envs
    decay_total = int(cfg.get("TOTAL_TIMESTEPS_DECAY", total))
    cfg["NUM_UPDATES_DECAY"] = max(1, decay_total // num_steps // num_envs)

    if (num_steps * num_envs) % num_minibatches != 0:
        raise ValueError("NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS")

    env_name = str(cfg["ENV_NAME"]).lower()
    train_mods = cfg.get("GAME_MODS") or cfg.get("TRAIN_MODS")
    if train_mods is not None and not isinstance(train_mods, list):
        train_mods = [train_mods]
    if isinstance(train_mods, list) and len(train_mods) == 0:
        train_mods = None

    def apply_wrappers(base_env):
        wrapped = AtariWrapper(
            base_env,
            sticky_actions=0.0 if not cfg.get("STICKY_ACTIONS", False) else 0.25,
            episodic_life=bool(cfg.get("EPISODIC_LIFE", True)),
            first_fire=bool(cfg.get("FIRST_FIRE", True)),
            noop_max=int(cfg.get("NOOP_MAX", 30)),
            full_action_space=False,
        )
        if object_centric:
            wrapped = ObjectCentricWrapper(
                wrapped,
                frame_stack_size=frame_stack,
                frame_skip=frame_skip,
                clip_reward=bool(cfg.get("CLIP_REWARD", True)),
            )
            wrapped = NormalizeObservationWrapper(wrapped)
            wrapped = FlattenObservationWrapper(wrapped)
        else:
            resize_shape = cfg.get("PIXEL_RESIZE_SHAPE", [84, 84])
            if isinstance(resize_shape, list):
                resize_shape = tuple(resize_shape)
            wrapped = PixelObsWrapper(
                wrapped,
                do_pixel_resize=bool(cfg.get("PIXEL_RESIZE", True)),
                pixel_resize_shape=tuple(resize_shape),
                grayscale=bool(cfg.get("PIXEL_GRAYSCALE", True)),
                use_native_downscaling=bool(cfg.get("USE_NATIVE_DOWNSCALING", True)),
                smooth_image=bool(cfg.get("SMOOTH_IMAGE", False)),
                frame_stack_size=frame_stack,
                frame_skip=frame_skip,
                max_pooling=bool(cfg.get("PIXEL_MAX_POOLING", True)),
                clip_reward=bool(cfg.get("CLIP_REWARD", True)),
            )
        return LogWrapper(wrapped)

    env = apply_wrappers(jaxatari.make(env_name, mods=train_mods))

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy = jnp.argmax(q_vals, axis=-1)
        return jnp.where(
            jax.random.uniform(rng_e, greedy.shape) < eps,
            jax.random.randint(rng_a, greedy.shape, 0, q_vals.shape[-1]),
            greedy,
        )

    def train(rng):
        original_rng = rng[0]
        save_root = cfg.get("SAVE_PATH") or "./outputs/models/flat_pqn"
        env_for_csv = str(cfg.get("ENV_NAME", "seaquest")).lower()
        training_metrics_csv = (
            Path(save_root) / env_for_csv / "training_metrics.csv"
            if (
                bool(cfg.get("SAVE_TRAINING_METRICS_CSV", True))
                and not str(save_root).startswith("gs://")
            )
            else None
        )
        eps_scheduler = optax.linear_schedule(
            cfg["EPS_START"],
            cfg["EPS_FINISH"],
            float(cfg.get("EPS_DECAY", 0.2)) * cfg["NUM_UPDATES_DECAY"],
        )
        lr = cfg["LR"]
        if cfg.get("LR_LINEAR_DECAY", False):
            lr = optax.linear_schedule(
                init_value=cfg["LR"],
                end_value=1e-20,
                transition_steps=cfg["NUM_UPDATES_DECAY"] * num_minibatches * num_epochs,
            )

        network = QNetwork(
            action_dim=env.action_space().n,
            hidden_size=int(cfg.get("HIDDEN_SIZE", 128)),
            num_layers=int(cfg.get("NUM_LAYERS", 2)),
            norm_type=str(cfg.get("NORM_TYPE", "layer_norm")),
            norm_input=bool(cfg.get("NORM_INPUT", False)),
            object_centric=object_centric,
        )

        def create_agent(key):
            init_x = jnp.zeros((1, *env.observation_space().shape))
            variables = network.init(key, init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(float(cfg.get("MAX_GRAD_NORM", 0.8))),
                optax.radam(learning_rate=lr),
            )
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=variables["params"],
                batch_stats=variables["batch_stats"],
                tx=tx,
            )

        rng, init_rng = jax.random.split(rng)
        train_state = create_agent(init_rng)

        rng, reset_rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        expl_state = (obsv, env_state)

        def _update_step(runner_state, _unused):
            train_state, expl_state, rng = runner_state

            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    last_obs,
                    train=False,
                )
                eps = jnp.full(num_envs, eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, num_envs), q_vals, eps
                )
                new_obs, new_env_state, reward, term, trunc, info = jax.vmap(env.step)(
                    env_state, new_action
                )
                done = jnp.logical_or(term, trunc)
                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=float(cfg.get("REW_SCALE", 1.0)) * reward,
                    done=done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            rng, step_rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env, (*expl_state, step_rng), None, num_steps
            )
            expl_state = tuple(expl_state)
            train_state = train_state.replace(
                timesteps=train_state.timesteps + num_steps * num_envs
            )

            last_q = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + cfg["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + cfg["GAMMA"] * cfg["LAMBDA"] * delta
                )
                lambda_returns = (1 - transition.done) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + cfg["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):
                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        chosen = jnp.take_along_axis(
                            q_vals, jnp.expand_dims(minibatch.action, -1), -1
                        ).squeeze(-1)
                        loss = 0.5 * jnp.square(chosen - target).mean()
                        return loss, (updates, chosen)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, key):
                    x = x.reshape(-1, *x.shape[2:])
                    x = jax.random.permutation(key, x)
                    return x.reshape(num_minibatches, -1, *x.shape[1:])

                rng, shuffle_rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, shuffle_rng), transitions
                )
                targets = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, shuffle_rng), lambda_targets
                )
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )
                return (train_state, rng), (loss, qvals)

            rng, learn_rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, learn_rng), None, num_epochs
            )
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics = {
                "env_step": train_state.timesteps,
                "env/step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update({k: v.mean() for k, v in infos.items()})
            # LogWrapper latches on env_done + env_reward → full-game unclipped score.
            # Alias to the same charts/* keys as hierarchical skills so W&B overlays match.
            game_ret = metrics.get("returned_episode_returns")
            if game_ret is not None:
                metrics["charts/avg_game_return"] = game_ret
                metrics["env/game_return"] = game_ret
                metrics["returned_game_returns"] = game_ret
            game_len = metrics.get("returned_episode_lengths")
            if game_len is not None:
                metrics["charts/avg_game_length"] = game_len

            def _log_cb(m, _rng):
                log_dict = {
                    k: (float(v) if hasattr(v, "shape") else v)
                    for k, v in m.items()
                }
                # Flatten any remaining JAX scalars.
                clean = {}
                for k, v in log_dict.items():
                    try:
                        clean[k] = float(v)
                    except (TypeError, ValueError):
                        continue
                if training_metrics_csv is not None:
                    try:
                        append_training_metrics_csv(training_metrics_csv, clean)
                    except Exception:
                        pass
                if cfg.get("WANDB_MODE", "disabled") == "disabled":
                    return
                try:
                    wandb.log(clean, step=int(clean.get("env/step", clean.get("env_step", 0))))
                except Exception:
                    pass

            jax.debug.callback(_log_cb, metrics, original_rng)

            if is_rtpt_enabled(cfg):
                jax.debug.callback(rtpt_step_callback, train_state.n_updates)

            return (train_state, expl_state, rng), metrics

        runner_state = (train_state, expl_state, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, cfg["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def train_flat_pqn(config: dict[str, Any]) -> Path:
    """
    Train flat PQN and save under ``{SAVE_PATH}/{env}/``.

    Returns the directory containing the seed-0 checkpoint + config yaml.
    """
    cfg = normalize_flat_pqn_config(dict(config))
    cfg.setdefault("NUM_SEEDS", 1)
    cfg.setdefault("TOTAL_TIMESTEPS_DECAY", cfg["TOTAL_TIMESTEPS"])
    cfg.setdefault("NORM_TYPE", "layer_norm")
    cfg.setdefault("NORM_INPUT", False)
    cfg.setdefault("WANDB_MODE", "disabled")
    cfg.setdefault("ENTITY", "paul-seitz-tu-darmstadt")
    cfg.setdefault("PROJECT", "concept_based_learning")
    cfg.setdefault("ALG_NAME", "flat_pqn")

    alg_name = str(cfg.get("ALG_NAME", "flat_pqn"))
    env_name = str(cfg["ENV_NAME"]).lower()
    oc_tag = "oc" if cfg.get("OBJECT_CENTRIC", True) else "pixel"
    run_name = cfg.get("WANDB_RUN_NAME") or cfg.get("NAME") or f"{alg_name}_{env_name}_{oc_tag}"

    wandb.init(
        entity=cfg["ENTITY"],
        project=cfg["PROJECT"],
        tags=list(cfg.get("WANDB_TAGS") or []) + [alg_name.upper(), env_name.upper()],
        name=str(run_name),
        config=cfg,
        mode=cfg["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(int(cfg["SEED"]))
    rngs = jax.random.split(rng, int(cfg["NUM_SEEDS"]))
    t0 = time.time()
    train_vjit = jax.jit(jax.vmap(make_train(cfg)))
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Flat PQN finished in {time.time() - t0:.1f}s")

    save_root = resolve_config_path(cfg.get("SAVE_PATH", "./outputs/models/flat_pqn"))
    save_dir = Path(save_root) / env_name
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        cfg,
        str(save_dir / f"{alg_name}_{env_name}_{oc_tag}_seed{cfg['SEED']}_config.yaml"),
    )

    model_state = outs["runner_state"][0]
    ckpt_paths: list[Path] = []
    for i in range(int(cfg["NUM_SEEDS"])):
        params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
        batch_stats = jax.tree_util.tree_map(lambda x: x[i], model_state.batch_stats)
        ckpt = save_dir / f"{alg_name}_{env_name}_{oc_tag}_seed{cfg['SEED']}_vmap{i}.safetensors"
        _save_params(params, ckpt)
        _save_params(batch_stats, str(ckpt).replace(".safetensors", "_bs.safetensors"))
        ckpt_paths.append(ckpt)
        print(f"Flat PQN checkpoint: {ckpt}")

    wandb.finish()
    return save_dir
