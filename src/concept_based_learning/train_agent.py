"""
Hierarchical PQN-style trainer for Concept-Based Learning.

This file implements:
- A shared `QNetwork` MLP used for skills (System 1)
- A fixed, heuristic meta-policy (System 2) imported from a local file
- A `Transition` container that stores egocentric observations
- PQN-style update logic with Lambda-returns and SGD epochs
- Off-policy updates: all skills learn from the chosen action simultaneously
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from concept_based_learning.egocentric_transformation import get_ego_obs
from concept_based_learning.environments.datatypes import ConceptObservation
from concept_based_learning.environments.seaquest.seaquest_policy import get_heuristic_target, NO_TARGET_IDX
from src.concept_based_learning.environments.seaquest.seaquest_label import extract_all_skills_rewards, seaquest_action_mapper


class QNetwork(nn.Module):
    """
    Simple MLP Q-network used for Skills (System 1).
    Maps ego-centric features → action-values over ego actions.
    """

    action_dim: int
    hidden_size: int = 64
    num_layers: int = 3
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        # Simple LayerNorm / identity normalization
        if self.norm_type == "layer_norm":
            normalize = lambda v: nn.LayerNorm()(v)
        else:
            normalize = lambda v: v

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Transition:
    """
    Single transition in the hierarchical replay rollouts.

    Note that rewards are stored for **all** skills so we can do off-policy
    learning: every skill updates on every transition, regardless of whether
    it was active.
    """

    target_idx: chex.Array      # selected target object index (meta action)
    ego_obs: chex.Array         # ego-centric observation [dist, angle]
    next_ego_obs: chex.Array    # next-step ego-centric observation
    ego_action: chex.Array      # selected egocentric action index
    active_skill: chex.Array    # which skill (concept) is currently active
    q_val: chex.Array           # all skills' Q-values [num_skills, num_envs, ego_action_dim]
    rewards: chex.Array         # rewards for ALL skills [num_envs, num_skills]; num_skills=3 (no NOOP)
    done: chex.Array            # done flags from the env
    env_reward: chex.Array      # raw environment reward (game score increment)


class CustomTrainState(TrainState):
    """
    Extension of Flax's TrainState to keep track of:
    - `batch_stats` for normalization layers (if needed)
    - counters for timesteps, updates, and gradient steps
    """

    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env, test_env, renderer=None):
    """
    Factory that creates a `train(rng, params, batch_stats)` function.

    Required config keys:
    - TOTAL_TIMESTEPS, NUM_STEPS, NUM_ENVS, NUM_MINIBATCHES, NUM_EPOCHS
    - GAMMA, LAMBDA, LR, MAX_GRAD_NORM
    - NUM_SKILLS, MAX_OBJECTS, EGO_ACTION_DIM, HIDDEN_SIZE, NUM_LAYERS
    - EPS_START, EPS_FINISH, EPS_DECAY
    """

    # ---------------------------------------------------------------
    # Basic derived hyperparameters
    # ---------------------------------------------------------------
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0, (
        "NUM_MINIBATCHES must divide NUM_STEPS * NUM_ENVS"
    )

    NUM_SKILLS = config["NUM_SKILLS"]
    MAX_OBJECTS = config["MAX_OBJECTS"]
    EGO_ACTION_DIM = config["EGO_ACTION_DIM"]

    # Map semantic label (0=NOOP, 1=DESTROY, 2=COLLECT, 3=AVOID) → skill index (0, 1, 2).
    # We have no NOOP policy; NOOP (0) falls back to skill 0 so indexing is always in range.
    def label_to_skill_idx(label):
        return jnp.clip(label - 1, 0, NUM_SKILLS - 1)

    # Shared Q-network for skills (used by both train and eval_rollout)
    skill_network = QNetwork(
        action_dim=EGO_ACTION_DIM,
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
    )

    # Vectorized env helpers
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(jax.random.split(rng, n_envs))
    vmap_step = lambda env_state, action: jax.vmap(env.step)(env_state, action)

    def eps_greedy_exploration(rng, q_vals, eps, mask=None):
        """
        Epsilon-greedy with optional invalid action masking.

        - `q_vals`: [..., action_dim]
        - `mask`:  [..., action_dim] boolean; False = invalid.
        """
        rng_a, rng_e = jax.random.split(rng)
        valid_q_vals = q_vals if mask is None else jnp.where(mask, q_vals, -jnp.inf)
        greedy_actions = jnp.argmax(valid_q_vals, axis=-1)

        if mask is not None:
            random_logits = jnp.where(mask, 0.0, -jnp.inf)
            random_actions = jax.random.categorical(rng_a, random_logits)
        else:
            random_actions = jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            )

        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosen_actions

    def train(rng, params, batch_stats):
        """
        Main training loop entry point.

        The signature `(rng, params, batch_stats)` is kept for compatibility
        with prior training drivers, although `params` / `batch_stats` are
        re-initialized internally.
        """
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"], config["EPS_FINISH"], config["EPS_DECAY"]
        )
        lr = config["LR"]

        # ===============================================================
        # 1) AGENT INITIALIZATION
        # ===============================================================
        def create_agent(rng_key, network, obs_len):
            """
            Create one `CustomTrainState` for a Q-network with input length `obs_len`.
            """
            init_x = jnp.zeros(obs_len)
            network_variables = network.init(rng_key, init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables.get("batch_stats", {}),
                tx=tx,
            )

        # System 1: Skills (one Q-network per concept)
        # Input: flattened ego features; see `get_ego_obs` for exact length.
        rng_keys = jax.random.split(rng, NUM_SKILLS)

        # Determine ego feature length from egocentric transformation helper.
        # We call it once with dummy observation to infer dimension.
        # NOTE: vmap_reset(1) returns a batched observation (batch dim 1); unbatch for get_ego_obs.
        rng_probe, rng = jax.random.split(rng)
        probe_obs_batched, _ = vmap_reset(1)(rng_probe)
        probe_obs = jax.tree_util.tree_map(lambda x: x[0], probe_obs_batched)
        probe_ego = get_ego_obs(
            probe_obs.raw_observation, probe_obs.objects, jnp.array(0, dtype=jnp.int32)
        )
        ego_feat_dim = int(probe_ego.shape[-1])

        train_states = jax.vmap(create_agent, in_axes=(0, None, None))(
            rng_keys, skill_network, ego_feat_dim
        )

        # ===============================================================
        # 2) TRAINING LOOP BODY
        # ===============================================================
        def _update_step(runner_state, _):
            """
            One update iteration:
            - Roll out `NUM_STEPS` trajectories
            - Build Transition objects
            - Compute PQN-style lambda returns and perform SGD epochs
            """
            train_states, expl_state, rng = runner_state

            # -----------------------------
            # 2.1 SAMPLE PHASE
            # -----------------------------
            def _step_env(carry, _t):
                """
                Step the vectorized environment once.

                Uses the **fixed** heuristic meta-policy to choose a target
                object for each environment, then runs all skills on the
                corresponding egocentric state.
                """
                last_obs, env_state, rng = carry
                rng, rng_skill = jax.random.split(rng, 2)

                # SYSTEM 2: Heuristic meta-policy over objects
                target_obj_idxs = jax.vmap(get_heuristic_target)(
                    last_obs.raw_observation, last_obs.objects
                )
                # When no target (NO_TARGET_IDX), use index 0 for safe indexing; we force NOOP action below
                safe_target_idxs = jnp.where(
                    target_obj_idxs == NO_TARGET_IDX, 0, target_obj_idxs
                )

                # Map object → active skill: semantic label (0–3) → skill index (0–2); no NOOP policy
                raw_labels = jax.vmap(
                    lambda labels, idx: labels[-1, idx]
                )(last_obs.objects.semantic_label, safe_target_idxs)
                active_skills = label_to_skill_idx(raw_labels)

                # Egocentric transform: from player/target positions → ego features (per env)
                ego_obses = jax.vmap(get_ego_obs)(
                    last_obs.raw_observation, last_obs.objects, safe_target_idxs
                )

                # Run ALL skills; exploration is per-skill and per-env
                def compute_skill_actions(train_state):
                    q_vals = skill_network.apply(
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                        ego_obses,
                        train=False,
                    )
                    _rngs = jax.random.split(rng_skill, config["NUM_ENVS"])
                    eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                    ego_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
                    return ego_action, q_vals

                all_ego_actions, all_skill_q_vals = jax.vmap(compute_skill_actions)(train_states)

                # Select ego actions from the active skill (one per env)
                chosen_ego_actions = all_ego_actions[
                    active_skills, jnp.arange(config["NUM_ENVS"])
                ]
                # When no target on screen, do NOOP (ego action 0 → global action 0)
                chosen_ego_actions = jnp.where(
                    target_obj_idxs == NO_TARGET_IDX, 0, chosen_ego_actions
                )

                global_actions = jax.vmap(seaquest_action_mapper)(
                    chosen_ego_actions, last_obs.raw_observation
                )
                global_actions = jnp.where(
                    target_obj_idxs == NO_TARGET_IDX, 0, global_actions
                )

                new_obs, new_env_state, raw_rewards, new_done, info = vmap_step(
                    env_state, global_actions
                )

                # Compute life loss per environment (positive when a life is lost)
                prev_lives = env_state.atari_state.env_state.lives
                next_lives = new_env_state.atari_state.env_state.lives
                life_lost = prev_lives - next_lives

                # We need to extract the target object specifically for the current environments
                def get_target_obj(objects, idx):
                    # objects: per-env, leaves (stack_size, max_objects) or (max_objects,); idx = target object index
                    return jax.tree_util.tree_map(
                        lambda x: jnp.take(x, idx, axis=-1), objects
                    )

                prev_targets = jax.vmap(get_target_obj)(last_obs.objects, safe_target_idxs)
                next_targets = jax.vmap(get_target_obj)(new_obs.objects, safe_target_idxs)

                # Map across the batch of environments
                skill_rewards = jax.vmap(extract_all_skills_rewards)(
                    last_obs.raw_observation.player,  # prev_player
                    new_obs.raw_observation.player,   # next_player
                    prev_targets,                     # prev_target 
                    next_targets                      # next_target
                )

                # Next-step ego obs for TD targets (per env)
                next_ego_obses = jax.vmap(get_ego_obs)(
                    new_obs.raw_observation, new_obs.objects, safe_target_idxs
                )

                transition = Transition(
                    target_idx=target_obj_idxs,
                    ego_obs=ego_obses,
                    next_ego_obs=next_ego_obses,
                    ego_action=chosen_ego_actions,
                    active_skill=active_skills,
                    q_val=all_skill_q_vals,
                    rewards=skill_rewards,
                    done=new_done,
                    env_reward=raw_rewards,
                )

                return (new_obs, new_env_state, rng), (transition, info)

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            # ---------------------------------------------------------
            # 2.2 UPDATE PHASE (PQN & Off-Policy)
            # ---------------------------------------------------------
            def _update_skill(train_state, skill_idx, rng):
                """
                PQN-style Q(λ) update for a single skill.

                - Builds λ-returns using the skill-specific rewards.
                - Reuses data for multiple SGD epochs over shuffled minibatches.
                - OFF-POLICY: the action used is the one chosen by the **active** skill,
                  but all skills share that action for their updates.
                """
                # Update total timesteps counter
                train_state = train_state.replace(
                    timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
                )

                # Q-values at final next_ego_obs
                last_q = skill_network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    transitions.next_ego_obs[-1],
                    train=False,
                )
                last_q_max = jnp.max(last_q, axis=-1)

                def _get_target(lambda_returns_and_next_q, transition):
                    lambda_returns, next_q = lambda_returns_and_next_q

                    # Skill-specific reward
                    step_reward = transition.rewards[..., skill_idx]

                    target_bootstrap = (
                        step_reward + config["GAMMA"] * (1 - transition.done) * next_q
                    )
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (
                        (1 - transition.done) * lambda_returns
                        + transition.done * step_reward
                    )

                    # Next Q from stored q_val for this skill
                    next_q_val = jnp.max(transition.q_val[skill_idx], axis=-1)
                    return (lambda_returns, next_q_val), lambda_returns

                last_q_max = last_q_max * (1 - transitions.done[-1])
                # Last step reward for this skill only: (NUM_ENVS,) to match last_q_max.
                # Use dynamic_slice so the vmapped skill_idx is applied correctly.
                last_step_reward = jnp.squeeze(
                    jax.lax.dynamic_slice_in_dim(
                        transitions.rewards[-1], skill_idx, 1, axis=-1
                    ),
                    axis=-1,
                )
                lambda_returns = last_step_reward + config["GAMMA"] * last_q_max

                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q_max),
                    jax.tree.map(lambda x: x[:-1], transitions),
                    reverse=True,
                )
                lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

                # SGD Epochs
                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_and_target):
                        train_state, rng = carry
                        minibatch, target = minibatch_and_target

                        def _loss_fn(params):
                            q_vals, updates = skill_network.apply(
                                {"params": params, "batch_stats": train_state.batch_stats},
                                minibatch.ego_obs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                            # OFF-POLICY: all skills use the same `ego_action` indices.
                            chosen_action_qvals = jnp.take_along_axis(
                                q_vals,
                                jnp.expand_dims(minibatch.ego_action, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)

                            # Do not learn from no-target (NOOP) steps; they use dummy object indexing
                            valid = (minibatch.target_idx != NO_TARGET_IDX)
                            sq_err = jnp.square(chosen_action_qvals - target)
                            loss = (sq_err * valid).sum() / (valid.sum() + 1e-8)
                            return loss, (updates, chosen_action_qvals)

                        (loss, (updates, qvals)), grads = jax.value_and_grad(
                            _loss_fn, has_aux=True
                        )(train_state.params)

                        train_state = train_state.apply_gradients(grads=grads)
                        train_state = train_state.replace(
                            grad_steps=train_state.grad_steps + 1,
                            batch_stats=updates["batch_stats"],
                        )
                        return (train_state, rng), (loss, qvals)

                    def preprocess_transition(x, rng):
                        """
                        Turn [num_steps, num_envs, ...] into:
                        - Flattened batch of size num_steps*num_envs
                        - Shuffled
                        - Reshaped into [NUM_MINIBATCHES, batch_size_per_minibatch, ...]
                        """
                        x = x.reshape(-1, *x.shape[2:])
                        x = jax.random.permutation(rng, x)
                        x = x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                        return x

                    rng, _rng = jax.random.split(rng)
                    minibatches = jax.tree.map(
                        lambda x: preprocess_transition(x, _rng), transitions
                    )
                    targets_batch = preprocess_transition(lambda_targets, _rng)

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), (loss, qvals) = jax.lax.scan(
                        _learn_phase, (train_state, rng), (minibatches, targets_batch)
                    )

                    return (train_state, rng), (loss, qvals)

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
                )

                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                metrics = {
                    "td_loss": loss[-1].mean(),
                    "qvals": qvals[-1].mean(),
                }
                return metrics, train_state

            # Vectorized update across all skills
            rngs = jax.random.split(rng, NUM_SKILLS)
            skill_metrics, train_states = jax.vmap(_update_skill, in_axes=(0, 0, 0))(
                train_states, jnp.arange(NUM_SKILLS), rngs
            )

            # Flatten per-skill metrics: "skill_i/key"
            metrics = {
                f"skill_{i}/{k}": v[i] for k, v in skill_metrics.items() for i in range(NUM_SKILLS)
            }

            # Mean skill rewards over rollout [num_steps, num_envs]
            mean_skill_rewards = jnp.mean(transitions.rewards, axis=(0, 1))  # [NUM_SKILLS]
            for i in range(NUM_SKILLS):
                metrics[f"skill_{i}/reward"] = mean_skill_rewards[i]

            # Environment reward (mean raw game reward over rollout)
            metrics["env/reward_mean"] = jnp.mean(transitions.env_reward)

            # If env info dict provides episode statistics (e.g. returns), log their means as well
            try:
                info_metrics = {f"info/{k}": jnp.mean(v) for k, v in infos.items()}
                metrics.update(info_metrics)
            except AttributeError:
                # infos is not a mapping; ignore
                pass

            # Global counters
            metrics["update_steps"] = train_states.n_updates[0]
            metrics["env/step"] = train_states.timesteps[0]

            # Stream metrics to wandb via JAX host callback (one log per update)
            if config.get("WANDB_MODE", "disabled") != "disabled":
                def _wandb_log_cb(m):
                    try:
                        import wandb
                    except ImportError:
                        return
                    if wandb.run is None:
                        return
                    # Convert JAX values to Python scalars
                    log_dict = {}
                    for key, value in m.items():
                        value_np = np.asarray(value)
                        if value_np.size == 1:
                            log_dict[key] = float(value_np.ravel()[0])
                        else:
                            log_dict[key] = float(value_np.mean())
                    step = int(log_dict.get("update_steps", wandb.run.step or 0))
                    wandb.log(log_dict, step=step)

                jax.debug.callback(_wandb_log_cb, metrics)

            runner_state = (train_states, tuple(expl_state), rng)
            return runner_state, metrics

        # Initial reset
        rng, _rng = jax.random.split(rng)
        expl_state = vmap_reset(config["NUM_ENVS"])(_rng)
        runner_state = (train_states, expl_state, _rng)

        # Unroll NUM_UPDATES update steps
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    def eval_rollout(runner_state, step_number, max_frames=None):
        """
        Run one evaluation episode on test_env with the current policy (greedy),
        collect (states, active_agents, combined_qs, dones) for video recording.
        runner_state may be batched by NUM_SEEDS; we use the first seed.
        """
        max_frames = max_frames or config.get("VIDEO_MAX_FRAMES", 2000)
        train_states, _expl_state, rng = runner_state

        # Use first seed if batched
        def _first(x):
            if hasattr(x, "ndim") and x.ndim > 0:
                return x[0]
            return x

        train_states = jax.tree_util.tree_map(_first, train_states)
        rng = _first(rng)
        rng = jax.random.fold_in(rng, 0)

        rng, rng_reset = jax.random.split(rng)
        obs, state = test_env.reset(rng_reset)

        states_list = []
        active_list = []
        combined_qs_list = []
        dones_list = []
        target_obj_idxs_list = []
        obj_x_list = []
        obj_y_list = []
        obj_w_list = []
        obj_h_list = []
        obj_active_list = []

        for _ in range(max_frames):
            target_obj_idxs = get_heuristic_target(obs.raw_observation, obs.objects)
            target_obj_idxs = int(jnp.ravel(target_obj_idxs)[0])

            if target_obj_idxs == NO_TARGET_IDX:
                # No object on screen → NOOP
                global_action = 0
                active_skills = 0  # display as first row in sidebar
                combined_q = np.array([0.0, 0.0, 0.0])  # dummy Q for sidebar
            else:
                raw_label = obs.objects.semantic_label[-1, target_obj_idxs]
                active_skills = label_to_skill_idx(raw_label)
                ego_obs = get_ego_obs(
                    obs.raw_observation, obs.objects, target_obj_idxs
                )
                ego_obs_batch = ego_obs[np.newaxis, ...]

                # One Q-network per skill; vmap over skills
                q_vals = jax.vmap(
                    lambda ts: skill_network.apply(
                        {"params": ts.params, "batch_stats": ts.batch_stats},
                        ego_obs_batch,
                        train=False,
                    )
                )(train_states)
                q_vals = jnp.squeeze(q_vals, axis=1)
                combined_q = np.array(jnp.max(q_vals, axis=-1))
                chosen_ego = jnp.argmax(q_vals[active_skills])
                global_action = seaquest_action_mapper(
                    chosen_ego, obs.raw_observation
                )
                active_skills = int(active_skills)

            obs, state, _reward, done, _info = test_env.step(state, global_action)

            states_list.append(state)
            active_list.append(active_skills)
            combined_qs_list.append(np.array(combined_q))
            dones_list.append(float(done))
            target_obj_idxs_list.append(target_obj_idxs)
            # Object bboxes after step so outline matches the rendered frame
            obj_x_list.append(np.array(obs.objects.x[-1]))
            obj_y_list.append(np.array(obs.objects.y[-1]))
            obj_w_list.append(np.array(obs.objects.width[-1]))
            obj_h_list.append(np.array(obs.objects.height[-1]))
            obj_active_list.append(np.array(obs.objects.active[-1]))

            if done:
                break

        def stack_trees(*trees):
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)

        stacked_states = stack_trees(*states_list)
        active_agents = jnp.array(active_list, dtype=jnp.int32)
        combined_qs = jnp.stack(combined_qs_list)
        dones = jnp.array(dones_list, dtype=jnp.float32)
        target_obj_idxs_stacked = np.array(target_obj_idxs_list, dtype=np.int32)
        objects_bbox = (
            np.stack(obj_x_list),
            np.stack(obj_y_list),
            np.stack(obj_w_list),
            np.stack(obj_h_list),
            np.stack(obj_active_list),
        )
        return stacked_states, active_agents, combined_qs, dones, target_obj_idxs_stacked, objects_bbox

    return train, eval_rollout