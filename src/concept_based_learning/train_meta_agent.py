"""
Hierarchical Meta-Policy (System 2) trainer for Concept-Based Learning.

This file implements:
- A training loop specifically for a learned Meta-Policy.
- Execution of actions through a FROZEN System 1 (Skills) network.
- Optimization based entirely on global environment rewards and global done signals.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import chex
import optax
from flax.training.train_state import TrainState

from concept_based_learning.egocentric_transformation import (
    EGO_OBS_CHANNELS_PER_FRAME,
    get_ego_obs,
)
from concept_based_learning.ego_action_space import fire_ego_action_indices
from concept_based_learning.train_agent import QNetwork as SkillQNetwork
#from concept_based_learning.basic_mlp_network import MetaQNetwork, MetaTransition
from concept_based_learning.affordance_pointer_network import MetaQNetwork, MetaTransition


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_meta_train(
    config,
    env,
    test_env,
    renderer=None,
    *,
    frozen_skill_params,
    no_target_idx,
    action_mapper,
    extract_life_lost,
    meta_mask_hook,
    forced_target_hook,
):
    """
    Factory that creates a `train(rng)` function for the Meta-Policy.
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
    step_offset_updates = int(config.get("STEP_OFFSET", 0))
    step_offset_timesteps = int(
        config.get(
            "STEP_OFFSET_TIMESTEPS",
            step_offset_updates * config["NUM_STEPS"] * config["NUM_ENVS"],
        )
    )
    restrict_fire_for_non_destroy = bool(config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False))
    _custom_fire = config.get("FIRE_EGO_ACTION_INDICES")
    if _custom_fire is not None:
        FIRE_EGO_ACTION_INDICES = jnp.array(list(_custom_fire), dtype=jnp.int32)
    else:
        FIRE_EGO_ACTION_INDICES = fire_ego_action_indices(EGO_ACTION_DIM)
    # Keep NOOP (ego action 0) always available, even if custom FIRE indices are misconfigured.
    NON_NOOP_FIRE_EGO_ACTION_INDICES = FIRE_EGO_ACTION_INDICES[FIRE_EGO_ACTION_INDICES != 0]
    META_USE_NOOP_TARGET = bool(config.get("META_USE_NOOP_TARGET", False))
    META_NOOP_TARGET_IDX = int(config.get("META_NOOP_TARGET_IDX", 0))
    meta_noop_target_enabled = META_USE_NOOP_TARGET and (0 <= META_NOOP_TARGET_IDX < MAX_OBJECTS)
    # Switching-cost / action-switch penalty for System 2.
    # Penalizes changing the selected target from the previous step to reduce
    # target jittering while still allowing deliberate switches.
    META_SWITCH_PENALTY = float(config.get("META_SWITCH_PENALTY", -0.5))
    META_DEATH_PENALTY = float(config.get("META_DEATH_PENALTY", -5.0))

    def _apply_meta_target_mask_overrides(mask):
        if meta_noop_target_enabled:
            mask = mask.at[:, META_NOOP_TARGET_IDX].set(True)
        return mask

    # ---------------------------------------------------------------
    # Instantiate Networks
    # ---------------------------------------------------------------
    # 1. System 1 (Skills) - FROZEN
    skill_network = SkillQNetwork(
        action_dim=EGO_ACTION_DIM,
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
    )

    # 2. System 2 (Meta-Policy) - LEARNED
    meta_network = MetaQNetwork(
        action_dim=MAX_OBJECTS,
        hidden_size=config.get("META_HIDDEN_SIZE", 256),
        num_skills=NUM_SKILLS 
    )

    # Vectorized env helpers
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(jax.random.split(rng, n_envs))
    vmap_step = lambda env_state, action: jax.vmap(env.step)(env_state, action)

    # Fail fast: checkpoint vs env ego-feature length (stack depth × channels/frame).
    # No silent slicing/padding — that misaligns frames and hides bugs.
    _p0 = jax.tree_util.tree_map(lambda x: x[0], frozen_skill_params)
    ckpt_ego_in = int(_p0["Dense_0"]["kernel"].shape[0])
    ckpt_ego_out = int(_p0["Dense_3"]["kernel"].shape[1])
    if int(EGO_ACTION_DIM) != ckpt_ego_out:
        raise ValueError(
            f"Frozen skill checkpoint expects EGO_ACTION_DIM={ckpt_ego_out} "
            f"(Dense_3 out), but config has EGO_ACTION_DIM={int(EGO_ACTION_DIM)}. "
            "Set `experiment.alg.EGO_ACTION_DIM` to match the skill checkpoint, "
            "or retrain skills with the desired action dim."
        )
    rng_ck = jax.random.PRNGKey(0)
    _probe_b, _ = vmap_reset(1)(rng_ck)
    _probe = jax.tree_util.tree_map(lambda x: x[0], _probe_b)
    _probe_ego = get_ego_obs(
        _probe.raw_observation, _probe.objects, jnp.array(0, dtype=jnp.int32)
    )
    env_ego_dim = int(_probe_ego.shape[-1])
    if env_ego_dim != ckpt_ego_in:
        _need_stack = (
            ckpt_ego_in // int(EGO_OBS_CHANNELS_PER_FRAME)
            if ckpt_ego_in % int(EGO_OBS_CHANNELS_PER_FRAME) == 0
            else "?"
        )
        raise ValueError(
            f"Frozen skill checkpoint expects MLP input size {ckpt_ego_in}, but this "
            f"environment's egocentric observation length is {env_ego_dim}. "
            f"Ego length is {int(EGO_OBS_CHANNELS_PER_FRAME)} × JAXAtari `AtariWrapper` frame stack. Set "
            f"`experiment.alg.FRAME_STACK_SIZE` so {int(EGO_OBS_CHANNELS_PER_FRAME)} × FRAME_STACK_SIZE matches the checkpoint "
            f"(e.g. checkpoint {ckpt_ego_in} → FRAME_STACK_SIZE={_need_stack}), and use the "
            "same value when training System 1."
        )

    def _merge_reset_where_done(new_tree, reset_tree, done):
        def _replace(new_leaf, reset_leaf):
            expand = (config["NUM_ENVS"],) + (1,) * (new_leaf.ndim - 1)
            return jnp.where(jnp.reshape(done, expand), reset_leaf, new_leaf)
        return jax.tree_util.tree_map(_replace, new_tree, reset_tree)

    def eps_greedy_exploration(rng, q_vals, eps, mask):
        """Epsilon-greedy with invalid action masking."""
        rng_a, rng_e = jax.random.split(rng)

        # Mask out inactive targets with -inf
        valid_q_vals = jnp.where(mask, q_vals, -jnp.inf)
        greedy_actions = jnp.argmax(valid_q_vals, axis=-1)

        # Random action logits
        # Use a large finite negative number instead of `-inf` to avoid NaNs
        # from `jax.random.categorical` when *all* actions are masked.
        random_logits = jnp.where(mask, 0.0, -1e9)
        random_actions = jax.random.categorical(rng_a, random_logits)

        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosen_actions

    def label_to_skill_idx(label):
        return jnp.clip(label - 1, 0, NUM_SKILLS - 1)

    def _params_for_skill(skill_idx):
        """
        Select one skill's parameter subtree from stacked frozen skill params.
        frozen_skill_params leaves are expected to be shaped [NUM_SKILLS, ...].
        """
        return jax.tree_util.tree_map(lambda x: x[skill_idx], frozen_skill_params)

    # ===============================================================
    # TRAINING LOOP
    # ===============================================================
    def train(rng, init_meta_params=None):
        eps_decay_cfg = config["EPS_DECAY"]
        # Backward-compatible interpretation:
        # - If EPS_DECAY <= 1.0, treat as fraction of NUM_UPDATES.
        # - Otherwise, treat as explicit number of updates.
        eps_decay_steps = (
            int(max(1, round(config["NUM_UPDATES"] * eps_decay_cfg)))
            if eps_decay_cfg <= 1.0
            else int(max(1, round(eps_decay_cfg)))
        )
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"], config["EPS_FINISH"], eps_decay_steps
        )
        lr = config.get("META_LR", config["LR"])

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )

        rng, _rng = jax.random.split(rng)

        # Probe observation to initialize the MetaNetwork
        probe_obs_batched, _ = vmap_reset(1)(_rng)
        probe_obs = jax.tree_util.tree_map(lambda x: x[0], probe_obs_batched)

        meta_vars = meta_network.init(
            rng, probe_obs.raw_observation, probe_obs.objects, train=False
        )
        meta_params = (
            meta_vars["params"] if init_meta_params is None else init_meta_params
        )
        train_state = CustomTrainState.create(
            apply_fn=meta_network.apply,
            params=meta_params,
            batch_stats={},
            tx=tx,
        )

        def _update_step(runner_state, _):
            train_state, expl_state, running_returns, rng = runner_state

            # -----------------------------
            # 2.1 SAMPLE PHASE
            # -----------------------------
            def _step_env(carry, _t):
                (
                    last_obs,
                    env_state,
                    running_returns,
                    rng,
                    completed_returns_sum,
                    num_completed,
                    last_targets,
                ) = carry

                rng, rng_rest = jax.random.split(rng)
                rng_reset, rng_step = jax.random.split(rng_rest)
                rng_policy, _ = jax.random.split(rng_step)

                # --- 1. System 2 (Meta-Policy) Forward Pass ---
                # vmap the meta-network over the batch of raw observations
                meta_q_vals = jax.vmap(
                    lambda obs, objs: meta_network.apply(
                        {"params": train_state.params}, obs, objs, train=False
                    )
                )(last_obs.raw_observation, last_obs.objects)

                # --- 2. Masking & Target Selection ---
                # Build active masks per environment using the same single-env
                # indexing convention as the heuristic policies (`obj.active[-1]`).
                active_masks = jax.vmap(lambda obj: obj.active[-1] > 0.5)(last_obs.objects)
                active_masks = meta_mask_hook(
                    last_obs.raw_observation, active_masks, config
                )
                active_masks = _apply_meta_target_mask_overrides(active_masks)

                _rngs = jax.random.split(rng_policy, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                target_obj_idxs = jax.vmap(eps_greedy_exploration)(
                    _rngs, meta_q_vals, eps, active_masks
                )
                target_obj_idxs = forced_target_hook(
                    last_obs.raw_observation, target_obj_idxs, config, rng_policy
                )

                # --- 3. System 1 (Skills) Forward Pass (FROZEN) ---
                def get_global_action(obs, target_idx, active_mask):
                    any_active = jnp.any(active_mask)
                    # Safe index to avoid out-of-bounds in PyTree lookups when screen is empty
                    safe_target = jnp.where(any_active, jnp.clip(target_idx, 0, MAX_OBJECTS - 1), 0)
                    is_noop_target = (
                        meta_noop_target_enabled
                        and (target_idx == jnp.asarray(META_NOOP_TARGET_IDX, dtype=target_idx.dtype))
                    )
                    force_noop = jnp.logical_or(jnp.logical_not(any_active), is_noop_target)

                    ego_obs = get_ego_obs(obs.raw_observation, obs.objects, safe_target)
                    raw_label = obs.objects.semantic_label[-1, safe_target]
                    skill_idx = label_to_skill_idx(raw_label)
                    skill_params = _params_for_skill(skill_idx)

                    # Forward pass through frozen skills
                    # (Note: batch_stats are ignored for the frozen network if not used)
                    skill_qs = skill_network.apply(
                        {"params": skill_params},
                        ego_obs,
                        train=False
                    )

                    # Optional: FIRE masking for non-destroy skills
                    if restrict_fire_for_non_destroy:
                        skill_qs = jax.lax.cond(
                            skill_idx == 0,
                            lambda qs: qs,
                            lambda qs: qs.at[NON_NOOP_FIRE_EGO_ACTION_INDICES].set(-jnp.inf),
                            skill_qs,
                        )

                    ego_action = jnp.argmax(skill_qs)
                    global_action = action_mapper(ego_action, obs.raw_observation, safe_target)

                    # Optional selectable NOOP target and empty-screen fallback.
                    return jnp.where(force_noop, 0, global_action)

                # Map action retrieval over the batch
                global_actions = jax.vmap(get_global_action)(last_obs, target_obj_idxs, active_masks)

                # --- 4. Step Environment ---
                new_obs, new_env_state, raw_rewards, new_done, info = vmap_step(
                    env_state, global_actions
                )

                # Detect whether this terminal transition was caused by a life loss.
                # Must be computed before auto-reset merges in fresh env states.
                life_lost = extract_life_lost(env_state, new_env_state) > 0
                done_due_to_life_lost = jnp.logical_and(new_done, life_lost)

                # Auto-reset
                reset_obs, reset_state = vmap_reset(config["NUM_ENVS"])(rng_reset)
                new_obs = _merge_reset_where_done(new_obs, reset_obs, new_done)
                new_env_state = _merge_reset_where_done(new_env_state, reset_state, new_done)

                # Episode return tracking
                episode_return = running_returns + raw_rewards
                done_float = new_done.astype(jnp.float32)
                completed_returns_sum += jnp.sum(episode_return * done_float)
                num_completed += jnp.sum(done_float)
                running_returns = episode_return * (1.0 - done_float)

                next_active_masks = jax.vmap(
                    lambda obj: obj.active[-1] > 0.5
                )(new_obs.objects)
                next_active_masks = meta_mask_hook(
                    new_obs.raw_observation, next_active_masks, config
                )
                next_active_masks = _apply_meta_target_mask_overrides(next_active_masks)

                # --- Switching penalty (System 2 only) ---
                # Apply a small negative reward if the meta-policy changes its target.
                # We only penalize switches when `last_targets` was a valid target.
                changed_target = target_obj_idxs != last_targets
                was_valid_target = jnp.logical_and(
                    last_targets >= 0, last_targets < MAX_OBJECTS
                )
                apply_penalty = jnp.logical_and(changed_target, was_valid_target)

                # ====================================================
                # INTERSECT HERE: Modify rewards for the network only
                # ====================================================
                # Important: keep `episode_return` (used for performance tracking)
                # based on `raw_rewards` above; only the System 2 TD target uses
                # `meta_rewards` below.
                #
                # Option A (default): Standard (no modification)
                # processed_rewards = raw_rewards
                # Option B: Reward Scaling (uncomment and scale penalty accordingly)
                # processed_rewards = raw_rewards / 100.0
                # Option C: Reward Clipping (uncomment and scale penalty accordingly)
                processed_rewards = jnp.sign(raw_rewards)
                # (Alternatively, `jnp.sign(raw_rewards)` is common.)
                # ====================================================
                # Strong global death penalty for System 2: makes suicidal
                # target choices mathematically unattractive.
                processed_rewards = processed_rewards + jnp.where(
                    done_due_to_life_lost,
                    META_DEATH_PENALTY,
                    0.0,
                ).astype(processed_rewards.dtype)

                # Apply the penalty to the PROCESSED reward
                meta_rewards = processed_rewards + jnp.where(
                    apply_penalty, META_SWITCH_PENALTY, 0.0
                ).astype(processed_rewards.dtype)

                transition = MetaTransition(
                    obs=last_obs.raw_observation,
                    objects=last_obs.objects,
                    action=target_obj_idxs,
                    reward=meta_rewards,
                    done=new_done,
                    next_obs=new_obs.raw_observation,
                    next_objects=new_obs.objects,
                    q_val=meta_q_vals,
                    active_mask=active_masks,
                    next_active_mask=next_active_masks,
                )

                return (
                    new_obs,
                    new_env_state,
                    running_returns,
                    rng,
                    completed_returns_sum,
                    num_completed,
                    target_obj_idxs,
                ), (transition, info)

            rng, _rng = jax.random.split(rng)
            init_last_targets = jnp.full(
                (config["NUM_ENVS"],), no_target_idx, dtype=jnp.int32
            )
            carry_final, (transitions, infos) = jax.lax.scan(
                _step_env,
                (
                    *expl_state,
                    running_returns,
                    _rng,
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    init_last_targets,
                ),
                None,
                config["NUM_STEPS"],
            )
            (
                last_obs,
                env_state,
                running_returns,
                rng,
                completed_returns_sum,
                num_completed,
                _last_targets,
            ) = carry_final
            expl_state = (last_obs, env_state)

            # ---------------------------------------------------------
            # 2.2 UPDATE PHASE (PQN)
            # ---------------------------------------------------------
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # Bootstrapping from next_obs:
            # `next_obs` is a raw-observation PyTree (not a plain array), so select
            # the final rollout step by slicing each leaf, then vmap over envs.
            last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], transitions.next_obs)
            last_next_objs = jax.tree_util.tree_map(lambda x: x[-1], transitions.next_objects)
            last_q = jax.vmap(
                lambda obs, objs: meta_network.apply(
                    {"params": train_state.params}, obs, objs, train=False
                )
            )(last_next_obs, last_next_objs)

            # Mask inactive targets before taking max
            next_active_mask = transitions.next_active_mask[-1]
            next_active_any = jnp.any(next_active_mask, axis=-1)
            masked_last_q = jnp.max(
                jnp.where(next_active_mask, last_q, -jnp.inf), axis=-1
            )
            # If there are no active targets, force bootstrap to a finite value
            # to prevent `inf * 0 -> NaN` during the TD loss computation.
            last_q = jnp.where(next_active_any, masked_last_q, 0.0)
            last_q = last_q * (1 - transitions.done[-1])

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q

                target_bootstrap = transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                delta = lambda_returns - next_q
                lambda_returns = target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                lambda_returns = (1 - transition.done) * lambda_returns + transition.done * transition.reward

                # Mask next_q for the previous step in the reverse scan
                next_active_mask = transition.active_mask
                next_active_any = jnp.any(next_active_mask, axis=-1)
                masked_next_q = jnp.max(
                    jnp.where(next_active_mask, transition.q_val, -jnp.inf),
                    axis=-1,
                )
                next_q = jnp.where(next_active_any, masked_next_q, 0.0)

                return (lambda_returns, next_q), lambda_returns

            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
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
                        q_vals = jax.vmap(
                            lambda obs, objs: meta_network.apply(
                                {"params": params}, obs, objs, train=True
                            )
                        )(minibatch.obs, minibatch.objects)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals, jnp.expand_dims(minibatch.action, axis=-1), axis=-1
                        ).squeeze(axis=-1)

                        # Only compute loss if there was at least one active object to choose from
                        valid = jnp.any(minibatch.active_mask, axis=-1)
                        sq_err = jnp.square(chosen_action_qvals - target)
                        loss = (sq_err * valid).sum() / (valid.sum() + 1e-8)

                        return loss, chosen_action_qvals

                    (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(-1, *x.shape[2:])
                    x = jax.random.permutation(rng, x)
                    x = x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree.map(lambda x: preprocess_transition(x, _rng), transitions)
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

            mean_episodic_return = completed_returns_sum / jnp.maximum(1.0, num_completed)
            reward_nonzero = (transitions.reward != 0).astype(jnp.float32)
            valid_target_mask = jnp.any(transitions.active_mask, axis=-1).astype(jnp.float32)
            metrics = {
                "meta_policy/td_loss": loss[-1].mean(),
                "meta_policy/qvals": qvals[-1].mean(),
                "env/reward_mean": jnp.mean(transitions.reward),
                "env/reward_nonzero_frac": jnp.mean(reward_nonzero),
                "env/reward_max": jnp.max(transitions.reward),
                "env/episodic_return": mean_episodic_return,
                "returned_episode_returns": mean_episodic_return,
                "meta_policy/valid_target_frac": jnp.mean(valid_target_mask),
                "update_steps": train_state.n_updates,
                "global/update_steps": step_offset_updates + train_state.n_updates,
                "env/step": step_offset_timesteps + train_state.timesteps,
            }

            # Wandb logging hook (if enabled)
            if config.get("WANDB_MODE", "disabled") != "disabled":
                def _wandb_log_cb(m):
                    try:
                        import wandb
                        if wandb.run is not None:
                            log_dict = {k: float(np.asarray(v).mean()) for k, v in m.items()}
                            step = config.get("STEP_OFFSET", 0) + int(log_dict.get("update_steps", 0))
                            wandb.log(log_dict, step=step)
                    except ImportError:
                        pass
                jax.debug.callback(_wandb_log_cb, metrics)

            runner_state = (train_state, tuple(expl_state), running_returns, rng)
            return runner_state, metrics

        # Initialization
        env_seed_base = config.get("ENV_SEED_BASE", config["SEED"])
        init_reset_keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(env_seed_base), i))(
            jnp.arange(config["NUM_ENVS"], dtype=jnp.uint32)
        )
        expl_state = jax.vmap(env.reset)(init_reset_keys)
        rng, _rng = jax.random.split(rng)
        running_returns = jnp.zeros(config["NUM_ENVS"], dtype=jnp.float32)
        runner_state = (train_state, expl_state, running_returns, _rng)

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    def eval_rollout(runner_state, eval_rng, step_number, max_frames=None):
        """
        Evaluation rollout using the Learned Meta-Policy and Frozen Skills.
        """
        max_frames = max_frames or config.get("VIDEO_MAX_FRAMES", 2000)
        train_states, _expl_state, _running_returns, _train_rng = runner_state

        def _first(x):
            return x[0] if hasattr(x, "ndim") and x.ndim > 0 else x

        train_state = jax.tree_util.tree_map(_first, train_states)
        rng = jax.random.fold_in(eval_rng, 0)
        rng, rng_reset = jax.random.split(rng)
        obs, state = test_env.reset(rng_reset)

        states_list, active_list, combined_qs_list = [], [], []
        dones_list, target_obj_idxs_list = [], []
        obj_x_list, obj_y_list, obj_w_list, obj_h_list, obj_active_list = [], [], [], [], []

        for _ in range(max_frames):
            # 1. System 2 (Meta-Policy) Selection
            meta_q_vals = meta_network.apply(
                {"params": train_state.params}, obs.raw_observation, obs.objects, train=False
            )
            active_mask = obs.objects.active[-1] > 0.5
            active_mask = meta_mask_hook(
                jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), obs.raw_observation),
                jnp.expand_dims(active_mask, 0),
                config,
            )[0]
            if meta_noop_target_enabled:
                active_mask = active_mask.at[META_NOOP_TARGET_IDX].set(True)

            any_active = jnp.any(active_mask)

            # Mask inactive targets and pick greedily
            valid_qs = jnp.where(active_mask, meta_q_vals, -jnp.inf)
            target_idx = jnp.argmax(valid_qs)
            target_idx = forced_target_hook(
                jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), obs.raw_observation),
                jnp.expand_dims(target_idx, 0),
                config,
            )[0]
            target_idx = jnp.where(any_active, target_idx, no_target_idx)
            target_obj_idxs = int(target_idx)

            if target_obj_idxs == no_target_idx or (
                meta_noop_target_enabled and target_obj_idxs == META_NOOP_TARGET_IDX
            ):
                global_action = 0
                active_skills = 0
                combined_q = np.array([0.0, 0.0, 0.0])
            else:
                # 2. System 1 (Skills) Execution
                raw_label = obs.objects.semantic_label[-1, target_obj_idxs]
                active_skills = label_to_skill_idx(raw_label)
                ego_obs = get_ego_obs(obs.raw_observation, obs.objects, target_obj_idxs)
                skill_params = _params_for_skill(active_skills)

                skill_qs = skill_network.apply(
                    {"params": skill_params}, jnp.expand_dims(ego_obs, 0), train=False
                )[0]

                if restrict_fire_for_non_destroy and int(active_skills) != 0:
                    skill_qs = skill_qs.at[NON_NOOP_FIRE_EGO_ACTION_INDICES].set(-jnp.inf)

                chosen_ego = jnp.argmax(skill_qs)
                global_action = action_mapper(chosen_ego, obs.raw_observation, target_obj_idxs)
                # Keep video sidebar shape consistent with skill trainer:
                # one value per skill (here we only evaluate active skill, so we
                # place its max-Q into that slot and keep others at 0).
                combined_q = np.zeros((NUM_SKILLS,), dtype=np.float32)
                combined_q[int(active_skills)] = float(jnp.max(skill_qs))

            obs, state, _reward, done, _info = test_env.step(state, global_action)

            states_list.append(state)
            active_list.append(int(active_skills))
            combined_qs_list.append(combined_q)
            dones_list.append(float(done))
            target_obj_idxs_list.append(target_obj_idxs)
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
            np.stack(obj_x_list), np.stack(obj_y_list),
            np.stack(obj_w_list), np.stack(obj_h_list), np.stack(obj_active_list),
        )
        return stacked_states, active_agents, combined_qs, dones, target_obj_idxs_stacked, objects_bbox

    return train, eval_rollout
