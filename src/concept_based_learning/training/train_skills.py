"""
Hierarchical PQN-style trainer for Concept-Based Learning (skill training path).

Implements:
- Meta-policy-driven (learned or heuristic) target selection (System 2)
- PQN-style update logic with Lambda-returns and SGD epochs
- Off-policy updates: all skills learn from the chosen action simultaneously
"""

import jax
import jax.numpy as jnp
import numpy as np

import optax

from concept_based_learning.observations.egocentric import get_ego_obs
from concept_based_learning.observations.action_space import fire_ego_action_indices
from concept_based_learning.networks.skill_network import QNetwork, CustomTrainState, Transition
from concept_based_learning.networks.meta_network import MetaQNetwork
from concept_based_learning.environments.datatypes import ConceptObservation
from concept_based_learning.training.schedules import (
    DEFAULT_SKILL_EPS_DECAY,
    DEFAULT_SKILL_EPS_FINISH,
    DEFAULT_SKILL_EPS_START,
    eps_decay_transition_steps,
)
from concept_based_learning.training.trajectory_cuts import (
    DEFAULT_SKILL_TARGET_CUT,
    normalize_skill_target_cut,
    skill_target_switch_cut,
    skill_target_switch_cut_env,
    skill_target_switch_cut_env_padded,
)
from concept_based_learning.training.rtpt_progress import is_rtpt_enabled, rtpt_step_callback


def make_train(
    config,
    env,
    test_env,
    renderer=None,
    *,
    frozen_meta_params=None,
    get_heuristic_target,
    no_target_idx,
    action_mapper,
    inverse_action_mapper,
    extract_all_skills_rewards,
    extract_life_lost,
    fixed_target_idx_from_config,
    meta_mask_hook,
    forced_target_hook,
):
    """
    Factory that creates a `train(rng, params, batch_stats)` function.

    Required config keys:
    - TOTAL_TIMESTEPS, NUM_STEPS, NUM_ENVS, NUM_MINIBATCHES, NUM_EPOCHS
    - GAMMA, LAMBDA, LR, MAX_GRAD_NORM
    - NUM_SKILLS, MAX_OBJECTS, EGO_ACTION_DIM, HIDDEN_SIZE, NUM_LAYERS
    - EPS_START, EPS_FINISH, EPS_DECAY (skill head; defaults in schedules.py)
    """

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
    relabel_all_objects = bool(config.get("RELABEL_ALL_OBJECTS", False))
    relabel_k_objects = config.get("RELABEL_K_OBJECTS", None)
    skill_bootstrap = str(config.get("SKILL_BOOTSTRAP", "q_lambda")).lower()
    relabel_k_objects = (
        None
        if relabel_k_objects is None
        else int(max(1, min(MAX_OBJECTS, int(relabel_k_objects))))
    )
    skill_gamma = float(config.get("SKILL_GAMMA", config["GAMMA"]))
    skill_reward_lambda = float(config.get("SKILL_REWARD_LAMBDA", 0.0))
    skill_target_cut = normalize_skill_target_cut(
        config.get("SKILL_TARGET_CUT", DEFAULT_SKILL_TARGET_CUT)
    )

    _raw_skill_names = config.get("SKILL_NAMES")
    if _raw_skill_names is None or len(_raw_skill_names) != NUM_SKILLS:
        skill_names = [f"skill_{i}" for i in range(NUM_SKILLS)]
    else:
        skill_names = [str(n) for n in _raw_skill_names]

    log_freeway_car_hits = str(config.get("ENV_NAME", "")).lower() == "freeway"
    freeway_env_steps_per_update = config["NUM_STEPS"] * config["NUM_ENVS"]

    def label_to_skill_idx(label):
        return jnp.clip(label - 1, 0, NUM_SKILLS - 1)

    restrict_fire_for_non_destroy = bool(
        config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False)
    )

    _custom_fire = config.get("FIRE_EGO_ACTION_INDICES")
    if _custom_fire is not None:
        FIRE_EGO_ACTION_INDICES = jnp.array(list(_custom_fire), dtype=jnp.int32)
    else:
        FIRE_EGO_ACTION_INDICES = fire_ego_action_indices(EGO_ACTION_DIM)
    NON_NOOP_FIRE_EGO_ACTION_INDICES = FIRE_EGO_ACTION_INDICES[FIRE_EGO_ACTION_INDICES != 0]

    fixed_target_idx = fixed_target_idx_from_config(config)
    symbolic_collect_enabled = bool(config.get("SYMBOLIC_COLLECT_FORCE_UP", False))
    symbolic_collect_skill_idx = int(config.get("COLLECT_SKILL_INDEX", 1))
    symbolic_collect_up_action = int(config.get("SYMBOLIC_COLLECT_UP_ACTION", 1))

    skill_network = QNetwork(
        action_dim=EGO_ACTION_DIM,
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
    )
    use_learned_meta_policy = frozen_meta_params is not None
    if use_learned_meta_policy:
        meta_network = MetaQNetwork(
            action_dim=MAX_OBJECTS,
            hidden_size=config.get("META_HIDDEN_SIZE", 256),
            num_skills=NUM_SKILLS,
        )
        meta_eps_start = float(
            config.get("SKILL_FINETUNE_META_EPS_START", config.get("EPS_START", 0.2))
        )
        meta_eps_finish = float(
            config.get("SKILL_FINETUNE_META_EPS_FINISH", config.get("EPS_FINISH", 0.01))
        )
        meta_eps_decay_cfg = float(
            config.get("SKILL_FINETUNE_META_EPS_DECAY", config.get("EPS_DECAY", 0.8))
        )

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(jax.random.split(rng, n_envs))
    vmap_step = lambda env_state, action: jax.vmap(env.step)(env_state, action)

    def _merge_reset_where_done(new_tree, reset_tree, done):
        def _replace(new_leaf, reset_leaf):
            expand = (config["NUM_ENVS"],) + (1,) * (new_leaf.ndim - 1)
            return jnp.where(jnp.reshape(done, expand), reset_leaf, new_leaf)
        return jax.tree_util.tree_map(_replace, new_tree, reset_tree)

    def eps_greedy_exploration(rng, q_vals, eps, mask=None):
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
        eps_scheduler = optax.linear_schedule(
            float(config.get("EPS_START", DEFAULT_SKILL_EPS_START)),
            float(config.get("EPS_FINISH", DEFAULT_SKILL_EPS_FINISH)),
            eps_decay_transition_steps(
                config.get("EPS_DECAY", DEFAULT_SKILL_EPS_DECAY),
                config["NUM_UPDATES"],
            ),
        )
        if use_learned_meta_policy:
            meta_eps_scheduler = optax.linear_schedule(
                meta_eps_start,
                meta_eps_finish,
                eps_decay_transition_steps(meta_eps_decay_cfg, config["NUM_UPDATES"]),
            )
        lr = config["LR"]

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )

        def create_agent(rng_key, network, obs_len, init_params=None):
            if init_params is None:
                init_x = jnp.zeros(obs_len)
                network_variables = network.init(rng_key, init_x, train=False)
                params_local = network_variables["params"]
                batch_stats_local = network_variables.get("batch_stats", {})
            else:
                params_local = init_params
                batch_stats_local = {}

            return CustomTrainState.create(
                apply_fn=network.apply,
                params=params_local,
                batch_stats=batch_stats_local,
                tx=tx,
            )

        rng_keys = jax.random.split(rng, NUM_SKILLS)

        rng_probe, rng = jax.random.split(rng)
        probe_obs_batched, _ = vmap_reset(1)(rng_probe)
        probe_obs = jax.tree_util.tree_map(lambda x: x[0], probe_obs_batched)
        probe_ego = get_ego_obs(
            probe_obs.raw_observation, probe_obs.objects, jnp.array(0, dtype=jnp.int32)
        )
        ego_feat_dim = int(probe_ego.shape[-1])

        if params is None:
            train_states = jax.vmap(create_agent, in_axes=(0, None, None, None))(
                rng_keys, skill_network, ego_feat_dim, None
            )
        else:
            train_states = jax.vmap(create_agent, in_axes=(0, None, None, 0))(
                rng_keys, skill_network, ego_feat_dim, params
            )

        def _update_step(runner_state, _):
            train_states, expl_state, running_returns, rng = runner_state

            def _step_env(carry, _t):
                (
                    last_obs,
                    env_state,
                    running_returns,
                    rng,
                    completed_returns_sum,
                    num_completed,
                    last_targets,
                    car_hits_sum,
                ) = carry

                rng, rng_rest = jax.random.split(rng)
                rng_reset, rng_step = jax.random.split(rng_rest)
                rng_skill, rng_policy = jax.random.split(rng_step)

                policy_rngs = jax.random.split(rng_policy, config["NUM_ENVS"])
                if use_learned_meta_policy:
                    active_masks = jax.vmap(lambda obj: obj.active[-1] > 0.5)(
                        last_obs.objects
                    )
                    active_masks = meta_mask_hook(
                        last_obs.raw_observation,
                        active_masks,
                        config,
                        objects=last_obs.objects,
                    )

                    any_active = jnp.any(active_masks, axis=-1)
                    fallback_mask = jnp.zeros_like(active_masks).at[:, 0].set(True)
                    safe_masks = jnp.where(any_active[:, None], active_masks, fallback_mask)
                    meta_q_vals = jax.vmap(
                        lambda obs, objs: meta_network.apply(
                            {"params": frozen_meta_params}, obs, objs, train=False
                        )
                    )(last_obs.raw_observation, last_obs.objects)
                    meta_eps = jnp.full(
                        config["NUM_ENVS"],
                        meta_eps_scheduler(train_states.n_updates[0]),
                    )
                    target_obj_idxs = jax.vmap(eps_greedy_exploration)(
                        policy_rngs, meta_q_vals, meta_eps, safe_masks
                    )
                    target_obj_idxs = forced_target_hook(
                        last_obs.raw_observation,
                        target_obj_idxs,
                        config,
                        rng_policy,
                    )
                    target_obj_idxs = jnp.where(
                        any_active,
                        target_obj_idxs.astype(jnp.int32),
                        jnp.full_like(target_obj_idxs, no_target_idx, dtype=jnp.int32),
                    )
                else:
                    if fixed_target_idx is not None:
                        target_obj_idxs = jnp.full(
                            (config["NUM_ENVS"],),
                            fixed_target_idx,
                            dtype=jnp.int32,
                        )
                    else:
                        target_obj_idxs = jax.vmap(get_heuristic_target)(
                            last_obs.raw_observation,
                            last_obs.objects,
                            last_targets,
                            policy_rngs,
                        )

                safe_target_idxs = jnp.where(
                    target_obj_idxs == no_target_idx, 0, target_obj_idxs
                )

                raw_labels = jax.vmap(
                    lambda labels, idx: labels[-1, idx]
                )(last_obs.objects.semantic_label, safe_target_idxs)
                active_skills = label_to_skill_idx(raw_labels)

                ego_obses = jax.vmap(get_ego_obs)(
                    last_obs.raw_observation, last_obs.objects, safe_target_idxs
                )

                def compute_skill_actions(train_state, skill_idx):
                    q_vals = skill_network.apply(
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                        ego_obses,
                        train=False,
                    )
                    _rngs = jax.random.split(rng_skill, config["NUM_ENVS"])
                    eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))

                    invalid_ego_actions = config.get("INVALID_EGO_ACTIONS", [])

                    def build_mask_for_skill(idx):
                        base_row = jnp.ones((EGO_ACTION_DIM,), dtype=bool)

                        if invalid_ego_actions:
                            invalid_arr = jnp.array(invalid_ego_actions, dtype=jnp.int32)
                            base_row = base_row.at[invalid_arr].set(False)
                        base_row = base_row.at[0].set(True)

                        return jax.lax.cond(
                            idx == 0,
                            lambda r: r,
                            lambda r: r.at[NON_NOOP_FIRE_EGO_ACTION_INDICES].set(False) if restrict_fire_for_non_destroy else r,
                            base_row,
                        )

                    mask_row = build_mask_for_skill(skill_idx)
                    mask = jnp.broadcast_to(
                        mask_row, (config["NUM_ENVS"], EGO_ACTION_DIM)
                    )

                    ego_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps, mask)
                    return ego_action, q_vals

                all_ego_actions, all_skill_q_vals = jax.vmap(
                    compute_skill_actions
                )(train_states, jnp.arange(NUM_SKILLS, dtype=jnp.int32))

                chosen_ego_actions = all_ego_actions[
                    active_skills, jnp.arange(config["NUM_ENVS"])
                ]
                if symbolic_collect_enabled:
                    symbolic_collect_ego_action = jax.vmap(inverse_action_mapper)(
                        jnp.full(
                            (config["NUM_ENVS"],),
                            symbolic_collect_up_action,
                            dtype=jnp.int32,
                        ),
                        last_obs.raw_observation,
                        safe_target_idxs,
                        last_obs.objects,
                    )
                    chosen_ego_actions = jnp.where(
                        active_skills == symbolic_collect_skill_idx,
                        symbolic_collect_ego_action,
                        chosen_ego_actions,
                    )
                no_target = target_obj_idxs == no_target_idx
                chosen_ego_actions = jnp.where(no_target, 0, chosen_ego_actions)

                global_actions = jax.vmap(action_mapper)(
                    chosen_ego_actions,
                    last_obs.raw_observation,
                    target_obj_idxs,
                    last_obs.objects,
                )
                if symbolic_collect_enabled:
                    global_actions = jnp.where(
                        active_skills == symbolic_collect_skill_idx,
                        jnp.full_like(global_actions, symbolic_collect_up_action),
                        global_actions,
                    )
                global_actions = jnp.where(no_target, 0, global_actions)

                new_obs, new_env_state, raw_rewards, new_done, info = vmap_step(
                    env_state, global_actions
                )

                reset_obs, reset_state = vmap_reset(config["NUM_ENVS"])(rng_reset)
                new_obs = _merge_reset_where_done(new_obs, reset_obs, new_done)
                new_env_state = _merge_reset_where_done(new_env_state, reset_state, new_done)

                life_lost = extract_life_lost(env_state, new_env_state)
                car_hits_sum = car_hits_sum + jnp.sum(life_lost > 0)

                def get_target_obj(objects, idx):
                    return jax.tree_util.tree_map(
                        lambda x: jnp.take(x, idx, axis=-1), objects
                    )

                prev_targets = jax.vmap(get_target_obj)(last_obs.objects, safe_target_idxs)
                next_targets = jax.vmap(get_target_obj)(new_obs.objects, safe_target_idxs)

                rewards_and_masks = jax.vmap(
                    lambda pr, nr, pt, nt, ll, inf: extract_all_skills_rewards(
                        pr, nr, pt, nt, ll, inf, config
                    )
                )(
                    last_obs.raw_observation,
                    new_obs.raw_observation,
                    prev_targets,
                    next_targets,
                    life_lost,
                    info,
                )
                skill_rewards, skill_done = rewards_and_masks
                base_env_reward = jnp.sign(raw_rewards)
                skill_rewards = (
                    skill_reward_lambda * base_env_reward[:, jnp.newaxis]
                    + (1.0 - skill_reward_lambda) * skill_rewards
                )

                next_ego_obses = jax.vmap(get_ego_obs)(
                    new_obs.raw_observation, new_obs.objects, safe_target_idxs
                )

                if relabel_all_objects:
                    obj_ids = jnp.arange(MAX_OBJECTS, dtype=jnp.int32)
                    ego_obs_all_full = jax.vmap(
                        lambda raw, objs: jax.vmap(
                            lambda obj_idx: get_ego_obs(raw, objs, obj_idx)
                        )(obj_ids)
                    )(last_obs.raw_observation, last_obs.objects)
                    next_ego_obs_all_full = jax.vmap(
                        lambda raw, objs: jax.vmap(
                            lambda obj_idx: get_ego_obs(raw, objs, obj_idx)
                        )(obj_ids)
                    )(new_obs.raw_observation, new_obs.objects)
                    ego_action_all_full = jax.vmap(
                        lambda game_action, raw, objs: jax.vmap(
                            lambda obj_idx: inverse_action_mapper(
                                game_action, raw, obj_idx, objs
                            )
                        )(obj_ids)
                    )(global_actions, last_obs.raw_observation, last_obs.objects)

                    def _all_object_rewards(pr, nr, prev_objs, next_objs, ll, inf):
                        def _single_object(obj_idx):
                            prev_obj = jax.tree_util.tree_map(
                                lambda x: jnp.take(x, obj_idx, axis=-1), prev_objs
                            )
                            next_obj = jax.tree_util.tree_map(
                                lambda x: jnp.take(x, obj_idx, axis=-1), next_objs
                            )
                            return extract_all_skills_rewards(
                                pr, nr, prev_obj, next_obj, ll, inf, config
                            )

                        return jax.vmap(_single_object)(obj_ids)

                    rewards_all_full, skill_done_all_full = jax.vmap(
                        _all_object_rewards
                    )(
                        last_obs.raw_observation,
                        new_obs.raw_observation,
                        last_obs.objects,
                        new_obs.objects,
                        life_lost,
                        info,
                    )
                    rewards_all_full = (
                        skill_reward_lambda * base_env_reward[:, jnp.newaxis, jnp.newaxis]
                        + (1.0 - skill_reward_lambda) * rewards_all_full
                    )
                    relabel_mask_full = jax.vmap(lambda obj: obj.active[-1] > 0.5)(
                        last_obs.objects
                    )
                    relabel_mask_full = meta_mask_hook(
                        last_obs.raw_observation,
                        relabel_mask_full,
                        config,
                        objects=last_obs.objects,
                    )

                    if relabel_k_objects is not None:
                        rng, rng_relabel = jax.random.split(rng)
                        relabel_scores = jnp.where(
                            relabel_mask_full,
                            jax.random.uniform(
                                rng_relabel, shape=relabel_mask_full.shape
                            ),
                            -1.0,
                        )
                        gather_idx = jnp.flip(
                            jnp.argsort(relabel_scores, axis=-1), axis=-1
                        )[:, :relabel_k_objects]

                        def _gather_objects(arr):
                            idx = gather_idx
                            while idx.ndim < arr.ndim:
                                idx = idx[..., None]
                            return jnp.take_along_axis(arr, idx, axis=1)

                        ego_obs_all = _gather_objects(ego_obs_all_full)
                        next_ego_obs_all = _gather_objects(next_ego_obs_all_full)
                        ego_action_all = _gather_objects(ego_action_all_full)
                        rewards_all = _gather_objects(rewards_all_full)
                        skill_done_all = _gather_objects(skill_done_all_full)
                        relabel_mask = _gather_objects(relabel_mask_full)
                    else:
                        ego_obs_all = ego_obs_all_full
                        next_ego_obs_all = next_ego_obs_all_full
                        ego_action_all = ego_action_all_full
                        rewards_all = rewards_all_full
                        skill_done_all = skill_done_all_full
                        relabel_mask = relabel_mask_full
                else:
                    base_valid = target_obj_idxs != no_target_idx
                    ego_obs_all = ego_obses[:, jnp.newaxis, :]
                    next_ego_obs_all = next_ego_obses[:, jnp.newaxis, :]
                    ego_action_all = chosen_ego_actions[:, jnp.newaxis]
                    rewards_all = skill_rewards[:, jnp.newaxis, :]
                    skill_done_all = skill_done[:, jnp.newaxis, :]
                    relabel_mask = base_valid[:, jnp.newaxis]

                episode_return = running_returns + raw_rewards
                done_float = new_done.astype(jnp.float32)

                completed_returns_sum = completed_returns_sum + jnp.sum(
                    episode_return * done_float
                )
                num_completed = num_completed + jnp.sum(done_float)

                running_returns = episode_return * (1.0 - done_float)

                transition = Transition(
                    target_idx=target_obj_idxs,
                    ego_obs=ego_obses,
                    next_ego_obs=next_ego_obses,
                    ego_action=chosen_ego_actions,
                    active_skill=active_skills,
                    q_val=all_skill_q_vals,
                    rewards=skill_rewards,
                    skill_done=skill_done,
                    ego_obs_all=ego_obs_all,
                    next_ego_obs_all=next_ego_obs_all,
                    ego_action_all=ego_action_all,
                    rewards_all=rewards_all,
                    skill_done_all=skill_done_all,
                    relabel_mask=relabel_mask,
                    done=new_done,
                    env_reward=raw_rewards,
                )

                return (
                    new_obs,
                    new_env_state,
                    running_returns,
                    rng,
                    completed_returns_sum,
                    num_completed,
                    target_obj_idxs,
                    car_hits_sum,
                ), (transition, info)

            rng, _rng = jax.random.split(rng)
            init_completed_returns = jnp.array(0.0, dtype=jnp.float32)
            init_num_completed = jnp.array(0.0, dtype=jnp.float32)
            init_car_hits_sum = jnp.array(0.0, dtype=jnp.float32)
            init_last_targets = jnp.full(
                (config["NUM_ENVS"],),
                no_target_idx,
                dtype=jnp.int32,
            )
            carry_final, (transitions, infos) = jax.lax.scan(
                _step_env,
                (
                    *expl_state,
                    running_returns,
                    _rng,
                    init_completed_returns,
                    init_num_completed,
                    init_last_targets,
                    init_car_hits_sum,
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
                car_hits_sum,
            ) = carry_final
            expl_state = (last_obs, env_state)

            def _update_skill(train_state, skill_idx, rng):
                train_state = train_state.replace(
                    timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                use_one_step_td = skill_bootstrap == "one_step"

                if relabel_all_objects:
                    # Lambda returns over [NUM_STEPS, NUM_ENVS, MAX_OBJECTS].
                    # Bootstrap Q-values come from the online network applied to
                    # next_ego_obs_all (all steps, all objects at once).
                    next_q_all = skill_network.apply(
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                        transitions.next_ego_obs_all,
                        train=False,
                    )
                    # [NUM_STEPS, NUM_ENVS, MAX_OBJECTS]
                    next_q_max_all = jnp.max(next_q_all, axis=-1)

                    # Seed the backward scan from the last step.
                    last_step_done_all = jnp.logical_or(
                        transitions.skill_done_all[-1, :, :, skill_idx],
                        transitions.done[-1, :, jnp.newaxis],
                    ).astype(jnp.float32)
                    last_next_q = next_q_max_all[-1] * (1.0 - last_step_done_all)
                    last_reward_all = transitions.rewards_all[-1, :, :, skill_idx]
                    last_lambda = last_reward_all + skill_gamma * last_next_q

                    target_switch_cut = skill_target_switch_cut(
                        transitions.target_idx, MAX_OBJECTS, skill_target_cut
                    )

                    def _get_target_all(carry, xs):
                        lambda_ret, next_q = carry
                        # xs: per-step slices, shapes [NUM_ENVS, MAX_OBJECTS]
                        step_reward, step_done_skill, step_done_ep, step_target_cut, q_next_step = xs

                        step_done = jnp.logical_or(
                            step_done_skill, step_done_ep[:, jnp.newaxis]
                        )
                        step_done = jnp.logical_or(
                            step_done, step_target_cut.astype(jnp.bool_)
                        ).astype(jnp.float32)

                        target_bootstrap = step_reward + skill_gamma * (1.0 - step_done) * next_q
                        delta = lambda_ret - next_q
                        lambda_ret = target_bootstrap + skill_gamma * config["LAMBDA"] * delta
                        lambda_ret = (1.0 - step_done) * lambda_ret + step_done * step_reward

                        return (lambda_ret, q_next_step), lambda_ret

                    xs_all = (
                        transitions.rewards_all[:-1, :, :, skill_idx],
                        transitions.skill_done_all[:-1, :, :, skill_idx],
                        transitions.done[:-1],
                        target_switch_cut,
                        next_q_max_all[:-1],
                    )
                    _, targets_all = jax.lax.scan(
                        _get_target_all,
                        (last_lambda, next_q_max_all[-1]),
                        xs_all,
                        reverse=True,
                    )
                    # [NUM_STEPS, NUM_ENVS, MAX_OBJECTS]
                    train_targets = jnp.concatenate(
                        (targets_all, last_lambda[jnp.newaxis]), axis=0
                    )
                elif use_one_step_td:
                    next_q_single = skill_network.apply(
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                        transitions.next_ego_obs,
                        train=False,
                    )
                    next_q_max_single = jnp.max(next_q_single, axis=-1)
                    target_switch_cut_padded = skill_target_switch_cut_env_padded(
                        transitions.target_idx, skill_target_cut
                    )
                    step_done_single = jnp.logical_or(
                        transitions.skill_done[..., skill_idx],
                        transitions.done,
                    )
                    step_done_single = jnp.logical_or(
                        step_done_single,
                        target_switch_cut_padded.astype(jnp.bool_),
                    ).astype(jnp.float32)
                    train_targets = (
                        transitions.rewards[..., skill_idx]
                        + skill_gamma * (1.0 - step_done_single) * next_q_max_single
                    )[..., jnp.newaxis]
                else:
                    # Lambda returns over [NUM_STEPS, NUM_ENVS] using fresh online Qs.
                    next_q_single_all = skill_network.apply(
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                        transitions.next_ego_obs,
                        train=False,
                    )
                    next_q_max_single = jnp.max(next_q_single_all, axis=-1)  # [T, E]

                    last_step_done = jnp.logical_or(
                        transitions.skill_done[-1, :, skill_idx], transitions.done[-1]
                    ).astype(jnp.float32)
                    last_step_reward = transitions.rewards[-1, :, skill_idx]
                    lambda_returns = last_step_reward + skill_gamma * (1.0 - last_step_done) * next_q_max_single[-1]

                    target_switch_cut_env = skill_target_switch_cut_env(
                        transitions.target_idx, skill_target_cut
                    )

                    def _get_target(carry, xs):
                        lambda_ret, next_q = carry
                        step_reward, step_done_skill, step_done_ep, step_target_cut, q_next = xs

                        step_done = jnp.logical_or(step_done_skill, step_done_ep)
                        step_done = jnp.logical_or(
                            step_done, step_target_cut.astype(jnp.bool_)
                        ).astype(jnp.float32)
                        target_bootstrap = step_reward + skill_gamma * (1.0 - step_done) * next_q
                        delta = lambda_ret - next_q
                        lambda_ret = target_bootstrap + skill_gamma * config["LAMBDA"] * delta
                        lambda_ret = (1.0 - step_done) * lambda_ret + step_done * step_reward
                        return (lambda_ret, q_next), lambda_ret

                    _, targets = jax.lax.scan(
                        _get_target,
                        (lambda_returns, next_q_max_single[-1]),
                        (
                            transitions.rewards[:-1, :, skill_idx],
                            transitions.skill_done[:-1, :, skill_idx],
                            transitions.done[:-1],
                            target_switch_cut_env,
                            next_q_max_single[:-1],
                        ),
                        reverse=True,
                    )
                    lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))
                    train_targets = lambda_targets[..., jnp.newaxis]

                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_and_target):
                        train_state, rng = carry
                        minibatch, target = minibatch_and_target

                        def _loss_fn(params):
                            q_vals, updates = skill_network.apply(
                                {"params": params, "batch_stats": train_state.batch_stats},
                                minibatch["ego_obs_all"],
                                train=True,
                                mutable=["batch_stats"],
                            )
                            chosen_action_qvals = jnp.take_along_axis(
                                q_vals,
                                jnp.expand_dims(minibatch["ego_action_all"], axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)

                            if config.get("OFF_POLICY", True):
                                valid = minibatch["relabel_mask"]
                            else:
                                n_objects = minibatch["ego_obs_all"].shape[1]
                                # Single-object batches store the chosen target at slot 0;
                                # do not compare global target_idx to that lone slot.
                                if n_objects == 1:
                                    on_policy = (
                                        minibatch["active_skill"][:, jnp.newaxis]
                                        == skill_idx
                                    )
                                else:
                                    chosen_obj_mask = (
                                        jnp.arange(n_objects)[jnp.newaxis, :]
                                        == minibatch["target_idx"][:, jnp.newaxis]
                                    )
                                    on_policy = jnp.logical_and(
                                        chosen_obj_mask,
                                        minibatch["active_skill"][:, jnp.newaxis]
                                        == skill_idx,
                                    )
                                valid = jnp.logical_and(
                                    minibatch["relabel_mask"],
                                    on_policy,
                                )
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
                        num_steps, num_envs = x.shape[:2]
                        rest_shape = x.shape[2:]
                        x = x.reshape(num_steps * num_envs, *rest_shape)
                        x = jax.random.permutation(rng, x)
                        total_batch = num_steps * num_envs
                        assert total_batch % config["NUM_MINIBATCHES"] == 0
                        x = x.reshape(config["NUM_MINIBATCHES"], -1, *rest_shape)
                        return x

                    rng, _rng = jax.random.split(rng)
                    train_batch = {
                        "relabel_mask": transitions.relabel_mask,
                        "ego_obs_all": transitions.ego_obs_all,
                        "ego_action_all": transitions.ego_action_all,
                        "active_skill": transitions.active_skill,
                        "target_idx": transitions.target_idx,
                    }
                    minibatches = jax.tree.map(
                        lambda x: preprocess_transition(x, _rng), train_batch
                    )
                    targets_batch = preprocess_transition(train_targets, _rng)

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

            rngs = jax.random.split(rng, NUM_SKILLS)
            skill_metrics, train_states = jax.vmap(_update_skill, in_axes=(0, 0, 0))(
                train_states, jnp.arange(NUM_SKILLS), rngs
            )

            metrics = {
                f"skill_{skill_names[i]}/{k}": v[i]
                for k, v in skill_metrics.items()
                for i in range(NUM_SKILLS)
            }

            mean_skill_rewards = jnp.mean(transitions.rewards_all, axis=(0, 1, 2))
            for i in range(NUM_SKILLS):
                metrics[f"skill_{skill_names[i]}/reward"] = mean_skill_rewards[i]

            metrics["env/reward_mean"] = jnp.mean(transitions.env_reward)

            try:
                info_metrics = {f"info/{k}": jnp.mean(v) for k, v in infos.items()}
                metrics.update(info_metrics)
            except AttributeError:
                pass

            mean_episodic_return = completed_returns_sum / jnp.maximum(
                1.0, num_completed
            )
            metrics["returned_episode_returns"] = mean_episodic_return
            metrics["env/episodic_return"] = mean_episodic_return

            if log_freeway_car_hits:
                metrics["freeway/car_hit_rate"] = (
                    car_hits_sum / freeway_env_steps_per_update
                )
                metrics["freeway/car_hits_per_update"] = car_hits_sum

            metrics["update_steps"] = train_states.n_updates[0]
            metrics["global/update_steps"] = step_offset_updates + train_states.n_updates[0]
            metrics["env/step"] = step_offset_timesteps + train_states.timesteps[0]

            if config.get("WANDB_MODE", "disabled") != "disabled":
                def _wandb_log_cb(m):
                    try:
                        import wandb
                    except ImportError:
                        return
                    if wandb.run is None:
                        return
                    log_dict = {}
                    for key, value in m.items():
                        value_np = np.asarray(value)
                        if value_np.size == 1:
                            log_dict[key] = float(value_np.ravel()[0])
                        else:
                            log_dict[key] = float(value_np.mean())
                    local_step = int(log_dict.get("update_steps", 0))
                    step = config.get("STEP_OFFSET", 0) + local_step
                    wandb.log(log_dict, step=step)

                jax.debug.callback(_wandb_log_cb, metrics)

            if is_rtpt_enabled(config):
                jax.debug.callback(rtpt_step_callback)

            runner_state = (train_states, tuple(expl_state), running_returns, rng)
            return runner_state, metrics

        env_seed_base = config.get("ENV_SEED_BASE", config["SEED"])
        init_reset_keys = jax.vmap(
            lambda i: jax.random.fold_in(jax.random.PRNGKey(env_seed_base), i)
        )(jnp.arange(config["NUM_ENVS"], dtype=jnp.uint32))
        expl_state = jax.vmap(env.reset)(init_reset_keys)
        rng, _rng = jax.random.split(rng)
        running_returns = jnp.zeros(config["NUM_ENVS"], dtype=jnp.float32)
        runner_state = (train_states, expl_state, running_returns, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    def eval_rollout(runner_state, eval_rng, step_number, max_frames=None):
        """Run one greedy evaluation episode; returns rollout data for video recording."""
        max_frames = max_frames or config.get("VIDEO_MAX_FRAMES", 2000)
        train_states, _expl_state, _running_returns, _train_rng = runner_state

        def _first(x):
            if hasattr(x, "ndim") and x.ndim > 0:
                return x[0]
            return x

        train_states = jax.tree_util.tree_map(_first, train_states)
        rng = jax.random.fold_in(eval_rng, 0)

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

        rng, _rng_policy = jax.random.split(rng)
        prev_target_idx = no_target_idx

        for _ in range(max_frames):
            rng, step_key = jax.random.split(rng)
            if fixed_target_idx is not None:
                target_obj_idxs = fixed_target_idx
            else:
                target_obj_idxs = get_heuristic_target(
                    obs.raw_observation,
                    obs.objects,
                    prev_target_idx,
                    step_key,
                )
                target_obj_idxs = int(jnp.ravel(target_obj_idxs)[0])
            prev_target_idx = target_obj_idxs

            if target_obj_idxs == no_target_idx:
                global_action = 0
                active_skills = 0
                combined_q = np.array([0.0, 0.0, 0.0])
            else:
                raw_label = obs.objects.semantic_label[-1, target_obj_idxs]
                active_skills = label_to_skill_idx(raw_label)
                ego_obs = get_ego_obs(
                    obs.raw_observation, obs.objects, target_obj_idxs
                )
                ego_obs_batch = ego_obs[np.newaxis, ...]

                q_vals = jax.vmap(
                    lambda ts: skill_network.apply(
                        {"params": ts.params, "batch_stats": ts.batch_stats},
                        ego_obs_batch,
                        train=False,
                    )
                )(train_states)
                q_vals = jnp.squeeze(q_vals, axis=1)
                combined_q = np.array(jnp.max(q_vals, axis=-1))

                if restrict_fire_for_non_destroy and int(active_skills) != 0:
                    q_active = q_vals[active_skills]
                    q_active = q_active.at[FIRE_EGO_ACTION_INDICES].set(-jnp.inf)
                else:
                    q_active = q_vals[active_skills]

                chosen_ego = jnp.argmax(q_active)
                global_action = action_mapper(
                    chosen_ego,
                    obs.raw_observation,
                    target_obj_idxs,
                    obs.objects,
                )
                if symbolic_collect_enabled and int(active_skills) == symbolic_collect_skill_idx:
                    global_action = jnp.int32(symbolic_collect_up_action)
                active_skills = int(active_skills)

            obs, state, _reward, done, _info = test_env.step(state, global_action)

            states_list.append(state)
            active_list.append(active_skills)
            combined_qs_list.append(np.array(combined_q))
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
            np.stack(obj_x_list),
            np.stack(obj_y_list),
            np.stack(obj_w_list),
            np.stack(obj_h_list),
            np.stack(obj_active_list),
        )
        return stacked_states, active_agents, combined_qs, dones, target_obj_idxs_stacked, objects_bbox

    return train, eval_rollout
