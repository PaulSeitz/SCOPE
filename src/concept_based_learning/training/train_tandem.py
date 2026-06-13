import jax
import jax.numpy as jnp
import numpy as np

import optax

from concept_based_learning.observations.action_space import fire_ego_action_indices
from concept_based_learning.observations.egocentric import get_ego_obs
from concept_based_learning.networks.skill_network import QNetwork as SkillQNetwork, CustomTrainState
from concept_based_learning.networks.meta_factory import create_meta_network
from concept_based_learning.environments.datatypes import TandemTransition
from concept_based_learning.environments.meta_target_mask import (
    build_meta_target_mask,
    is_meta_fire_target,
    is_meta_noop_target,
    meta_fire_game_action,
    meta_fire_target_enabled,
    meta_fire_target_idx,
    meta_noop_target_enabled,
    meta_noop_target_idx,
)
from concept_based_learning.training.schedules import eps_decay_transition_steps
from concept_based_learning.training.rtpt_progress import is_rtpt_enabled, rtpt_step_callback


def make_tandem_train(
    config,
    env,
    test_env,
    renderer=None,
    *,
    no_target_idx,
    get_heuristic_target,
    action_mapper,
    inverse_action_mapper,
    extract_all_skills_rewards,
    extract_life_lost,
    meta_mask_hook,
    forced_target_hook,
):
    del test_env, renderer

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0, (
        "NUM_MINIBATCHES must divide NUM_STEPS * NUM_ENVS"
    )

    num_skills = config["NUM_SKILLS"]
    max_objects = config["MAX_OBJECTS"]
    ego_action_dim = config["EGO_ACTION_DIM"]
    skill_warmup_updates = int(config.get("SKILL_WARMUP_UPDATES", 0))
    meta_learn_during_warmup = bool(config.get("META_LEARN_DURING_WARMUP", False))
    meta_num_epochs = int(config.get("META_NUM_EPOCHS", config["NUM_EPOCHS"]))
    skill_freeze_after_updates = int(config.get("SKILL_FREEZE_AFTER_UPDATES", 0))
    meta_noop_enabled = meta_noop_target_enabled(config)
    meta_noop_idx = meta_noop_target_idx(config)
    meta_fire_enabled = meta_fire_target_enabled(config)
    meta_fire_idx = meta_fire_target_idx(config)
    meta_fire_action = int(meta_fire_game_action(config))
    relabel_all_objects = bool(config.get("RELABEL_ALL_OBJECTS", False))
    relabel_k_objects = config.get("RELABEL_K_OBJECTS", None)
    skill_bootstrap = str(config.get("SKILL_BOOTSTRAP", "q_lambda")).lower()
    relabel_k_objects = (
        None
        if relabel_k_objects is None
        else int(max(1, min(max_objects, int(relabel_k_objects))))
    )
    skill_gamma = float(config.get("SKILL_GAMMA", config["GAMMA"]))
    symbolic_collect_enabled = bool(config.get("SYMBOLIC_COLLECT_FORCE_UP", False))
    symbolic_collect_skill_idx = int(config.get("COLLECT_SKILL_INDEX", 1))
    symbolic_collect_up_action = int(config.get("SYMBOLIC_COLLECT_UP_ACTION", 1))
    skill_qlambda_watkins_cut = bool(
        config.get("SKILL_QLAMBDA_WATKINS_CUT", False)
    )
    skill_lambda = float(config.get("SKILL_LAMBDA", config["LAMBDA"]))

    log_freeway_car_hits = str(config.get("ENV_NAME", "")).lower() == "freeway"
    freeway_env_steps_per_update = config["NUM_STEPS"] * config["NUM_ENVS"]

    def label_to_skill_idx(label):
        return jnp.clip(label - 1, 0, num_skills - 1)

    def make_linear_schedule(start, finish, decay_cfg):
        return optax.linear_schedule(
            start, finish, eps_decay_transition_steps(decay_cfg, config["NUM_UPDATES"])
        )

    skill_eps_schedule = make_linear_schedule(
        config.get("SKILL_EPS_START", config["EPS_START"]),
        config.get("SKILL_EPS_FINISH", config["EPS_FINISH"]),
        config.get("SKILL_EPS_DECAY", config["EPS_DECAY"]),
    )
    meta_eps_schedule = make_linear_schedule(
        config.get("META_EPS_START", config["EPS_START"]),
        config.get("META_EPS_FINISH", config["EPS_FINISH"]),
        config.get("META_EPS_DECAY", config["EPS_DECAY"]),
    )

    skill_network = SkillQNetwork(
        action_dim=ego_action_dim,
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
    )
    meta_network = create_meta_network(config)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
        jax.random.split(rng, n_envs)
    )
    vmap_step = lambda env_state, action: jax.vmap(env.step)(env_state, action)

    def merge_reset_where_done(new_tree, reset_tree, done):
        def _replace(new_leaf, reset_leaf):
            expand = (config["NUM_ENVS"],) + (1,) * (new_leaf.ndim - 1)
            return jnp.where(jnp.reshape(done, expand), reset_leaf, new_leaf)

        return jax.tree_util.tree_map(_replace, new_tree, reset_tree)

    def eps_greedy_exploration(rng, q_vals, eps, mask=None):
        rng_a, rng_e = jax.random.split(rng)
        valid_q_vals = q_vals if mask is None else jnp.where(mask, q_vals, -jnp.inf)
        greedy_actions = jnp.argmax(valid_q_vals, axis=-1)
        if mask is not None:
            random_logits = jnp.where(mask, 0.0, -1e9)
            random_actions = jax.random.categorical(rng_a, random_logits)
        else:
            random_actions = jax.random.randint(
                rng_a,
                shape=greedy_actions.shape,
                minval=0,
                maxval=q_vals.shape[-1],
            )
        return jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )

    def preprocess_transition(x, rng_perm):
        num_steps, num_envs = x.shape[:2]
        rest_shape = x.shape[2:]
        x = x.reshape(num_steps * num_envs, *rest_shape)
        x = jax.random.permutation(rng_perm, x)
        x = x.reshape(config["NUM_MINIBATCHES"], -1, *rest_shape)
        return x

    def train(rng, init_meta_params=None, init_skill_params=None):
        lr_skill = config.get("SKILL_LR", config["LR"])
        lr_meta = config.get("META_LR", config["LR"] * 0.1)

        tx_skill = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr_skill),
        )
        tx_meta = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr_meta),
        )

        rng, _rng = jax.random.split(rng)
        probe_obs_batched, _ = vmap_reset(1)(_rng)
        probe_obs = jax.tree_util.tree_map(lambda x: x[0], probe_obs_batched)
        probe_ego = get_ego_obs(
            probe_obs.raw_observation,
            probe_obs.objects,
            jnp.array(0, dtype=jnp.int32),
        )

        meta_vars = meta_network.init(
            rng, probe_obs.raw_observation, probe_obs.objects, train=False
        )
        meta_state = CustomTrainState.create(
            apply_fn=meta_network.apply,
            params=meta_vars["params"] if init_meta_params is None else init_meta_params,
            batch_stats={},
            tx=tx_meta,
        )

        def create_skill_state(rng_key, init_p):
            vars = skill_network.init(rng_key, probe_ego, train=False)
            return CustomTrainState.create(
                apply_fn=skill_network.apply,
                params=vars["params"] if init_p is None else init_p,
                batch_stats=vars.get("batch_stats", {}),
                tx=tx_skill,
            )

        rng_keys = jax.random.split(rng, num_skills)
        skill_states = jax.vmap(
            create_skill_state,
            in_axes=(0, 0 if init_skill_params is not None else None),
        )(rng_keys, init_skill_params)

        def _update_step(runner_state, _):
            meta_state, skill_states, expl_state, running_returns, rng = runner_state

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
                rng, rng_meta, rng_skill = jax.random.split(rng, 3)

                meta_q_vals = jax.vmap(
                    lambda obs, objs: meta_network.apply(
                        {"params": meta_state.params}, obs, objs, train=False
                    )
                )(last_obs.raw_observation, last_obs.objects)
                active_masks = jax.vmap(
                    lambda objs, raw: build_meta_target_mask(
                        objs, raw, config, hook=meta_mask_hook
                    )
                )(last_obs.objects, last_obs.raw_observation)
                active_any = jnp.any(active_masks, axis=-1)

                eps_meta = jnp.full(
                    config["NUM_ENVS"],
                    meta_eps_schedule(jnp.maximum(0, meta_state.n_updates - skill_warmup_updates)),
                )
                policy_rngs = jax.random.split(rng_meta, config["NUM_ENVS"])
                use_heuristic_warmup = meta_state.n_updates < skill_warmup_updates
                target_obj_idxs = jax.vmap(eps_greedy_exploration)(
                    policy_rngs,
                    meta_q_vals,
                    eps_meta,
                    active_masks,
                )
                heuristic_targets = jax.vmap(get_heuristic_target)(
                    last_obs.raw_observation,
                    last_obs.objects,
                    last_targets,
                    policy_rngs,
                )
                target_obj_idxs = jnp.where(
                    use_heuristic_warmup & (heuristic_targets != no_target_idx),
                    heuristic_targets,
                    target_obj_idxs,
                )
                target_obj_idxs = target_obj_idxs.astype(jnp.int32)
                target_obj_idxs = jnp.where(
                    active_any,
                    target_obj_idxs,
                    jnp.full_like(target_obj_idxs, no_target_idx, dtype=jnp.int32),
                )
                target_obj_idxs = jax.lax.cond(
                    use_heuristic_warmup,
                    lambda x: x,
                    lambda x: forced_target_hook(last_obs.raw_observation, x, config, rng_meta),
                    target_obj_idxs,
                )
                safe_target_idxs = jnp.where(
                    target_obj_idxs == no_target_idx, 0, target_obj_idxs
                )

                raw_labels = jax.vmap(lambda labels, idx: labels[-1, idx])(
                    last_obs.objects.semantic_label, safe_target_idxs
                )
                active_skills = label_to_skill_idx(raw_labels)
                ego_obses = jax.vmap(get_ego_obs)(
                    last_obs.raw_observation, last_obs.objects, safe_target_idxs
                )

                def compute_skill_actions(state, skill_idx):
                    q_vals = skill_network.apply(
                        {"params": state.params, "batch_stats": state.batch_stats},
                        ego_obses,
                        train=False,
                    )
                    eps_skill = jnp.full(
                        config["NUM_ENVS"],
                        skill_eps_schedule(meta_state.n_updates),
                    )

                    invalid_ego_actions = config.get("INVALID_EGO_ACTIONS", [])
                    restrict_fire_for_non_destroy = bool(
                        config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False)
                    )
                    _custom_fire = config.get("FIRE_EGO_ACTION_INDICES")

                    if _custom_fire is not None:
                        FIRE_EGO_ACTION_INDICES = jnp.array(
                            list(_custom_fire), dtype=jnp.int32
                        )
                    else:
                        FIRE_EGO_ACTION_INDICES = fire_ego_action_indices(
                            ego_action_dim
                        )

                    def build_mask_for_skill(idx):
                        base_row = jnp.ones((ego_action_dim,), dtype=bool)
                        if invalid_ego_actions:
                            invalid_arr = jnp.array(
                                invalid_ego_actions, dtype=jnp.int32
                            )
                            base_row = base_row.at[invalid_arr].set(False)
                        base_row = base_row.at[0].set(True)
                        return jax.lax.cond(
                            idx == 0,
                            lambda r: r,
                            lambda r: (
                                r.at[FIRE_EGO_ACTION_INDICES].set(False).at[0].set(True)
                                if restrict_fire_for_non_destroy
                                else r
                            ),
                            base_row,
                        )

                    mask_row = build_mask_for_skill(skill_idx)
                    mask = jnp.broadcast_to(mask_row, (config["NUM_ENVS"], ego_action_dim))

                    ego_action = jax.vmap(eps_greedy_exploration)(
                        jax.random.split(rng_skill, config["NUM_ENVS"]),
                        q_vals,
                        eps_skill,
                        mask=mask,
                    )
                    return ego_action, q_vals

                all_ego_actions, all_skill_q_vals = jax.vmap(compute_skill_actions)(
                    skill_states, jnp.arange(num_skills, dtype=jnp.int32)
                )
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
                meta_noop = is_meta_noop_target(
                    target_obj_idxs,
                    config,
                    noop_idx=meta_noop_idx,
                    enabled=meta_noop_enabled,
                )
                meta_fire = is_meta_fire_target(
                    target_obj_idxs,
                    config,
                    fire_idx=meta_fire_idx,
                    enabled=meta_fire_enabled,
                )
                force_noop = jnp.logical_or(target_obj_idxs == no_target_idx, meta_noop)
                chosen_ego_actions = jnp.where(force_noop, 0, chosen_ego_actions)

                global_actions = jax.vmap(action_mapper)(
                    chosen_ego_actions, last_obs.raw_observation, target_obj_idxs, last_obs.objects
                )
                global_actions = jnp.where(
                    meta_fire,
                    jnp.full_like(global_actions, meta_fire_action),
                    global_actions,
                )
                if symbolic_collect_enabled:
                    global_actions = jnp.where(
                        active_skills == symbolic_collect_skill_idx,
                        jnp.full_like(global_actions, symbolic_collect_up_action),
                        global_actions,
                    )
                global_actions = jnp.where(force_noop, 0, global_actions)

                new_obs, new_env_state, env_reward, new_done, info = vmap_step(
                    env_state, global_actions
                )
                life_lost = extract_life_lost(env_state, new_env_state)
                car_hits_sum = car_hits_sum + jnp.sum(life_lost > 0)
                done_due_to_life_lost = jnp.logical_and(new_done, life_lost > 0)

                episode_return = running_returns + env_reward
                done_float = new_done.astype(jnp.float32)
                completed_returns_sum = completed_returns_sum + jnp.sum(
                    episode_return * done_float
                )
                num_completed = num_completed + jnp.sum(done_float)
                running_returns = episode_return * (1.0 - done_float)

                prev_targets = jax.vmap(
                    lambda objs, idx: jax.tree_util.tree_map(
                        lambda x: jnp.take(x, idx, axis=-1), objs
                    )
                )(last_obs.objects, safe_target_idxs)
                next_targets = jax.vmap(
                    lambda objs, idx: jax.tree_util.tree_map(
                        lambda x: jnp.take(x, idx, axis=-1), objs
                    )
                )(new_obs.objects, safe_target_idxs)
                skill_rewards, skill_done = jax.vmap(
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
                if meta_noop_enabled:
                    skill_rewards = jnp.where(
                        meta_noop[:, jnp.newaxis], 0.0, skill_rewards
                    )
                if meta_fire_enabled:
                    skill_rewards = jnp.where(
                        meta_fire[:, jnp.newaxis], 0.0, skill_rewards
                    )

                changed_target = target_obj_idxs != last_targets
                was_valid_target = jnp.logical_and(
                    last_targets >= 0, last_targets < max_objects
                )
                apply_penalty = jnp.logical_and(changed_target, was_valid_target)

                # Blended meta reward: λ * env_reward + (1-λ) * active_skill_reward.
                # λ=1.0 → full env reward (default); λ=0.0 → full skill reward.
                # skill_rewards is already noop-zeroed above; no extra masking needed.
                meta_reward_lambda = float(config.get("META_REWARD_LAMBDA", 1.0))
                base_env_reward = jnp.sign(env_reward)
                active_skill_reward = skill_rewards[
                    jnp.arange(config["NUM_ENVS"]), active_skills
                ]
                blended_reward = (
                    meta_reward_lambda * base_env_reward
                    + (1.0 - meta_reward_lambda) * active_skill_reward
                )
                meta_death_penalty = float(config.get("META_DEATH_PENALTY", -5.0))
                meta_switch_penalty = float(config.get("META_SWITCH_PENALTY", -0.5))
                meta_reward = (
                    blended_reward
                    + jnp.where(
                        done_due_to_life_lost, meta_death_penalty, 0.0
                    ).astype(blended_reward.dtype)
                    + jnp.where(
                        apply_penalty, meta_switch_penalty, 0.0
                    ).astype(blended_reward.dtype)
                )

                rng, rng_reset = jax.random.split(rng)
                reset_obs, reset_state = vmap_reset(config["NUM_ENVS"])(rng_reset)
                next_obs_for_rollout = merge_reset_where_done(
                    new_obs, reset_obs, new_done
                )
                next_state_for_rollout = merge_reset_where_done(
                    new_env_state, reset_state, new_done
                )
                next_active_masks = jax.vmap(
                    lambda objs, raw: build_meta_target_mask(
                        objs, raw, config, hook=meta_mask_hook
                    )
                )(next_obs_for_rollout.objects, next_obs_for_rollout.raw_observation)
                next_ego_obses = jax.vmap(get_ego_obs)(
                    next_obs_for_rollout.raw_observation,
                    next_obs_for_rollout.objects,
                    safe_target_idxs,
                )

                if relabel_all_objects:
                    obj_ids = jnp.arange(max_objects, dtype=jnp.int32)
                    ego_obs_all_full = jax.vmap(
                        lambda raw, objs: jax.vmap(
                            lambda obj_idx: get_ego_obs(raw, objs, obj_idx)
                        )(obj_ids)
                    )(last_obs.raw_observation, last_obs.objects)
                    next_ego_obs_all_full = jax.vmap(
                        lambda raw, objs: jax.vmap(
                            lambda obj_idx: get_ego_obs(raw, objs, obj_idx)
                        )(obj_ids)
                    )(next_obs_for_rollout.raw_observation, next_obs_for_rollout.objects)
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

                    skill_rewards_all_full, skill_done_all_full = jax.vmap(
                        _all_object_rewards
                    )(
                        last_obs.raw_observation,
                        new_obs.raw_observation,
                        last_obs.objects,
                        new_obs.objects,
                        life_lost,
                        info,
                    )
                    relabel_mask_full = active_masks
                    if meta_noop_enabled:
                        relabel_mask_full = relabel_mask_full.at[:, meta_noop_idx].set(
                            False
                        )
                    if meta_fire_enabled:
                        relabel_mask_full = relabel_mask_full.at[:, meta_fire_idx].set(
                            False
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
                        skill_rewards_all = _gather_objects(skill_rewards_all_full)
                        skill_done_all = _gather_objects(skill_done_all_full)
                        relabel_mask = _gather_objects(relabel_mask_full)
                    else:
                        ego_obs_all = ego_obs_all_full
                        next_ego_obs_all = next_ego_obs_all_full
                        ego_action_all = ego_action_all_full
                        skill_rewards_all = skill_rewards_all_full
                        skill_done_all = skill_done_all_full
                        relabel_mask = relabel_mask_full
                else:
                    base_valid = target_obj_idxs != no_target_idx
                    if meta_noop_enabled:
                        base_valid = jnp.logical_and(base_valid, jnp.logical_not(meta_noop))
                    if meta_fire_enabled:
                        base_valid = jnp.logical_and(base_valid, jnp.logical_not(meta_fire))
                    ego_obs_all = ego_obses[:, jnp.newaxis, :]
                    next_ego_obs_all = next_ego_obses[:, jnp.newaxis, :]
                    ego_action_all = chosen_ego_actions[:, jnp.newaxis]
                    skill_rewards_all = skill_rewards[:, jnp.newaxis, :]
                    skill_done_all = skill_done[:, jnp.newaxis, :]
                    relabel_mask = base_valid[:, jnp.newaxis]

                transition = TandemTransition(
                    done=new_done,
                    obs=last_obs.raw_observation,
                    objects=last_obs.objects,
                    next_obs=next_obs_for_rollout.raw_observation,
                    next_objects=next_obs_for_rollout.objects,
                    target_idx=target_obj_idxs,
                    meta_q_val=meta_q_vals,
                    meta_reward=meta_reward,
                    active_mask=active_masks,
                    next_active_mask=next_active_masks,
                    ego_obs=ego_obses,
                    next_ego_obs=next_ego_obses,
                    ego_action=chosen_ego_actions,
                    active_skill=active_skills,
                    skill_q_val=all_skill_q_vals,
                    skill_rewards=skill_rewards,
                    skill_done=skill_done,
                    ego_obs_all=ego_obs_all,
                    next_ego_obs_all=next_ego_obs_all,
                    ego_action_all=ego_action_all,
                    skill_rewards_all=skill_rewards_all,
                    skill_done_all=skill_done_all,
                    relabel_mask=relabel_mask,
                    env_reward=base_env_reward,
                )
                return (
                    next_obs_for_rollout,
                    next_state_for_rollout,
                    running_returns,
                    rng,
                    completed_returns_sum,
                    num_completed,
                    target_obj_idxs,
                    car_hits_sum,
                ), transition

            rng, _rng = jax.random.split(rng)
            init_targets = jnp.full(
                (config["NUM_ENVS"],), no_target_idx, dtype=jnp.int32
            )
            init_car_hits_sum = jnp.array(0.0, dtype=jnp.float32)
            carry_final, transitions = jax.lax.scan(
                _step_env,
                (
                    *expl_state,
                    running_returns,
                    _rng,
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    init_targets,
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

            meta_state = meta_state.replace(
                timesteps=meta_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            skip_meta_update = jnp.logical_and(
                meta_state.n_updates < skill_warmup_updates,
                not meta_learn_during_warmup,
            )

            last_next_obs = jax.tree_util.tree_map(
                lambda x: x[-1], transitions.next_obs
            )
            last_next_objects = jax.tree_util.tree_map(
                lambda x: x[-1], transitions.next_objects
            )
            last_meta_q = jax.vmap(
                lambda obs, objs: meta_network.apply(
                    {"params": meta_state.params}, obs, objs, train=False
                )
            )(last_next_obs, last_next_objects)
            next_active_mask = transitions.next_active_mask[-1]
            next_active_any = jnp.any(next_active_mask, axis=-1)
            masked_last_q = jnp.max(
                jnp.where(next_active_mask, last_meta_q, -jnp.inf), axis=-1
            )
            last_q = jnp.where(next_active_any, masked_last_q, 0.0)
            last_q = last_q * (1 - transitions.done[-1])

            def _meta_get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.meta_reward
                    + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    (1 - transition.done) * lambda_returns
                    + transition.done * transition.meta_reward
                )

                prev_active_mask = transition.active_mask
                prev_active_any = jnp.any(prev_active_mask, axis=-1)
                masked_next_q = jnp.max(
                    jnp.where(prev_active_mask, transition.meta_q_val, -jnp.inf),
                    axis=-1,
                )
                next_q = jnp.where(prev_active_any, masked_next_q, 0.0)
                return (lambda_returns, next_q), lambda_returns

            meta_lambda_returns = transitions.meta_reward[-1] + config["GAMMA"] * last_q
            _, meta_targets_scan = jax.lax.scan(
                _meta_get_target,
                (meta_lambda_returns, last_q),
                jax.tree.map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            meta_lambda_targets = jnp.concatenate(
                (meta_targets_scan, meta_lambda_returns[jnp.newaxis])
            )

            def _meta_learn_epoch(carry, _):
                m_state, rng_meta_upd = carry
                rng_meta_upd, rng_perm = jax.random.split(rng_meta_upd)

                meta_batch = {
                    "obs": transitions.obs,
                    "objects": transitions.objects,
                    "target_idx": transitions.target_idx,
                    "active_mask": transitions.active_mask,
                }
                minibatches = jax.tree.map(
                    lambda x: preprocess_transition(x, rng_perm),
                    meta_batch,
                )
                target_batches = preprocess_transition(meta_lambda_targets, rng_perm)

                def _meta_learn_phase(carry_phase, minibatch_and_target):
                    m_state_phase, rng_phase = carry_phase
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        q_vals = jax.vmap(
                            lambda obs, objs: meta_network.apply(
                                {"params": params}, obs, objs, train=True
                            )
                        )(minibatch["obs"], minibatch["objects"])
                        chosen_q = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch["target_idx"], -1),
                            axis=-1,
                        ).squeeze(axis=-1)
                        valid = jnp.any(minibatch["active_mask"], axis=-1)
                        loss = (jnp.square(chosen_q - target) * valid).sum() / (
                            valid.sum() + 1e-8
                        )
                        return loss, chosen_q

                    (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        m_state_phase.params
                    )
                    update_scale = jnp.where(skip_meta_update, 0.0, 1.0)
                    grads = jax.tree_util.tree_map(lambda g: g * update_scale, grads)
                    m_state_phase = m_state_phase.apply_gradients(grads=grads)
                    m_state_phase = m_state_phase.replace(
                        grad_steps=m_state_phase.grad_steps + 1
                    )
                    return (m_state_phase, rng_phase), (loss, qvals)

                (m_state, rng_meta_upd), (loss, qvals) = jax.lax.scan(
                    _meta_learn_phase,
                    (m_state, rng_meta_upd),
                    (minibatches, target_batches),
                )
                return (m_state, rng_meta_upd), (loss, qvals)

            (meta_state, rng), (meta_loss, meta_qvals) = jax.lax.scan(
                _meta_learn_epoch,
                (meta_state, rng),
                None,
                meta_num_epochs,
            )
            meta_state = meta_state.replace(n_updates=meta_state.n_updates + 1)

            def _update_single_skill(s_state, skill_idx, rng_s):
                s_state = s_state.replace(
                    timesteps=s_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                next_q_all = skill_network.apply(
                    {"params": s_state.params, "batch_stats": s_state.batch_stats},
                    transitions.next_ego_obs_all,
                    train=False,
                )  # [T, E, O, A]
                next_q_max = jnp.max(next_q_all, axis=-1)  # [T, E, O]

                step_reward = transitions.skill_rewards_all[..., skill_idx]  # [T, E, O]
                step_done = jnp.logical_or(
                    transitions.skill_done_all[..., skill_idx],
                    transitions.done[..., jnp.newaxis],
                ).astype(jnp.float32)  # [T, E, O]

                if skill_bootstrap == "one_step":
                    skill_targets = step_reward + (
                        skill_gamma * (1.0 - step_done) * next_q_max
                    )  # [T, E, O]
                else:
                    if skill_qlambda_watkins_cut:
                        q_all_t = skill_network.apply(
                            {"params": s_state.params, "batch_stats": s_state.batch_stats},
                            transitions.ego_obs_all,
                            train=False,
                        )  # [T, E, O, A]
                        greedy_t = jnp.argmax(q_all_t, axis=-1)  # [T, E, O]
                        cut = (
                            greedy_t == transitions.ego_action_all
                        ).astype(jnp.float32)
                    else:
                        cut = jnp.ones_like(step_reward)  # [T, E, O]

                    last_target = (
                        step_reward[-1]
                        + skill_gamma * (1.0 - step_done[-1]) * next_q_max[-1]
                    )  # [E, O]

                    def _skill_get_target(g_future, xs):
                        r_t, d_t, nq_t, cut_t = xs  # each [E, O]
                        boot_t = r_t + skill_gamma * (1.0 - d_t) * nq_t
                        g_t = boot_t + skill_gamma * skill_lambda * (1.0 - d_t) * cut_t * (
                            g_future - nq_t
                        )
                        g_t = (1.0 - d_t) * g_t + d_t * r_t
                        return g_t, g_t

                    xs = (
                        step_reward[:-1],
                        step_done[:-1],
                        next_q_max[:-1],
                        cut[:-1],
                    )
                    _, targets_body = jax.lax.scan(
                        _skill_get_target, last_target, xs, reverse=True
                    )
                    skill_targets = jnp.concatenate(
                        [targets_body, last_target[jnp.newaxis]], axis=0
                    )  # [T, E, O]

                def _learn_epoch(carry, _):
                    s_state_epoch, rng_epoch = carry
                    rng_epoch, rng_perm = jax.random.split(rng_epoch)

                    train_batch = {
                        "relabel_mask": transitions.relabel_mask,
                        "ego_obs_all": transitions.ego_obs_all,
                        "ego_action_all": transitions.ego_action_all,
                        "active_skill": transitions.active_skill,
                        "target_idx": transitions.target_idx,
                    }
                    minibatches = jax.tree.map(
                        lambda x: preprocess_transition(x, rng_perm),
                        train_batch,
                    )
                    targets_batch = preprocess_transition(skill_targets, rng_perm)

                    def _learn_phase(carry_phase, minibatch_and_target):
                        s_state_phase, rng_phase = carry_phase
                        minibatch, target = minibatch_and_target

                        def _loss_fn(params):
                            q_vals, updates = skill_network.apply(
                                {
                                    "params": params,
                                    "batch_stats": s_state_phase.batch_stats,
                                },
                                minibatch["ego_obs_all"],
                                train=True,
                                mutable=["batch_stats"],
                            )
                            chosen_q = jnp.take_along_axis(
                                q_vals,
                                jnp.expand_dims(minibatch["ego_action_all"], axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)
                            if config.get("OFF_POLICY", True):
                                valid = minibatch["relabel_mask"]
                            else:
                                n_objects = minibatch["ego_obs_all"].shape[1]
                                chosen_obj_mask = (
                                    jnp.arange(n_objects)[jnp.newaxis, :]
                                    == minibatch["target_idx"][:, jnp.newaxis]
                                )
                                valid = jnp.logical_and(
                                    minibatch["relabel_mask"],
                                    jnp.logical_and(
                                        chosen_obj_mask,
                                        minibatch["active_skill"][:, jnp.newaxis] == skill_idx,
                                    ),
                                )
                            loss = (jnp.square(chosen_q - target) * valid).sum() / (
                                valid.sum() + 1e-8
                            )
                            return loss, (updates, chosen_q)

                        (loss, (updates, qvals)), grads = jax.value_and_grad(
                            _loss_fn, has_aux=True
                        )(s_state_phase.params)
                        grads = jax.tree_util.tree_map(
                            lambda g: g * jnp.where(skills_frozen, 0.0, 1.0), grads
                        )
                        s_state_phase = s_state_phase.apply_gradients(grads=grads)
                        s_state_phase = s_state_phase.replace(
                            grad_steps=s_state_phase.grad_steps + 1,
                            batch_stats=updates["batch_stats"],
                        )
                        return (s_state_phase, rng_phase), (loss, qvals)

                    (s_state_epoch, rng_epoch), (loss, qvals) = jax.lax.scan(
                        _learn_phase,
                        (s_state_epoch, rng_epoch),
                        (minibatches, targets_batch),
                    )
                    return (s_state_epoch, rng_epoch), (loss, qvals)

                (s_state, rng_s), (skill_loss, skill_qvals) = jax.lax.scan(
                    _learn_epoch,
                    (s_state, rng_s),
                    None,
                    config["NUM_EPOCHS"],
                )
                s_state = s_state.replace(n_updates=s_state.n_updates + 1)
                return s_state, (
                    skill_loss[-1].mean(),
                    skill_qvals[-1].mean(),
                )

            skills_frozen = jnp.logical_and(
                skill_freeze_after_updates > 0,
                meta_state.n_updates >= skill_freeze_after_updates,
            )

            rngs = jax.random.split(rng, num_skills)
            skill_states, skill_metrics = jax.vmap(
                _update_single_skill,
                in_axes=(0, 0, 0),
            )(skill_states, jnp.arange(num_skills), rngs)

            mean_episodic_return = completed_returns_sum / jnp.maximum(1.0, num_completed)
            valid_target_mask = (transitions.target_idx != no_target_idx).astype(
                jnp.float32
            )
            reward_nonzero = (transitions.env_reward != 0).astype(jnp.float32)
            skill_reward_nonzero = (
                jnp.abs(transitions.skill_rewards_all) > 0
            ).astype(jnp.float32)
            if config.get("OFF_POLICY", True):
                skill_valid_mask = jnp.broadcast_to(
                    transitions.relabel_mask[..., None], transitions.skill_rewards_all.shape
                )
            else:
                skill_valid_mask = jnp.logical_and(
                    transitions.relabel_mask[..., None] > 0,
                    transitions.active_skill[:, :, None, None]
                    == jnp.arange(num_skills, dtype=jnp.int32)[None, None, None, :],
                )
                skill_valid_mask = skill_valid_mask.astype(jnp.float32)

            _act = transitions.relabel_mask
            _both_active = jnp.logical_and(_act[1:], _act[:-1])
            _no_reset = jnp.logical_not(
                transitions.done[:-1].astype(bool)[..., jnp.newaxis]
            )
            _valid_pair = jnp.logical_and(_both_active, _no_reset)
            slot_stability = _valid_pair.sum() / (_no_reset.sum() + 1e-8)

            metrics = {
                "meta_policy/td_loss": meta_loss[-1].mean(),
                "meta_policy/qvals": meta_qvals[-1].mean(),
                "meta_policy/reward_mean": jnp.mean(transitions.meta_reward),
                "env/reward_mean": jnp.mean(transitions.env_reward),
                "env/reward_nonzero_frac": jnp.mean(reward_nonzero),
                "env/reward_max": jnp.max(transitions.env_reward),
                "env/done_frac": jnp.mean(transitions.done.astype(jnp.float32)),
                "env/episodic_return": mean_episodic_return,
                "returned_episode_returns": mean_episodic_return,
                "meta_policy/valid_target_frac": jnp.mean(valid_target_mask),
                "meta_policy/targetable_objects_mean": jnp.mean(
                    jnp.sum(transitions.active_mask.astype(jnp.float32), axis=-1)
                ),
                "meta_policy/eps": meta_eps_schedule(meta_state.n_updates),
                "skill_policy/eps": skill_eps_schedule(meta_state.n_updates),
                "skill/td_loss_mean": jnp.mean(skill_metrics[0]),
                "skill/qvals_mean": jnp.mean(skill_metrics[1]),
                "skill/valid_transition_frac": jnp.mean(skill_valid_mask),
                "skill/reward_nonzero_frac": jnp.mean(skill_reward_nonzero),
                "skill/relabel_objects_mean": jnp.mean(
                    jnp.sum(transitions.relabel_mask.astype(jnp.float32), axis=-1)
                ),
                "skill/slot_stability": slot_stability,
                "warmup/heuristic_active": (
                    meta_state.n_updates < skill_warmup_updates
                ).astype(jnp.float32),
                "warmup/meta_frozen": skip_meta_update.astype(jnp.float32),
                "skill/frozen": skills_frozen.astype(jnp.float32),
                "update_steps": meta_state.n_updates,
                "global/update_steps": meta_state.n_updates,
                "env/step": meta_state.timesteps,
            }

            if log_freeway_car_hits:
                metrics["freeway/car_hit_rate"] = (
                    car_hits_sum / freeway_env_steps_per_update
                )
                metrics["freeway/car_hits_per_update"] = car_hits_sum

            if config.get("WANDB_MODE", "disabled") != "disabled":
                def _wandb_log_cb(m):
                    try:
                        import wandb
                        if wandb.run is not None:
                            log_dict = {k: float(np.asarray(v).mean()) for k, v in m.items()}
                            step = int(log_dict.get("global/update_steps", 0))
                            wandb.log(log_dict, step=step)
                    except Exception:
                        pass

                jax.debug.callback(_wandb_log_cb, metrics)

            if is_rtpt_enabled(config):
                jax.debug.callback(rtpt_step_callback)

            return (meta_state, skill_states, expl_state, running_returns, rng), metrics

        rng, _rng = jax.random.split(rng)
        init_expl = jax.vmap(env.reset)(jax.random.split(_rng, config["NUM_ENVS"]))
        init_running_returns = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32)
        runner_state = (meta_state, skill_states, init_expl, init_running_returns, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
