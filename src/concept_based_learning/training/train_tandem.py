import time

import jax
import jax.numpy as jnp
import numpy as np

import optax

from concept_based_learning.observations.action_space import (
    fire_ego_action_indices,
    make_realized_ego_mask_fn,
    mask_q_with_legal_ego,
)
from concept_based_learning.observations.egocentric import get_ego_obs
from concept_based_learning.networks.skill_network import (
    OBJECT_CONDITIONED_SKILL_INPUT_MODES,
    PIXEL_SKILL_INPUT_MODES,
    QNetwork as SkillQNetwork,
    CustomTrainState,
)
from concept_based_learning.networks.meta_factory import create_meta_network
from concept_based_learning.networks.meta_network import flatten_raw_observation
from concept_based_learning.environments.datatypes import (
    TandemTransition,
    label_to_skill_idx as _label_to_skill_idx,
    project_skill_outcomes,
)
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
from concept_based_learning.environments.meta_target_persistence import (
    commit_meta_targets,
    compute_meta_target_locked,
    meta_q_stickiness_enabled,
    meta_target_persist_enabled,
)
from concept_based_learning.training.schedules import (
    DEFAULT_SKILL_EPS_DECAY,
    DEFAULT_SKILL_EPS_FINISH,
    DEFAULT_SKILL_EPS_START,
    eps_decay_transition_steps,
    skill_learning_rate_schedule,
)
from concept_based_learning.training.trajectory_cuts import (
    DEFAULT_SKILL_TARGET_CUT,
    normalize_skill_target_cut,
    skill_target_switch_cut,
    skill_target_switch_cut_env,
    skill_target_switch_cut_env_padded,
)
from concept_based_learning.training.rtpt_progress import is_rtpt_enabled, rtpt_step_callback
from concept_based_learning.training.skill_bootstrap import (
    map_over_skills,
    resolve_skill_chunk_steps,
    skill_network_q_max,
)
from concept_based_learning.training.input_debug import (
    input_debug_enabled,
    maybe_jax_check_probe_inputs,
    maybe_jax_check_rollout_transitions,
    maybe_jax_check_skill_minibatch,
)
from concept_based_learning.training.episode_stats import (
    episodic_log_metrics,
    init_full_game_episode_stats,
    init_per_env_episode_stats,
    is_game_over,
    logging_env_reward,
    remaining_lives_from_semantic_state,
    update_full_game_episode_stats,
    update_per_env_episode_stats,
)
from concept_based_learning.training.logging_metrics import (
    flag_combination_occupancy_metrics,
    info_log_metrics,
    metrics_to_wandb_log_dict,
    player_progress_metrics,
    reset_flag_combo_cumulative,
    resolve_skill_names,
    selected_skill_outcome_metrics,
)


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
    ego_mapper_noop_action=0,
    ego_mapper_fire_action=1,
):
    del test_env, renderer

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0, (
        "NUM_MINIBATCHES must divide NUM_STEPS * NUM_ENVS"
    )
    reset_flag_combo_cumulative()
    if bool(config.get("USE_DESTROY_ONLY_EGO_EXTRAS", False)):
        raise NotImplementedError(
            "USE_DESTROY_ONLY_EGO_EXTRAS is skills_only for now; tandem still "
            "vmaps a shared ego width. Train skills_only or turn the flag off."
        )

    num_skills = config["NUM_SKILLS"]
    max_objects = config["MAX_OBJECTS"]
    ego_action_dim = config["EGO_ACTION_DIM"]
    use_legal_ego_mask = bool(config.get("EGO_LEGAL_ACTION_MASK", False))
    legal_ego_mask_fn = make_realized_ego_mask_fn(
        action_mapper,
        noop_action=int(ego_mapper_noop_action),
        fire_action=int(ego_mapper_fire_action),
        ego_action_dim=int(ego_action_dim),
    )
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
    _skill_chunk_configured = int(config.get("SKILL_BOOTSTRAP_CHUNK_STEPS", 0))
    skill_gamma = float(config.get("SKILL_GAMMA", config["GAMMA"]))
    symbolic_collect_enabled = bool(config.get("SYMBOLIC_COLLECT_FORCE_UP", False))
    symbolic_collect_skill_idx = int(config.get("COLLECT_SKILL_INDEX", 1))
    symbolic_collect_up_action = int(config.get("SYMBOLIC_COLLECT_UP_ACTION", 1))
    skill_input_mode = str(config.get("SKILL_INPUT_MODE", "ego")).lower().strip()
    use_pixel_skill = skill_input_mode in PIXEL_SKILL_INPUT_MODES
    use_object_skill = skill_input_mode in OBJECT_CONDITIONED_SKILL_INPUT_MODES
    skill_bootstrap_chunk_steps = resolve_skill_chunk_steps(
        num_steps=int(config["NUM_STEPS"]),
        num_envs=int(config["NUM_ENVS"]),
        configured=_skill_chunk_configured,
        use_pixel=use_pixel_skill,
    )
    if use_pixel_skill:
        print(
            f"pixel skill CNN: chunk_steps={skill_bootstrap_chunk_steps} "
            f"(~{skill_bootstrap_chunk_steps * int(config['NUM_ENVS'])} frames/chunk); "
            "skill heads sequential (not vmapped)",
            flush=True,
        )
    meta_input_mode = str(config.get("META_INPUT_MODE", "flat")).lower().strip()
    debug_inputs = input_debug_enabled(config)
    skill_embed_dim = int(config.get("SKILL_EMBED_DIM", 256))
    skill_pixel_target_embed_dim = int(config.get("SKILL_PIXEL_TARGET_EMBED_DIM", 16))
    skill_pixel_channels = config.get("SKILL_PIXEL_CNN_CHANNELS", (32, 64))
    if isinstance(skill_pixel_channels, list):
        skill_pixel_channels = tuple(skill_pixel_channels)
    skill_pixel_kernel = config.get("SKILL_PIXEL_CNN_KERNEL_SIZE", (5, 5))
    if isinstance(skill_pixel_kernel, list):
        skill_pixel_kernel = tuple(skill_pixel_kernel)
    skill_lambda = float(config.get("SKILL_LAMBDA", config["LAMBDA"]))
    skill_reward_lambda = float(config.get("SKILL_REWARD_LAMBDA", 0.0))
    # Default True preserves historical tandem TD targets (DQN/PQN Atari sign-clip).
    # Set CLIP_REWARD: false to train on raw env returns (also disables wrapper clip).
    clip_env_reward = bool(config.get("CLIP_REWARD", True))
    skill_target_cut = normalize_skill_target_cut(
        config.get("SKILL_TARGET_CUT", DEFAULT_SKILL_TARGET_CUT)
    )
    use_target_persistence = meta_target_persist_enabled(config)
    use_q_stickiness = meta_q_stickiness_enabled(config)
    use_meta_target_overrides = use_target_persistence or use_q_stickiness

    log_freeway_car_hits = str(config.get("ENV_NAME", "")).lower() == "freeway"
    freeway_env_steps_per_update = config["NUM_STEPS"] * config["NUM_ENVS"]
    skill_names = resolve_skill_names(config, num_skills)

    def label_to_skill_idx(label):
        return _label_to_skill_idx(label, num_skills)

    def make_linear_schedule(start, finish, decay_cfg):
        return optax.linear_schedule(
            start, finish, eps_decay_transition_steps(decay_cfg, config["NUM_UPDATES"])
        )

    skill_eps_schedule = make_linear_schedule(
        config.get("SKILL_EPS_START", config.get("EPS_START", DEFAULT_SKILL_EPS_START)),
        config.get("SKILL_EPS_FINISH", config.get("EPS_FINISH", DEFAULT_SKILL_EPS_FINISH)),
        config.get("SKILL_EPS_DECAY", config.get("EPS_DECAY", DEFAULT_SKILL_EPS_DECAY)),
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
        input_mode=skill_input_mode,
        max_objects=max_objects,
        embed_dim=skill_embed_dim,
        pixel_cnn_channels=tuple(int(c) for c in skill_pixel_channels),
        pixel_cnn_kernel_size=tuple(int(k) for k in skill_pixel_kernel),
        pixel_target_embed_dim=skill_pixel_target_embed_dim,
        pixel_paint_target_outline=bool(
            config.get("SKILL_PIXEL_PAINT_TARGET_OUTLINE", True)
        ),
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

    def _skill_inputs(raw_obs, objects, target_idxs):
        if skill_input_mode in ("ego", "ego_embed"):
            return jax.vmap(get_ego_obs)(raw_obs, objects, target_idxs)
        if skill_input_mode == "oc_embed":
            return jax.vmap(flatten_raw_observation)(raw_obs)
        if skill_input_mode in PIXEL_SKILL_INPUT_MODES:
            return jax.vmap(get_ego_obs)(raw_obs, objects, target_idxs)
        raise ValueError(
            f"Unknown SKILL_INPUT_MODE={skill_input_mode!r}. "
            "Expected one of: ego, ego_embed, oc_embed, pixel_embed, pixel_embed_cond."
        )

    def _skill_input_single(raw_obs, objects, target_idx):
        if skill_input_mode in ("ego", "ego_embed"):
            return get_ego_obs(raw_obs, objects, target_idx)
        if skill_input_mode == "oc_embed":
            return flatten_raw_observation(raw_obs)
        if skill_input_mode in PIXEL_SKILL_INPUT_MODES:
            return get_ego_obs(raw_obs, objects, target_idx)
        raise ValueError(
            f"Unknown SKILL_INPUT_MODE={skill_input_mode!r}. "
            "Expected one of: ego, ego_embed, oc_embed, pixel_embed, pixel_embed_cond."
        )

    def preprocess_transition(x, rng_perm):
        num_steps, num_envs = x.shape[:2]
        rest_shape = x.shape[2:]
        x = x.reshape(num_steps * num_envs, *rest_shape)
        x = jax.random.permutation(rng_perm, x)
        x = x.reshape(config["NUM_MINIBATCHES"], -1, *rest_shape)
        return x

    def _unpack_runner_state(runner_state):
        (
            meta_state,
            skill_states,
            expl_state,
            running_returns,
            returned_episode_returns,
            returned_episode_lengths,
            episode_lengths,
            full_game_running,
            returned_game_returns,
            returned_game_lengths,
            game_lengths,
            rng,
        ) = runner_state
        collect_carry = (
            expl_state,
            running_returns,
            returned_episode_returns,
            returned_episode_lengths,
            episode_lengths,
            full_game_running,
            returned_game_returns,
            returned_game_lengths,
            game_lengths,
        )
        return meta_state, skill_states, collect_carry, rng

    def _pack_runner_state(meta_state, skill_states, collect_carry, rng):
        (
            expl_state,
            running_returns,
            returned_episode_returns,
            returned_episode_lengths,
            episode_lengths,
            full_game_running,
            returned_game_returns,
            returned_game_lengths,
            game_lengths,
        ) = collect_carry
        return (
            meta_state,
            skill_states,
            expl_state,
            running_returns,
            returned_episode_returns,
            returned_episode_lengths,
            episode_lengths,
            full_game_running,
            returned_game_returns,
            returned_game_lengths,
            game_lengths,
            rng,
        )

    def init_runner(rng, init_meta_params=None, init_skill_params=None):
        """R0 — initialize meta/skill train states and env carry for one seed."""
        lr_skill = skill_learning_rate_schedule(config, config["NUM_UPDATES"])
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
        probe_target = jnp.array(0, dtype=jnp.int32)
        probe_ego = _skill_inputs(
            jax.tree_util.tree_map(
                lambda x: jnp.expand_dims(x, axis=0), probe_obs.raw_observation
            ),
            jax.tree_util.tree_map(
                lambda x: jnp.expand_dims(x, axis=0), probe_obs.objects
            ),
            jnp.expand_dims(probe_target, axis=0),
        )[0]
        probe_pixel = (
            probe_obs.pixel_observation
            if probe_obs.pixel_observation is not None
            else probe_obs.raw_observation
        )
        if debug_inputs:
            maybe_jax_check_probe_inputs(
                config,
                probe_ego=probe_ego,
                probe_pixel=probe_pixel,
                probe_target=probe_target,
                meta_pixel=probe_obs.pixel_observation,
            )

        meta_vars = meta_network.init(
            rng,
            probe_obs.raw_observation,
            probe_obs.objects,
            train=False,
            pixel_obs=probe_obs.pixel_observation,
        )
        meta_state = CustomTrainState.create(
            apply_fn=meta_network.apply,
            params=meta_vars["params"] if init_meta_params is None else init_meta_params,
            batch_stats={},
            tx=tx_meta,
        )

        def create_skill_state(rng_key, init_p):
            vars = skill_network.init(
                rng_key,
                probe_ego,
                train=False,
                target_idx=probe_target,
                pixel_obs=probe_pixel,
                objects=probe_obs.objects,
            )
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

        rng, _rng = jax.random.split(rng)
        init_expl = jax.vmap(env.reset)(jax.random.split(_rng, config["NUM_ENVS"]))
        init_running_returns = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.float32)
        returned_episode_returns, returned_episode_lengths, episode_lengths = (
            init_per_env_episode_stats(config["NUM_ENVS"])
        )
        (
            full_game_running,
            returned_game_returns,
            returned_game_lengths,
            game_lengths,
        ) = init_full_game_episode_stats(config["NUM_ENVS"])
        return _pack_runner_state(
            meta_state,
            skill_states,
            (
                init_expl,
                init_running_returns,
                returned_episode_returns,
                returned_episode_lengths,
                episode_lengths,
                full_game_running,
                returned_game_returns,
                returned_game_lengths,
                game_lengths,
            ),
            rng,
        )

    def collect_rollout(meta_state, skill_states, collect_carry, rng):
        """R1 — env rollout; returns updated carry, transitions, persist carry, car_hits, infos."""
        (
            expl_state,
            running_returns,
            returned_episode_returns,
            returned_episode_lengths,
            episode_lengths,
            full_game_running,
            returned_game_returns,
            returned_game_lengths,
            game_lengths,
        ) = collect_carry

        def _step_env(carry, _t):
            (
                last_obs,
                env_state,
                running_returns,
                rng,
                last_targets,
                car_hits_sum,
                persist_steps_remaining,
                suspended_mission_idxs,
                returned_episode_returns,
                returned_episode_lengths,
                episode_lengths,
                full_game_running,
                returned_game_returns,
                returned_game_lengths,
                game_lengths,
            ) = carry
            rng, rng_meta, rng_skill = jax.random.split(rng, 3)

            meta_target_locked = compute_meta_target_locked(
                last_targets,
                suspended_mission_idxs,
                last_obs.objects,
                config,
                no_target_idx=no_target_idx,
                max_objects=max_objects,
                persist_steps_remaining=persist_steps_remaining,
            )
            pixel_inputs = (
                last_obs.pixel_observation
                if last_obs.pixel_observation is not None
                else last_obs.raw_observation
            )
            meta_q_vals = jax.vmap(
                lambda obs, objs, pix, locked: meta_network.apply(
                    {"params": meta_state.params},
                    obs,
                    objs,
                    train=False,
                    target_locked=locked,
                    pixel_obs=pix,
                )
            )(
                last_obs.raw_observation,
                last_obs.objects,
                pixel_inputs,
                meta_target_locked,
            )
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
            if use_meta_target_overrides:
                def _apply_meta_target_overrides(proposed):
                    return commit_meta_targets(
                        proposed,
                        last_targets,
                        last_obs.objects,
                        meta_q_vals,
                        active_masks,
                        config,
                        no_target_idx=no_target_idx,
                        max_objects=max_objects,
                        persist_steps_remaining=persist_steps_remaining,
                        suspended_mission_idxs=suspended_mission_idxs,
                    )

                (
                    target_obj_idxs,
                    persist_steps_remaining,
                    suspended_mission_idxs,
                ) = jax.lax.cond(
                    use_heuristic_warmup,
                    lambda proposed: (
                        proposed,
                        persist_steps_remaining,
                        suspended_mission_idxs,
                    ),
                    _apply_meta_target_overrides,
                    target_obj_idxs,
                )
            safe_target_idxs = jnp.where(
                target_obj_idxs == no_target_idx, 0, target_obj_idxs
            )

            raw_labels = jax.vmap(lambda labels, idx: labels[-1, idx])(
                last_obs.objects.semantic_label, safe_target_idxs
            )
            active_skills = label_to_skill_idx(raw_labels)
            ego_obses = _skill_inputs(
                last_obs.raw_observation, last_obs.objects, safe_target_idxs
            )
            pixel_obses = (
                last_obs.pixel_observation
                if last_obs.pixel_observation is not None
                else last_obs.raw_observation
            )

            def compute_skill_actions(state, skill_idx):
                q_vals = skill_network.apply(
                    {"params": state.params, "batch_stats": state.batch_stats},
                    ego_obses,
                    train=False,
                    target_idx=safe_target_idxs,
                    pixel_obs=pixel_obses if use_pixel_skill else None,
                    objects=last_obs.objects if use_object_skill else None,
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
                if use_legal_ego_mask:
                    dyn = jax.vmap(legal_ego_mask_fn)(
                        last_obs.raw_observation,
                        safe_target_idxs,
                        last_obs.objects,
                    )
                    mask = jnp.logical_and(mask, dyn)

                ego_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_skill, config["NUM_ENVS"]),
                    q_vals,
                    eps_skill,
                    mask=mask,
                )
                return ego_action, q_vals

            all_ego_actions, all_skill_q_vals = map_over_skills(
                compute_skill_actions,
                skill_states,
                jnp.arange(num_skills, dtype=jnp.int32),
                sequential=use_pixel_skill,
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

            step_obs, step_env_state, env_reward, new_done, info = vmap_step(
                env_state, global_actions
            )
            prev_lives = jax.vmap(remaining_lives_from_semantic_state)(env_state)
            lives_after_step = jax.vmap(remaining_lives_from_semantic_state)(
                step_env_state
            )
            life_lost = extract_life_lost(env_state, step_env_state)
            game_over = is_game_over(new_done, life_lost, prev_lives, lives_after_step)
            reset_mask = jnp.logical_or(
                game_over,
                jnp.logical_and(new_done, life_lost <= 0),
            )
            car_hits_sum = car_hits_sum + jnp.sum(life_lost > 0)
            done_due_to_life_lost = jnp.logical_and(new_done, life_lost > 0)
            new_obs = step_obs
            new_env_state = step_env_state

            # Prefer info["env_reward"] so CLIP_REWARD does not distort logged returns.
            log_rewards = logging_env_reward(info, env_reward)
            (
                running_returns,
                returned_episode_returns,
                returned_episode_lengths,
                episode_lengths,
            ) = update_per_env_episode_stats(
                running_returns,
                returned_episode_returns,
                returned_episode_lengths,
                episode_lengths,
                log_rewards,
                new_done,
            )
            (
                full_game_running,
                returned_game_returns,
                returned_game_lengths,
                game_lengths,
            ) = update_full_game_episode_stats(
                full_game_running,
                returned_game_returns,
                returned_game_lengths,
                game_lengths,
                log_rewards,
                game_over,
            )

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
            skill_rewards, skill_done = project_skill_outcomes(
                skill_rewards, skill_done, num_skills
            )
            if meta_noop_enabled:
                skill_rewards = jnp.where(
                    meta_noop[:, jnp.newaxis], 0.0, skill_rewards
                )
            if meta_fire_enabled:
                skill_rewards = jnp.where(
                    meta_fire[:, jnp.newaxis], 0.0, skill_rewards
                )

            base_env_reward = (
                jnp.sign(env_reward) if clip_env_reward else env_reward
            )
            skill_rewards_for_training = (
                skill_reward_lambda * base_env_reward[:, jnp.newaxis]
                + (1.0 - skill_reward_lambda) * skill_rewards
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
                new_obs, reset_obs, reset_mask
            )
            next_state_for_rollout = merge_reset_where_done(
                new_env_state, reset_state, reset_mask
            )
            target_obj_idxs = jnp.where(
                new_done,
                jnp.full_like(target_obj_idxs, no_target_idx),
                target_obj_idxs,
            )
            persist_steps_remaining = jnp.where(
                new_done, jnp.zeros_like(persist_steps_remaining), persist_steps_remaining
            )
            suspended_mission_idxs = jnp.where(
                new_done,
                jnp.full_like(suspended_mission_idxs, no_target_idx),
                suspended_mission_idxs,
            )
            next_active_masks = jax.vmap(
                lambda objs, raw: build_meta_target_mask(
                    objs, raw, config, hook=meta_mask_hook
                )
            )(next_obs_for_rollout.objects, next_obs_for_rollout.raw_observation)
            next_ego_obses = _skill_inputs(
                next_obs_for_rollout.raw_observation,
                next_obs_for_rollout.objects,
                safe_target_idxs,
            )

            if relabel_all_objects:
                obj_ids = jnp.arange(max_objects, dtype=jnp.int32)
                ego_obs_all_full = jax.vmap(
                    lambda raw, objs: jax.vmap(
                        lambda obj_idx: _skill_input_single(raw, objs, obj_idx)
                    )(obj_ids)
                )(last_obs.raw_observation, last_obs.objects)
                next_ego_obs_all_full = jax.vmap(
                    lambda raw, objs: jax.vmap(
                        lambda obj_idx: _skill_input_single(raw, objs, obj_idx)
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
                skill_rewards_all_full, skill_done_all_full = project_skill_outcomes(
                    skill_rewards_all_full, skill_done_all_full, num_skills
                )
                skill_rewards_all_full = (
                    skill_reward_lambda * base_env_reward[:, jnp.newaxis, jnp.newaxis]
                    + (1.0 - skill_reward_lambda) * skill_rewards_all_full
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
                    object_idx_all = gather_idx
                else:
                    ego_obs_all = ego_obs_all_full
                    next_ego_obs_all = next_ego_obs_all_full
                    ego_action_all = ego_action_all_full
                    skill_rewards_all = skill_rewards_all_full
                    skill_done_all = skill_done_all_full
                    relabel_mask = relabel_mask_full
                    object_idx_all = jnp.broadcast_to(
                        obj_ids[None, :],
                        (config["NUM_ENVS"], max_objects),
                    )
            else:
                base_valid = target_obj_idxs != no_target_idx
                if meta_noop_enabled:
                    base_valid = jnp.logical_and(base_valid, jnp.logical_not(meta_noop))
                if meta_fire_enabled:
                    base_valid = jnp.logical_and(base_valid, jnp.logical_not(meta_fire))
                ego_obs_all = ego_obses[:, jnp.newaxis, :]
                next_ego_obs_all = next_ego_obses[:, jnp.newaxis, :]
                ego_action_all = chosen_ego_actions[:, jnp.newaxis]
                skill_rewards_all = skill_rewards_for_training[:, jnp.newaxis, :]
                skill_done_all = skill_done[:, jnp.newaxis, :]
                relabel_mask = base_valid[:, jnp.newaxis]
                object_idx_all = safe_target_idxs[:, jnp.newaxis]

            transition = TandemTransition(
                done=new_done,
                obs=last_obs.raw_observation,
                objects=last_obs.objects,
                pixel_obs=(
                    last_obs.pixel_observation
                    if last_obs.pixel_observation is not None
                    else last_obs.raw_observation
                ),
                next_obs=next_obs_for_rollout.raw_observation,
                next_objects=next_obs_for_rollout.objects,
                next_pixel_obs=(
                    next_obs_for_rollout.pixel_observation
                    if next_obs_for_rollout.pixel_observation is not None
                    else next_obs_for_rollout.raw_observation
                ),
                target_idx=target_obj_idxs,
                meta_q_val=meta_q_vals,
                meta_reward=meta_reward,
                active_mask=active_masks,
                next_active_mask=next_active_masks,
                meta_target_locked=meta_target_locked,
                ego_obs=ego_obses,
                next_ego_obs=next_ego_obses,
                ego_action=chosen_ego_actions,
                active_skill=active_skills,
                skill_q_val=all_skill_q_vals,
                skill_rewards=skill_rewards_for_training,
                skill_done=skill_done,
                ego_obs_all=ego_obs_all,
                next_ego_obs_all=next_ego_obs_all,
                ego_action_all=ego_action_all,
                skill_rewards_all=skill_rewards_all,
                skill_done_all=skill_done_all,
                relabel_mask=relabel_mask,
                object_idx_all=object_idx_all,
                env_reward=base_env_reward,
            )
            return (
                next_obs_for_rollout,
                next_state_for_rollout,
                running_returns,
                rng,
                target_obj_idxs,
                car_hits_sum,
                persist_steps_remaining,
                suspended_mission_idxs,
                returned_episode_returns,
                returned_episode_lengths,
                episode_lengths,
                full_game_running,
                returned_game_returns,
                returned_game_lengths,
                game_lengths,
            ), (transition, info)


        rng, _rng = jax.random.split(rng)
        init_targets = jnp.full(
            (config["NUM_ENVS"],), no_target_idx, dtype=jnp.int32
        )
        init_persist_steps = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
        init_suspended_mission = jnp.full(
            (config["NUM_ENVS"],), no_target_idx, dtype=jnp.int32
        )
        init_car_hits_sum = jnp.array(0.0, dtype=jnp.float32)
        carry_final, (transitions, infos) = jax.lax.scan(
            _step_env,
            (
                *expl_state,
                running_returns,
                _rng,
                init_targets,
                init_car_hits_sum,
                init_persist_steps,
                init_suspended_mission,
                returned_episode_returns,
                returned_episode_lengths,
                episode_lengths,
                full_game_running,
                returned_game_returns,
                returned_game_lengths,
                game_lengths,
            ),
            None,
            config["NUM_STEPS"],
        )
        if debug_inputs:
            maybe_jax_check_rollout_transitions(
                config,
                transitions=transitions,
                update_idx=meta_state.n_updates,
                max_objects=max_objects,
                no_target_idx=no_target_idx,
                skill_input_mode=skill_input_mode,
                meta_input_mode=meta_input_mode,
                relabel_all_objects=relabel_all_objects,
            )
        (
            last_obs,
            env_state,
            running_returns,
            rng,
            _last_targets,
            car_hits_sum,
            _persist_steps_remaining,
            _suspended_mission_idxs,
            returned_episode_returns,
            returned_episode_lengths,
            episode_lengths,
            full_game_running,
            returned_game_returns,
            returned_game_lengths,
            game_lengths,
        ) = carry_final
        expl_state = (last_obs, env_state)

        persist_carry = (
            _last_targets,
            _persist_steps_remaining,
            _suspended_mission_idxs,
        )
        new_collect_carry = (
            expl_state,
            running_returns,
            returned_episode_returns,
            returned_episode_lengths,
            episode_lengths,
            full_game_running,
            returned_game_returns,
            returned_game_lengths,
            game_lengths,
        )
        return new_collect_carry, transitions, persist_carry, car_hits_sum, infos

    def meta_bootstrap_and_train(meta_state, transitions, rng, persist_carry):
        """R2 — meta λ-bootstrap + PQN (transitions read-only)."""
        (
            _last_targets,
            _persist_steps_remaining,
            _suspended_mission_idxs,
        ) = persist_carry

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
        last_next_pixels = jax.tree_util.tree_map(
            lambda x: x[-1], transitions.next_pixel_obs
        )
        last_meta_target_locked = compute_meta_target_locked(
            _last_targets,
            _suspended_mission_idxs,
            last_next_objects,
            config,
            no_target_idx=no_target_idx,
            max_objects=max_objects,
            persist_steps_remaining=_persist_steps_remaining,
        )
        last_meta_q = jax.vmap(
            lambda obs, objs, pix, locked: meta_network.apply(
                {"params": meta_state.params},
                obs,
                objs,
                train=False,
                target_locked=locked,
                pixel_obs=pix,
            )
        )(
            last_next_obs,
            last_next_objects,
            last_next_pixels,
            last_meta_target_locked,
        )
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
                "pixel_obs": transitions.pixel_obs,
                "target_idx": transitions.target_idx,
                "active_mask": transitions.active_mask,
                "meta_target_locked": transitions.meta_target_locked,
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
                        lambda obs, objs, pix, locked: meta_network.apply(
                            {"params": params},
                            obs,
                            objs,
                            train=True,
                            target_locked=locked,
                            pixel_obs=pix,
                        )
                    )(
                        minibatch["obs"],
                        minibatch["objects"],
                        minibatch["pixel_obs"],
                        minibatch["meta_target_locked"],
                    )
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

        return meta_state, rng, meta_loss, meta_qvals, skip_meta_update

    def skills_bootstrap_and_train(skill_states, transitions, meta_n_updates, rng):
        """R3 — batched skill bootstrap + PQN; meta_n_updates is post-R2."""
        def _update_single_skill(s_state, skill_idx, rng_s):
            s_state = s_state.replace(
                timesteps=s_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            use_one_step_td = skill_bootstrap == "one_step"

            if relabel_all_objects:
                # Match train_skills: per-object λ targets with per-slot switch cuts.
                target_idx_all = transitions.object_idx_all
                next_pixel_obs_all = (
                    transitions.next_pixel_obs if use_pixel_skill else None
                )
                next_objects_all = (
                    transitions.next_objects if use_object_skill else None
                )
                next_q_max_all = skill_network_q_max(
                    skill_network.apply,
                    s_state.params,
                    s_state.batch_stats,
                    transitions.next_ego_obs_all,
                    target_idx=target_idx_all,
                    pixel_obs=next_pixel_obs_all,
                    objects=next_objects_all,
                    chunk_steps=skill_bootstrap_chunk_steps,
                    train=False,
                    chunk=True,
                )

                last_step_done_all = jnp.logical_or(
                    transitions.skill_done_all[-1, :, :, skill_idx],
                    transitions.done[-1, :, jnp.newaxis],
                ).astype(jnp.float32)
                last_next_q = next_q_max_all[-1] * (1.0 - last_step_done_all)
                last_reward_all = transitions.skill_rewards_all[-1, :, :, skill_idx]
                last_lambda = last_reward_all + skill_gamma * last_next_q

                target_switch_cut = skill_target_switch_cut(
                    transitions.target_idx,
                    transitions.next_ego_obs_all.shape[2],
                    skill_target_cut,
                )

                def _get_target_all(carry, xs):
                    lambda_ret, next_q = carry
                    (
                        step_reward,
                        step_done_skill,
                        step_done_ep,
                        step_target_cut,
                        q_next_step,
                    ) = xs
                    step_done = jnp.logical_or(
                        step_done_skill, step_done_ep[:, jnp.newaxis]
                    )
                    step_done = jnp.logical_or(
                        step_done, step_target_cut.astype(jnp.bool_)
                    ).astype(jnp.float32)
                    target_bootstrap = (
                        step_reward + skill_gamma * (1.0 - step_done) * next_q
                    )
                    delta = lambda_ret - next_q
                    lambda_ret = target_bootstrap + skill_gamma * skill_lambda * delta
                    lambda_ret = (1.0 - step_done) * lambda_ret + step_done * step_reward
                    return (lambda_ret, q_next_step), lambda_ret

                xs_all = (
                    transitions.skill_rewards_all[:-1, :, :, skill_idx],
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
                skill_targets = jnp.concatenate(
                    (targets_all, last_lambda[jnp.newaxis]), axis=0
                )
            elif use_one_step_td:
                next_q_max_single = skill_network_q_max(
                    skill_network.apply,
                    s_state.params,
                    s_state.batch_stats,
                    transitions.next_ego_obs_all,
                    target_idx=transitions.object_idx_all,
                    pixel_obs=transitions.next_pixel_obs if use_pixel_skill else None,
                    objects=transitions.next_objects if use_object_skill else None,
                    chunk_steps=skill_bootstrap_chunk_steps,
                    train=False,
                    chunk=use_pixel_skill,
                )
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
                skill_targets = (
                    transitions.skill_rewards[..., skill_idx]
                    + skill_gamma * (1.0 - step_done_single) * next_q_max_single.squeeze(-1)
                )[..., jnp.newaxis]
            else:
                next_q_max_single = skill_network_q_max(
                    skill_network.apply,
                    s_state.params,
                    s_state.batch_stats,
                    transitions.next_ego_obs_all,
                    target_idx=transitions.object_idx_all,
                    pixel_obs=transitions.next_pixel_obs if use_pixel_skill else None,
                    objects=transitions.next_objects if use_object_skill else None,
                    chunk_steps=skill_bootstrap_chunk_steps,
                    train=False,
                    chunk=use_pixel_skill,
                )

                last_step_done = jnp.logical_or(
                    transitions.skill_done[-1, :, skill_idx],
                    transitions.done[-1],
                ).astype(jnp.float32)
                last_step_reward = transitions.skill_rewards[-1, :, skill_idx]
                last_next_q = next_q_max_single[-1].squeeze(-1)
                lambda_returns = (
                    last_step_reward + skill_gamma * (1.0 - last_step_done) * last_next_q
                )

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
                    lambda_ret = target_bootstrap + skill_gamma * skill_lambda * delta
                    lambda_ret = (1.0 - step_done) * lambda_ret + step_done * step_reward
                    return (lambda_ret, q_next), lambda_ret

                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_next_q),
                    (
                        transitions.skill_rewards[:-1, :, skill_idx],
                        transitions.skill_done[:-1, :, skill_idx],
                        transitions.done[:-1],
                        target_switch_cut_env,
                        next_q_max_single[:-1].squeeze(-1),
                    ),
                    reverse=True,
                )
                skill_targets = jnp.concatenate(
                    (targets, lambda_returns[jnp.newaxis]), axis=0
                )[..., jnp.newaxis]

            def _learn_epoch(carry, _):
                s_state_epoch, rng_epoch = carry
                rng_epoch, rng_perm = jax.random.split(rng_epoch)

                train_batch = {
                    "relabel_mask": transitions.relabel_mask,
                    "ego_obs_all": transitions.ego_obs_all,
                    "ego_action_all": transitions.ego_action_all,
                    "active_skill": transitions.active_skill,
                    "target_idx": transitions.target_idx,
                    "object_idx_all": transitions.object_idx_all,
                }
                if use_pixel_skill:
                    train_batch["pixel_obs"] = transitions.pixel_obs
                if use_object_skill:
                    train_batch["objects"] = transitions.objects
                minibatches = jax.tree.map(
                    lambda x: preprocess_transition(x, rng_perm),
                    train_batch,
                )
                targets_batch = preprocess_transition(skill_targets, rng_perm)

                def _learn_phase(carry_phase, minibatch_and_target):
                    s_state_phase, rng_phase = carry_phase
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        n_objects = minibatch["ego_obs_all"].shape[1]
                        mb_target_idx = minibatch["object_idx_all"]
                        mb_pixel_obs = (
                            minibatch["pixel_obs"] if use_pixel_skill else None
                        )
                        mb_objects = (
                            minibatch["objects"] if use_object_skill else None
                        )
                        if debug_inputs:
                            maybe_jax_check_skill_minibatch(
                                config,
                                update_idx=meta_n_updates,
                                ego_obs=minibatch["ego_obs_all"],
                                target_idx=mb_target_idx,
                                pixel_obs=mb_pixel_obs,
                                use_pixel_skill=use_pixel_skill,
                                site="skill/train_minibatch",
                            )
                        q_vals, updates = skill_network.apply(
                            {
                                "params": params,
                                "batch_stats": s_state_phase.batch_stats,
                            },
                            minibatch["ego_obs_all"],
                            train=True,
                            mutable=["batch_stats"],
                            target_idx=mb_target_idx,
                            pixel_obs=mb_pixel_obs,
                            objects=mb_objects,
                        )
                        chosen_q = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch["ego_action_all"], axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)
                        if config.get("OFF_POLICY", True):
                            valid = minibatch["relabel_mask"]
                        else:
                            if n_objects == 1:
                                on_policy = (
                                    minibatch["active_skill"][:, jnp.newaxis]
                                    == skill_idx
                                )
                            else:
                                chosen_obj_mask = (
                                    minibatch["object_idx_all"]
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
            meta_n_updates >= skill_freeze_after_updates,
        )

        rngs = jax.random.split(rng, num_skills)
        skill_states, skill_metrics = map_over_skills(
            _update_single_skill,
            skill_states,
            jnp.arange(num_skills),
            rngs,
            sequential=use_pixel_skill,
        )

        return skill_states, skill_metrics, skills_frozen

    def compute_update_metrics(
        transitions,
        meta_state,
        meta_loss,
        meta_qvals,
        skip_meta_update,
        skill_metrics,
        skills_frozen,
        car_hits_sum,
        returned_episode_returns,
        returned_episode_lengths,
        returned_game_returns,
        returned_game_lengths,
        infos,
    ):
        """R4 — pure metric tensors for one update (host logging outside JIT)."""
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
        metrics.update(
            episodic_log_metrics(
                returned_episode_returns,
                returned_episode_lengths,
                returned_game_returns=returned_game_returns,
                returned_game_lengths=returned_game_lengths,
            )
        )

        skill_valid = transitions.target_idx != no_target_idx
        if meta_noop_enabled:
            skill_valid = jnp.logical_and(
                skill_valid, transitions.target_idx != meta_noop_idx
            )
        if meta_fire_enabled:
            skill_valid = jnp.logical_and(
                skill_valid, transitions.target_idx != meta_fire_idx
            )
        metrics.update(
            selected_skill_outcome_metrics(
                transitions.active_skill,
                transitions.skill_done,
                transitions.skill_rewards,
                skill_names,
                valid_mask=skill_valid,
                env_reward=transitions.env_reward,
                skill_reward_lambda=skill_reward_lambda,
            )
        )
        if skill_input_mode in ("ego", "ego_embed") or skill_input_mode in PIXEL_SKILL_INPUT_MODES:
            metrics.update(
                flag_combination_occupancy_metrics(
                    transitions.ego_obs,
                    transitions.active_skill,
                    skill_names,
                    valid_mask=skill_valid,
                )
            )
        metrics.update(info_log_metrics(infos))
        metrics.update(player_progress_metrics(transitions.obs))

        if log_freeway_car_hits:
            metrics["freeway/car_hit_rate"] = (
                car_hits_sum / freeway_env_steps_per_update
            )
            metrics["freeway/car_hits_per_update"] = car_hits_sum

        return metrics

    collect_vmap = jax.vmap(collect_rollout, in_axes=(0, 0, 0, 0))
    meta_vmap = jax.vmap(meta_bootstrap_and_train, in_axes=(0, 0, 0, 0))
    skills_vmap = jax.vmap(skills_bootstrap_and_train, in_axes=(0, 0, 0, 0))
    metrics_vmap = jax.vmap(
        compute_update_metrics,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )

    collect_jit = jax.jit(
        collect_vmap,
        # Only donate mutable carry + rng; meta/skill states are read-only in R1 and
        # must stay alive for R2/R3 (donating them reuses buffers and deletes refs).
        donate_argnums=(2, 3),
    )
    meta_jit = jax.jit(
        meta_vmap,
        donate_argnums=(0, 2, 3),
    )
    skills_jit = jax.jit(
        skills_vmap,
        donate_argnums=(0, 2, 3),
    )
    init_fn = jax.jit(jax.vmap(init_runner, in_axes=(0, None, None)))
    metrics_jit = jax.jit(metrics_vmap)

    def compile_regions(
        rng,
        init_meta_params=None,
        init_skill_params=None,
        *,
        timings: dict[str, float] | None = None,
        benchmark_exec: bool = False,
    ):
        """Lower/compile all region jits once (uses init output shapes for collect/meta/skills).

        Donated region jits must not call ``lower`` after an exec that deleted inputs.
        We compile every region from init refs plus ``eval_shape`` stand-ins for cross-region
        outputs (no copy+exec probes). Optional ``benchmark_exec`` runs one training-style
        update after all regions are compiled (for smoke timing only).
        """
        def _mark(name: str, t0: float) -> None:
            if timings is not None:
                timings[name] = time.time() - t0

        t0 = time.time()
        init_fn.lower(rng, init_meta_params, init_skill_params).compile()
        _mark("init_fn", t0)

        t0 = time.time()
        runner_state = init_fn(rng, init_meta_params, init_skill_params)
        _mark("init_fn_exec", t0)
        meta_state, skill_states, collect_carry, rng_carry = _unpack_runner_state(
            runner_state
        )

        t0 = time.time()
        collect_jit.lower(meta_state, skill_states, collect_carry, rng_carry).compile()
        _mark("collect_jit", t0)

        new_collect, transitions, persist_carry, car_hits_sum, infos = jax.eval_shape(
            collect_vmap,
            meta_state,
            skill_states,
            collect_carry,
            rng_carry,
        )

        t0 = time.time()
        meta_jit.lower(meta_state, transitions, rng_carry, persist_carry).compile()
        _mark("meta_jit", t0)

        (
            meta_state_shape,
            _rng_shape,
            meta_loss,
            meta_qvals,
            skip_meta,
        ) = jax.eval_shape(
            meta_vmap,
            meta_state,
            transitions,
            rng_carry,
            persist_carry,
        )
        meta_n_updates = meta_state.n_updates

        t0 = time.time()
        skills_jit.lower(
            skill_states, transitions, meta_n_updates, rng_carry
        ).compile()
        _mark("skills_jit", t0)

        _, skill_metrics, skills_frozen = jax.eval_shape(
            skills_vmap,
            skill_states,
            transitions,
            meta_n_updates,
            rng_carry,
        )

        t0 = time.time()
        metrics_jit.lower(
            transitions,
            meta_state_shape,
            meta_loss,
            meta_qvals,
            skip_meta,
            skill_metrics,
            skills_frozen,
            car_hits_sum,
            new_collect[2],
            new_collect[3],
            new_collect[6],
            new_collect[7],
            infos,
        ).compile()
        _mark("metrics_jit", t0)

        if not benchmark_exec:
            return

        runner_state = init_fn(rng, init_meta_params, init_skill_params)
        meta_state, skill_states, collect_carry, rng_carry = _unpack_runner_state(
            runner_state
        )
        t0 = time.time()
        new_collect, transitions, persist_carry, car_hits_sum, infos = collect_jit(
            meta_state, skill_states, collect_carry, rng_carry
        )
        _mark("collect_jit_exec", t0)

        t0 = time.time()
        meta_state, rng_carry, meta_loss, meta_qvals, skip_meta = meta_jit(
            meta_state, transitions, rng_carry, persist_carry
        )
        _mark("meta_jit_exec", t0)

        meta_n_updates = meta_state.n_updates
        t0 = time.time()
        skill_states, skill_metrics, skills_frozen = skills_jit(
            skill_states, transitions, meta_n_updates, rng_carry
        )
        _mark("skills_jit_exec", t0)

        t0 = time.time()
        jax.block_until_ready(
            metrics_jit(
                transitions,
                meta_state,
                meta_loss,
                meta_qvals,
                skip_meta,
                skill_metrics,
                skills_frozen,
                car_hits_sum,
                new_collect[2],
                new_collect[3],
                new_collect[6],
                new_collect[7],
                infos,
            )
        )
        _mark("metrics_jit_exec", t0)

    def run_training(rng, init_meta_params=None, init_skill_params=None):
        """Python update loop: R1 → R2 → R3 → R4 (metrics/wandb/RTPT on host)."""
        runner_state = init_fn(rng, init_meta_params, init_skill_params)
        all_metrics = []
        for _ in range(config["NUM_UPDATES"]):
            meta_state, skill_states, collect_carry, rng_carry = _unpack_runner_state(
                runner_state
            )
            new_collect, transitions, persist_carry, car_hits_sum, infos = collect_jit(
                meta_state, skill_states, collect_carry, rng_carry
            )
            meta_state, rng_carry, meta_loss, meta_qvals, skip_meta = meta_jit(
                meta_state, transitions, rng_carry, persist_carry
            )
            meta_n_updates = meta_state.n_updates
            skill_states, skill_metrics, skills_frozen = skills_jit(
                skill_states, transitions, meta_n_updates, rng_carry
            )
            metrics = metrics_jit(
                transitions,
                meta_state,
                meta_loss,
                meta_qvals,
                skip_meta,
                skill_metrics,
                skills_frozen,
                car_hits_sum,
                new_collect[2],
                new_collect[3],
                new_collect[6],
                new_collect[7],
                infos,
            )
            all_metrics.append(metrics)

            if config.get("WANDB_MODE", "disabled") != "disabled":
                try:
                    import wandb

                    if wandb.run is not None:
                        log_dict = metrics_to_wandb_log_dict(metrics)
                        step = int(
                            log_dict.get(
                                "env/step", log_dict.get("global/update_steps", 0)
                            )
                        )
                        wandb.log(log_dict, step=step)
                except Exception:
                    pass

            if is_rtpt_enabled(config):
                rtpt_step_callback()

            runner_state = _pack_runner_state(
                meta_state, skill_states, new_collect, rng_carry
            )

        stacked_metrics = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0), *all_metrics
        )
        return {"runner_state": runner_state, "metrics": stacked_metrics}

    return {
        "init_fn": init_fn,
        "collect_jit": collect_jit,
        "meta_jit": meta_jit,
        "skills_jit": skills_jit,
        "metrics_jit": metrics_jit,
        "compile_regions": compile_regions,
        "run_training": run_training,
        "num_updates": config["NUM_UPDATES"],
    }

