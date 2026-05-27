import jax
import jax.numpy as jnp
import numpy as np

import optax

from concept_based_learning.observations.action_space import fire_ego_action_indices
from concept_based_learning.observations.egocentric import get_ego_obs
from concept_based_learning.networks.skill_network import QNetwork as SkillQNetwork, CustomTrainState
from concept_based_learning.networks.meta_factory import create_meta_network
from concept_based_learning.environments.datatypes import (
    TandemTransition,
    CompositionSlots,
)
from concept_based_learning.environments.meta_target_mask import (
    build_meta_target_mask,
    is_meta_noop_target,
    meta_noop_target_enabled,
    meta_noop_target_idx,
)
from concept_based_learning.environments.freeway.freeway_label import (
    _CAR_ZONE_LEFT,
    _CAR_ZONE_RIGHT,
)
from concept_based_learning.training.schedules import eps_decay_transition_steps
from concept_based_learning import composition as comp


def make_tandem_train(
    config,
    env,
    test_env,
    renderer=None,
    *,
    no_target_idx,
    get_heuristic_target,
    action_mapper,
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
    _raw_skill_names = config.get("SKILL_NAMES")
    if _raw_skill_names is None or len(_raw_skill_names) != num_skills:
        skill_names = [f"skill_{i}" for i in range(num_skills)]
    else:
        skill_names = [str(n) for n in _raw_skill_names]
    avoid_slot_idx = (
        skill_names.index("AVOID")
        if "AVOID" in skill_names and config.get("COMPOSITION_SLOT_MODE", "per_label") == "per_label"
        else None
    )
    skill_warmup_updates = int(config.get("SKILL_WARMUP_UPDATES", 0))
    meta_learn_during_warmup = bool(config.get("META_LEARN_DURING_WARMUP", False))
    meta_num_epochs = int(config.get("META_NUM_EPOCHS", config["NUM_EPOCHS"]))
    skill_freeze_after_updates = int(config.get("SKILL_FREEZE_AFTER_UPDATES", 0))
    meta_noop_enabled = meta_noop_target_enabled(config)
    meta_noop_idx = meta_noop_target_idx(config)

    # Game action space size (for Q-decomposition scatter).
    try:
        num_game_actions = int(env.action_space().n)
    except Exception:
        num_game_actions = int(config.get("NUM_GAME_ACTIONS", 18))

    # Q-decomposition composition config. K=1 keeps current behavior bit-equivalent.
    COMPOSITION_K = int(config.get("COMPOSITION_K", 1))
    COMPOSITION_TAU = float(config.get("COMPOSITION_TAU", 1.0))
    COMPOSITION_SLOT_MODE = str(config.get("COMPOSITION_SLOT_MODE", "per_label"))
    COMPOSITION_NORMALIZE = str(config.get("COMPOSITION_NORMALIZE_SKILL_Q", "zscore"))
    COMPOSITION_WEIGHT_THRESHOLD = float(config.get("COMPOSITION_SLOT_WEIGHT_THRESHOLD", 0.0))
    if COMPOSITION_K > 1 and COMPOSITION_SLOT_MODE == "per_label":
        assert COMPOSITION_K == num_skills, (
            f"per_label slot mode requires COMPOSITION_K == NUM_SKILLS "
            f"(got K={COMPOSITION_K}, NUM_SKILLS={num_skills})"
        )
    composition_enabled = COMPOSITION_K > 1
    META_LOSS_MODE = str(config.get("COMPOSITION_META_LOSS", "hybrid"))
    if META_LOSS_MODE not in ("hybrid", "soft_q"):
        raise ValueError(
            f"COMPOSITION_META_LOSS must be 'hybrid' or 'soft_q' (got {META_LOSS_MODE!r})"
        )

    def _meta_state_value(meta_q, mask):
        """V(s) over active objects. 'hybrid' = max; 'soft_q' = τ logΣexp(Q/τ).

        meta_q : [..., max_objects]
        mask   : [..., max_objects] bool
        Returns V : [...]
        """
        if META_LOSS_MODE == "hybrid":
            return jnp.max(jnp.where(mask, meta_q, -jnp.inf), axis=-1)
        # soft_q: V = τ log Σ exp(Q/τ) over masked entries. Use stable form.
        masked_q = jnp.where(mask, meta_q / COMPOSITION_TAU, -jnp.inf)
        m = jnp.max(masked_q, axis=-1, keepdims=True)
        # Avoid -inf - -inf = nan when no entry is active; mask-out later.
        m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
        exp_ = jnp.where(mask, jnp.exp(masked_q - m_safe), 0.0)
        z = jnp.sum(exp_, axis=-1)
        # If no active entry, return 0 (caller separately gates with any_active).
        log_z = jnp.where(z > 0, jnp.log(z), 0.0)
        return COMPOSITION_TAU * (log_z + jnp.squeeze(m_safe, axis=-1))

    def label_to_skill_idx(label):
        return jnp.clip(label - 1, 0, num_skills - 1)

    def make_linear_schedule(start, finish, decay_cfg):
        return optax.linear_schedule(
            start, finish, eps_decay_transition_steps(decay_cfg, config["NUM_UPDATES"])
        )

    skill_eps_schedule = make_linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"],
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

                def compute_skill_actions(state, skill_idx, ego_obs_batch):
                    q_vals = skill_network.apply(
                        {"params": state.params, "batch_stats": state.batch_stats},
                        ego_obs_batch,
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

                all_ego_actions, all_skill_q_vals = jax.vmap(
                    compute_skill_actions, in_axes=(0, 0, None)
                )(skill_states, jnp.arange(num_skills, dtype=jnp.int32), ego_obses)

                meta_noop = is_meta_noop_target(
                    target_obj_idxs,
                    config,
                    noop_idx=meta_noop_idx,
                    enabled=meta_noop_enabled,
                )
                force_noop = jnp.logical_or(target_obj_idxs == no_target_idx, meta_noop)

                if not composition_enabled:
                    # ---- K=1 path: bit-equivalent to pre-composition behavior ----
                    chosen_ego_actions = all_ego_actions[
                        active_skills, jnp.arange(config["NUM_ENVS"])
                    ]
                    chosen_ego_actions = jnp.where(force_noop, 0, chosen_ego_actions)

                    global_actions = jax.vmap(action_mapper)(
                        chosen_ego_actions,
                        last_obs.raw_observation,
                        target_obj_idxs,
                        last_obs.objects,
                    )
                    global_actions = jnp.where(force_noop, 0, global_actions)

                    # Per-slot fields are slot 0 only (legacy K=1 capture).
                    slot_target_idxs = target_obj_idxs[:, None]
                    slot_active = (~force_noop)[:, None]
                    slot_weights = (~force_noop).astype(jnp.float32)[:, None]
                    slot_skill_idx = active_skills[:, None]
                    slot_ego_obses = ego_obses[:, None, ...]
                    slot_ego_actions = chosen_ego_actions[:, None]
                    slot_skill_q = jnp.transpose(all_skill_q_vals, (1, 0, 2))[:, None, :, :]
                else:
                    # ---- K>1 composition path ----
                    K = COMPOSITION_K

                    # Select K slots ε-greedily from the deterministic meta_q.
                    # During warmup, the heuristic produced a single target;
                    # we override with single-slot below.
                    semantic_labels_now = last_obs.objects.semantic_label[:, -1, :]
                    rngs_select = jax.random.split(rng_meta, config["NUM_ENVS"])
                    _meta_eps_scalar = meta_eps_schedule(
                        jnp.maximum(0, meta_state.n_updates - skill_warmup_updates)
                    )
                    slot_target_idxs, slot_active, slot_skill_idx = jax.vmap(
                        lambda rk, mq, am, lbl: comp.select_slots_eps(
                            rk, mq, am, lbl, no_target_idx,
                            mode=COMPOSITION_SLOT_MODE, K=K, eps=_meta_eps_scalar,
                        )
                    )(rngs_select, meta_q_vals, active_masks, semantic_labels_now)

                    # Override during warmup: degrade to single-slot (slot 0 = heuristic target).
                    def _warmup_slots(single_target_idxs):
                        # Build K-shaped arrays with only slot 0 populated.
                        K_target = jnp.stack(
                            [single_target_idxs]
                            + [jnp.full_like(single_target_idxs, no_target_idx)] * (K - 1),
                            axis=-1,
                        )
                        # Slot 0 is active iff single_target is a valid index.
                        slot0_active = single_target_idxs != no_target_idx
                        K_active = jnp.stack(
                            [slot0_active]
                            + [jnp.zeros_like(slot0_active, dtype=bool)] * (K - 1),
                            axis=-1,
                        )
                        safe_single = jnp.where(slot0_active, single_target_idxs, 0)
                        sk0 = label_to_skill_idx(
                            jax.vmap(lambda labels, idx: labels[-1, idx])(
                                last_obs.objects.semantic_label, safe_single
                            )
                        )
                        K_skill = jnp.stack(
                            [sk0] + [jnp.zeros_like(sk0)] * (K - 1), axis=-1
                        )
                        return K_target, K_active, K_skill

                    warmup_target_idxs, warmup_active, warmup_skill_idx = _warmup_slots(
                        target_obj_idxs
                    )
                    slot_target_idxs = jnp.where(
                        use_heuristic_warmup, warmup_target_idxs, slot_target_idxs
                    )
                    slot_active = jnp.where(
                        use_heuristic_warmup, warmup_active, slot_active
                    )
                    slot_skill_idx = jnp.where(
                        use_heuristic_warmup, warmup_skill_idx, slot_skill_idx
                    )

                    # Per-slot ego obs. [NUM_ENVS, K, ego_feat_dim].
                    safe_slot_targets = jnp.where(slot_active, slot_target_idxs, 0)
                    slot_ego_obses = jax.vmap(
                        lambda raw, objs, idxs: jax.vmap(
                            lambda i: get_ego_obs(raw, objs, i)
                        )(idxs)
                    )(last_obs.raw_observation, last_obs.objects, safe_slot_targets)

                    # Per-slot, per-skill Q values: run all skills against each slot's ego obs.
                    def all_skills_q_for_slot(ego_obs_for_slot):
                        _, q = jax.vmap(
                            compute_skill_actions, in_axes=(0, 0, None)
                        )(skill_states, jnp.arange(num_skills, dtype=jnp.int32), ego_obs_for_slot)
                        return q  # [num_skills, NUM_ENVS, ego_action_dim]

                    slot_q_kfirst = jax.vmap(all_skills_q_for_slot, in_axes=1)(slot_ego_obses)
                    # → [K, num_skills, NUM_ENVS, ego_action_dim]
                    slot_skill_q = jnp.transpose(slot_q_kfirst, (2, 0, 1, 3))
                    # → [NUM_ENVS, K, num_skills, ego_action_dim]

                    # Scatter each slot's active-skill ego-Q to global action space.
                    def scatter_one(slot_q, sk_idx, tgt_idx, raw_obs, objs):
                        active_ego_q = slot_q[sk_idx]
                        return comp.ego_q_to_global_q(
                            active_ego_q, raw_obs, tgt_idx, objs,
                            action_mapper, num_game_actions,
                        )
                    def scatter_one_env(slots_q, slots_sk, slots_tgt, raw_obs, objs):
                        return jax.vmap(
                            lambda q, s, t: scatter_one(q, s, t, raw_obs, objs)
                        )(slots_q, slots_sk, slots_tgt)
                    slot_global_q = jax.vmap(scatter_one_env)(
                        slot_skill_q, slot_skill_idx, safe_slot_targets,
                        last_obs.raw_observation, last_obs.objects,
                    )  # [NUM_ENVS, K, num_game_actions]

                    # Per-slot meta-Q. Inactive slots get -inf so they discard.
                    slot_meta_q = jnp.take_along_axis(meta_q_vals, safe_slot_targets, axis=-1)
                    slot_meta_q = jnp.where(slot_active, slot_meta_q, -jnp.inf)

                    # Compose → executed global action.
                    composed_q, slot_weights = jax.vmap(
                        lambda sq, mq, sa: comp.compose_global_q(
                            sq, mq, sa, tau=COMPOSITION_TAU, normalize=COMPOSITION_NORMALIZE,
                            weight_threshold=COMPOSITION_WEIGHT_THRESHOLD,
                        )
                    )(slot_global_q, slot_meta_q, slot_active)

                    # Fall back to NOOP when either no slot was originally active
                    # OR the weight threshold dropped every slot ("meta judged
                    # nothing worth attending to this frame").
                    any_slot_active = jnp.logical_and(
                        jnp.any(slot_active, axis=-1),
                        jnp.sum(slot_weights, axis=-1) > 0,
                    )
                    greedy_global_actions = jnp.argmax(composed_q, axis=-1).astype(jnp.int32)
                    # Action-level eps-greedy on the composed Q-vector. The
                    # skills' per-skill eps-greedy is computed but discarded
                    # under composition, so exploration must happen here.
                    eps_action = skill_eps_schedule(meta_state.n_updates)
                    rngs_env = jax.random.split(rng_skill, config["NUM_ENVS"])
                    def _eps_on_composed(env_rng, q_row, gact):
                        rng_u, rng_r = jax.random.split(env_rng)
                        random_logits = jnp.where(jnp.isfinite(q_row), 0.0, -jnp.inf)
                        rand_act = jax.random.categorical(rng_r, random_logits).astype(jnp.int32)
                        u = jax.random.uniform(rng_u, ())
                        return jnp.where(u < eps_action, rand_act, gact)
                    global_actions = jax.vmap(_eps_on_composed)(
                        rngs_env, composed_q, greedy_global_actions
                    )
                    global_actions = jnp.where(any_slot_active, global_actions, 0)

                    # Invert executed global action to per-slot ego actions.
                    def invert_one_env(g_act, slots_q, slots_sk, slots_tgt, slots_active, raw_obs, objs):
                        active_q = jax.vmap(lambda sq, s: sq[s])(slots_q, slots_sk)
                        return comp.invert_global_to_ego_action(
                            g_act, active_q, slots_tgt, slots_active,
                            raw_obs, objs, action_mapper,
                        )
                    slot_ego_actions = jax.vmap(invert_one_env)(
                        global_actions, slot_skill_q, slot_skill_idx,
                        safe_slot_targets, slot_active,
                        last_obs.raw_observation, last_obs.objects,
                    )

                    # Legacy single-slot fields drawn from slot 0 (keeps the
                    # existing loss path functional).
                    slot0_active = slot_active[:, 0]
                    slot0_target = slot_target_idxs[:, 0]
                    target_obj_idxs = jnp.where(slot0_active, slot0_target, no_target_idx)
                    ego_obses = slot_ego_obses[:, 0]
                    active_skills = slot_skill_idx[:, 0]
                    chosen_ego_actions = slot_ego_actions[:, 0]
                    all_skill_q_vals = jnp.transpose(slot_skill_q[:, 0], (1, 0, 2))
                    safe_target_idxs = jnp.where(target_obj_idxs == no_target_idx, 0, target_obj_idxs)
                    meta_noop = is_meta_noop_target(
                        target_obj_idxs, config,
                        noop_idx=meta_noop_idx, enabled=meta_noop_enabled,
                    )
                    force_noop = jnp.logical_or(target_obj_idxs == no_target_idx, meta_noop)

                new_obs, new_env_state, env_reward, new_done, info = vmap_step(
                    env_state, global_actions
                )
                life_lost = extract_life_lost(env_state, new_env_state)
                # Fire the meta-level "death" penalty on every collision/life-loss
                # frame, not only on episode-ending collisions. On lives-based
                # envs (Phoenix, Seaquest) with EPISODIC_LIFE=true, `new_done`
                # already coincides with every life-loss, so this is unchanged.
                # On Freeway (not lives-based, episode = fixed 8192 frames),
                # the previous gate (`new_done AND life_lost`) only fired if
                # the very last frame happened to be a collision — effectively
                # never. That left the meta with no negative training signal
                # for collisions, breaking AVOID-slot credit assignment.
                meta_collision_event = life_lost > 0

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

                # Per-slot rewards / next_ego_obs (K>1: one row per slot; K=1: unit K axis).
                if not composition_enabled:
                    slot_rewards = skill_rewards[:, None, :]
                    slot_skill_done = skill_done[:, None, :]
                else:
                    def get_target_obj_local(objects, idx):
                        return jax.tree_util.tree_map(
                            lambda x: jnp.take(x, idx, axis=-1), objects
                        )
                    slot_prev_targets = jax.vmap(
                        lambda objs, idxs: jax.vmap(
                            lambda i: get_target_obj_local(objs, i)
                        )(idxs)
                    )(last_obs.objects, safe_slot_targets)
                    slot_next_targets = jax.vmap(
                        lambda objs, idxs: jax.vmap(
                            lambda i: get_target_obj_local(objs, i)
                        )(idxs)
                    )(new_obs.objects, safe_slot_targets)

                    def per_slot_rewards(pr_raw, nr_raw, ll, inf_, sp, sn):
                        return jax.vmap(
                            lambda pt, nt: extract_all_skills_rewards(
                                pr_raw, nr_raw, pt, nt, ll, inf_, config
                            )
                        )(sp, sn)
                    slot_rewards, slot_skill_done = jax.vmap(per_slot_rewards)(
                        last_obs.raw_observation,
                        new_obs.raw_observation,
                        life_lost,
                        info,
                        slot_prev_targets,
                        slot_next_targets,
                    )
                    slot_rewards = jnp.where(
                        slot_active[..., None], slot_rewards, 0.0
                    )
                    slot_skill_done = jnp.where(
                        slot_active[..., None], slot_skill_done, True
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
                        meta_collision_event, meta_death_penalty, 0.0
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

                # Per-slot next_ego_obs.
                if not composition_enabled:
                    slot_next_ego_obses = next_ego_obses[:, None, ...]
                else:
                    slot_next_ego_obses = jax.vmap(
                        lambda raw, objs, idxs: jax.vmap(
                            lambda i: get_ego_obs(raw, objs, i)
                        )(idxs)
                    )(
                        next_obs_for_rollout.raw_observation,
                        next_obs_for_rollout.objects,
                        safe_slot_targets,
                    )

                # Composition slots. K=1 stores slot-0 mirror of legacy fields;
                # K>1 stores all K composed slots.
                if not composition_enabled:
                    slot_valid_f = (~force_noop).astype(jnp.float32)
                    composition = CompositionSlots(
                        target_idxs=target_obj_idxs[:, None],
                        slot_active=(~force_noop)[:, None],
                        slot_weights=slot_valid_f[:, None],
                        slot_skill_idx=active_skills[:, None],
                        ego_obses=ego_obses[:, None, ...],
                        next_ego_obses=slot_next_ego_obses,
                        slot_ego_actions=chosen_ego_actions[:, None],
                        slot_skill_q=jnp.transpose(all_skill_q_vals, (1, 0, 2))[:, None, :, :],
                        slot_rewards=slot_rewards,
                        slot_skill_done=slot_skill_done,
                    )
                    legacy_skill_rewards = skill_rewards
                    legacy_skill_done = skill_done
                else:
                    composition = CompositionSlots(
                        target_idxs=slot_target_idxs,
                        slot_active=slot_active,
                        slot_weights=slot_weights,
                        slot_skill_idx=slot_skill_idx,
                        ego_obses=slot_ego_obses,
                        next_ego_obses=slot_next_ego_obses,
                        slot_ego_actions=slot_ego_actions,
                        slot_skill_q=slot_skill_q,
                        slot_rewards=slot_rewards,
                        slot_skill_done=slot_skill_done,
                    )
                    legacy_skill_rewards = slot_rewards[:, 0, :]
                    legacy_skill_done = slot_skill_done[:, 0, :]
                    next_ego_obses = slot_next_ego_obses[:, 0]

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
                    skill_rewards=legacy_skill_rewards,
                    skill_done=legacy_skill_done,
                    env_reward=base_env_reward,
                    composition=composition,
                )
                return (
                    next_obs_for_rollout,
                    next_state_for_rollout,
                    running_returns,
                    rng,
                    completed_returns_sum,
                    num_completed,
                    target_obj_idxs,
                ), transition

            rng, _rng = jax.random.split(rng)
            init_targets = jnp.full(
                (config["NUM_ENVS"],), no_target_idx, dtype=jnp.int32
            )
            carry_final, transitions = jax.lax.scan(
                _step_env,
                (
                    *expl_state,
                    running_returns,
                    _rng,
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    init_targets,
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
            v_next = _meta_state_value(last_meta_q, next_active_mask)
            last_q = jnp.where(next_active_any, v_next, 0.0)
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
                v_prev = _meta_state_value(transition.meta_q_val, prev_active_mask)
                next_q = jnp.where(prev_active_any, v_prev, 0.0)
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
                    # Per-slot targets and active mask (each slot contributes one
                    # Bellman update sharing the global meta_reward target). With
                    # K=1 this collapses to the legacy single-target behavior.
                    "slot_target_idxs": transitions.composition.target_idxs,
                    "slot_active": transitions.composition.slot_active,
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
                        # Hybrid K-aware Bellman: each active slot's chosen
                        # target receives an MSE update against the same lambda
                        # target (sharing the global return signal across slots).
                        slot_targets = minibatch["slot_target_idxs"]  # [B, K]
                        slot_active = minibatch["slot_active"]        # [B, K]
                        safe_slot_targets = jnp.where(slot_active, slot_targets, 0)
                        # Gather per-slot Q-values: [B, K]
                        chosen_q_per_slot = jnp.take_along_axis(
                            q_vals, safe_slot_targets, axis=-1
                        )
                        env_valid = jnp.any(minibatch["active_mask"], axis=-1)
                        # Per-slot validity = env has any active object AND this slot active.
                        valid = jnp.logical_and(slot_active, env_valid[:, None])
                        # MSE between per-slot chosen Q and the (broadcast) lambda target.
                        sq_err = jnp.square(chosen_q_per_slot - target[:, None])
                        loss = (sq_err * valid).sum() / (valid.sum() + 1e-8)
                        # Slot-0 chosen_q for legacy logging.
                        chosen_q = chosen_q_per_slot[:, 0]
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

                # K-flattened per-slot data: each (env, slot) is an independent
                # training tuple. K=1 collapses to the legacy single-trajectory shape.
                comp_data = transitions.composition
                def _kflat(x):
                    T, E, K = x.shape[:3]
                    return x.reshape(T, E * K, *x.shape[3:])

                slot_ego_obs_k       = _kflat(comp_data.ego_obses)
                slot_next_ego_obs_k  = _kflat(comp_data.next_ego_obses)
                slot_ego_action_k    = _kflat(comp_data.slot_ego_actions)
                slot_target_idx_k    = _kflat(comp_data.target_idxs)
                slot_active_k        = _kflat(comp_data.slot_active)
                slot_skill_idx_k     = _kflat(comp_data.slot_skill_idx)
                slot_rewards_k       = _kflat(comp_data.slot_rewards)
                slot_skill_done_k    = _kflat(comp_data.slot_skill_done)
                slot_skill_q_k       = _kflat(comp_data.slot_skill_q)

                last_q = skill_network.apply(
                    {"params": s_state.params, "batch_stats": s_state.batch_stats},
                    slot_next_ego_obs_k[-1],
                    train=False,
                )
                last_q_max = jnp.max(last_q, axis=-1)
                last_step_done = slot_skill_done_k[-1, :, skill_idx]
                last_q_max = last_q_max * (1 - last_step_done)
                last_step_reward = slot_rewards_k[-1, :, skill_idx]
                lambda_returns = last_step_reward + config["GAMMA"] * last_q_max

                scan_inputs = {
                    "rewards":      slot_rewards_k,
                    "skill_done":   slot_skill_done_k,
                    "slot_skill_q": slot_skill_q_k,
                }

                def _skill_get_target(lambda_returns_and_next_q, transition):
                    lambda_ret, next_q = lambda_returns_and_next_q
                    step_reward = transition["rewards"][..., skill_idx]
                    step_done = transition["skill_done"][..., skill_idx]
                    target_bootstrap = (
                        step_reward + config["GAMMA"] * (1 - step_done) * next_q
                    )
                    delta = lambda_ret - next_q
                    lambda_ret = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_ret = (1 - step_done) * lambda_ret + step_done * step_reward
                    next_q_val = jnp.max(transition["slot_skill_q"][..., skill_idx, :], axis=-1)
                    return (lambda_ret, next_q_val), lambda_ret

                _, targets = jax.lax.scan(
                    _skill_get_target,
                    (lambda_returns, last_q_max),
                    jax.tree.map(lambda x: x[:-1], scan_inputs),
                    reverse=True,
                )
                skill_lambda_targets = jnp.concatenate(
                    (targets, lambda_returns[jnp.newaxis])
                )

                def _learn_epoch(carry, _):
                    s_state_epoch, rng_epoch = carry
                    rng_epoch, rng_perm = jax.random.split(rng_epoch)

                    train_batch = {
                        "target_idx": slot_target_idx_k,
                        "ego_obs": slot_ego_obs_k,
                        "ego_action": slot_ego_action_k,
                        "active_skill": slot_skill_idx_k,
                        "slot_active": slot_active_k,
                    }
                    minibatches = jax.tree.map(
                        lambda x: preprocess_transition(x, rng_perm),
                        train_batch,
                    )
                    targets_batch = preprocess_transition(skill_lambda_targets, rng_perm)

                    def _learn_phase(carry_phase, minibatch_and_target):
                        s_state_phase, rng_phase = carry_phase
                        minibatch, target = minibatch_and_target

                        def _loss_fn(params):
                            q_vals, updates = skill_network.apply(
                                {
                                    "params": params,
                                    "batch_stats": s_state_phase.batch_stats,
                                },
                                minibatch["ego_obs"],
                                train=True,
                                mutable=["batch_stats"],
                            )
                            chosen_q = jnp.take_along_axis(
                                q_vals,
                                jnp.expand_dims(minibatch["ego_action"], axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)
                            base_valid = jnp.logical_and(
                                minibatch["slot_active"],
                                minibatch["target_idx"] != no_target_idx,
                            )
                            if config.get("OFF_POLICY", True):
                                valid = base_valid
                            else:
                                valid = jnp.logical_and(
                                    base_valid,
                                    minibatch["active_skill"] == skill_idx,
                                )
                            if meta_noop_enabled:
                                valid = jnp.logical_and(
                                    valid, minibatch["target_idx"] != meta_noop_idx
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
            skill_reward_nonzero = (jnp.abs(transitions.skill_rewards) > 0).astype(
                jnp.float32
            )
            # K-aware composition aggregates for diagnostics (don't mutate
            # legacy variables above so existing masks remain valid).
            comp_rewards = transitions.composition.slot_rewards  # (T, E, K, num_skills)
            comp_active = transitions.composition.slot_active    # (T, E, K)
            comp_weights = transitions.composition.slot_weights  # (T, E, K)
            active_count_total = jnp.maximum(comp_active.sum(), 1.0)
            comp_skill_reward_mean = (
                (comp_rewards * comp_active[..., None]).sum() / active_count_total
            ) / max(1, num_skills)
            if config.get("OFF_POLICY", True):
                skill_valid_mask = jnp.broadcast_to(
                    valid_target_mask[..., None], transitions.skill_rewards.shape
                )
            else:
                skill_valid_mask = jnp.logical_and(
                    valid_target_mask[..., None] > 0,
                    transitions.active_skill[..., None]
                    == jnp.arange(num_skills, dtype=jnp.int32),
                )
                skill_valid_mask = skill_valid_mask.astype(jnp.float32)

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
                # Always-on progress signals (never zero between completed
                # episodes). episode_completions tells you whether the metric
                # above is even computed; reward_sum_per_env_per_update is a
                # dense per-window throughput proxy you can plot smoothly.
                "env/episode_completions": num_completed,
                "env/reward_sum_per_env_per_update": jnp.sum(transitions.env_reward) / config["NUM_ENVS"],
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
                # Per-skill breakdown — see if AVOID is actually receiving its
                # reward signal under composition.
                **{f"skill_{skill_names[i]}/td_loss": skill_metrics[0][i] for i in range(num_skills)},
                **{f"skill_{skill_names[i]}/qvals": skill_metrics[1][i] for i in range(num_skills)},
                # Per-skill reward mean across all active slots.
                **{
                    f"skill_{skill_names[i]}/reward": (
                        (comp_rewards[..., i] * comp_active.astype(jnp.float32)).sum()
                        / active_count_total
                    )
                    for i in range(num_skills)
                },
                # Per-skill nonzero-reward rate across all active slots.
                **{
                    f"skill_{skill_names[i]}/reward_nonzero_frac": (
                        ((jnp.abs(comp_rewards[..., i]) > 0).astype(jnp.float32)
                         * comp_active.astype(jnp.float32)).sum()
                        / active_count_total
                    )
                    for i in range(num_skills)
                },
                "warmup/heuristic_active": (
                    meta_state.n_updates < skill_warmup_updates
                ).astype(jnp.float32),
                "warmup/meta_frozen": skip_meta_update.astype(jnp.float32),
                "skill/frozen": skills_frozen.astype(jnp.float32),
                "update_steps": meta_state.n_updates,
                "global/update_steps": meta_state.n_updates,
                "env/step": meta_state.timesteps,
                # Composition diagnostics (K=1 reduces to trivial single-slot values).
                "composition/slot_active_rate": comp_active.astype(jnp.float32).mean(),
                "composition/avg_active_slots_per_step": comp_active.astype(jnp.float32).sum(axis=-1).mean(),
                # Effective-K: slots that survived the weight threshold (= the
                # meta actually attended to them). Equals avg_active_slots when
                # COMPOSITION_SLOT_WEIGHT_THRESHOLD = 0.
                "composition/avg_effective_slots_per_step": (comp_weights > 0).astype(jnp.float32).sum(axis=-1).mean(),
                "composition/skill_reward_mean_active_slots": comp_skill_reward_mean,
            }
            # Shannon entropy of slot weights (only steps with any active slot).
            _safe_w = jnp.clip(comp_weights, 1e-12, 1.0)
            _entropy = -(comp_weights * jnp.log(_safe_w)).sum(axis=-1)  # (T, E)
            _any_active = jnp.any(comp_active, axis=-1).astype(jnp.float32)
            metrics["composition/slot_weight_entropy"] = (
                (_entropy * _any_active).sum() / jnp.maximum(_any_active.sum(), 1.0)
            )
            if composition_enabled and COMPOSITION_SLOT_MODE == "per_label":
                for i in range(num_skills):
                    metrics[f"composition/slot_weight_{skill_names[i]}_mean"] = (
                        comp_weights[..., i].mean()
                    )
            if (
                composition_enabled
                and config.get("ENV_NAME") == "freeway"
                and avoid_slot_idx is not None
            ):
                # Variance of slot ego-Q across ego actions for the AVOID slot.
                # Split by whether the selected car overlaps the chicken column.
                comp_skill_q = transitions.composition.slot_skill_q
                # per_label: slot k uses skill k, so AVOID slot ego-Q is [..., 2, 2, :]
                avoid_ego_q = comp_skill_q[..., avoid_slot_idx, avoid_slot_idx, :]
                avoid_ego_q_var = jnp.var(avoid_ego_q, axis=-1)
                avoid_targets = transitions.composition.target_idxs[..., avoid_slot_idx]
                safe_avoid_targets = jnp.where(
                    comp_active[..., avoid_slot_idx], avoid_targets, 0
                )
                selected_car_x = jnp.take_along_axis(
                    transitions.objects.x[..., -1, :],
                    safe_avoid_targets[..., None],
                    axis=-1,
                ).squeeze(-1)
                car_in_lane = jnp.logical_and(
                    selected_car_x >= _CAR_ZONE_LEFT,
                    selected_car_x <= _CAR_ZONE_RIGHT,
                )
                avoid_active = comp_active[..., avoid_slot_idx]
                in_lane_mask = avoid_active & car_in_lane
                far_away_mask = avoid_active & jnp.logical_not(car_in_lane)
                in_lane_f = in_lane_mask.astype(jnp.float32)
                far_away_f = far_away_mask.astype(jnp.float32)
                metrics["composition/freeway_avoid_ego_q_var_in_lane"] = (
                    (avoid_ego_q_var * in_lane_f).sum()
                    / jnp.maximum(in_lane_f.sum(), 1.0)
                )
                metrics["composition/freeway_avoid_ego_q_var_far_away"] = (
                    (avoid_ego_q_var * far_away_f).sum()
                    / jnp.maximum(far_away_f.sum(), 1.0)
                )

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
