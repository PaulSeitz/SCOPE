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
from concept_based_learning.networks.meta_factory import create_meta_network
from concept_based_learning.environments.datatypes import (
    ConceptObservation,
    CompositionSlots,
)
from concept_based_learning import composition as comp
from concept_based_learning.environments.meta_target_mask import (
    build_meta_target_mask,
    is_meta_noop_target,
    meta_noop_target_enabled,
    meta_noop_target_idx,
)
from concept_based_learning.training.schedules import eps_decay_transition_steps


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
    - EPS_START, EPS_FINISH, EPS_DECAY
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

    _raw_skill_names = config.get("SKILL_NAMES")
    if _raw_skill_names is None or len(_raw_skill_names) != NUM_SKILLS:
        skill_names = [f"skill_{i}" for i in range(NUM_SKILLS)]
    else:
        skill_names = [str(n) for n in _raw_skill_names]

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

    meta_noop_enabled = meta_noop_target_enabled(config)
    meta_noop_idx = meta_noop_target_idx(config)

    fixed_target_idx = fixed_target_idx_from_config(config)

    # Game action space size (used by Q-decomposition scatter).
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
    if COMPOSITION_K > 1:
        if COMPOSITION_SLOT_MODE == "per_label":
            assert COMPOSITION_K == NUM_SKILLS, (
                f"per_label slot mode requires COMPOSITION_K == NUM_SKILLS "
                f"(got K={COMPOSITION_K}, NUM_SKILLS={NUM_SKILLS})"
            )
    composition_enabled = COMPOSITION_K > 1
    composition_uses_heuristic_slots = False
    # K>1 composition supports two slot sources:
    # 1. Learned meta_q drives select_slots (preferred).
    # 2. Single heuristic target expanded to K slots via per-label nearest
    #    fallback; slot weights are uniform across active slots.
    # Fixed-target mode is incompatible with K>1.
    if composition_enabled and fixed_target_idx is not None:
        import warnings
        warnings.warn(
            "COMPOSITION_K>1 is incompatible with a fixed target; "
            "falling back to K=1 for this run.",
            UserWarning,
            stacklevel=2,
        )
        composition_enabled = False
    elif composition_enabled and frozen_meta_params is None:
        composition_uses_heuristic_slots = True

    skill_network = QNetwork(
        action_dim=EGO_ACTION_DIM,
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
    )
    use_learned_meta_policy = frozen_meta_params is not None
    if use_learned_meta_policy:
        meta_network = create_meta_network(config)
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
            config["EPS_START"],
            config["EPS_FINISH"],
            eps_decay_transition_steps(config["EPS_DECAY"], config["NUM_UPDATES"]),
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
                ) = carry

                rng, rng_rest = jax.random.split(rng)
                rng_reset, rng_step = jax.random.split(rng_rest)
                rng_skill, rng_policy = jax.random.split(rng_step)

                policy_rngs = jax.random.split(rng_policy, config["NUM_ENVS"])

                # active_masks is needed for both the learned-meta path and the
                # heuristic-K (K>1 no-meta) composition path; compute once.
                active_masks = jax.vmap(
                    lambda objs, raw: build_meta_target_mask(
                        objs, raw, config, hook=meta_mask_hook
                    )
                )(last_obs.objects, last_obs.raw_observation)

                if use_learned_meta_policy:
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

                def compute_skill_actions(train_state, skill_idx, ego_obs_batch):
                    q_vals = skill_network.apply(
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                        ego_obs_batch,
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
                    compute_skill_actions, in_axes=(0, 0, None)
                )(train_states, jnp.arange(NUM_SKILLS, dtype=jnp.int32), ego_obses)

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

                    # Per-slot data for slot 0 only (legacy K=1 capture; rest is K=1 default).
                    slot_target_idxs = target_obj_idxs[:, None]
                    slot_active = (~force_noop)[:, None]
                    slot_weights = (~force_noop).astype(jnp.float32)[:, None]
                    slot_skill_idx = active_skills[:, None]
                    slot_ego_obses = ego_obses[:, None, ...]
                    slot_ego_actions = chosen_ego_actions[:, None]
                    # all_skill_q_vals: [num_skills, num_envs, EGO_ACTION_DIM] →
                    # [num_envs, K=1, num_skills, EGO_ACTION_DIM]
                    slot_skill_q = jnp.transpose(all_skill_q_vals, (1, 0, 2))[:, None, :, :]
                else:
                    # ---- K>1 composition path ----
                    K = COMPOSITION_K
                    semantic_labels_now = last_obs.objects.semantic_label[:, -1, :]

                    if composition_uses_heuristic_slots:
                        # No learned meta: expand a single heuristic target into
                        # K slots via per-label nearest fallback.
                        player_x = last_obs.raw_observation.player.x[:, -1] + (
                            last_obs.raw_observation.player.width[:, -1] / 2.0
                        )
                        player_y = last_obs.raw_observation.player.y[:, -1] + (
                            last_obs.raw_observation.player.height[:, -1] / 2.0
                        )
                        obj_cx = last_obs.objects.x[:, -1, :] + (
                            last_obs.objects.width[:, -1, :] / 2.0
                        )
                        obj_cy = last_obs.objects.y[:, -1, :] + (
                            last_obs.objects.height[:, -1, :] / 2.0
                        )
                        slot_target_idxs, slot_active, slot_skill_idx = jax.vmap(
                            lambda h, am, lbl, px, py, ox, oy: comp.slots_from_heuristic_target(
                                h, am, lbl, px, py, ox, oy,
                                no_target_idx=no_target_idx,
                                mode=COMPOSITION_SLOT_MODE, K=K,
                            )
                        )(
                            target_obj_idxs, active_masks, semantic_labels_now,
                            player_x, player_y, obj_cx, obj_cy,
                        )
                    else:
                        # 1. Select K slots ε-greedily over the deterministic meta_q.
                        rngs_select = jax.random.split(rng_policy, config["NUM_ENVS"])
                        _meta_eps_scalar = (
                            meta_eps_scheduler(train_states.n_updates[0])
                            if use_learned_meta_policy
                            else 0.0
                        )
                        slot_target_idxs, slot_active, slot_skill_idx = jax.vmap(
                            lambda rk, mq, am, lbl: comp.select_slots_eps(
                                rk, mq, am, lbl, no_target_idx,
                                mode=COMPOSITION_SLOT_MODE, K=K, eps=_meta_eps_scalar,
                            )
                        )(rngs_select, meta_q_vals, active_masks, semantic_labels_now)
                    # slot_target_idxs: [NUM_ENVS, K], slot_active: [NUM_ENVS, K], slot_skill_idx: [NUM_ENVS, K]

                    # 2. Per-slot ego observation. Result: [NUM_ENVS, K, ego_feat_dim].
                    safe_slot_targets = jnp.where(slot_active, slot_target_idxs, 0)
                    def per_slot_ego(raw_obs, objs, idxs):
                        return jax.vmap(lambda i: get_ego_obs(raw_obs, objs, i))(idxs)
                    slot_ego_obses = jax.vmap(per_slot_ego)(
                        last_obs.raw_observation, last_obs.objects, safe_slot_targets
                    )

                    # 3. Per-slot, per-skill Q values. Run each skill over each slot's ego obs.
                    # Reuse compute_skill_actions but vmap over slots externally to share rng.
                    def all_skills_q_for_slot(ego_obs_for_slot):
                        # ego_obs_for_slot: [NUM_ENVS, ego_feat_dim]
                        _, q = jax.vmap(
                            compute_skill_actions, in_axes=(0, 0, None)
                        )(train_states, jnp.arange(NUM_SKILLS, dtype=jnp.int32), ego_obs_for_slot)
                        return q  # [num_skills, NUM_ENVS, EGO_ACTION_DIM]
                    # vmap over the K axis: input [NUM_ENVS, K, ego_feat_dim] -> in_axes=1
                    slot_q_kfirst = jax.vmap(all_skills_q_for_slot, in_axes=1)(slot_ego_obses)
                    # slot_q_kfirst: [K, num_skills, NUM_ENVS, EGO_ACTION_DIM]
                    # Reorder to [NUM_ENVS, K, num_skills, EGO_ACTION_DIM].
                    slot_skill_q = jnp.transpose(slot_q_kfirst, (2, 0, 1, 3))

                    # 4. For each (env, slot): pick the active skill's Q, scatter to global.
                    def scatter_one(slot_q, sk_idx, tgt_idx, raw_obs, objs):
                        # slot_q: [num_skills, EGO_ACTION_DIM]; pick active skill's row.
                        active_ego_q = slot_q[sk_idx]
                        return comp.ego_q_to_global_q(
                            active_ego_q, raw_obs, tgt_idx, objs,
                            action_mapper, num_game_actions,
                        )
                    def scatter_one_env(slots_q, slots_sk, slots_tgt, raw_obs, objs):
                        # slots_q: [K, num_skills, EGO_ACTION_DIM]
                        return jax.vmap(
                            lambda q, s, t: scatter_one(q, s, t, raw_obs, objs)
                        )(slots_q, slots_sk, slots_tgt)
                    slot_global_q = jax.vmap(scatter_one_env)(
                        slot_skill_q, slot_skill_idx, safe_slot_targets,
                        last_obs.raw_observation, last_obs.objects,
                    )  # [NUM_ENVS, K, num_game_actions]

                    # 5. Per-slot meta-Q at the chosen object.
                    if composition_uses_heuristic_slots:
                        # No learned meta: equal logits → softmax gives uniform
                        # weights over active slots.
                        slot_meta_q = jnp.where(slot_active, 0.0, -jnp.inf)
                    else:
                        slot_meta_q = jnp.take_along_axis(meta_q_vals, safe_slot_targets, axis=-1)
                        slot_meta_q = jnp.where(slot_active, slot_meta_q, -jnp.inf)

                    # 6. Compose: weighted sum over slots → global Q per env.
                    composed_q, slot_weights = jax.vmap(
                        lambda sq, mq, sa: comp.compose_global_q(
                            sq, mq, sa, tau=COMPOSITION_TAU, normalize=COMPOSITION_NORMALIZE,
                            weight_threshold=COMPOSITION_WEIGHT_THRESHOLD,
                        )
                    )(slot_global_q, slot_meta_q, slot_active)
                    # composed_q: [NUM_ENVS, num_game_actions], slot_weights: [NUM_ENVS, K]

                    # 7. Executed global action: argmax over composed Q. Fall
                    # back to NOOP when either no slot was active (nothing to
                    # attend to) OR when the slot-weight threshold dropped every
                    # slot (the meta judged nothing worth attending to). Detect
                    # the latter via the post-threshold weights sum.
                    any_slot_active = jnp.logical_and(
                        jnp.any(slot_active, axis=-1),
                        jnp.sum(slot_weights, axis=-1) > 0,
                    )
                    greedy_global_actions = jnp.argmax(composed_q, axis=-1).astype(jnp.int32)
                    # Action-level eps-greedy on the composed Q-vector. Reaches
                    # the *global* action space directly — skills' per-skill
                    # eps-greedy is computed but discarded under composition.
                    eps_action = eps_scheduler(train_states.n_updates[0])
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

                    # 8. Invert executed action to per-slot ego actions (for replay).
                    def invert_one_env(g_act, slots_q, slots_sk, slots_tgt, slots_active, raw_obs, objs):
                        # Pull the per-slot active-skill ego-Q [K, EGO_ACTION_DIM].
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

                    # 9. Populate legacy single-slot fields from slot 0 so existing
                    # loss/lambda-return code (which still reads target_idx, ego_obs,
                    # ego_action, q_val, active_skill) keeps training on a valid
                    # transition. We use slot 0 because it carries DESTROY in per_label
                    # mode (matching today's "skill 0 is DESTROY" convention) and is
                    # the obvious first-class slot in topk.
                    slot0_active = slot_active[:, 0]
                    slot0_target = slot_target_idxs[:, 0]
                    target_obj_idxs = jnp.where(slot0_active, slot0_target, no_target_idx)
                    ego_obses = slot_ego_obses[:, 0]
                    active_skills = slot_skill_idx[:, 0]
                    chosen_ego_actions = slot_ego_actions[:, 0]
                    # slot_skill_q is [NUM_ENVS, K, num_skills, EGO_ACTION_DIM] →
                    # all_skill_q_vals[skill, env, A] = slot_skill_q[env, 0, skill, A]
                    all_skill_q_vals = jnp.transpose(slot_skill_q[:, 0], (1, 0, 2))
                    # Re-derive force_noop from slot-0 (legacy fields).
                    safe_target_idxs = jnp.where(target_obj_idxs == no_target_idx, 0, target_obj_idxs)
                    meta_noop = is_meta_noop_target(
                        target_obj_idxs, config,
                        noop_idx=meta_noop_idx, enabled=meta_noop_enabled,
                    )
                    force_noop = jnp.logical_or(target_obj_idxs == no_target_idx, meta_noop)

                new_obs, new_env_state, raw_rewards, new_done, info = vmap_step(
                    env_state, global_actions
                )

                reset_obs, reset_state = vmap_reset(config["NUM_ENVS"])(rng_reset)
                new_obs = _merge_reset_where_done(new_obs, reset_obs, new_done)
                new_env_state = _merge_reset_where_done(new_env_state, reset_state, new_done)

                life_lost = extract_life_lost(env_state, new_env_state)

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
                if meta_noop_enabled:
                    skill_rewards = jnp.where(
                        meta_noop[:, jnp.newaxis], 0.0, skill_rewards
                    )

                next_ego_obses = jax.vmap(get_ego_obs)(
                    new_obs.raw_observation, new_obs.objects, safe_target_idxs
                )

                # Per-slot rewards / next_ego_obs (K>1: one row per slot; K=1: a unit K axis).
                if not composition_enabled:
                    slot_next_ego_obses = next_ego_obses[:, None, ...]
                    slot_rewards = skill_rewards[:, None, :]
                    slot_skill_done = skill_done[:, None, :]
                else:
                    # safe_slot_targets is [NUM_ENVS, K].
                    # Extract per-slot prev/next targets and rewards by vmapping over slots.
                    slot_prev_targets = jax.vmap(
                        lambda objs, idxs: jax.vmap(lambda i: get_target_obj(objs, i))(idxs)
                    )(last_obs.objects, safe_slot_targets)
                    slot_next_targets = jax.vmap(
                        lambda objs, idxs: jax.vmap(lambda i: get_target_obj(objs, i))(idxs)
                    )(new_obs.objects, safe_slot_targets)

                    def per_slot_rewards(pr_raw, nr_raw, ll, inf_, slot_prev, slot_next):
                        # slot_prev/slot_next: per-slot tree, leading axis = K.
                        return jax.vmap(
                            lambda pt, nt: extract_all_skills_rewards(
                                pr_raw, nr_raw, pt, nt, ll, inf_, config
                            )
                        )(slot_prev, slot_next)
                    slot_rewards, slot_skill_done = jax.vmap(per_slot_rewards)(
                        last_obs.raw_observation,
                        new_obs.raw_observation,
                        life_lost,
                        info,
                        slot_prev_targets,
                        slot_next_targets,
                    )
                    # slot_rewards: [NUM_ENVS, K, num_skills], slot_skill_done: [NUM_ENVS, K, num_skills]
                    # Zero inactive slots.
                    slot_rewards = jnp.where(
                        slot_active[..., None], slot_rewards, 0.0
                    )
                    slot_skill_done = jnp.where(
                        slot_active[..., None], slot_skill_done, True
                    )

                    # Per-slot next_ego_obs.
                    slot_next_ego_obses = jax.vmap(
                        lambda raw, objs, idxs: jax.vmap(lambda i: get_ego_obs(raw, objs, i))(idxs)
                    )(new_obs.raw_observation, new_obs.objects, safe_slot_targets)

                episode_return = running_returns + raw_rewards
                done_float = new_done.astype(jnp.float32)

                completed_returns_sum = completed_returns_sum + jnp.sum(
                    episode_return * done_float
                )
                num_completed = num_completed + jnp.sum(done_float)

                running_returns = episode_return * (1.0 - done_float)

                # Composition slot capture. Under K=1 this carries the single
                # executed (target, skill, ego_obs, ego_action) in a unit-K
                # axis; under K>1 it carries all K slots used in the composed
                # action. Loss code can read from either the legacy fields or
                # this dataclass depending on the K-awareness of the migration.
                if not composition_enabled:
                    slot_valid = (~force_noop).astype(jnp.float32)
                    composition = CompositionSlots(
                        target_idxs=target_obj_idxs[:, None],
                        slot_active=(~force_noop)[:, None],
                        slot_weights=slot_valid[:, None],
                        slot_skill_idx=active_skills[:, None],
                        ego_obses=ego_obses[:, None, ...],
                        next_ego_obses=slot_next_ego_obses,
                        slot_ego_actions=chosen_ego_actions[:, None],
                        slot_skill_q=jnp.transpose(all_skill_q_vals, (1, 0, 2))[:, None, :, :],
                        slot_rewards=slot_rewards,
                        slot_skill_done=slot_skill_done,
                    )
                    # Pull legacy fields from slot 0 (here, the only slot).
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
                    # Legacy fields drawn from slot 0 — keeps the old loss path
                    # functional under K>1 (it will train on slot-0 transitions
                    # only until the loss is migrated to read from composition).
                    legacy_skill_rewards = slot_rewards[:, 0, :]
                    legacy_skill_done = slot_skill_done[:, 0, :]
                    # Also reflect slot-0 next_ego_obs into the legacy field.
                    next_ego_obses = slot_next_ego_obses[:, 0]

                transition = Transition(
                    target_idx=target_obj_idxs,
                    ego_obs=ego_obses,
                    next_ego_obs=next_ego_obses,
                    ego_action=chosen_ego_actions,
                    active_skill=active_skills,
                    q_val=all_skill_q_vals,
                    rewards=legacy_skill_rewards,
                    skill_done=legacy_skill_done,
                    done=new_done,
                    env_reward=raw_rewards,
                    composition=composition,
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
            init_completed_returns = jnp.array(0.0, dtype=jnp.float32)
            init_num_completed = jnp.array(0.0, dtype=jnp.float32)
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

            def _update_skill(train_state, skill_idx, rng):
                train_state = train_state.replace(
                    timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
                )

                # Build K-flattened views of the per-slot data: each (env, slot)
                # becomes an independent training tuple. With K=1 this is
                # exactly the legacy layout (effective batch unchanged).
                comp_data = transitions.composition
                # Shapes carry (NUM_STEPS, NUM_ENVS, K, ...). Collapse axes 1, 2.
                def _kflat(x):
                    # x: (T, E, K, ...) → (T, E*K, ...)
                    T, E, K = x.shape[:3]
                    return x.reshape(T, E * K, *x.shape[3:])

                slot_ego_obs_k       = _kflat(comp_data.ego_obses)         # (T, E*K, ego_feat)
                slot_next_ego_obs_k  = _kflat(comp_data.next_ego_obses)    # (T, E*K, ego_feat)
                slot_ego_action_k    = _kflat(comp_data.slot_ego_actions)  # (T, E*K)
                slot_target_idx_k    = _kflat(comp_data.target_idxs)       # (T, E*K)
                slot_active_k        = _kflat(comp_data.slot_active)       # (T, E*K)
                slot_skill_idx_k     = _kflat(comp_data.slot_skill_idx)    # (T, E*K)
                slot_rewards_k       = _kflat(comp_data.slot_rewards)      # (T, E*K, num_skills)
                slot_skill_done_k    = _kflat(comp_data.slot_skill_done)   # (T, E*K, num_skills)
                slot_skill_q_k       = _kflat(comp_data.slot_skill_q)      # (T, E*K, num_skills, ego_action_dim)

                # Bootstrap: max Q at the post-rollout next_ego_obs for this skill.
                last_q = skill_network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    slot_next_ego_obs_k[-1],
                    train=False,
                )
                last_q_max = jnp.max(last_q, axis=-1)

                # In-scan transition tuple: only the fields _get_target needs.
                # Bundle as a dict so jax.tree.map / scan see a flat pytree.
                scan_inputs = {
                    "rewards":    slot_rewards_k,       # (T, E*K, num_skills)
                    "skill_done": slot_skill_done_k,    # (T, E*K, num_skills)
                    "slot_skill_q": slot_skill_q_k,     # (T, E*K, num_skills, ego_action_dim)
                }

                def _get_target(lambda_returns_and_next_q, transition):
                    lambda_returns, next_q = lambda_returns_and_next_q

                    step_reward = transition["rewards"][..., skill_idx]
                    step_done = transition["skill_done"][..., skill_idx]

                    target_bootstrap = (
                        step_reward + config["GAMMA"] * (1 - step_done) * next_q
                    )
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (
                        (1 - step_done) * lambda_returns + step_done * step_reward
                    )

                    # next_q for the *previous* step's bootstrap: greedy Q of
                    # this step's skill on this slot's ego obs.
                    next_q_val = jnp.max(transition["slot_skill_q"][..., skill_idx, :], axis=-1)
                    return (lambda_returns, next_q_val), lambda_returns

                last_step_done = slot_skill_done_k[-1, :, skill_idx]
                last_q_max = last_q_max * (1 - last_step_done)
                last_step_reward = slot_rewards_k[-1, :, skill_idx]
                lambda_returns = last_step_reward + config["GAMMA"] * last_q_max

                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q_max),
                    jax.tree.map(lambda x: x[:-1], scan_inputs),
                    reverse=True,
                )
                lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_and_target):
                        train_state, rng = carry
                        minibatch, target = minibatch_and_target

                        def _loss_fn(params):
                            q_vals, updates = skill_network.apply(
                                {"params": params, "batch_stats": train_state.batch_stats},
                                minibatch["ego_obs"],
                                train=True,
                                mutable=["batch_stats"],
                            )
                            chosen_action_qvals = jnp.take_along_axis(
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
                        "target_idx": slot_target_idx_k,
                        "ego_obs": slot_ego_obs_k,
                        "ego_action": slot_ego_action_k,
                        "active_skill": slot_skill_idx_k,
                        "slot_active": slot_active_k,
                    }
                    minibatches = jax.tree.map(
                        lambda x: preprocess_transition(x, _rng), train_batch
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

            rngs = jax.random.split(rng, NUM_SKILLS)
            skill_metrics, train_states = jax.vmap(_update_skill, in_axes=(0, 0, 0))(
                train_states, jnp.arange(NUM_SKILLS), rngs
            )

            metrics = {
                f"skill_{skill_names[i]}/{k}": v[i]
                for k, v in skill_metrics.items()
                for i in range(NUM_SKILLS)
            }

            # Per-skill reward mean across all *active* slots (K-aware). K=1
            # reduces to the legacy mean across (steps, envs).
            comp_rewards = transitions.composition.slot_rewards  # (T, E, K, num_skills)
            comp_active = transitions.composition.slot_active    # (T, E, K)
            active_count = jnp.maximum(comp_active.sum(), 1.0)
            mean_skill_rewards = (
                (comp_rewards * comp_active[..., None]).sum(axis=(0, 1, 2))
                / active_count
            )
            for i in range(NUM_SKILLS):
                metrics[f"skill_{skill_names[i]}/reward"] = mean_skill_rewards[i]

            metrics["env/reward_mean"] = jnp.mean(transitions.env_reward)

            # Slot diagnostics: effective-K entropy and per-slot activity rate.
            slot_w = transitions.composition.slot_weights  # (T, E, K)
            slot_active_f = comp_active.astype(jnp.float32)
            metrics["composition/slot_active_rate"] = slot_active_f.mean()
            metrics["composition/avg_active_slots_per_step"] = slot_active_f.sum(axis=-1).mean()
            # Effective-K = slots that survived the weight threshold.
            metrics["composition/avg_effective_slots_per_step"] = (slot_w > 0).astype(jnp.float32).sum(axis=-1).mean()
            # Shannon entropy of slot weights (only steps with any active slot).
            safe_w = jnp.clip(slot_w, 1e-12, 1.0)
            entropy = -(slot_w * jnp.log(safe_w)).sum(axis=-1)  # (T, E)
            any_active = jnp.any(comp_active, axis=-1).astype(jnp.float32)
            denom = jnp.maximum(any_active.sum(), 1.0)
            metrics["composition/slot_weight_entropy"] = (entropy * any_active).sum() / denom

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
