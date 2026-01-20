# src/agent/orchestrator.py
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
import chex
from typing import Tuple
import optax

from src.rl_utils import RLTrainState
from src.agent.components.meta_policy import MetaPolicy
from src.agent.components.sub_policy import SubPolicy
from src.agent.components.instance_selector import InstanceSelector
from src.config import Config
from src.environments.seaquest.datatypes import ConceptObs, AgentState
import src.environments.seaquest.seaquest_constants as consts

def heuristic_meta(obs):
    """
    Rulebook for Phase 1 & 2.
    # TODO: if it does not work maybe change order of concepts and pull DESTROY forward
    Priority: Surface (if low oxy) > Collect (if divers) > Destroy (if enemies) > Default
    """
    oxy_level = obs.global_features[0]
    diver_count = obs.global_features[4]
    enemy_count = obs.global_features[3]
    
    is_low_oxy = oxy_level < 0.3
    has_divers = diver_count > 0
    has_enemies = enemy_count > 0
    
    c_collect = consts.LABEL_COLLECT
    c_destroy = consts.LABEL_DESTROY
    c_default = consts.LABEL_DEFAULT
    
    concept = jnp.where(is_low_oxy, c_default,
               jnp.where(has_divers, c_collect,
                jnp.where(has_enemies, c_destroy, c_default)))
    return concept

def heuristic_selector(obs, concept_id, player_pos):
    """
    Rulebook for Phase 1.
    Logic: Pick NEAREST valid object for the chosen concept.
    """
    relevant_col = obs.object_labels[:, concept_id]
    is_valid = (relevant_col > 0.5) & (obs.object_mask > 0.5)
    
    obj_pos = obs.objects[:, 0:2]
    dists = jnp.sum((obj_pos - player_pos)**2, axis=-1)
    
    safe_dists = jnp.where(is_valid, dists, 1e9)
    
    target_idx = jnp.argmin(safe_dists)
    return target_idx

@struct.dataclass
class RelationalAgent:
    """
    The 'Brain' that coordinates the hierarchy.
    Implements soft hysteresis (stickiness) to prevent rapid flickering.
    """
    # Config is metadata (Static), so we tell JAX not to trace it
    config: Config = struct.field(pytree_node=False)
    
    # These contain the arrays (params) JAX needs to update
    meta_state: RLTrainState = None
    selector_state: RLTrainState = None
    sub_state: RLTrainState = None

    @classmethod
    def create(cls, key: chex.PRNGKey, config: Config, sample_obs: ConceptObs):
        key_m, key_s, key_sub = jax.random.split(key, 3)
        
        # 1. Meta Policy
        # Objects are flattened: [max_objs, stack_size * n_obj_feats]
        obj_dim = config.environment.frame_stack_size * config.environment.n_obj_feats
        meta_state = MetaPolicy.create_train_state(
            key_m,
            num_concepts=config.concepts.label_high_level_concept + 1,
            learning_rate=config.training.meta_policy_lr,
            global_dim=sample_obs.global_features.shape[-1],
            obj_dim=obj_dim,
            max_objs=config.environment.max_objs
        )

        # 2. Instance Selector
        num_concepts = config.concepts.label_high_level_concept + 1
        selector_state = InstanceSelector.create_train_state(
            key_s,
            learning_rate=config.training.selector_lr,
            global_dim=sample_obs.global_features.shape[-1],
            obj_dim=obj_dim,
            max_objs=config.environment.max_objs,
            num_concepts=num_concepts,
            hidden_dim=config.model.selector_hidden_dims[0] if len(config.model.selector_hidden_dims) > 0 else 64
        )

        # 3. Sub Policy
        sub_state = SubPolicy.create_train_state(
            key_sub,
            num_actions=config.environment.n_actions,
            learning_rate=config.training.sub_policy_lr,
            hidden_dims=config.model.hidden_dims,
            cap_dim=config.concepts.n_capabilities,
            num_concepts=config.concepts.label_high_level_concept + 1,
            global_dim=sample_obs.global_features.shape[-1]
        )
        
        return cls(config=config, meta_state=meta_state, selector_state=selector_state, sub_state=sub_state)

    @jax.jit
    def select_action(self, 
                      obs: ConceptObs, 
                      prev_concept: chex.Array, 
                      prev_target_idx: chex.Array,
                      key: chex.PRNGKey, 
                      epsilon_meta: float,
                      epsilon_sub: float,
                      global_step: int):
        
        # --- 0. Batch Dimension Check ---
        # We ensure obs features have a batch dimension for consistent processing.
        is_unbatched = obs.global_features.ndim == 1
        
        if is_unbatched:
            obs_in = jax.tree.map(lambda x: x[None, ...], obs)
            prev_concept_in = jnp.expand_dims(prev_concept, 0)
            prev_target_idx_in = jnp.expand_dims(prev_target_idx, 0)
        else:
            obs_in = obs
            prev_concept_in = prev_concept
            prev_target_idx_in = prev_target_idx

        k1, k_sub = jax.random.split(key)
        k1a, k1b, k_drill = jax.random.split(k1, 3)
        k2a, k2b = jax.random.split(k_sub)

        # --- CURRICULUM PHASES ---
        phase_1_steps = self.config.training.phase_1_steps
        phase_2_steps = self.config.training.phase_2_steps
        
        is_phase_1 = global_step < phase_1_steps
        is_phase_2 = (global_step >= phase_1_steps) & (global_step < phase_2_steps)
        
        # --- FORCED CONCEPT MODE ---
        # If forced concept mode is enabled, always select the forced concept
        force_concept_mode = self.config.training.force_concept_mode
        force_concept_id = self.config.training.force_concept_id
        
        # --- Layer 1: Strategic Decision (Meta) ---
        q_values_meta = self.meta_state.apply_fn(
            self.meta_state.params, 
            obs_in.global_features,
            obs_in.objects,
            obs_in.object_labels,
            obs_in.object_mask
        )
        
        batch_size = q_values_meta.shape[0]
        num_concepts = q_values_meta.shape[1]
        
        # --- Action Validity Masking ---
        labels_T = jnp.transpose(obs_in.object_labels, (0, 2, 1))
        object_exists = obs_in.object_mask[:, None, :] > 0.5
        has_label = labels_T * object_exists
        label_exists = jnp.any(has_label > 0.5, axis=-1)
        
        # 2. Construct the Mask
        # By default, concepts require their corresponding objects
        valid_mask = label_exists

        
        # Exception C: High Level / Monitor
        # These don't target specific objects, so they are always valid
        # Ensure Valid Mask allows LABEL_DEFAULT
        valid_mask = valid_mask.at[:, consts.LABEL_DEFAULT].set(True)
        
        # valid_mask = valid_mask.at[:, consts.LABEL_HIGH_LEVEL_CONCEPT].set(True)
        any_valid = jnp.any(valid_mask, axis=-1, keepdims=True)
        fallback_mask = jax.nn.one_hot(jnp.full((batch_size,), consts.LABEL_DEFAULT), num_concepts)
        valid_mask = jnp.where(any_valid, valid_mask, fallback_mask)
        
        # 3. Apply Mask to Q-Values
        large_neg = -1e9
        masked_q_values = q_values_meta + (1.0 - valid_mask.astype(jnp.float32)) * large_neg
        
        prev_concept_mask = jax.nn.one_hot(prev_concept_in, num_classes=num_concepts)
        adjusted_meta_q = masked_q_values + (prev_concept_mask * self.config.training.meta_stickiness)
        
        greedy_concept = jnp.argmax(adjusted_meta_q, axis=-1)
        
        # Random Selection (Must also respect mask!)
        # Instead of pure random, we sample from Softmax of mask
        # Add large noise to masked_q and argmax
        random_noise = jax.random.uniform(k1a, shape=masked_q_values.shape)
        noisy_q = masked_q_values + random_noise * 1e6
        random_concept = jnp.argmax(noisy_q, axis=-1)
        
        rng_mix = jax.random.uniform(k1b, shape=greedy_concept.shape)
        net_concept = jnp.where(rng_mix < epsilon_meta, random_concept, greedy_concept)

        # B. THE DRILL SERGEANT (Sticky Random Valid Choice for Phase 1)
        # During Phase 1, we randomly sample from valid concepts (except DEFAULT TODO: I think this is necessary, but can test..) and stick to them
        # until they become invalid. This ensures all Sub-Policies get training data.
        
        # 1. Check if previous concept is still valid
        # prev_concept_in is [Batch], valid_mask is [Batch, Num_Concepts]
        prev_concept_idx = prev_concept_in[:, None]  # [Batch, 1]
        prev_is_valid = jnp.take_along_axis(valid_mask, prev_concept_idx, axis=1).squeeze(1) > 0.5
        
        # 2. Pick new random concept from valid ones
        # Add uniform noise to valid_mask and argmax to get random valid concept
        drill_noise = jax.random.uniform(k_drill, shape=valid_mask.shape)
        # Only consider valid concepts (multiply by valid_mask)
        weighted_valid = valid_mask.astype(jnp.float32) * drill_noise
        # Remove DEFAULT from valid concepts
        weighted_valid = weighted_valid.at[:, consts.LABEL_DEFAULT].set(0.0)
        new_random_concept = jnp.argmax(weighted_valid, axis=-1)
        
        # 3. Resolve Drill Concept: stick to previous if still valid, otherwise pick new
        drill_concept = jnp.where(prev_is_valid, prev_concept_in, new_random_concept)
        
        # C. Heuristic Output (for Phase 2)
        if obs_in.global_features.ndim == 1:
            rule_concept = jnp.expand_dims(heuristic_meta(obs_in), 0)
        else:
            rule_concept = jax.vmap(heuristic_meta)(obs_in)
        
        # D. Switch based on Phase
        # Phase 1: Use Sticky Drill (random valid, persists until invalid)
        # Phase 2: Use Heuristic (priority-based)
        # Phase 3: Use Network (full AI)
        normal_selected_concept = jnp.where(
            is_phase_1, 
            drill_concept,
            jnp.where(is_phase_2, rule_concept, net_concept)
        )
        
        # E. Apply Forced Concept Mode (if enabled)
        # If force_concept_mode is True and force_concept_id is set, override the selection
        # Ensure selected_concept has the same shape as normal_selected_concept
        if force_concept_mode and force_concept_id is not None:
            # Always create with batch dimension to match normal_selected_concept shape
            selected_concept = jnp.full(normal_selected_concept.shape, force_concept_id, dtype=jnp.int32)
        else:
            selected_concept = normal_selected_concept

        # --- Layer 2: Tactical Decision (Selector) ---
        # PREPARE COMMAND: Convert integer concept to One-Hot Vector
        # This acts as the "Command Signal" for the Selector
        cmd_one_hot = jax.nn.one_hot(selected_concept, num_classes=num_concepts)
        
        # A. Network Output (Q-Values) - Conditioned on Command
        # Inspector rates EVERY object, knowing what command it's executing
        raw_scores = self.selector_state.apply_fn(
            self.selector_state.params,
            obs_in.global_features,
            obs_in.objects,
            obs_in.object_mask,
            cmd_one_hot
        )
        
        # B. The Semantic Gate (Pre-Masking Logic)
        # Even though the network *knows* the command, we still strictly enforce 
        # that it cannot pick invalid objects (e.g. collecting a Shark).
        # Extract the column for the active concept
        # obs.object_labels: [Batch, N, Num_Concepts]
        concept_mask = jnp.take_along_axis(
            obs_in.object_labels,
            selected_concept[:, None, None],
            axis=2
        ).squeeze(2)  # Result: [Batch, N] (1.0 = Valid, 0.0 = Invalid)
        
        # C. Apply Gate
        # If label says "No", score becomes -1e9 (Ignored)
        masked_scores = raw_scores + (1.0 - concept_mask) * large_neg
        
        # D. Greedy Selection
        greedy_target = jnp.argmax(masked_scores, axis=-1)
        
        # E. Epsilon Greedy (Exploration) for Selector
        # Add random noise to masked scores to pick random valid object
        noise = jax.random.uniform(k2a, shape=masked_scores.shape, minval=0.0, maxval=1.0)
        # Only add noise to valid targets (prevent picking invalid via noise)
        noisy_scores = masked_scores + (noise * concept_mask * 100.0)
        
        random_valid_target = jnp.argmax(noisy_scores, axis=-1)
        
        rng_mix_sel = jax.random.uniform(k2b, shape=greedy_target.shape)
        net_target_idx = jnp.where(rng_mix_sel < epsilon_sub, random_valid_target, greedy_target)
        
        # B. Heuristic Output
        p_x = obs_in.global_features[:, 5]
        p_y = obs_in.global_features[:, 6]
        player_pos = jnp.stack([p_x, p_y], axis=-1)
        
        def sel_heuristic_single(b_idx):
            return heuristic_selector(
                jax.tree.map(lambda x: x[b_idx], obs_in),
                selected_concept[b_idx],
                player_pos[b_idx]
            )
        
        rule_target_idx = jax.vmap(sel_heuristic_single)(jnp.arange(batch_size))
        
        # C. Switch (Phase 1 forces Heuristic)
        use_rule_sel = is_phase_1
        target_idx = jnp.where(use_rule_sel, rule_target_idx, net_target_idx)
        target_obj_feats = jnp.take_along_axis(obs_in.objects, target_idx[:, None, None], axis=1).squeeze(1)

        # --- Layer 3: Relative Vector Calculation ---
        target_pos = target_obj_feats[:, 0:2]
        raw_relative = target_pos - player_pos 
        
        # 4. Zero-out vector for DEFAULT Concept
        # If we are in "Default" mode, we don't care about the target object.
        # We mask the relative vector to 0.0 so the Sub-Policy learns purely from Globals.
        is_default = (selected_concept == consts.LABEL_DEFAULT)
        relative_vec = jnp.where(is_default[:, None], jnp.zeros_like(raw_relative), raw_relative) 

        # --- Layer 4: Motor Control (Sub) ---
        num_concepts = self.config.concepts.label_high_level_concept + 1
        concept_one_hot = jax.nn.one_hot(selected_concept, num_classes=num_concepts)
        expanded_capabilities = jnp.concatenate([obs_in.capabilities, concept_one_hot], axis=-1)
        
        # Standard Sub-Policy execution (used for all concepts initially)
        action_std, sub_q_std = SubPolicy.select_action_eps_greedy(
            self.sub_state, obs_in.global_features, relative_vec, expanded_capabilities, epsilon_sub, k_sub
        )

        # Implementation of the Aggregation Logic for AVOID
        def get_avoidance_q(b_idx):
            # 1. Identify threats
            labels = obs_in.object_labels[b_idx]
            is_threat = (labels[:, consts.LABEL_AVOID] > 0.5) & (obs_in.object_mask[b_idx] > 0.5)
            
            # 2. Get relative vectors for ALL threats
            all_target_pos = obs_in.objects[b_idx, :, 0:2]
            all_rel_vecs = all_target_pos - player_pos[b_idx]
            
            # 3. Context for AVOID
            avoid_one_hot = jax.nn.one_hot(consts.LABEL_AVOID, num_classes=num_concepts)
            avoid_caps = jnp.concatenate([obs_in.capabilities[b_idx], avoid_one_hot], axis=-1)
            n_objs = obs_in.objects.shape[1]
            avoid_caps_batch = jnp.tile(avoid_caps[None, :], (n_objs, 1))
            glob_batch = jnp.tile(obs_in.global_features[b_idx][None, :], (n_objs, 1))
            
            # 4. Batch execute Sub-Policy
            all_q = self.sub_state.apply_fn(self.sub_state.params, glob_batch, all_rel_vecs, avoid_caps_batch)
            
            # 5. Mask out non-threats and sum Q-values
            masked_q = all_q * is_threat[:, None]
            return jnp.sum(masked_q, axis=0)

        all_avoid_q = jax.vmap(get_avoidance_q)(jnp.arange(batch_size))
        
        # Action selection for aggregated avoidance
        greedy_avoid_action = jnp.argmax(all_avoid_q, axis=-1)
        k_avoid, _ = jax.random.split(k_sub)
        rng_mix_sub = jax.random.uniform(k_avoid, shape=greedy_avoid_action.shape)
        action_avoid = jnp.where(rng_mix_sub < epsilon_sub, 
                                 jax.random.randint(k_avoid, shape=greedy_avoid_action.shape, minval=0, maxval=sub_q_std.shape[-1]), 
                                 greedy_avoid_action)

        is_avoid = (selected_concept == consts.LABEL_AVOID)
        final_action = jnp.where(is_avoid, action_avoid, action_std)
        final_q = jnp.where(is_avoid[:, None], all_avoid_q, sub_q_std)

        # --- 5. Auto-Squeeze Logic ---
        if is_unbatched:
            return AgentState(
                action=final_action.squeeze(0),
                selected_concept=selected_concept.squeeze(0),
                selected_target_idx=target_idx.squeeze(0),
                log_probs=jnp.max(final_q, axis=-1).squeeze(0)
            )
        else:
            return AgentState(
                action=final_action,
                selected_concept=selected_concept,
                selected_target_idx=target_idx,
                log_probs=jnp.max(final_q, axis=-1)
            )