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
    selector_state: TrainState = None
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
        selector_net = InstanceSelector(
            num_concepts=config.concepts.label_high_level_concept + 1,
            embed_dim=config.model.selector_hidden_dims[0]
        )
        # Dummy inputs for init
        # We ensure these have the Batch dimension [1, ...]
        # Objects are flattened: [max_objs, stack_size * n_obj_feats]
        d_glob = jnp.zeros((1, sample_obs.global_features.shape[-1]))
        d_obj = jnp.zeros((1, config.environment.max_objs, obj_dim))
        d_labels = jnp.zeros((1, config.environment.max_objs), dtype=jnp.int32)
        d_mask = jnp.zeros((1, config.environment.max_objs))
        
        sel_params = selector_net.init(key_s, d_glob, d_obj, d_labels, d_mask)
        sel_tx = optax.adam(config.training.selector_lr)
        selector_state = TrainState.create(apply_fn=selector_net.apply, params=sel_params, tx=sel_tx)

        # 3. Sub Policy
        sub_state = SubPolicy.create_train_state(
            key_sub,
            num_actions=config.environment.n_actions,
            learning_rate=config.training.sub_policy_lr,
            hidden_dims=config.model.hidden_dims,
            cap_dim=config.concepts.n_capabilities,
            num_concepts=config.concepts.label_high_level_concept + 1
        )
        
        return cls(config=config, meta_state=meta_state, selector_state=selector_state, sub_state=sub_state)

    @jax.jit
    def select_action(self, 
                      obs: ConceptObs, 
                      prev_concept: chex.Array, 
                      prev_target_idx: chex.Array,
                      key: chex.PRNGKey, 
                      epsilon_meta: float,
                      epsilon_sub: float):
        
        # --- 0. Auto-Batching Logic ---
        # If input is unbatched (rank 1 global features), expand dims to [1, ...]
        is_unbatched = obs.global_features.ndim == 1
        
        if is_unbatched:
            # Expand all leaves in the ConceptObs struct
            obs_in = jax.tree.map(lambda x: x[None, ...], obs)
            # Expand scalars
            prev_concept_in = jnp.expand_dims(prev_concept, 0)
            prev_target_idx_in = jnp.expand_dims(prev_target_idx, 0)
        else:
            obs_in = obs
            prev_concept_in = prev_concept
            prev_target_idx_in = prev_target_idx

        k1, k2 = jax.random.split(key)
        k1a, k1b = jax.random.split(k1)

        # --- Layer 1: Strategic Decision (Meta) ---
        q_values_meta = self.meta_state.apply_fn(
            self.meta_state.params, 
            obs_in.global_features,
            obs_in.objects,
            obs_in.object_labels,
            obs_in.object_mask
        )
        
        # --- Action Validity Masking ---
        # 1. Identify which concepts are possible right now
        # Check existence of objects for each label
        # labels: [Batch, Max_Objs]
        
        batch_size = q_values_meta.shape[0]
        num_concepts = q_values_meta.shape[1]
        
        # Count objects per label
        # We want a mask: [Batch, Num_Concepts]
        # label_counts[b, c] > 0 if label 'c' exists in batch 'b'
        # We use vmap or broadcasting
        
        # Expand labels: [Batch, 1, Max_Objs]
        labels_exp = obs_in.object_labels[:, None, :]
        # Expand concepts: [1, Num_Concepts, 1]
        concepts_exp = jnp.arange(num_concepts)[None, :, None]
        # Check match: [Batch, Num_Concepts, Max_Objs]
        has_label = (labels_exp == concepts_exp) & (obs_in.object_mask[:, None, :] > 0.5)
        # Reduce: [Batch, Num_Concepts]
        label_exists = jnp.any(has_label, axis=-1)
        
        # 2. Construct the Mask
        # By default, concepts require their corresponding objects
        valid_mask = label_exists
        
        # Exception A: Concept 0 (Player)
        # If we want to BLOCK it, set to False.
        # If we want to ALLOW it (as a fallback), set to True.
        # You requested to BLOCK it.
        valid_mask = valid_mask.at[:, consts.LABEL_PLAYER].set(False) 
        
        # Exception B: Concept 1 (Tool/Projectile) - usually exists if Player exists
        valid_mask = valid_mask.at[:, consts.LABEL_PLAYER_TOOL].set(False) # Block tool concept too? usually internal
        
        # Exception C: High Level / Monitor
        # These don't target specific objects, so they are always valid
        valid_mask = valid_mask.at[:, consts.LABEL_MONITOR_STATISTIC].set(True)
        # valid_mask = valid_mask.at[:, consts.LABEL_HIGH_LEVEL_CONCEPT].set(True)

        # Safety: If EVERYTHING is masked (e.g. empty screen), allow 'Monitor' or 'Destroy' (blind fire)
        any_valid = jnp.any(valid_mask, axis=-1, keepdims=True)
        # If nothing valid, force Monitor (Index 5) or similar safe state
        fallback_mask = jax.nn.one_hot(jnp.full((batch_size,), consts.LABEL_MONITOR_STATISTIC), num_concepts)
        valid_mask = jnp.where(any_valid, valid_mask, fallback_mask)
        
        # 3. Apply Mask to Q-Values
        large_neg = -1e9
        masked_q_values = q_values_meta + (1.0 - valid_mask.astype(jnp.float32)) * large_neg
        
        # --- Standard Selection Logic ---
        
        # Stickiness (Apply to masked Q)
        prev_concept_mask = jax.nn.one_hot(prev_concept_in, num_classes=num_concepts)
        adjusted_meta_q = masked_q_values + (prev_concept_mask * self.config.training.meta_stickiness)
        
        greedy_concept = jnp.argmax(adjusted_meta_q, axis=-1)
        
        # Random Selection (Must also respect mask!)
        # Instead of pure random, we sample from Softmax of mask
        # Add large noise to masked_q and argmax
        random_noise = jax.random.uniform(k1a, shape=masked_q_values.shape)
        noisy_q = masked_q_values + random_noise * 1e6 # Noise dominates Q, but Mask dominates Noise
        random_concept = jnp.argmax(noisy_q, axis=-1)
        
        rng_mix = jax.random.uniform(k1b, shape=greedy_concept.shape)
        concept_id = jnp.where(rng_mix < epsilon_meta, random_concept, greedy_concept)

        # --- Layer 2: Tactical Decision (Selector) ---
        attn_weights = self.selector_state.apply_fn(
            self.selector_state.params,
            obs_in.global_features,
            obs_in.objects,
            obs_in.object_labels,
            obs_in.object_mask
        )
        
        # Get scores for the active concept
        active_scores = jnp.take_along_axis(attn_weights, concept_id[:, None, None], axis=1).squeeze(1)
        
        # [Stickiness]
        concept_unchanged = (concept_id == prev_concept_in).astype(jnp.float32)
        prev_target_mask = jax.nn.one_hot(prev_target_idx_in, num_classes=active_scores.shape[-1])
        
        stickiness_bonus = prev_target_mask * self.config.training.selector_stickiness * concept_unchanged[:, None]
        adjusted_scores = active_scores + stickiness_bonus
        
        # Select target (Argmax on adjusted scores)
        target_idx = jnp.argmax(adjusted_scores, axis=-1)
        target_obj_feats = jnp.take_along_axis(obs_in.objects, target_idx[:, None, None], axis=1).squeeze(1)

        # --- Layer 3: Translation ---
        player_pos = obs_in.objects[:, 0, 0:2] 
        target_pos = target_obj_feats[:, 0:2]
        relative_vec = target_pos - player_pos 

        # --- Layer 4: Motor Control (Sub) ---
        num_concepts = self.config.concepts.label_high_level_concept + 1
        concept_one_hot = jax.nn.one_hot(concept_id, num_classes=num_concepts)
        expanded_capabilities = jnp.concatenate([obs_in.capabilities, concept_one_hot], axis=-1)
        
        action, sub_q = SubPolicy.select_action_eps_greedy(
            self.sub_state,
            relative_vec,
            expanded_capabilities,
            epsilon_sub,
            k2
        )

        # --- 5. Auto-Squeeze Logic ---
        if is_unbatched:
            return AgentState(
                action=action.squeeze(0),
                selected_concept=concept_id.squeeze(0),
                selected_target_idx=target_idx.squeeze(0),
                log_probs=jnp.max(sub_q, axis=-1).squeeze(0)
            )
        else:
            return AgentState(
                action=action,
                selected_concept=concept_id,
                selected_target_idx=target_idx,
                log_probs=jnp.max(sub_q, axis=-1)
            )