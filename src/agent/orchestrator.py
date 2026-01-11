# src/agents/orchestrator.py
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import chex
from typing import Tuple

from src.agent.components.meta_policy import MetaPolicy, MetaPolicyQNetwork
from src.agent.components.sub_policy import SubPolicy, SubPolicyQNetwork
from src.agent.components.instance_selector import InstanceSelector
from src.config import Config
from src.environments.seaquest.datatypes import ConceptObs, AgentState
import optax

class RelationalAgent:
    """
    The 'Brain' that coordinates the hierarchy.
    Implements k-step persistence to prevent Meta-Policy flickering.
    """
    def __init__(self, 
                 config: Config, 
                 meta_state: TrainState,
                 selector_state: TrainState,
                 sub_state: TrainState,
                 meta_interval: int = 10): # New: Persistence Interval
        self.config = config
        self.meta_state = meta_state
        self.selector_state = selector_state
        self.sub_state = sub_state
        self.meta_interval = meta_interval

    @classmethod
    def create(cls, key: chex.PRNGKey, config: Config, sample_obs: ConceptObs):
        key_m, key_s, key_sub = jax.random.split(key, 3)
        
        # 1. Meta Policy (Global -> Concept)
        meta_state = MetaPolicy.create_train_state(
            key_m,
            num_concepts=config.concepts.label_high_level_concept + 1,
            learning_rate=config.training.meta_policy_lr,
            global_dim=sample_obs.global_features.shape[-1]
        )

        # 2. Instance Selector (Objects -> Target Index)
        # Manually create TrainState as instance_selector.py lacked the helper
        selector_net = InstanceSelector(
            num_concepts=config.concepts.label_high_level_concept + 1,
            embed_dim=64 # Should move to config
        )
        # Dummy inputs
        d_glob = jnp.zeros((1, sample_obs.global_features.shape[-1]))
        d_obj = jnp.zeros((1, config.environment.max_objs, config.environment.n_obj_feats))
        d_mask = jnp.zeros((1, config.environment.max_objs))
        
        sel_params = selector_net.init(key_s, d_glob, d_obj, d_mask)
        sel_tx = optax.adam(config.training.selector_lr)
        selector_state = TrainState.create(apply_fn=selector_net.apply, params=sel_params, tx=sel_tx)

        # 3. Sub Policy (Relative -> Action)
        sub_state = SubPolicy.create_train_state(
            key_sub,
            num_actions=18, # Atari standard
            learning_rate=config.training.sub_policy_lr,
            hidden_dims=config.model.hidden_dims,
            cap_dim=config.concepts.n_capabilities
        )
        
        return cls(config, meta_state, selector_state, sub_state)

    @jax.jit
    def select_action(self, 
                      obs: ConceptObs, 
                      prev_concept: chex.Array, 
                      key: chex.PRNGKey, 
                      epsilon_meta: float,
                      epsilon_sub: float,
                      inertia_bias: float = 0.5): # <--- The "Stickiness" Parameter
        
        k1, k2 = jax.random.split(key)
        k1a, k1b = jax.random.split(k1)

        # --- Layer 1: Strategic Decision (Meta) ---
        # 1. Get Raw Q-Values
        q_values_meta = self.meta_state.apply_fn(
            self.meta_state.params, 
            obs.global_features
        )
        
        # 2. Apply "Stickiness" (Hysteresis)
        # Create a one-hot vector for the concept we chose last step
        prev_one_hot = jax.nn.one_hot(prev_concept, num_classes=q_values_meta.shape[-1])
        
        # Add bonus: If Q(Avoid) is HUGE, it will still overpower this bonus.
        # If Q(Collect) and Q(Surface) are close, this keeps us on track.
        adjusted_q = q_values_meta + (prev_one_hot * inertia_bias)
        
        # 3. Select Concept (Epsilon Greedy on ADJUSTED values)
        greedy_concept = jnp.argmax(adjusted_q, axis=-1)
        
        random_concept = jax.random.randint(k1a, shape=greedy_concept.shape, 
                                            minval=0, maxval=q_values_meta.shape[-1])
        
        rng_mix = jax.random.uniform(k1b, shape=greedy_concept.shape)
        concept_id = jnp.where(rng_mix < epsilon_meta, random_concept, greedy_concept)

        # --- Layer 2: Tactical Decision (Selector) ---
        # "Which object is the best target for that concept?"
        attn_weights = self.selector_state.apply_fn(
            self.selector_state.params,
            obs.global_features,
            obs.objects,
            obs.object_mask
        )
        
        # Target selection depends on the (potentially persistent) concept_id
        target_obj_feats, target_idx = InstanceSelector.get_target_object(
            obs.objects, attn_weights, concept_id
        )

        # --- Layer 3: Translation (Geometry) ---
        # Calculate relative vector: Target - Player
        # Assuming Object 0 is Player (Standard in your Wrapper)
        player_pos = obs.objects[:, 0, 0:2] 
        target_pos = target_obj_feats[:, 0:2]
        relative_vec = target_pos - player_pos 

        # --- Layer 4: Motor Control (Sub-Policy) ---
        action, sub_q = SubPolicy.select_action_eps_greedy(
            self.sub_state,
            relative_vec,
            obs.capabilities,
            epsilon_sub,
            k2
        )

        return AgentState(
            action=action,
            selected_concept=concept_id,
            selected_target_idx=target_idx,
            log_probs=jnp.max(sub_q, axis=-1)
        )