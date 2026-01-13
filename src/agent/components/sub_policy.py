# References:
# Adapted from 'Symbolic Options' (https://github.com/remunds/symbolic_options)
# Specifically adapting the QNetwork architecture for Relational Inputs.

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import chex
from typing import Tuple
from src.rl_utils import RLTrainState

class SubPolicyQNetwork(nn.Module):
    """
    The Executor Network.
    
    Instead of processing the entire screen, it processes a focused 'Concept'.
    I.e. Collecting an object, avoiding an object, etc.
    
    Inputs:
        relative_vec: [Batch, 2]  (dx, dy) to the target.
        capabilities: [Batch, N]  (can_fire, etc.) flags relevant to the concept.
        
    Outputs:
        q_values: [Batch, Num_Actions]
    """
    num_actions: int # correspods to the output layer of the network
    hidden_dims: Tuple[int, ...]

    @nn.compact
    def __call__(self, relative_vec: chex.Array, capabilities: chex.Array):
        # --- FIX: Normalize Inputs ---
        # Seaquest screen is roughly 160x210. 
        # relative_vec is (target - player), so range is approx [-160, 160].
        # We divide by 160.0 to squash it to ~[-1.0, 1.0]
        norm_rel_vec = relative_vec / 160.0
        
        # 1. Input Fusion
        # capabilities are [0, 1].
        x = jnp.concatenate([norm_rel_vec, capabilities], axis=-1)

        # 2. The Logic Body (MLP with LayerNorm)
        # Low number of hidden dims since the concepts are already low level.
        # LayerNorm is CRITICAL for PQN stability - prevents target collapse
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.LayerNorm()(x)  # CRITICAL FOR PQN STABILITY
            x = nn.relu(x) # Simple ReLU is robust for Q-learning

        # 3. Action Head
        # No activation at the end (Q-values are unbounded regression targets)
        q_values = nn.Dense(features=self.num_actions)(x)
        
        return q_values

class SubPolicy:
    """
    Namespace for Sub-Policy utility functions.
    This replaces a stateful Class to keep JAX purity.
    """

    @staticmethod
    def create_train_state(key, num_actions: int, learning_rate: float, 
                           hidden_dims: Tuple[int, ...], cap_dim: int = 4, 
                           num_concepts: int = 7) -> RLTrainState:
        """
        Initializes the model parameters and optimizer with target network support.
        Note: capabilities input includes concept context, so total input size is cap_dim + num_concepts.
        """
        net = SubPolicyQNetwork(num_actions=num_actions, hidden_dims=hidden_dims)
        
        # Dummy inputs for shape inference
        # Shape: [1, 2] and [1, cap_dim + num_concepts] (includes concept one-hot)
        dummy_rel = jnp.zeros((1, 2))
        expanded_cap_dim = cap_dim + num_concepts
        dummy_cap = jnp.zeros((1, expanded_cap_dim))
        
        params = net.init(key, dummy_rel, dummy_cap)
        
        tx = optax.adam(learning_rate=learning_rate)
        
        return RLTrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=tx
        )

    @staticmethod
    @jax.jit
    def select_action_eps_greedy(train_state: RLTrainState, 
                                 relative_vec: chex.Array, 
                                 capabilities: chex.Array, 
                                 epsilon: float, 
                                 key: chex.PRNGKey):
        """
        Standard Epsilon-Greedy selection for Q-Learning.
        """
        # 1. Get Q-Values
        q_values = train_state.apply_fn(train_state.params, relative_vec, capabilities)
        
        # 2. Greedy Choice (Argmax)
        greedy_action = jnp.argmax(q_values, axis=-1)
        
        # 3. Random Choice
        random_action = jax.random.randint(key, shape=greedy_action.shape, 
                                           minval=0, maxval=q_values.shape[-1])
        
        # 4. Mix based on Epsilon
        # If rand < eps -> Random, else Greedy
        rng_mix = jax.random.uniform(key, shape=greedy_action.shape)
        action = jnp.where(rng_mix < epsilon, random_action, greedy_action)
        
        return action, q_values

    @staticmethod
    @jax.jit
    def get_q_values(train_state: RLTrainState, 
                     relative_vec: chex.Array, 
                     capabilities: chex.Array):
        """
        Helper for the update step (calculating loss).
        """
        return train_state.apply_fn(train_state.params, relative_vec, capabilities)