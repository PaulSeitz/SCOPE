import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import optax
from src.rl_utils import RLTrainState

class InstanceSelector(nn.Module):
    """
    Command-Conditioned Scorer.
    Input: Global Context + Command (Concept) + Object
    Output: Relevance Score
    """
    hidden_dim: int = 64
    num_concepts: int = 4

    @nn.compact
    def __call__(self, global_features: chex.Array, objects: chex.Array, object_mask: chex.Array, concept_one_hot: chex.Array):
        """
        Args:
            global_features: [Batch, G]
            objects: [Batch, N, F]
            object_mask: [Batch, N]
            concept_one_hot: [Batch, Num_Concepts]
        Returns:
            q_values: [Batch, N] (Raw scores)
        """
        n_objects = objects.shape[1]

        # 1. Expand Globals AND Command to match object count
        # We broadcast the "Orders" to every "Inspector"
        # [Batch, G] -> [Batch, N, G]
        global_expanded = jnp.tile(global_features[:, None, :], (1, n_objects, 1))
        
        # [Batch, C] -> [Batch, N, C]
        command_expanded = jnp.tile(concept_one_hot[:, None, :], (1, n_objects, 1))
        
        # 2. Fuse Everything: Context + Command + Object
        # The network now knows: "We are in danger (Context), Command is Fight (Command), This is a Shark (Object)"
        x = jnp.concatenate([global_expanded, command_expanded, objects], axis=-1)
        
        # 3. Shared Network (PQN-compliant: LayerNorm after activation for internal normalization)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        h = nn.LayerNorm()(h)
        
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        h = nn.LayerNorm()(h)
        
        # 4. Score (Head)
        # Features entering head are now normalized (zero-mean, unit-variance)
        q_values = nn.Dense(1)(h).squeeze(-1)
        
        # 5. Existence Masking (Standard)
        large_neg = -1e9
        q_values = jnp.where(object_mask > 0.5, q_values, large_neg)
        
        return q_values

    @staticmethod
    def create_train_state(key, learning_rate, global_dim, obj_dim, max_objs, num_concepts, hidden_dim=64,
                          max_grad_norm=None, lr_linear_decay=False, total_steps=None):
        from src.rl_utils import create_optimizer
        
        net = InstanceSelector(hidden_dim=hidden_dim, num_concepts=num_concepts)
        
        # Dummy inputs
        d_glob = jnp.zeros((1, global_dim))
        d_obj = jnp.zeros((1, max_objs, obj_dim))
        d_mask = jnp.zeros((1, max_objs))
        d_conc = jnp.zeros((1, num_concepts))
        
        params = net.init(key, d_glob, d_obj, d_mask, d_conc)
        tx = create_optimizer(
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            lr_linear_decay=lr_linear_decay,
            total_steps=total_steps
        )
        return RLTrainState.create(apply_fn=net.apply, params=params, tx=tx)