 
# src/agent/components/meta_policy.py
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from src.rl_utils import RLTrainState

# --- Config ---
# TODO: might be better to use attention mechanism (instead of self-attention) as it learns faster.
# Question is, is there viable information in the relation between objects that helps the decision making or are the relations player <-> objects enough?
# TODO: if using self-attention might be worth switching from max-pooling to just giving the player token to the decision mlp as a sort of 'summary' (similar to what BERT does in their models)
class MetaPolicyQNetwork(nn.Module):
    """
    Relational Meta-Policy.
    Uses Self-Attention to understand the 'Board State' before making a strategic decision.
    """
    num_concepts: int
    hidden_dim: int = 128
    num_heads: int = 2
    
    @nn.compact
    def __call__(self, global_features, objects, object_labels, object_mask):
        # --- 1. Normalize Inputs (Crucial) ---
        # Globals: [Batch, 10]. Oxygen is ~100. Scale down.
        # We assume globals are [Oxygen, Score, Held_Divers, ...]
        # A rough division by 100.0 puts them in [0, 1] range.
        x_glob = global_features / 100.0
        
        # --- 2. Embeddings ---
        # Embed Global Context
        emb_glob = nn.Dense(self.hidden_dim)(x_glob)
        emb_glob = nn.LayerNorm()(emb_glob)
        
        # Embed Objects (Shared weights for all 35 objects)
        emb_objs = nn.Dense(self.hidden_dim)(objects) # [Batch, 35, Dim]
        
        # Linear projection of the object labels (since they are already multi-hot encoded)
        label_embeddings = nn.Dense(features=self.hidden_dim, use_bias=False)(object_labels)
        
        # Combine Object Features + Label
        x_items = emb_objs + label_embeddings
        x_items = nn.LayerNorm()(x_items)

        # --- 3. Relational Reasoning (Transformer Block) ---
        # Create attention mask: Don't look at padding objects (where mask is 0)
        # Mask shape for MultiHeadDotProductAttention: [Batch, 1, 1, Objects]
        attn_mask = object_mask[:, None, None, :]
        
        # Self-Attention: Objects look at other objects
        # "Is there a shark near the diver?"
        x_attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.hidden_dim)(
            inputs_q=x_items,
            inputs_kv=x_items,
            mask=attn_mask
        )
        x_items = x_items + x_attn # Residual
        x_items = nn.LayerNorm()(x_items)
        
        # Feed Forward block
        x_ff = nn.Dense(self.hidden_dim)(x_items)
        x_ff = nn.relu(x_ff)
        x_ff = nn.Dense(self.hidden_dim)(x_ff)
        x_items = x_items + x_ff # Residual
        x_items = nn.LayerNorm()(x_items)

        # --- 4. Aggregation ---
        # We need to summarize the 35 objects into one "Situation Vector".
        # We use Max-Pooling masked by existence.
        # Set padding to -infinity before max
        x_items = x_items + (1.0 - object_mask[:, :, None]) * -1e9
        situation_vector = jnp.max(x_items, axis=1) # [Batch, Dim]

        # --- 5. Decision ---
        # Combine "My Status" (Globals) with "Situation" (Objects)
        z = jnp.concatenate([emb_glob, situation_vector], axis=-1)
        
        z = nn.Dense(self.hidden_dim)(z)
        z = nn.relu(z)
        z = nn.Dense(self.num_concepts)(z)
        
        return z

    @staticmethod
    def create_train_state(key, num_concepts, learning_rate, global_dim, obj_dim, max_objs):
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        # Dummy init with correct shapes
        d_glob = jnp.zeros((1, global_dim))
        d_objs = jnp.zeros((1, max_objs, obj_dim))
        # object_labels shape: [Batch, Max_Objs, Num_Concepts] (multi-hot encoding)
        d_labels = jnp.zeros((1, max_objs, num_concepts), dtype=jnp.float32)
        d_mask = jnp.zeros((1, max_objs))
        
        params = net.init(key, d_glob, d_objs, d_labels, d_mask)
        tx = optax.adam(learning_rate)
        return RLTrainState.create(apply_fn=net.apply, params=params, tx=tx)

class MetaPolicy:
    @staticmethod
    def create_train_state(key, num_concepts, learning_rate, global_dim, obj_dim=None, max_objs=None):
        """
        Creates RLTrainState for Meta Policy with target network support.
        Delegates to MetaPolicyQNetwork.create_train_state for consistency.
        """
        if obj_dim is None or max_objs is None:
            # Fallback for backward compatibility
            obj_dim = 4  # Default fallback
            max_objs = 35  # Default fallback
        
        return MetaPolicyQNetwork.create_train_state(
            key=key,
            num_concepts=num_concepts,
            learning_rate=learning_rate,
            global_dim=global_dim,
            obj_dim=obj_dim,
            max_objs=max_objs
        )

    @staticmethod
    @jax.jit
    def select_concept_eps_greedy(train_state, global_features, objects, object_labels, object_mask, epsilon, key):
        q_values = train_state.apply_fn(train_state.params, global_features, objects, object_labels, object_mask)
        
        greedy = jnp.argmax(q_values, axis=-1)
        random = jax.random.randint(key, shape=greedy.shape, minval=0, maxval=q_values.shape[-1])
        rng_mix = jax.random.uniform(key, shape=greedy.shape)
        return jnp.where(rng_mix < epsilon, random, greedy), q_values