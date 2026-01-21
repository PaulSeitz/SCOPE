 
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
    TODO: pull hidden_dim sizes etc into config
    """
    num_concepts: int
    hidden_dim: int = 64
    num_heads: int = 2
    
    @nn.compact
    def __call__(self, global_features, objects, object_mask):
        # --- 1. Embeddings ---
        # Embed Global Context
        emb_glob = nn.Dense(self.hidden_dim)(global_features)
        emb_glob = nn.LayerNorm()(emb_glob)  # Internal normalization after embedding
        
        # Embed Objects (Shared weights for all objects)
        # Objects now include spatial features (x, y, w, h) + labels, so embedding learns joint representation
        x_items = nn.Dense(self.hidden_dim)(objects) # [Batch, Max_Objs, Dim]
        x_items = nn.LayerNorm()(x_items)  # Internal normalization after embedding

        # --- 2. Add Learnable Summary Token (BERT-style [CLS] token) ---
        # Create a learnable "summary token" that will aggregate information from all objects
        # This is similar to BERT's [CLS] token - it learns to summarize the entire scene
        batch_size = objects.shape[0]
        summary_token = self.param('summary_token', 
                                   nn.initializers.zeros_init(),
                                   (1, self.hidden_dim))  # [1, hidden_dim]
        summary_token_batch = jnp.tile(summary_token[None, :, :], (batch_size, 1, 1))  # [Batch, 1, hidden_dim]
        
        # Prepend summary token to object sequence: [Batch, 1+Max_Objs, hidden_dim]
        x_sequence = jnp.concatenate([summary_token_batch, x_items], axis=1)
        
        # Update mask: summary token is always valid (mask=1), then object mask
        summary_mask = jnp.ones((batch_size, 1))  # [Batch, 1]
        sequence_mask = jnp.concatenate([summary_mask, object_mask], axis=1)  # [Batch, 1+Max_Objs]

        # --- 3. Relational Reasoning (Transformer Block) ---
        # Create attention mask: Don't look at padding objects (where mask is 0)
        # Mask shape for MultiHeadDotProductAttention: [Batch, 1, 1, 1+Max_Objs]
        attn_mask = sequence_mask[:, None, None, :]
        
        # Self-Attention: Summary token and objects attend to each other
        # The summary token learns to aggregate relevant information from all objects
        x_attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.hidden_dim)(
            inputs_q=x_sequence,
            inputs_kv=x_sequence,
            mask=attn_mask
        )
        x_sequence = x_sequence + x_attn # Residual
        x_sequence = nn.LayerNorm()(x_sequence)
        
        # Feed Forward block
        x_ff = nn.Dense(self.hidden_dim)(x_sequence)
        x_ff = nn.relu(x_ff)
        x_ff = nn.Dense(self.hidden_dim)(x_ff)
        x_sequence = x_sequence + x_ff # Residual
        x_sequence = nn.LayerNorm()(x_sequence)

        # --- 4. Extract Situation Vector from Summary Token ---
        # The summary token (first token) now contains the aggregated situation information
        situation_vector = x_sequence[:, 0, :]  # [Batch, hidden_dim]

        # --- 5. Decision ---
        # Combine "My Status" (Globals) with "Situation" (Summary Token)
        z = jnp.concatenate([emb_glob, situation_vector], axis=-1)
        
        z = nn.Dense(self.hidden_dim)(z)
        z = nn.relu(z)
        z = nn.LayerNorm()(z)  # <--- PQN: Ensure features entering head are normalized
        
        z = nn.Dense(self.num_concepts)(z)  # Head
        return z

    @staticmethod
    def create_train_state(key, num_concepts, learning_rate, global_dim, obj_dim, max_objs,
                          max_grad_norm=None, lr_linear_decay=False, total_steps=None):
        from src.rl_utils import create_optimizer
        
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        # Dummy init with correct shapes
        # obj_dim now includes labels (spatial features + num_concepts)
        d_glob = jnp.zeros((1, global_dim))
        d_objs = jnp.zeros((1, max_objs, obj_dim))
        d_mask = jnp.zeros((1, max_objs))
        
        params = net.init(key, d_glob, d_objs, d_mask)
        tx = create_optimizer(
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            lr_linear_decay=lr_linear_decay,
            total_steps=total_steps
        )
        return RLTrainState.create(apply_fn=net.apply, params=params, tx=tx)

class MetaPolicy:
    @staticmethod
    def create_train_state(key, num_concepts, learning_rate, global_dim, obj_dim=None, max_objs=None,
                          max_grad_norm=None, lr_linear_decay=False, total_steps=None):
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
            max_objs=max_objs,
            max_grad_norm=max_grad_norm,
            lr_linear_decay=lr_linear_decay,
            total_steps=total_steps
        )

    @staticmethod
    @jax.jit
    def select_concept_eps_greedy(train_state, global_features, objects, object_mask, epsilon, key):
        q_values = train_state.apply_fn(train_state.params, global_features, objects, object_mask)
        
        greedy = jnp.argmax(q_values, axis=-1)
        random = jax.random.randint(key, shape=greedy.shape, minval=0, maxval=q_values.shape[-1])
        rng_mix = jax.random.uniform(key, shape=greedy.shape)
        return jnp.where(rng_mix < epsilon, random, greedy), q_values