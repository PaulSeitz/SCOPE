 
# src/agents/meta_policy.py
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import chex

# --- Config ---
HIDDEN_DIMS = (64, 64) # Keep it small. Strategy is usually low-dimensional.
# TODO: might be better to use attention mechanism (instead of self-attention) as it learns faster.
# Question is, is there viable information in the relation between objects that helps the decision making or are the relations player <-> objects enough?
# TODO: if using self-attention might be worth switching from max-pooling to just giving the player token to the decision mlp as a sort of 'summary' (similar to what BERT does in their models)
class MetaPolicyQNetwork(nn.Module):
    """
    A Strategic Commander that can 'see' the board.
    
    It uses a Transformer-style attention block to process the variable list of objects
    into a fixed-size 'Strategic Summary', allowing it to detect concepts like 
    'Blocking' or 'Cluster' without breaking Transfer.
    """
    num_concepts: int
    embed_dim: int = 64
    num_heads: int = 4
    num_labels: int = 10 # Max label index (Player, Shark, Diver...)

    @nn.compact
    def __call__(self, 
                 global_features: chex.Array, 
                 objects: chex.Array,       # [Batch, Max_Objs, 4] (x,y,w,h)
                 object_labels: chex.Array, # [Batch, Max_Objs] (Ints)
                 object_mask: chex.Array):
        
        # --- 1. Semantic Embedding (The "What") ---
        # Map integer labels (0, 1, 2...) to dense vectors [Batch, Max_Objs, 32]
        # This allows the network to learn "Sharkness" as a vector concept.
        label_embedding = nn.Embed(num_embeddings=self.num_labels, features=32)(object_labels)
        
        # --- 2. Geometric Projection (The "Where") ---
        # Map raw geometry [x,y,w,h] to dense vectors [Batch, Max_Objs, 32]
        geo_embedding = nn.Dense(32)(objects)
        geo_embedding = nn.relu(geo_embedding)
        
        # --- 3. Fusion (The "Token") ---
        # Combine What + Where. 
        # Result: [Batch, Max_Objs, 64]
        # Now the token represents "A Shark at (10,10)"
        x = jnp.concatenate([label_embedding, geo_embedding], axis=-1)
        
        # --- 4. Relational Reasoning (Self-Attention) ---
        # Now, when the Player token queries the Shark token:
        # - The Query comes from the "Player" part of the embedding.
        # - The Key comes from the "Shark" part of the embedding.
        # - The Attention Score will be high because the network learns Player <-> Shark matters.
        mask_2d = nn.make_attention_mask(object_mask > 0.5, object_mask > 0.5)
        
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim
        )(inputs_q=x, inputs_k=x, inputs_v=x, mask=mask_2d)
        # Residual + Norm
        x = nn.LayerNorm()(x + attended)
        
        # --- 3. Strategic Aggregation (Pooling) ---
        # Collapse the N objects into a single "Situation Vector".
        # We use Max Pooling to detect "Is there ANY critical threat?" regardless of count.
        # Apply large negative to masked items so they aren't picked by Max
        x = x + (1.0 - object_mask[:, :, None]) * -1e9
        situation_vector = jnp.max(x, axis=1) # [Batch, Embed_Dim]
        
        # --- 4. Fusion (Global Stats + Visual Situation) ---
        # Combine the expert dashboard (Oxygen, Score) with the learned situation
        combined = jnp.concatenate([global_features, situation_vector], axis=-1)
        
        # --- 5. Decision Head ---
        z = nn.Dense(128)(combined)
        z = nn.relu(z)
        q_values = nn.Dense(self.num_concepts)(z)
        
        return q_values

class MetaPolicy:
    @staticmethod
    def create_train_state(key, num_concepts, learning_rate, global_dim, obj_dim, max_objs):
        net = MetaPolicyQNetwork(num_concepts=num_concepts)
        
        # Dummy inputs
        d_glob = jnp.zeros((1, global_dim))
        d_objs = jnp.zeros((1, max_objs, obj_dim))
        d_labels = jnp.zeros((1, max_objs), dtype=jnp.int32)
        d_mask = jnp.zeros((1, max_objs))
        
        params = net.init(key, d_glob, d_objs, d_labels, d_mask)
        tx = optax.adam(learning_rate)
        return TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    @staticmethod
    @jax.jit
    def select_concept_eps_greedy(train_state, global_features, objects, object_labels, object_mask, epsilon, key):
        q_values = train_state.apply_fn(train_state.params, global_features, objects, object_labels, object_mask)
        
        greedy = jnp.argmax(q_values, axis=-1)
        random = jax.random.randint(key, shape=greedy.shape, minval=0, maxval=q_values.shape[-1])
        rng_mix = jax.random.uniform(key, shape=greedy.shape)
        return jnp.where(rng_mix < epsilon, random, greedy), q_values