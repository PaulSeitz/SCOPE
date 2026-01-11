# src/agents/instance_selector.py
# References:
# Adapted from 'Attention Is All You Need' for Object-Centric RL.
# This module acts as the bridge between Abstract Command (Meta) and Concrete Target (Sub).

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import chex

class InstanceSelector(nn.Module):
    """
    The Attention Mechanism. Using the previously selected concept to execute, selects the target object to act on (i.e. for collect it tries to find a suitable diver).
    TODO: maybe we can cut out the concept selection step and only expand this to select an object to act on and then act according to the label?
    
    Query:  Global Context (What do we need?)
    Keys:   All Objects (What is available?)
    Values: Object Features (Where are they?)
    """
    num_concepts: int
    embed_dim: int = 64

    @nn.compact
    def __call__(self, global_features: chex.Array, objects: chex.Array, object_mask: chex.Array):
        """
        Args:
            global_features: [Batch, N_Globals]
            objects: [Batch, Max_Objs, Obj_Feats]
            object_mask: [Batch, Max_Objs] (1=Real, 0=Padding)
            
        Returns:
            attention_weights: [Batch, Num_Concepts, Max_Objs] 
                               (Separate attention distribution for EACH concept)
        """
        batch_size = global_features.shape[0]
        n_objects = objects.shape[1]

        # --- 1. The Query (What the concept is looking for) ---
        # generate a unique Query Vector for EVERY concept simultaneously.
        # This allows us to parallelize decision making.
        # Input: [Batch, Global_Dim] -> Output: [Batch, Num_Concepts, Embed_Dim]
        
        # separate projection for each concept to allow distinct "search criteria"
        q_proj = nn.Dense(self.num_concepts * self.embed_dim, name='query_proj')
        queries = q_proj(global_features).reshape(batch_size, self.num_concepts, self.embed_dim)

        # --- 2. The Keys (What the objects are) ---
        # Input: [Batch, Max_Objs, Obj_Feats] -> Output: [Batch, Max_Objs, Embed_Dim]
        k_proj = nn.Dense(self.embed_dim, name='key_proj')
        keys = k_proj(objects)

        # --- 3. Attention Score (Matching) ---
        # Equation: Q * K^T
        # Shapes: [B, Concepts, D] @ [B, D, Objs] -> [B, Concepts, Objs]
        # This gives us a score matrix: "How much does Concept C want Object O?"
        scores = jnp.matmul(queries, keys.swapaxes(1, 2))
        
        # Scale for stability
        scores = scores / jnp.sqrt(self.embed_dim)

        # --- 4. Masking (Don't look at ghosts) ---
        # Mask shape: [Batch, Objs] -> Broadcast to [Batch, 1, Objs]
        mask_expanded = object_mask[:, None, :]
        
        # Apply -Infinity to invalid objects so Softmax creates 0 probability
        large_neg = -1e9
        scores = scores + (1.0 - mask_expanded) * large_neg

        # --- 5. Probabilities ---
        # Dim -1 is the Object dimension
        attn_weights = nn.softmax(scores, axis=-1)
        
        return attn_weights

    @staticmethod
    def get_target_object(objects: chex.Array, attn_weights: chex.Array, concept_id: chex.Array):
        """
        Helper to extract the actual target features based on the attention.
        This handles the "Soft" vs "Hard" selection logic.
        """
        # attn_weights: [Batch, Num_Concepts, Max_Objs]
        # concept_id: [Batch] (The active concept chosen by Meta)
        
        # 1. Select the Attention Map for the ACTIVE concept
        # Shape: [Batch, Max_Objs]
        active_attn = jnp.take_along_axis(attn_weights, concept_id[:, None, None], axis=1).squeeze(1)
        
        # 2. Hard Selection (Argmax) - for selecting the discrete target
        # We use this to compute the Relative Vector
        target_idx = jnp.argmax(active_attn, axis=-1)
        
        # 3. Extract Target Features
        # Shape: [Batch, Obj_Feats]
        target_obj = jnp.take_along_axis(objects, target_idx[:, None, None], axis=1).squeeze(1)
        
        return target_obj, target_idx