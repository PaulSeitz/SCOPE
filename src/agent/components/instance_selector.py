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
import src.environments.seaquest.seaquest_constants as consts

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
    def __call__(self, global_features: chex.Array, objects: chex.Array, object_labels: chex.Array, object_mask: chex.Array):
        """
        Args:
            global_features: [Batch, G]
            objects: [Batch, N, F]
            object_labels: [Batch, N, Num_Concepts] (Multi-Hot Encoding)
            object_mask: [Batch, N] (0/1)
            
        Returns:
            attention_weights: [Batch, Num_Concepts, Max_Objs] 
                               (Separate attention distribution for EACH concept)
        """
        batch_size = global_features.shape[0]

        # --- 1. Embeddings ---
        q_glob = nn.Dense(self.embed_dim)(global_features)
        
        # Objects are already normalized to [0, 1] in seaquest_wrapper
        k_objs = nn.Dense(self.embed_dim)(objects)
        
        # --- 2. Concept Queries [Batch, Num_Concepts, Dim] ---
        concept_queries = nn.Dense(self.num_concepts * self.embed_dim)(q_glob)
        concept_queries = concept_queries.reshape(batch_size, self.num_concepts, self.embed_dim)
        
        # --- 3. Raw Scores [Batch, Num_Concepts, Max_Objs] ---
        # (B, C, D) @ (B, O, D).T -> (B, C, O)
        scores = jnp.einsum('bcd,bod->bco', concept_queries, k_objs)
        scores = scores / jnp.sqrt(self.embed_dim)

        # --- 4. SEMANTIC MASKING---
        
        # object_labels: [Batch, Max_Objs, Num_Concepts] (The Multi-Hot vectors)
        # We want to know: For Concept C, is the bit 'C' set in object_labels?
        # Since object_labels is multi-hot [0, 0, 1, 0...], we can just extract the channel 'C'.
        
        # 1. Transpose labels to [Batch, Num_Concepts, Max_Objs] to match scores shape
        # This directly gives us the label match mask:
        # If object 'o' has label 'c', then labels_T[b, c, o] = 1.0
        labels_T = jnp.transpose(object_labels, (0, 2, 1))
        
        # The transposed labels matrix IS the label match mask
        # labels_T[b, c, o] = 1.0 if object 'o' has concept 'c' in its multi-hot encoding
        label_match_mask = labels_T
        
        # B. Existence Check
        # Handle both batched [Batch, Max_Objs] and unbatched [Max_Objs] cases
        if object_mask.ndim == 1:
            # Unbatched: object_mask is [Max_Objs]
            # label_match_mask is [Num_Concepts, Max_Objs] (no batch dim)
            # Expand to [1, Max_Objs] which will broadcast with [Num_Concepts, Max_Objs]
            existence = (object_mask[None, :] > 0.5).astype(jnp.float32)  # [1, Max_Objs] broadcasts to [Num_Concepts, Max_Objs]
        else:
            # Batched: object_mask is [Batch, Max_Objs]
            # label_match_mask is [Batch, Num_Concepts, Max_Objs]
            # Expand to [Batch, 1, Max_Objs] which will broadcast with [Batch, Num_Concepts, Max_Objs]
            existence = (object_mask[:, None, :] > 0.5).astype(jnp.float32)  # [Batch, 1, Max_Objs]
        
        # C. Final Mask construction
        # We simply demand that the Object has the Concept label set in its multi-hot encoding.
        # This handles all concepts uniformly:
        # - Collect/Avoid/Destroy: Must match label
        # - Monitor: Will try to find objects with Monitor label (if any exist)
        # - Tool: Will try to find objects with Tool label (Player Missile)
        # Use multiplication instead of & since both arrays are float32
        valid_mask = label_match_mask * existence
        
        # D. Fallback Safety
        # If a row has NO valid targets, we must allow Index 0 to avoid NaNs in Softmax.
        # This happens if:
        # 1. The Meta policy picked "Collect" but there are no divers.
        # 2. The concept is "Monitor" and there are no objects labeled "Monitor".
        any_valid = jnp.any(valid_mask, axis=-1, keepdims=True)
        
        # Fallback: Allow Index 0. 
        # (The downstream logic in Orchestrator handles the case where target is irrelevant)
        fallback_mask = jax.nn.one_hot(jnp.zeros((batch_size, self.num_concepts), dtype=jnp.int32), 
                                       num_classes=objects.shape[1]) > 0.5
                                       
        valid_mask = jnp.where(any_valid, valid_mask, fallback_mask)

        # Apply Mask (-1e9 is safer than -inf for Softmax stability)
        large_neg = -1e9
        scores = scores + (1.0 - valid_mask.astype(jnp.float32)) * large_neg

        # --- 5. Softmax ---
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