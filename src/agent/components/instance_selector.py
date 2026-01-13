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
            object_labels: [Batch, N] (Integer IDs)
            object_mask: [Batch, N] (0/1)
            
        Returns:
            attention_weights: [Batch, Num_Concepts, Max_Objs] 
                               (Separate attention distribution for EACH concept)
        """
        batch_size = global_features.shape[0]
        n_objects = objects.shape[1]

        # --- 1. Embeddings ---
        q_glob = nn.Dense(self.embed_dim)(global_features)
        
        # Normalize objects for consistency
        norm_objects = objects / 210.0
        k_objs = nn.Dense(self.embed_dim)(norm_objects)
        
        # --- 2. Concept Queries [Batch, Num_Concepts, Dim] ---
        concept_queries = nn.Dense(self.num_concepts * self.embed_dim)(q_glob)
        concept_queries = concept_queries.reshape(batch_size, self.num_concepts, self.embed_dim)
        
        # --- 3. Raw Scores [Batch, Num_Concepts, Max_Objs] ---
        # (B, C, D) @ (B, O, D).T -> (B, C, O)
        scores = jnp.einsum('bcd,bod->bco', concept_queries, k_objs)
        scores = scores / jnp.sqrt(self.embed_dim)

        # --- 4. THE REVISED MASKING LOGIC ---
        
        # A. Concept-Specific Targeting Rules
        concept_ids = jnp.arange(self.num_concepts)[None, :, None] # [1, C, 1]
        
        # Rule 1: "Strict Matching" Concepts (Must target specific label)
        # Collect(2), Avoid(3), Destroy(4)
        is_strict_concept = (concept_ids == consts.LABEL_COLLECT) | \
                            (concept_ids == consts.LABEL_AVOID) | \
                            (concept_ids == consts.LABEL_DESTROY)
                            
        # Rule 2: "Self-Targeting" Concepts (Target the Player/Self)
        # Player(0), Tool(1), Monitor(5)
        # Note: We group 'Monitor' here. If I want to monitor, I look at myself.
        is_self_concept = ~is_strict_concept

        # B. Object Identity Checks
        # We assume Object 0 is always Player
        obj_ids = jnp.arange(objects.shape[1])[None, None, :]
        is_player_obj = (obj_ids == 0)
        
        # C. Labels
        labels_expanded = object_labels[:, None, :] # [B, 1, O]
        label_match = (labels_expanded == concept_ids)
        
        # --- D. Construct the Valid Mask ---
        
        # Case 1: Strict Concepts -> Must match Label AND Not be Player
        # (Technically Label check covers 'Not Player' usually, but safe to be explicit)
        mask_strict = is_strict_concept & label_match & (~is_player_obj)
        
        # Case 2: Self Concepts -> Must be Player
        mask_self = is_self_concept & is_player_obj
        
        # Combine
        final_semantic_mask = mask_strict | mask_self
        
        # E. Existence Check
        existence = object_mask[:, None, :] > 0.5
        
        # Final Mask
        valid_mask = final_semantic_mask & existence
        
        # F. Fallback Safety
        # If a row has NO valid targets (e.g., Collect selected but no Divers), 
        # fall back to Player (Index 0) to prevent Uniform Crash.
        # This ensures 'scores' has at least one valid entry (Index 0).
        any_valid = jnp.any(valid_mask, axis=-1, keepdims=True)
        fallback_mask = is_player_obj # Fallback to Player
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