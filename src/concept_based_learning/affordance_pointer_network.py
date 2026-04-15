import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

@chex.dataclass(frozen=True)
class MetaTransition:
    """Transition strictly for the Meta-Policy (System 2) using global rewards."""
    obs: Any                    # The raw Jaxatari observation PyTree
    objects: Any                # Object PyTree aligned with obs
    action: chex.Array          # Selected target_idx
    reward: chex.Array          # Global game reward
    done: chex.Array            # Global game over flag
    next_obs: Any               # Next raw Jaxatari observation
    next_objects: Any           # Next object PyTree aligned with next_obs
    q_val: chex.Array           # Meta Q-values
    active_mask: chex.Array     # Which objects were valid targets
    next_active_mask: chex.Array

def flatten_raw_observation(raw_obs):
    """
    Takes a raw Jaxatari observation PyTree (e.g., SeaquestObservation),
    extracts the most recent frame, and flattens all numerical leaves into a 1D array.
    This works generically across all Jaxatari environments.
    """
    # Grab the last frame of the stack for all leaves
    last_frame_obs = jax.tree_util.tree_map(lambda x: x[-1] if jnp.ndim(x) > 0 else x, raw_obs)

    # Flatten the PyTree into a list of arrays
    leaves, _ = jax.tree_util.tree_flatten(last_frame_obs)

    # Flatten each leaf and concatenate them all into a single 1D vector
    flat_arrays = [jnp.atleast_1d(leaf).flatten() for leaf in leaves]
    return jnp.concatenate(flat_arrays).astype(jnp.float32)

class MetaQNetwork(nn.Module):
    """
    System 2: Affordance-Masked Pointer Network.
    Evaluates the raw environment state and outputs Q-values for each possible target index
    by factorizing the decision into Strategic Intent (Affordance) and Spatial Relevance (Pointer).
    """
    action_dim: int  # This will be your MAX_OBJECTS
    hidden_size: int = 256
    num_layers: int = 3
    norm_type: str = "layer_norm"
    num_skills: int = 3 # Added to handle semantic label mapping

    @nn.compact
    def __call__(self, raw_obs: Any, objects: Any, train: bool):
        # 1) Global state features (Shared Query Context)
        global_features = flatten_raw_observation(raw_obs)

        # Setup the PQN normalization function
        if self.norm_type == "layer_norm":
            normalize = lambda v: nn.LayerNorm()(v)
        elif self.norm_type == "batch_norm":
            normalize = lambda v: nn.BatchNorm(use_running_average=not train)(v)
        else:
            normalize = lambda v: v

        # ==========================================
        # Component A: Affordance Head (Intent)
        # ==========================================
        aff_x = global_features
        for _ in range(self.num_layers):
            aff_x = nn.Dense(self.hidden_size)(aff_x)
            aff_x = normalize(aff_x)
            aff_x = nn.relu(aff_x)
            
        # Outputs a base Q-value for each of the 3 skills/intents
        intent_q_values = nn.Dense(self.num_skills)(aff_x) # Shape: [NUM_SKILLS]

        # ==========================================
        # Component B: Object Feature Extraction
        # ==========================================
        last_frame_objs = jax.tree_util.tree_map(
            lambda x: x[-1] if jnp.ndim(x) > 0 else x, objects
        )
        
        # Extract Semantic Labels to map objects to their intents (1=Destroy, 2=Collect, 3=Avoid)
        labels = jnp.asarray(last_frame_objs.semantic_label, dtype=jnp.int32)
        skill_indices = jnp.clip(labels - 1, 0, self.num_skills - 1)

        leaves, _ = jax.tree_util.tree_flatten(last_frame_objs)

        def _leaf_to_object_features(leaf):
            arr = jnp.asarray(leaf, dtype=jnp.float32)
            arr = jnp.squeeze(arr)

            if arr.ndim == 0:
                return jnp.broadcast_to(arr, (self.action_dim, 1))
            if arr.ndim == 1:
                if arr.shape[0] == self.action_dim:
                    return arr[:, None]
                if arr.shape[0] == 1:
                    return jnp.broadcast_to(arr, (self.action_dim, 1))
                return jnp.broadcast_to(arr[None, :], (self.action_dim, arr.shape[0]))
            if arr.shape[0] == self.action_dim:
                return arr.reshape(self.action_dim, -1)
            if arr.shape[-1] == self.action_dim:
                return jnp.moveaxis(arr, -1, 0).reshape(self.action_dim, -1)

            flat = arr.reshape(-1)
            return jnp.broadcast_to(flat[None, :], (self.action_dim, flat.shape[0]))

        obj_features_list = [_leaf_to_object_features(leaf) for leaf in leaves]
        
        # Shape: [MAX_OBJECTS, object_feature_dim]
        obj_matrix = jnp.concatenate(obj_features_list, axis=-1).astype(jnp.float32)

        # ==========================================
        # Component C: Pointer Head (Spatial Cross-Attention)
        # ==========================================
        # Query (Global Strategy)
        q_x = global_features
        for _ in range(self.num_layers):
            q_x = nn.Dense(self.hidden_size)(q_x)
            q_x = normalize(q_x)
            q_x = nn.relu(q_x)
        query = nn.Dense(self.hidden_size)(q_x) # Shape: [hidden_size]

        # Keys (Object Spatial Data)
        k_x = obj_matrix
        for _ in range(self.num_layers):
            k_x = nn.Dense(self.hidden_size)(k_x)
            k_x = normalize(k_x)
            k_x = nn.relu(k_x)
        keys = nn.Dense(self.hidden_size)(k_x) # Shape: [MAX_OBJECTS, hidden_size]

        # Scaled Dot-Product Attention
        spatial_scores = jnp.sum(query[None, :] * keys, axis=-1) / jnp.sqrt(self.hidden_size)

        # ==========================================
        # Component D: Factorized Combination
        # ==========================================
        # The base Q-value is inherited from the Affordance Head based on the object's label
        base_q = intent_q_values[skill_indices] 

        # Final Q-value is the Strategic Intent + Spatial Relevance
        q_values = base_q + spatial_scores
        
        return q_values