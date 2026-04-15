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
    System 2: Evaluates the raw environment state and outputs Q-values 
    for each possible target index.
    """
    action_dim: int  # This will be your MAX_OBJECTS
    hidden_size: int = 256
    num_layers: int = 3
    norm_type: str = "layer_norm" # Brought back from your pqn_gymnax.py

    @nn.compact
    def __call__(self, raw_obs: Any, objects: Any, train: bool):
        # 1) Global state features (shared context for all objects)
        global_features = flatten_raw_observation(raw_obs)

        # 2) Per-object matrix from last frame: [MAX_OBJECTS, object_feature_dim]
        last_frame_objs = jax.tree_util.tree_map(
            lambda x: x[-1] if jnp.ndim(x) > 0 else x, objects
        )
        leaves, _ = jax.tree_util.tree_flatten(last_frame_objs)

        def _leaf_to_object_features(leaf):
            arr = jnp.asarray(leaf, dtype=jnp.float32)
            arr = jnp.squeeze(arr)

            # Scalar metadata (or squeezed [1]) -> broadcast to all objects.
            if arr.ndim == 0:
                return jnp.broadcast_to(arr, (self.action_dim, 1))

            # Vector either aligns with objects or is global metadata.
            if arr.ndim == 1:
                if arr.shape[0] == self.action_dim:
                    return arr[:, None]
                if arr.shape[0] == 1:
                    return jnp.broadcast_to(arr, (self.action_dim, 1))
                return jnp.broadcast_to(arr[None, :], (self.action_dim, arr.shape[0]))

            # Matrix/tensor where object axis can be leading or trailing.
            if arr.shape[0] == self.action_dim:
                return arr.reshape(self.action_dim, -1)
            if arr.shape[-1] == self.action_dim:
                return jnp.moveaxis(arr, -1, 0).reshape(self.action_dim, -1)

            # Fallback: treat as global tensor and broadcast to every object row.
            flat = arr.reshape(-1)
            return jnp.broadcast_to(flat[None, :], (self.action_dim, flat.shape[0]))

        obj_features_list = [_leaf_to_object_features(leaf) for leaf in leaves]
        obj_matrix = jnp.concatenate(obj_features_list, axis=-1).astype(jnp.float32)

        # 3) Broadcast global features per object and concatenate
        global_matrix = jnp.broadcast_to(
            global_features, (self.action_dim, global_features.shape[-1])
        )
        x = jnp.concatenate([global_matrix, obj_matrix], axis=-1)

        # Setup the PQN normalization function
        if self.norm_type == "layer_norm":
            normalize = lambda v: nn.LayerNorm()(v)
        elif self.norm_type == "batch_norm":
            normalize = lambda v: nn.BatchNorm(use_running_average=not train)(v)
        else:
            normalize = lambda v: v

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)  # <-- The secret sauce of PQN stability!
            x = nn.relu(x)

        # 4) Per-object scalar Q-values -> [MAX_OBJECTS]
        q_values = nn.Dense(1)(x)
        return jnp.squeeze(q_values, axis=-1)
