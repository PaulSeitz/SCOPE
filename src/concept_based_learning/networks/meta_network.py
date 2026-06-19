import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


def append_target_locked_feature(
    global_features: jnp.ndarray,
    target_locked,
    *,
    enabled: bool,
) -> jnp.ndarray:
    """Optionally append a scalar 0/1 persistence-lock flag to global meta features."""
    if not enabled:
        return global_features
    if target_locked is None:
        locked = jnp.array(0.0, dtype=jnp.float32)
    else:
        locked = jnp.asarray(target_locked, dtype=jnp.float32).reshape(())
    return jnp.concatenate([global_features, jnp.atleast_1d(locked)])


def flatten_raw_observation(raw_obs):
    """
    Extracts the most recent frame from a raw Jaxatari observation PyTree and
    flattens all numerical leaves into a 1D array.
    """
    last_frame_obs = jax.tree_util.tree_map(lambda x: x[-1] if jnp.ndim(x) > 0 else x, raw_obs)
    leaves, _ = jax.tree_util.tree_flatten(last_frame_obs)
    flat_arrays = [jnp.atleast_1d(leaf).flatten() for leaf in leaves]
    return jnp.concatenate(flat_arrays).astype(jnp.float32)


class MetaQNetwork(nn.Module):
    """
    System 2 MLP: evaluates the raw environment state and outputs Q-values
    for each possible target index.
    """

    action_dim: int  # MAX_OBJECTS
    hidden_size: int = 256
    num_layers: int = 3
    norm_type: str = "layer_norm"
    num_skills: int = 3  # unused in MLP path, kept for API compatibility
    use_target_locked_input: bool = False

    @nn.compact
    def __call__(self, raw_obs: Any, objects: Any, train: bool, target_locked=None):
        global_features = flatten_raw_observation(raw_obs)
        global_features = append_target_locked_feature(
            global_features,
            target_locked,
            enabled=self.use_target_locked_input,
        )

        last_frame_objs = jax.tree_util.tree_map(
            lambda x: x[-1] if jnp.ndim(x) > 0 else x, objects
        )
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
        obj_matrix = jnp.concatenate(obj_features_list, axis=-1).astype(jnp.float32)

        global_matrix = jnp.broadcast_to(
            global_features, (self.action_dim, global_features.shape[-1])
        )
        x = jnp.concatenate([global_matrix, obj_matrix], axis=-1)

        if self.norm_type == "layer_norm":
            normalize = lambda v: nn.LayerNorm()(v)
        elif self.norm_type == "batch_norm":
            normalize = lambda v: nn.BatchNorm(use_running_average=not train)(v)
        else:
            normalize = lambda v: v

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        q_values = nn.Dense(1)(x)
        return jnp.squeeze(q_values, axis=-1)
