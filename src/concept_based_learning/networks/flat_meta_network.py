"""Meta-policy MLP: flattened raw observation -> Q-values over object indices."""

import flax.linen as nn
import jax.numpy as jnp
from typing import Any

from concept_based_learning.networks.meta_network import flatten_raw_observation


class FlatMetaQNetwork(nn.Module):
    """
    System 2 head with a single trunk over the raw Jaxatari observation.

    Outputs ``action_dim`` Q-values (one per object slot). Object tensors are
    accepted for API compatibility with other meta architectures but are not used;
    entity state is assumed to be present in ``raw_obs``.
    """

    action_dim: int
    hidden_size: int = 256
    num_layers: int = 3
    norm_type: str = "layer_norm"
    num_skills: int = 3  # unused; kept for factory API parity

    @nn.compact
    def __call__(self, raw_obs: Any, objects: Any, train: bool):
        del objects
        x = flatten_raw_observation(raw_obs)

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

        return nn.Dense(self.action_dim)(x).astype(jnp.float32)
