"""Meta-policy MLP: flattened raw observation -> Q-values over object indices."""

import flax.linen as nn
import jax.numpy as jnp
from typing import Any

from concept_based_learning.networks.meta_network import (
    append_target_locked_feature,
    flatten_raw_observation,
)


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
    use_target_locked_input: bool = False
    meta_input_mode: str = "flat"
    meta_embed_dim: int = 256
    pixel_cnn_channels: tuple = (32, 64)
    pixel_cnn_kernel_size: tuple = (5, 5)

    @nn.compact
    def __call__(
        self,
        raw_obs: Any,
        objects: Any,
        train: bool,
        target_locked=None,
        pixel_obs=None,
    ):
        del objects
        mode = str(self.meta_input_mode).lower().strip()
        if mode in ("flat", "flatten"):
            x = flatten_raw_observation(raw_obs)
        elif mode in ("oc_embed", "object_embed", "object_embedding"):
            x = flatten_raw_observation(raw_obs)
            x = nn.Dense(self.meta_embed_dim, name="oc_embed")(x)
            x = nn.relu(x)
        elif mode in ("oc_simple", "simple_oc", "oc_slim") or str(mode).startswith(
            "oc_simple"
        ):
            raise ValueError(
                f"META_INPUT_MODE={mode!r} requires META_ARCHITECTURE=per_object_mlp "
                "(FlatMetaQNetwork ignores the object matrix)."
            )
        elif mode in ("pixel_embed", "pixel_embedding", "cnn"):
            if pixel_obs is None:
                raise ValueError(
                    "meta_input_mode='pixel_embed' requires pixel_obs from the combined wrapper."
                )
            img = pixel_obs.astype(jnp.float32) / 255.0
            img = jnp.reshape(img, (img.shape[1], img.shape[2], -1))
            for i, channels in enumerate(self.pixel_cnn_channels):
                img = nn.Conv(
                    features=int(channels),
                    kernel_size=self.pixel_cnn_kernel_size,
                    strides=(2, 2),
                    padding="SAME",
                    name=f"pixel_conv_{i}",
                )(img)
                img = nn.relu(img)
            x = img.reshape(-1)
            x = nn.Dense(self.meta_embed_dim, name="pixel_embed_proj")(x)
            x = nn.relu(x)
        else:
            raise ValueError(
                f"Unknown meta_input_mode={self.meta_input_mode!r}. "
                "Expected one of: flat, oc_embed, oc_simple, pixel_embed."
            )
        x = append_target_locked_feature(
            x,
            target_locked,
            enabled=self.use_target_locked_input,
        )

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
