import jax.numpy as jnp
from typing import Any
import math

import chex
import flax.linen as nn
import jax
from flax.training.train_state import TrainState


# Skill input modes whose forward pass reads pixel_obs (require PixelAndObjectObsWrapper).
PIXEL_SKILL_INPUT_MODES = ("pixel_embed", "pixel_embed_cond")


class QNetwork(nn.Module):
    """Simple MLP Q-network for skills (System 1). Maps ego-centric features → action-values."""

    action_dim: int
    hidden_size: int = 64
    num_layers: int = 3
    norm_type: str = "layer_norm"
    input_mode: str = "ego"
    max_objects: int = 35
    embed_dim: int = 256
    pixel_cnn_channels: tuple = (32, 64)
    pixel_cnn_kernel_size: tuple = (5, 5)
    pixel_target_embed_dim: int = 16

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool,
        *,
        target_idx=None,
        pixel_obs=None,
    ):
        def _align_target_dims(target_oh: jnp.ndarray, feat: jnp.ndarray) -> jnp.ndarray:
            """Broadcast target one-hot to match feature tensor rank."""
            if target_oh.ndim == feat.ndim:
                return target_oh
            if target_oh.ndim == feat.ndim - 1:
                target_oh = jnp.expand_dims(target_oh, axis=-2)
                out_shape = feat.shape[:-1] + (target_oh.shape[-1],)
                return jnp.broadcast_to(target_oh, out_shape)
            raise ValueError(
                f"Target one-hot rank mismatch: target_oh.ndim={target_oh.ndim}, "
                f"feature.ndim={feat.ndim}."
            )

        mode = str(self.input_mode).lower().strip()
        if target_idx is None:
            target_idx = jnp.array(0, dtype=jnp.int32)
        target_idx = jnp.asarray(target_idx, dtype=jnp.int32)
        target_oh = jax.nn.one_hot(
            jnp.clip(target_idx, 0, self.max_objects - 1), self.max_objects
        ).astype(jnp.float32)

        if mode == "ego":
            pass
        elif mode in ("oc_embed", "object_embed", "object_embedding"):
            x = x.astype(jnp.float32)
            x = nn.Dense(self.embed_dim, name="oc_embed")(x)
            x = nn.relu(x)
            target_oh = _align_target_dims(target_oh, x)
            x = jnp.concatenate([x, target_oh], axis=-1)
        elif mode in ("pixel_embed", "pixel_embedding", "cnn", "pixel_embed_cond"):
            target_conditioned = mode == "pixel_embed_cond"
            if pixel_obs is None:
                raise ValueError(
                    f"input_mode={mode!r} requires pixel_obs from combined observation wrappers."
                )
            img = pixel_obs.astype(jnp.float32) / 255.0
            # pixel_obs shape is [..., T, H, W, C]. Leading dims are shared screen
            # batch axes (e.g. time/env), not per-object slots. When target_idx has
            # extra object axes (relabel-all-objects):
            #   pixel_embed:      encode pixels once, broadcast the embedding.
            #   pixel_embed_cond: broadcast the *image* per target so the CNN sees
            #                     the target index as tiled input channels. The conv
            #                     stack then runs once per (screen, target) pair.
            if img.ndim < 4:
                raise ValueError(
                    f"pixel_obs must have rank >=4 with trailing [T,H,W,C], got rank {img.ndim}."
                )
            pixel_lead_shape = img.shape[:-4]
            target_lead_shape = target_idx.shape
            t, h, w, c = img.shape[-4:]
            extra_target_axes = len(target_lead_shape) - len(pixel_lead_shape)
            if extra_target_axes < 0:
                raise ValueError(
                    f"target_idx rank {len(target_lead_shape)} must be >= pixel lead rank "
                    f"{len(pixel_lead_shape)}."
                )

            if target_conditioned:
                lead_shape = target_lead_shape
                for _ in range(extra_target_axes):
                    img = jnp.expand_dims(img, axis=len(pixel_lead_shape))
                img = jnp.broadcast_to(img, lead_shape + (t, h, w, c))
            else:
                lead_shape = pixel_lead_shape

            n = math.prod(lead_shape) if lead_shape else 1
            img = jnp.reshape(img, (n, t, h, w, c))
            img = jnp.transpose(img, (0, 2, 3, 1, 4))  # [N,H,W,T,C]
            img = jnp.reshape(img, (n, h, w, t * c))   # [N,H,W,T*C]

            if target_conditioned:
                # Embed the target one-hot and tile it spatially so the conv stack
                # is conditioned on which object slot to focus on.
                target_emb = nn.Dense(
                    self.pixel_target_embed_dim, name="pixel_target_embed"
                )(jnp.reshape(target_oh, (n, self.max_objects)))
                target_emb = nn.relu(target_emb)
                target_planes = jnp.broadcast_to(
                    target_emb[:, None, None, :],
                    (n, h, w, self.pixel_target_embed_dim),
                )
                img = jnp.concatenate([img, target_planes], axis=-1)

            for i, channels in enumerate(self.pixel_cnn_channels):
                img = nn.Conv(
                    features=int(channels),
                    kernel_size=self.pixel_cnn_kernel_size,
                    strides=(2, 2),
                    padding="SAME",
                    name=f"pixel_conv_{i}",
                )(img)
                img = nn.relu(img)
            pixel_embed = img.reshape(img.shape[0], -1)
            pixel_embed = nn.Dense(self.embed_dim, name="pixel_embed_proj")(pixel_embed)
            pixel_embed = nn.relu(pixel_embed)
            if lead_shape:
                pixel_embed = jnp.reshape(pixel_embed, lead_shape + (pixel_embed.shape[-1],))
            else:
                pixel_embed = pixel_embed[0]

            if not target_conditioned and extra_target_axes > 0:
                for _ in range(extra_target_axes):
                    pixel_embed = jnp.expand_dims(pixel_embed, axis=-2)
                x = jnp.broadcast_to(
                    pixel_embed,
                    target_lead_shape + (pixel_embed.shape[-1],),
                )
            else:
                x = pixel_embed
            target_oh = _align_target_dims(target_oh, x)
            x = jnp.concatenate([x, target_oh], axis=-1)
        else:
            raise ValueError(
                f"Unknown SKILL_INPUT_MODE={self.input_mode!r}. "
                "Expected one of: ego, oc_embed, pixel_embed, pixel_embed_cond."
            )

        if self.norm_type == "layer_norm":
            normalize = lambda v: nn.LayerNorm()(v)
        else:
            normalize = lambda v: v

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Transition:
    """Single transition in hierarchical replay rollouts. Rewards stored for all skills."""

    target_idx: chex.Array
    ego_obs: chex.Array
    next_ego_obs: chex.Array
    pixel_obs: chex.Array
    next_pixel_obs: chex.Array
    ego_action: chex.Array
    active_skill: chex.Array
    q_val: chex.Array       # all skills' Q-values [num_skills, num_envs, ego_action_dim]
    rewards: chex.Array     # rewards for ALL skills [num_envs, num_skills]
    skill_done: chex.Array  # [num_envs, num_skills]
    ego_obs_all: chex.Array
    next_ego_obs_all: chex.Array
    ego_action_all: chex.Array
    rewards_all: chex.Array
    skill_done_all: chex.Array
    relabel_mask: chex.Array
    done: chex.Array
    env_reward: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
