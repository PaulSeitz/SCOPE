import jax.numpy as jnp
from typing import Any
import math

import chex
import flax.linen as nn
import jax
from flax.training.train_state import TrainState

from concept_based_learning.observations.target_conditioning import (
    DEFAULT_OUTLINE_THICKNESS,
    build_oc_object_matrix,
    gather_object_rows,
    paint_target_outline_pixels,
)


# Skill input modes whose forward pass reads pixel_obs (require PixelAndObjectObsWrapper).
PIXEL_SKILL_INPUT_MODES = ("pixel_embed", "pixel_embed_cond")
# Modes that need labeled ``objects`` for index-invariant target cues.
OBJECT_CONDITIONED_SKILL_INPUT_MODES = (
    "oc_embed",
    "object_embed",
    "object_embedding",
) + PIXEL_SKILL_INPUT_MODES


class QNetwork(nn.Module):
    """MLP Q-network for skills (System 1).

    Input modes
    -----------
    ego / ego_embed
        Relative engineered features (``x``). Index is only used upstream to
        build those features — never fed as a slot id.
    oc_embed
        Full labeled ``objects`` matrix (same leaves meta uses). A **shared**
        ``Dense(embed_dim)`` embeds each object row, then the target row is
        gathered — so the Dense input width is the per-object feature size
        ``F`` (shared across envs), not ``M·F``. Requires ``objects``.
    pixel_embed / pixel_embed_cond
        CNN over pixels with a painted target outline. ``pixel_embed_cond`` is
        kept as a deprecated alias of ``pixel_embed`` (slot-channel conditioning
        was removed). Requires ``pixel_obs`` and ``objects``.
    """

    action_dim: int
    hidden_size: int = 64
    num_layers: int = 3
    norm_type: str = "layer_norm"
    input_mode: str = "ego"
    max_objects: int = 35
    embed_dim: int = 256
    pixel_cnn_channels: tuple = (32, 64)
    pixel_cnn_kernel_size: tuple = (5, 5)
    pixel_target_embed_dim: int = 16  # unused; kept for config compatibility
    pixel_outline_thickness: int = DEFAULT_OUTLINE_THICKNESS

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool,
        *,
        target_idx=None,
        pixel_obs=None,
        objects=None,
    ):
        del train  # LayerNorm path does not use batch stats.
        mode = str(self.input_mode).lower().strip()
        if target_idx is None:
            target_idx = jnp.array(0, dtype=jnp.int32)
        target_idx = jnp.asarray(target_idx, dtype=jnp.int32)

        if mode == "ego":
            pass
        elif mode in ("ego_embed", "ego_embedding"):
            # Same relative ego features as ``ego``, but with a learned projection
            # before the Q MLP (tests whether an embed helps beyond raw ego).
            x = x.astype(jnp.float32)
            x = nn.Dense(self.embed_dim, name="ego_embed")(x)
            x = nn.relu(x)
        elif mode in ("oc_embed", "object_embed", "object_embedding"):
            if objects is None:
                raise ValueError(
                    "input_mode='oc_embed' requires labeled objects "
                    "(slot one-hots were removed)."
                )
            # Per-object Dense (input F fixed across envs) → gather target row.
            obj_mat = build_oc_object_matrix(objects, max_objects=self.max_objects)
            obj_h = nn.Dense(self.embed_dim, name="oc_embed")(obj_mat)
            obj_h = nn.relu(obj_h)
            x = gather_object_rows(
                obj_h, target_idx, max_objects=self.max_objects
            )
        elif mode in ("pixel_embed", "pixel_embedding", "cnn", "pixel_embed_cond"):
            # ``pixel_embed_cond`` is an alias: target identity is the painted
            # outline, not a learned slot embedding.
            if pixel_obs is None:
                raise ValueError(
                    f"input_mode={mode!r} requires pixel_obs from combined observation wrappers."
                )
            if objects is None:
                raise ValueError(
                    f"input_mode={mode!r} requires labeled objects to paint the "
                    "target outline (slot one-hots were removed)."
                )
            img = paint_target_outline_pixels(
                pixel_obs,
                objects,
                target_idx,
                max_objects=self.max_objects,
                thickness=self.pixel_outline_thickness,
            )
            # img: [..., T, H, W, C] already in [0, 1]
            if img.ndim < 4:
                raise ValueError(
                    f"pixel_obs must have rank >=4 with trailing [T,H,W,C], got rank {img.ndim}."
                )
            lead_shape = img.shape[:-4]
            t, h, w, c = img.shape[-4:]
            n = math.prod(lead_shape) if lead_shape else 1
            img = jnp.reshape(img, (n, t, h, w, c))
            img = jnp.transpose(img, (0, 2, 3, 1, 4))  # [N,H,W,T,C]
            img = jnp.reshape(img, (n, h, w, t * c))  # [N,H,W,T*C]

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
                x = jnp.reshape(pixel_embed, lead_shape + (pixel_embed.shape[-1],))
            else:
                x = pixel_embed[0]
        else:
            raise ValueError(
                f"Unknown SKILL_INPUT_MODE={self.input_mode!r}. "
                "Expected one of: ego, ego_embed, oc_embed, pixel_embed, pixel_embed_cond."
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
    objects: Any
    next_objects: Any
    object_idx_all: chex.Array
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
