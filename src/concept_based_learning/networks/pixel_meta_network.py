"""Pixel-only selector pilots; object metadata never enters the scoring MLP.

pixel_slots: one shared scene encoding, one output per fixed target slot.
pixel_candidates: shared scalar scorer over candidate-marked image stacks.
Candidate marking reads only bbox/activity; labels and reward events are ignored.
Validity masking and label-to-skill routing stay outside both networks.
"""
from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from concept_based_learning.observations.target_conditioning import (
    normalize_pixel_obs,
    paint_target_outline_pixels,
)


def pixel_stack_channels(pixel_obs):
    """[..., T,H,W,C] -> [..., H,W,T*C], preserving spatial/frame order."""
    img = normalize_pixel_obs(pixel_obs)
    if img.ndim < 4:
        raise ValueError("Pixel selector requires [..., T,H,W,C].")
    lead = img.shape[:-4]
    t, h, w, c = img.shape[-4:]
    axes = tuple(range(len(lead))) + (len(lead)+1, len(lead)+2, len(lead), len(lead)+3)
    return jnp.transpose(img, axes).reshape(lead + (h, w, t*c))


def candidate_marked_stacks(pixel_obs, objects, *, action_dim, noop_idx):
    """Build one image per slot; NOOP gets a fixed black/white corner marker.

    The same latest-frame bbox marks every frame, as in pixel skills. Reserved
    NOOP uses the original scene plus a 6px black square / 4px white inset,
    rather than a candidate box. Inactive object slots are scored but must be
    masked by the caller, just as for other meta architectures.
    """
    if objects is None:
        raise ValueError("Candidate marking requires candidate bounding boxes.")
    if not 0 <= int(noop_idx) < int(action_dim):
        raise ValueError("A valid reserved NOOP index is required.")
    marked = paint_target_outline_pixels(
        pixel_obs, objects, jnp.arange(action_dim), max_objects=action_dim
    )
    original = normalize_pixel_obs(pixel_obs)
    marked = marked.at[noop_idx].set(original)
    marked = marked.at[noop_idx, :, :6, :6, :].set(0.0)
    marked = marked.at[noop_idx, :, 1:5, 1:5, :].set(1.0)
    return marked


class _PixelScorer(nn.Module):
    output_dim: int
    hidden_size: int = 256
    num_layers: int = 3
    norm_type: str = "layer_norm"
    embed_dim: int = 256
    channels: tuple = (32, 64)
    kernel_size: tuple = (5, 5)

    @nn.compact
    def __call__(self, pixel_obs):
        x = pixel_stack_channels(pixel_obs)
        for i, width in enumerate(self.channels):
            x = nn.relu(nn.Conv(int(width), self.kernel_size, strides=(2, 2),
                               padding="SAME", name=f"pixel_conv_{i}")(x))
        lead = x.shape[:-3]
        x = x.reshape(lead + (math.prod(x.shape[-3:]),))
        x = nn.relu(nn.Dense(self.embed_dim, name="pixel_embed_proj")(x))
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_size, name=f"hidden_{i}")(x)
            if self.norm_type == "layer_norm":
                x = nn.LayerNorm(name=f"norm_{i}")(x)
            elif self.norm_type != "none":
                raise ValueError("Pixel pilots support layer_norm or none.")
            x = nn.relu(x)
        return nn.Dense(self.output_dim, name="q_values")(x)


class _ChunkScorer(nn.Module):
    hidden_size: int
    num_layers: int
    norm_type: str
    embed_dim: int
    channels: tuple
    kernel_size: tuple

    @nn.compact
    def __call__(self, carry, images):
        q = _PixelScorer(1, self.hidden_size, self.num_layers, self.norm_type,
                         self.embed_dim, self.channels, self.kernel_size)(images)
        return carry, q[..., 0]


class PixelMetaQNetwork(nn.Module):
    """Unified training/eval API for raw-slot and marked-candidate pilot heads."""
    action_dim: int
    architecture: str = "pixel_slots"
    hidden_size: int = 256
    num_layers: int = 3
    norm_type: str = "layer_norm"
    meta_embed_dim: int = 256
    pixel_cnn_channels: tuple = (32, 64)
    pixel_cnn_kernel_size: tuple = (5, 5)
    noop_idx: int = -1
    candidate_chunk_size: int = 1

    @nn.compact
    def __call__(self, raw_obs: Any, objects: Any, train: bool,
                 target_locked=None, pixel_obs=None):
        del raw_obs, train, target_locked
        if pixel_obs is None:
            raise ValueError("Pixel selector requires pixel_obs.")
        if self.architecture == "pixel_slots":
            # Deliberately never inspect objects, labels, events, or raw state.
            return _PixelScorer(
                self.action_dim, self.hidden_size, self.num_layers, self.norm_type,
                self.meta_embed_dim, self.pixel_cnn_channels,
                self.pixel_cnn_kernel_size,
            )(pixel_obs)
        if self.architecture != "pixel_candidates":
            raise ValueError(f"Unknown pixel pilot architecture: {self.architecture}")
        images = candidate_marked_stacks(pixel_obs, objects,
                                         action_dim=self.action_dim,
                                         noop_idx=self.noop_idx)
        chunk = int(self.candidate_chunk_size)
        if chunk < 1:
            raise ValueError("META_CANDIDATE_CHUNK_SIZE must be positive.")
        count = (self.action_dim + chunk - 1) // chunk
        padding = count * chunk - self.action_dim
        images = jnp.pad(images, ((0, padding),) + ((0, 0),)*(images.ndim-1))
        images = images.reshape((count, chunk) + images.shape[1:])
        # Lifted scan shares every parameter across candidates. Rematerializing
        # each chunk limits activation storage during the backward pass.
        scorer = nn.scan(
            nn.remat(_ChunkScorer), variable_broadcast="params",
            split_rngs={"params": False}, in_axes=0, out_axes=0,
        )(self.hidden_size, self.num_layers, self.norm_type, self.meta_embed_dim,
          self.pixel_cnn_channels, self.pixel_cnn_kernel_size)
        _, q = scorer(None, images)
        return q.reshape(-1)[:self.action_dim]
