"""Construct meta-policy networks from ``META_ARCHITECTURE`` config."""

from __future__ import annotations

from typing import Any, Dict

import flax.linen as nn

from concept_based_learning.networks.flat_meta_network import FlatMetaQNetwork
from concept_based_learning.networks.meta_network import MetaQNetwork

# per_object_mlp: global + per-slot object features, shared row-wise MLP (default)
# flat_mlp: raw obs vector only -> Dense(MAX_OBJECTS)
META_ARCHITECTURE_CHOICES = ("per_object_mlp", "flat_mlp", "pixel_slots", "pixel_candidates")


def _meta_num_layers(config: Dict[str, Any], arch_key: str) -> int:
    """Resolve depth: architecture-specific key, then ``META_NUM_LAYERS``, then 3."""
    specific = config.get(arch_key, None)
    if specific is not None:
        return int(specific)
    return int(config.get("META_NUM_LAYERS", 3))


def _meta_network_kwargs(config: Dict[str, Any], *, num_layers: int) -> Dict[str, Any]:
    pixel_channels = config.get("META_PIXEL_CNN_CHANNELS", (32, 64))
    if isinstance(pixel_channels, list):
        pixel_channels = tuple(pixel_channels)
    pixel_kernel = config.get("META_PIXEL_CNN_KERNEL_SIZE", (5, 5))
    if isinstance(pixel_kernel, list):
        pixel_kernel = tuple(pixel_kernel)
    return dict(
        action_dim=int(config["MAX_OBJECTS"]),
        hidden_size=int(config.get("META_HIDDEN_SIZE", 256)),
        num_layers=num_layers,
        norm_type=str(config.get("META_NORM_TYPE", "layer_norm")),
        num_skills=int(config.get("NUM_SKILLS", 3)),
        use_target_locked_input=bool(config.get("META_TARGET_LOCKED_INPUT", False)),
        meta_input_mode=str(config.get("META_INPUT_MODE", "flat")),
        meta_embed_dim=int(config.get("META_EMBED_DIM", 256)),
        pixel_cnn_channels=tuple(int(c) for c in pixel_channels),
        pixel_cnn_kernel_size=tuple(int(k) for k in pixel_kernel),
    )


def create_meta_network(config: Dict[str, Any]) -> nn.Module:
    """Instantiate the meta Q-network selected by ``config['META_ARCHITECTURE']``."""
    arch = str(config.get("META_ARCHITECTURE", "per_object_mlp")).lower().strip()

    if arch in ("pixel_slots", "pixel_candidates"):
        from concept_based_learning.networks.pixel_meta_network import PixelMetaQNetwork
        from concept_based_learning.environments.meta_target_mask import meta_noop_target_idx

        mode = str(config.get("META_INPUT_MODE", "")).lower().strip()
        if mode not in ("pixel_embed", "pixel_embedding", "cnn"):
            raise ValueError(f"{arch} requires META_INPUT_MODE=pixel_embed.")
        if bool(config.get("META_TARGET_LOCKED_INPUT", False)):
            raise ValueError("Pixel-only pilots do not accept target-lock metadata.")
        if bool(config.get("META_USE_FIRE_TARGET", False)):
            raise ValueError("Pixel pilots support object targets and NOOP, not meta FIRE.")
        if not bool(config.get("META_USE_NOOP_TARGET", False)):
            raise ValueError("Pixel pilots require an explicit reserved NOOP slot.")
        kwargs = _meta_network_kwargs(
            config, num_layers=_meta_num_layers(config, "META_FLAT_NUM_LAYERS")
        )
        for key in ("num_skills", "use_target_locked_input", "meta_input_mode"):
            kwargs.pop(key)
        return PixelMetaQNetwork(
            **kwargs, architecture=arch, noop_idx=meta_noop_target_idx(config),
            candidate_chunk_size=int(config.get("META_CANDIDATE_CHUNK_SIZE", 1)),
        )

    if arch in ("per_object_mlp", "mlp", "default"):
        kwargs = _meta_network_kwargs(
            config, num_layers=_meta_num_layers(config, "META_PER_OBJECT_NUM_LAYERS")
        )
        return MetaQNetwork(**kwargs)

    if arch in ("flat_mlp", "flat", "raw_obs"):
        kwargs = _meta_network_kwargs(
            config, num_layers=_meta_num_layers(config, "META_FLAT_NUM_LAYERS")
        )
        return FlatMetaQNetwork(**kwargs)

    raise ValueError(
        f"Unknown META_ARCHITECTURE={arch!r}. "
        f"Choose one of: {', '.join(META_ARCHITECTURE_CHOICES)} "
        f"(aliases: mlp, flat, raw_obs)."
    )
