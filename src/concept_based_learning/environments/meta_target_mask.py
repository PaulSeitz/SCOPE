"""Build masks for which object indices the meta-policy may select."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_BLOCKER, LABEL_NOOP
from concept_based_learning.environments.meta_object_layout import (
    meta_fire_slot_index,
    meta_noop_slot_index,
)

# jax_phoenix: enemy state >= 4 means death animation (not a valid target).
PHOENIX_DYING_STATE_THRESHOLD = 4

MetaMaskHook = Callable[..., jnp.ndarray]


def _latest_object_frame(field: jnp.ndarray) -> jnp.ndarray:
    """Select the latest stacked frame from object fields.

    Supports per-env batches ``[num_envs, stack, ...]`` and single-env ``[stack, ...]``.
    """
    if field.ndim >= 3:
        return field[:, -1, ...]
    if field.ndim == 2:
        return field[-1]
    return field


def _config_max_objects(config: Optional[dict]) -> int:
    if not config:
        return 0
    raw = config.get("MAX_OBJECTS", 0)
    if raw is None:
        return 0
    return int(raw)


def _resolve_slot_idx(raw, *, default: int) -> int:
    """``None`` / ``\"last\"`` / negative → ``default`` (absolute last-slot helpers)."""
    if raw is None or raw == "last":
        return int(default)
    idx = int(raw)
    if idx < 0:
        return int(default)
    return idx


def meta_noop_target_idx(config: Optional[dict]) -> int:
    """Object index for learnable NOOP (default: last slot = ``MAX_OBJECTS - 1``)."""
    mo = _config_max_objects(config)
    default = meta_noop_slot_index(mo)
    if not config:
        return default
    return _resolve_slot_idx(config.get("META_NOOP_TARGET_IDX", None), default=default)


def meta_fire_target_idx(config: Optional[dict]) -> int:
    """Object index for learnable FIRE (default: first trailing reserved = ``MAX_OBJECTS - 2``)."""
    mo = _config_max_objects(config)
    # Assume 2 reserved when FIRE is used; callers with reserved=1 should not enable FIRE.
    default = meta_fire_slot_index(mo, num_reserved=2)
    if not config:
        return default
    return _resolve_slot_idx(config.get("META_FIRE_TARGET_IDX", None), default=default)


def meta_noop_target_enabled(config: Optional[dict]) -> bool:
    """True when a dedicated meta-policy slot is reserved for learnable NOOP."""
    if not config:
        return False
    use_noop = bool(config.get("META_USE_NOOP_TARGET", False))
    max_objects = _config_max_objects(config)
    idx = meta_noop_target_idx(config)
    return use_noop and max_objects > 0 and 0 <= idx < max_objects


def meta_fire_target_enabled(config: Optional[dict]) -> bool:
    """True when a dedicated meta-policy slot is reserved for learnable FIRE."""
    if not config:
        return False
    use_fire = bool(config.get("META_USE_FIRE_TARGET", False))
    max_objects = _config_max_objects(config)
    idx = meta_fire_target_idx(config)
    return use_fire and max_objects > 0 and 0 <= idx < max_objects


def meta_fire_game_action(config: Optional[dict]) -> int:
    """Atari game action emitted when the meta-policy selects the FIRE slot."""
    if not config:
        return 1
    return int(config.get("META_FIRE_GAME_ACTION", 1))


def apply_meta_noop_target_mask(mask: jnp.ndarray, config: Optional[dict]) -> jnp.ndarray:
    """Keep the NOOP slot selectable even when it has no real object."""
    if not meta_noop_target_enabled(config):
        return mask
    idx = meta_noop_target_idx(config)
    if mask.ndim == 1:
        return mask.at[idx].set(True)
    return mask.at[:, idx].set(True)


def apply_meta_fire_target_mask(mask: jnp.ndarray, config: Optional[dict]) -> jnp.ndarray:
    """Keep the FIRE slot selectable even when it has no real object."""
    if not meta_fire_target_enabled(config):
        return mask
    idx = meta_fire_target_idx(config)
    if mask.ndim == 1:
        return mask.at[idx].set(True)
    return mask.at[:, idx].set(True)


def is_meta_noop_target(
    target_idxs: jnp.ndarray,
    config: Optional[dict],
    *,
    noop_idx: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> jnp.ndarray:
    """Per-env boolean: meta selected the learnable NOOP slot."""
    if enabled is None:
        enabled = meta_noop_target_enabled(config)
    if not enabled:
        return jnp.zeros_like(target_idxs, dtype=bool)
    idx = meta_noop_target_idx(config) if noop_idx is None else noop_idx
    return target_idxs == jnp.asarray(idx, dtype=target_idxs.dtype)


def is_meta_fire_target(
    target_idxs: jnp.ndarray,
    config: Optional[dict],
    *,
    fire_idx: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> jnp.ndarray:
    """Per-env boolean: meta selected the learnable FIRE slot."""
    if enabled is None:
        enabled = meta_fire_target_enabled(config)
    if not enabled:
        return jnp.zeros_like(target_idxs, dtype=bool)
    idx = meta_fire_target_idx(config) if fire_idx is None else fire_idx
    return target_idxs == jnp.asarray(idx, dtype=target_idxs.dtype)


def build_meta_target_mask(
    objects: Any,
    raw_obs: Any = None,
    config: Optional[dict] = None,
    hook: Optional[MetaMaskHook] = None,
) -> jnp.ndarray:
    """
    Mask for meta-policy target selection (shape [num_envs, max_objects] or [max_objects]).

    An index is selectable when:
    - ``objects.active`` is set on the latest frame, and
    - ``semantic_label`` is not ``LABEL_NOOP`` (padding) and not ``LABEL_BLOCKER``
      (structural occluders for the line-of-sight feature), then
    - any environment-specific ``hook`` is applied.
    """
    active = _latest_object_frame(objects.active) > 0.5
    latest_labels = _latest_object_frame(objects.semantic_label)
    targetable_label = jnp.logical_and(
        latest_labels != LABEL_NOOP,
        latest_labels != LABEL_BLOCKER,
    )
    mask = jnp.logical_and(active, targetable_label)
    if hook is not None:
        mask = hook(raw_obs, mask, config, objects)
    mask = apply_meta_noop_target_mask(mask, config)
    return apply_meta_fire_target_mask(mask, config)


def phoenix_meta_mask_hook(raw_obs, active_masks, config, objects=None):
    """Drop dying enemies (state >= 4) from the meta-policy action set."""
    del raw_obs, config
    if objects is None or not hasattr(objects, "state"):
        return active_masks
    latest_state = _latest_object_frame(objects.state)
    is_dying = latest_state >= PHOENIX_DYING_STATE_THRESHOLD
    return jnp.logical_and(active_masks, jnp.logical_not(is_dying))
