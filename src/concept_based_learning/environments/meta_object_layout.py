"""Reserved object indices for learnable meta actions.

Meta slots are **appended** after game entities so NOOP is never index 0
(argmax ties / cold Qs would otherwise collapse onto idle).

Layout for ``num_reserved`` slots at the end of a ``max_objects`` vector:
  - ``num_reserved == 1``: ``[..., NOOP]``  → NOOP at ``max_objects - 1``
  - ``num_reserved >= 2``: ``[..., FIRE, …, NOOP]`` → FIRE at
    ``max_objects - num_reserved``, NOOP at ``max_objects - 1``
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_META_FIRE,
    LABEL_NOOP,
    LabeledObjectObservation,
)


def meta_noop_slot_index(max_objects: int) -> int:
    """Absolute index of the learnable NOOP slot (always last)."""
    mo = int(max_objects)
    return max(mo - 1, 0)


def meta_fire_slot_index(max_objects: int, num_reserved: int = 2) -> int:
    """Absolute index of the learnable FIRE slot (first of the trailing reserved block)."""
    mo = int(max_objects)
    nr = max(int(num_reserved), 2)
    return max(mo - nr, 0)


# Legacy names: historically NOOP was index 0 and FIRE index 1 (prepended).
# Prefer meta_noop_slot_index / meta_fire_slot_index with MAX_OBJECTS.
META_NOOP_OBJECT_IDX = -1  # sentinel: "last slot" (resolved via MAX_OBJECTS)
META_FIRE_OBJECT_IDX = -2  # sentinel: "first of trailing reserved block"


def append_meta_reserved_slots(
    objects: LabeledObjectObservation,
    num_reserved: int = 1,
) -> LabeledObjectObservation:
    """
    Append ``num_reserved`` synthetic meta slots after game entities.

    Last reserved slot: LABEL_NOOP (learnable NOOP).
    First reserved slot (when ``num_reserved >= 2``): LABEL_META_FIRE.
    """
    if num_reserved <= 0:
        return objects

    n = int(objects.x.shape[0])
    zf = jnp.zeros((num_reserved,), dtype=objects.x.dtype)
    zi = jnp.zeros((num_reserved,), dtype=jnp.int32)
    reserved_labels = jnp.full((num_reserved,), LABEL_NOOP, dtype=jnp.int32)
    if num_reserved >= 2:
        reserved_labels = reserved_labels.at[0].set(LABEL_META_FIRE)

    updates = {
        "x": jnp.concatenate([objects.x, zf]),
        "y": jnp.concatenate([objects.y, zf]),
        "width": jnp.concatenate([objects.width, zf]),
        "height": jnp.concatenate([objects.height, zf]),
        "active": jnp.concatenate([objects.active, zi]),
        "semantic_label": jnp.concatenate([objects.semantic_label, reserved_labels]),
    }
    for name in ("state", "orientation", "can_go_offscreen", "was_killed", "was_collected"):
        if hasattr(objects, name):
            leaf = getattr(objects, name)
            if leaf.ndim > 0 and int(leaf.shape[0]) == n:
                updates[name] = jnp.concatenate([leaf, zi.astype(leaf.dtype)])
    updates["visual_id"] = jnp.arange(n + num_reserved, dtype=jnp.int32)
    return objects.replace(**updates)


def prepend_meta_reserved_slots(
    objects: LabeledObjectObservation,
    num_reserved: int = 1,
) -> LabeledObjectObservation:
    """Deprecated alias: meta slots are appended (NOOP last), not prepended."""
    return append_meta_reserved_slots(objects, num_reserved=num_reserved)


def prepend_meta_noop_slot(objects: LabeledObjectObservation) -> LabeledObjectObservation:
    """Backward-compatible alias: reserve only the NOOP slot at the end."""
    return append_meta_reserved_slots(objects, num_reserved=1)
