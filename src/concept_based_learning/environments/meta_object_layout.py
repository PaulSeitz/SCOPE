"""Reserved object indices for learnable meta actions; game entities follow."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_META_FIRE,
    LABEL_NOOP,
    LabeledObjectObservation,
)

# Index 0 is always the meta-policy NOOP slot (inactive; selectable via mask override).
META_NOOP_OBJECT_IDX = 0
# Index 1 is the optional learnable FIRE slot (inactive; selectable via mask override).
META_FIRE_OBJECT_IDX = 1


def prepend_meta_reserved_slots(
    objects: LabeledObjectObservation,
    num_reserved: int = 1,
) -> LabeledObjectObservation:
    """
    Prepend ``num_reserved`` synthetic meta slots at indices 0..num_reserved-1.

    Slot 0: LABEL_NOOP (learnable NOOP)
    Slot 1 (when num_reserved >= 2): LABEL_META_FIRE (learnable FIRE)
    """
    if num_reserved <= 0:
        return objects

    n = int(objects.x.shape[0])
    zf = jnp.zeros((num_reserved,), dtype=objects.x.dtype)
    zi = jnp.zeros((num_reserved,), dtype=jnp.int32)
    reserved_labels = jnp.full((num_reserved,), LABEL_NOOP, dtype=jnp.int32)
    if num_reserved >= 2:
        reserved_labels = reserved_labels.at[META_FIRE_OBJECT_IDX].set(LABEL_META_FIRE)

    updates = {
        "x": jnp.concatenate([zf, objects.x]),
        "y": jnp.concatenate([zf, objects.y]),
        "width": jnp.concatenate([zf, objects.width]),
        "height": jnp.concatenate([zf, objects.height]),
        "active": jnp.concatenate([zi, objects.active]),
        "semantic_label": jnp.concatenate([reserved_labels, objects.semantic_label]),
    }
    for name in ("state", "orientation", "can_go_offscreen", "was_killed", "was_collected"):
        if hasattr(objects, name):
            leaf = getattr(objects, name)
            if leaf.ndim > 0 and int(leaf.shape[0]) == n:
                updates[name] = jnp.concatenate([zi.astype(leaf.dtype), leaf])
    updates["visual_id"] = jnp.arange(n + num_reserved, dtype=jnp.int32)
    return objects.replace(**updates)


def prepend_meta_noop_slot(objects: LabeledObjectObservation) -> LabeledObjectObservation:
    """Backward-compatible alias: reserve only the NOOP slot at index 0."""
    return prepend_meta_reserved_slots(objects, num_reserved=1)
