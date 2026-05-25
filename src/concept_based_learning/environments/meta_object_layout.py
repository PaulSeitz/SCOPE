"""Reserved object index 0 for learnable meta NOOP; game entities start at index 1."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_NOOP, LabeledObjectObservation

# Index 0 is always the meta-policy NOOP slot (inactive; selectable via mask override).
META_NOOP_OBJECT_IDX = 0


def prepend_meta_noop_slot(objects: LabeledObjectObservation) -> LabeledObjectObservation:
    """Shift game objects right by one so index ``META_NOOP_OBJECT_IDX`` is reserved."""
    n = int(objects.x.shape[0])
    zf = jnp.zeros((1,), dtype=objects.x.dtype)
    zi = jnp.zeros((1,), dtype=jnp.int32)

    updates = {
        "x": jnp.concatenate([zf, objects.x]),
        "y": jnp.concatenate([zf, objects.y]),
        "width": jnp.concatenate([zf, objects.width]),
        "height": jnp.concatenate([zf, objects.height]),
        "active": jnp.concatenate([zi, objects.active]),
        "semantic_label": jnp.concatenate(
            [jnp.full((1,), LABEL_NOOP, dtype=jnp.int32), objects.semantic_label]
        ),
    }
    for name in ("state", "visual_id", "orientation"):
        if hasattr(objects, name):
            leaf = getattr(objects, name)
            if leaf.ndim > 0 and int(leaf.shape[0]) == n:
                updates[name] = jnp.concatenate([zi.astype(leaf.dtype), leaf])
    return objects.replace(**updates)
