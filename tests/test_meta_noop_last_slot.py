"""Meta reserved slots are appended; NOOP is last."""

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_META_FIRE,
    LABEL_NOOP,
    LabeledObjectObservation,
)
from concept_based_learning.environments.meta_object_layout import (
    append_meta_reserved_slots,
    meta_fire_slot_index,
    meta_noop_slot_index,
)
from concept_based_learning.environments.meta_target_mask import (
    build_meta_target_mask,
    meta_fire_target_idx,
    meta_noop_target_idx,
)


def _toy_objects(n_game: int) -> LabeledObjectObservation:
    return LabeledObjectObservation(
        x=jnp.arange(n_game, dtype=jnp.int32),
        y=jnp.arange(n_game, dtype=jnp.int32) + 10,
        width=jnp.ones(n_game, dtype=jnp.int32),
        height=jnp.ones(n_game, dtype=jnp.int32),
        active=jnp.ones(n_game, dtype=jnp.int32),
        semantic_label=jnp.ones(n_game, dtype=jnp.int32),  # DESTROY
        state=jnp.zeros(n_game, dtype=jnp.int32),
        can_go_offscreen=jnp.zeros(n_game, dtype=jnp.int32),
        was_killed=jnp.zeros(n_game, dtype=jnp.int32),
        was_collected=jnp.zeros(n_game, dtype=jnp.int32),
        visual_id=jnp.arange(n_game, dtype=jnp.int32),
    )


def test_append_meta_noop_last_single_reserved():
    objs = append_meta_reserved_slots(_toy_objects(4), num_reserved=1)
    assert objs.x.shape[0] == 5
    assert int(objs.semantic_label[-1]) == LABEL_NOOP
    assert int(objs.active[-1]) == 0
    assert int(objs.x[0]) == 0  # game slot 0 unchanged


def test_append_meta_fire_then_noop():
    objs = append_meta_reserved_slots(_toy_objects(3), num_reserved=2)
    assert objs.x.shape[0] == 5
    assert int(objs.semantic_label[-2]) == LABEL_META_FIRE
    assert int(objs.semantic_label[-1]) == LABEL_NOOP
    assert meta_noop_slot_index(5) == 4
    assert meta_fire_slot_index(5, num_reserved=2) == 3


def test_meta_noop_idx_defaults_to_last():
    cfg = {"MAX_OBJECTS": 31, "META_USE_NOOP_TARGET": True}
    assert meta_noop_target_idx(cfg) == 30
    assert meta_fire_target_idx(cfg) == 29
    cfg_null = {**cfg, "META_NOOP_TARGET_IDX": None}
    assert meta_noop_target_idx(cfg_null) == 30


def test_mask_force_enables_trailing_noop():
    n_game = 4
    max_objects = 5
    objs = append_meta_reserved_slots(_toy_objects(n_game), num_reserved=1)
    # stack dim for mask helper
    class _O:
        active = objs.active[None, ...]  # fake wrong shape
    # build_meta_target_mask expects stacked [stack, max] or [max]
    stacked = jax_tree_stack(objs)
    cfg = {
        "MAX_OBJECTS": max_objects,
        "META_USE_NOOP_TARGET": True,
        "META_NOOP_TARGET_IDX": None,
    }
    mask = build_meta_target_mask(stacked, None, cfg)
    assert bool(mask[-1]) is True
    assert bool(mask[0]) is True  # active DESTROY game object


def jax_tree_stack(objs):
    """Add a trivial stack axis of size 1."""
    return objs.replace(
        **{
            name: jnp.expand_dims(getattr(objs, name), 0)
            for name in (
                "x",
                "y",
                "width",
                "height",
                "active",
                "semantic_label",
                "state",
                "can_go_offscreen",
                "was_killed",
                "was_collected",
                "visual_id",
            )
            if hasattr(objs, name)
        }
    )
