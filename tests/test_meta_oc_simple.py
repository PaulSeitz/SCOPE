"""Tests for META_INPUT_MODE=oc_simple* feature packing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from concept_based_learning.environments.datatypes import (
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NOOP,
    NUM_LABELS,
    LabeledObjectObservation,
)
from concept_based_learning.networks.meta_factory import create_meta_network
from concept_based_learning.observations.meta_oc_simple import (
    build_meta_oc_simple_features,
    build_meta_oc_simple_global,
    build_meta_oc_simple_object_matrix,
    build_meta_oc_simple_rel_object_matrix,
    build_meta_role_pool_features,
)


class _Box:
    def __init__(self, x, y, w, h):
        self.x = jnp.asarray(x, dtype=jnp.float32)
        self.y = jnp.asarray(y, dtype=jnp.float32)
        self.width = jnp.asarray(w, dtype=jnp.float32)
        self.height = jnp.asarray(h, dtype=jnp.float32)


class _Raw:
    def __init__(self):
        self.player = _Box(
            [10.0, 12.0],
            [40.0, 42.0],
            [8.0, 8.0],
            [10.0, 10.0],
        )
        self.oxygen_level = jnp.asarray([50.0, 48.0], dtype=jnp.float32)
        self.collected_divers = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
        self.lives = jnp.asarray([3.0, 2.0], dtype=jnp.float32)


def _objects(max_objects: int = 4) -> LabeledObjectObservation:
    x = jnp.asarray([0.0, 20.0, 30.0, 0.0], dtype=jnp.float32)
    y = jnp.asarray([0.0, 50.0, 60.0, 0.0], dtype=jnp.float32)
    w = jnp.asarray([0.0, 8.0, 10.0, 0.0], dtype=jnp.float32)
    h = jnp.asarray([0.0, 8.0, 6.0, 0.0], dtype=jnp.float32)
    active = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    labels = jnp.asarray(
        [LABEL_NOOP, LABEL_COLLECT, LABEL_DESTROY, LABEL_NOOP], dtype=jnp.int32
    )
    zeros = jnp.zeros((max_objects,), dtype=jnp.int32)
    return LabeledObjectObservation(
        x=x,
        y=y,
        width=w,
        height=h,
        active=active,
        visual_id=jnp.arange(max_objects, dtype=jnp.int32),
        state=zeros,
        orientation=zeros.astype(jnp.float32),
        semantic_label=labels,
        can_go_offscreen=zeros,
        was_killed=zeros,
        was_collected=zeros,
    )


def _stack_objects(objs: LabeledObjectObservation) -> LabeledObjectObservation:
    return LabeledObjectObservation(
        **{
            f: jnp.stack([getattr(objs, f), getattr(objs, f)], axis=0)
            for f in (
                "x",
                "y",
                "width",
                "height",
                "active",
                "visual_id",
                "state",
                "orientation",
                "semantic_label",
                "can_go_offscreen",
                "was_killed",
                "was_collected",
            )
        }
    )


def test_oc_simple_global_uses_latest_player_and_mission_scalars():
    g = build_meta_oc_simple_global(_Raw())
    assert g.shape == (6,)
    assert float(g[0]) == pytest.approx(12.0)
    assert float(g[4]) == pytest.approx(48.0)
    assert float(g[5]) == pytest.approx(3.0)


def test_oc_simple_inv_adds_lives_and_active_collect():
    g = build_meta_oc_simple_global(
        _Raw(),
        include_inventory_extras=True,
        objects=_objects(),
        max_objects=4,
    )
    assert g.shape == (8,)
    assert float(g[6]) == pytest.approx(2.0)  # lives
    assert float(g[7]) == pytest.approx(1.0)  # one active collect


def test_oc_simple_object_matrix_bbox_active_label_onehot_no_visual():
    mat = build_meta_oc_simple_object_matrix(_objects(), max_objects=4)
    assert mat.shape == (4, 5 + NUM_LABELS)
    assert float(mat[1, 5 + LABEL_COLLECT]) == pytest.approx(1.0)


def test_oc_simple_rel_zeros_inactive_geom_keeps_label():
    raw = _Raw()
    mat = build_meta_oc_simple_rel_object_matrix(raw, _objects(), max_objects=4)
    assert mat.shape == (4, 6 + NUM_LABELS)
    # Inactive NOOP: geometry zeroed, label OH still marks NOOP.
    assert float(jnp.sum(mat[0, :6])) == pytest.approx(0.0)
    assert float(mat[0, 6 + LABEL_NOOP]) == pytest.approx(1.0)
    assert float(mat[1, 5]) == pytest.approx(1.0)  # active
    assert float(mat[1, 6 + LABEL_COLLECT]) == pytest.approx(1.0)


def test_role_pool_counts_and_min_dist():
    feats = build_meta_role_pool_features(_Raw(), _objects(), max_objects=4)
    assert feats.shape == (8,)
    assert float(feats[0]) == pytest.approx(1.0)  # collect count
    assert float(feats[2]) == pytest.approx(1.0)  # destroy count


@pytest.mark.parametrize(
    "mode,gdim,odim",
    [
        ("oc_simple", 6, 5 + NUM_LABELS),
        ("oc_simple_rel", 6, 6 + NUM_LABELS),
        ("oc_simple_role", 6 + 8, 5 + NUM_LABELS),
        ("oc_simple_rel_role", 6 + 8, 6 + NUM_LABELS),
        ("oc_simple_inv", 8, 5 + NUM_LABELS),
    ],
)
def test_build_meta_oc_simple_features_shapes(mode, gdim, odim):
    g, o = build_meta_oc_simple_features(
        _Raw(), _objects(), max_objects=4, mode=mode
    )
    assert g.shape == (gdim,)
    assert o.shape == (4, odim)


@pytest.mark.parametrize(
    "mode",
    [
        "oc_simple",
        "oc_simple_rel",
        "oc_simple_role",
        "oc_simple_rel_role",
        "oc_simple_inv",
    ],
)
def test_meta_qnetwork_oc_simple_family_forward(mode):
    max_objects = 4
    net = create_meta_network(
        {
            "MAX_OBJECTS": max_objects,
            "META_ARCHITECTURE": "per_object_mlp",
            "META_INPUT_MODE": mode,
            "META_HIDDEN_SIZE": 32,
            "META_PER_OBJECT_NUM_LAYERS": 2,
            "META_NORM_TYPE": "layer_norm",
            "NUM_SKILLS": 3,
            "META_TARGET_LOCKED_INPUT": False,
        }
    )
    params = net.init(
        jax.random.PRNGKey(0), _Raw(), _stack_objects(_objects()), False
    )
    q = net.apply(params, _Raw(), _stack_objects(_objects()), False)
    assert q.shape == (max_objects,)
