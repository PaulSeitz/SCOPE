"""Tests for the line-of-sight blocked flag (raycast + ego channel wiring)."""

from collections import namedtuple

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_BLOCKER,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NOOP,
    static_blocker_arrays,
)
from concept_based_learning.environments.meta_target_mask import build_meta_target_mask
from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
    get_ego_obs,
)
from concept_based_learning.observations.line_of_sight import (
    line_of_sight_blocked,
    segment_intersects_box,
)

_Objects = namedtuple(
    "_Objects", ["x", "y", "width", "height", "active", "semantic_label"]
)


def _objects(rows):
    """rows: list of (x, y, w, h, active, label) — single frame [N] arrays."""
    cols = list(zip(*rows))
    return _Objects(
        x=jnp.array(cols[0], dtype=jnp.int32),
        y=jnp.array(cols[1], dtype=jnp.int32),
        width=jnp.array(cols[2], dtype=jnp.int32),
        height=jnp.array(cols[3], dtype=jnp.int32),
        active=jnp.array(cols[4], dtype=jnp.int32),
        semantic_label=jnp.array(cols[5], dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# segment_intersects_box
# ---------------------------------------------------------------------------


def test_segment_crosses_box():
    assert bool(segment_intersects_box(0.0, 5.0, 20.0, 5.0, 8.0, 0.0, 12.0, 10.0))


def test_segment_misses_box():
    assert not bool(segment_intersects_box(0.0, 5.0, 20.0, 5.0, 8.0, 20.0, 12.0, 30.0))


def test_segment_stops_before_box():
    # Box lies beyond the target endpoint (t > 1).
    assert not bool(segment_intersects_box(0.0, 5.0, 6.0, 5.0, 8.0, 0.0, 12.0, 10.0))


def test_segment_box_behind_player():
    # Box lies behind the start point (t < 0).
    assert not bool(segment_intersects_box(10.0, 5.0, 20.0, 5.0, 0.0, 0.0, 4.0, 10.0))


def test_vertical_segment_parallel_slab():
    # Vertical segment straight through a box (x-parallel slab handling).
    assert bool(segment_intersects_box(10.0, 0.0, 10.0, 20.0, 8.0, 8.0, 12.0, 12.0))
    # Vertical segment beside the box.
    assert not bool(segment_intersects_box(20.0, 0.0, 20.0, 20.0, 8.0, 8.0, 12.0, 12.0))


# ---------------------------------------------------------------------------
# line_of_sight_blocked
# ---------------------------------------------------------------------------


def _los(objs, target_idx, p=(0.0, 5.0), t=(30.0, 5.0)):
    return bool(
        line_of_sight_blocked(
            jnp.float32(p[0]),
            jnp.float32(p[1]),
            jnp.float32(t[0]),
            jnp.float32(t[1]),
            jnp.int32(target_idx),
            objs.x,
            objs.y,
            objs.width,
            objs.height,
            objs.active,
            objs.semantic_label,
            y_is_bottom_edge=False,
        )
    )


def test_avoid_object_blocks():
    objs = _objects([
        (28, 3, 4, 4, 1, LABEL_COLLECT),   # target at slot 0
        (14, 3, 4, 4, 1, LABEL_AVOID),     # hazard on the ray
    ])
    assert _los(objs, target_idx=0)


def test_collect_object_does_not_block():
    # Divers can be shot/moved through.
    objs = _objects([
        (28, 3, 4, 4, 1, LABEL_DESTROY),
        (14, 3, 4, 4, 1, LABEL_COLLECT),
    ])
    assert not _los(objs, target_idx=0)


def test_target_itself_does_not_block():
    objs = _objects([
        (28, 3, 4, 4, 1, LABEL_DESTROY),
    ])
    assert not _los(objs, target_idx=0)


def test_inactive_blocker_does_not_block():
    objs = _objects([
        (28, 3, 4, 4, 1, LABEL_COLLECT),
        (14, 3, 4, 4, 0, LABEL_BLOCKER),
    ])
    assert not _los(objs, target_idx=0)


def test_static_blocker_wall_blocks():
    xs, ys, ws, hs, active, labels = static_blocker_arrays([(14, 0, 2, 20)])
    objs = _Objects(
        x=jnp.concatenate([jnp.array([28], dtype=jnp.int32), xs]),
        y=jnp.concatenate([jnp.array([3], dtype=jnp.int32), ys]),
        width=jnp.concatenate([jnp.array([4], dtype=jnp.int32), ws]),
        height=jnp.concatenate([jnp.array([4], dtype=jnp.int32), hs]),
        active=jnp.concatenate([jnp.array([1], dtype=jnp.int32), active]),
        semantic_label=jnp.concatenate(
            [jnp.array([LABEL_COLLECT], dtype=jnp.int32), labels]
        ),
    )
    assert _los(objs, target_idx=0)


def test_blocker_off_ray_does_not_block():
    objs = _objects([
        (28, 3, 4, 4, 1, LABEL_COLLECT),
        (14, 50, 4, 4, 1, LABEL_BLOCKER),
    ])
    assert not _los(objs, target_idx=0)


# ---------------------------------------------------------------------------
# Meta target mask
# ---------------------------------------------------------------------------


def test_meta_mask_excludes_blockers():
    objs = _Objects(
        x=jnp.zeros((2, 3), dtype=jnp.int32),
        y=jnp.zeros((2, 3), dtype=jnp.int32),
        width=jnp.ones((2, 3), dtype=jnp.int32),
        height=jnp.ones((2, 3), dtype=jnp.int32),
        active=jnp.ones((2, 3), dtype=jnp.int32),
        semantic_label=jnp.array(
            [[LABEL_NOOP, LABEL_DESTROY, LABEL_BLOCKER]] * 2, dtype=jnp.int32
        ),
    )
    mask = build_meta_target_mask(objs)
    assert mask.tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# Ego channel wiring
# ---------------------------------------------------------------------------


class _FakePlayer:
    def __init__(self, n_frames):
        z = jnp.zeros((n_frames,), dtype=jnp.float32)
        self.x = z
        self.y = z + 3.0
        self.width = z + 4.0
        self.height = z + 4.0
        self.orientation = z


class _FakeRawObs:
    def __init__(self, n_frames):
        self.player = _FakePlayer(n_frames)


def _stacked_objects(rows, n_frames=2):
    single = _objects(rows)
    return _Objects(*(jnp.stack([leaf] * n_frames) for leaf in single))


def _base_config(los_enabled):
    return {
        "USE_LEGACY_EGO_OBS": False,
        "USE_EGO_SIZE_FEATURES": False,
        "USE_EGO_VELOCITY_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": los_enabled,
        "USE_EGO_WEAPON_FEATURES": False,
    }


def test_ego_channel_count_with_los():
    configure_ego_obs_from_config(_base_config(False))
    assert ego_obs_channels_per_frame() == 3
    configure_ego_obs_from_config(_base_config(True))
    assert ego_obs_channels_per_frame() == 4


def test_ego_obs_los_channel_value():
    n_frames = 2
    raw = _FakeRawObs(n_frames)
    target_idx = jnp.int32(0)

    blocked_objs = _stacked_objects([
        (28, 3, 4, 4, 1, LABEL_COLLECT),   # target
        (14, 3, 4, 4, 1, LABEL_AVOID),     # hazard on the ray
    ], n_frames)
    clear_objs = _stacked_objects([
        (28, 3, 4, 4, 1, LABEL_COLLECT),
        (14, 50, 4, 4, 1, LABEL_AVOID),    # hazard far off the ray
    ], n_frames)

    configure_ego_obs_from_config(_base_config(True))
    ego_blocked = get_ego_obs(raw, blocked_objs, target_idx).reshape(n_frames, -1)
    ego_clear = get_ego_obs(raw, clear_objs, target_idx).reshape(n_frames, -1)
    # LOS flag is the last base channel (no skill extras configured).
    assert ego_blocked[:, -1].tolist() == [1.0, 1.0]
    assert ego_clear[:, -1].tolist() == [0.0, 0.0]

    # Restore defaults for other tests.
    configure_ego_obs_from_config({})
