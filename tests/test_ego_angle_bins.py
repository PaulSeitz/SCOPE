"""World-θ triangular bins replace sin/cos(ψ) in target-aligned ego pose."""

from __future__ import annotations

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest

from concept_based_learning.environments.datatypes import LABEL_COLLECT
from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    distance_feature_from_pixels,
    ego_angle_bins,
    ego_obs_channels_per_frame,
    ego_pose_channels_per_frame,
    get_ego_obs,
    set_ego_angle_bins,
    set_ego_distance_clip_max_pixels,
    skill_ego_flat_dim,
    triangular_angle_bins,
)

_Objects = namedtuple(
    "_Objects", ["x", "y", "width", "height", "active", "semantic_label"]
)


class _FakePlayer:
    def __init__(self, n_frames, *, x=0.0, y=3.0, w=4.0, h=4.0, ori=90.0):
        z = jnp.ones((n_frames,), dtype=jnp.float32)
        self.x = z * x
        self.y = z * y
        self.width = z * w
        self.height = z * h
        self.orientation = z * ori


class _FakeRawObs:
    def __init__(self, n_frames, **kwargs):
        self.player = _FakePlayer(n_frames, **kwargs)


def _stacked_objects(x, y, w=4.0, h=4.0, n_frames=2):
    row = _Objects(
        x=jnp.array([x], dtype=jnp.float32),
        y=jnp.array([y], dtype=jnp.float32),
        width=jnp.array([w], dtype=jnp.float32),
        height=jnp.array([h], dtype=jnp.float32),
        active=jnp.array([1], dtype=jnp.int32),
        semantic_label=jnp.array([LABEL_COLLECT], dtype=jnp.int32),
    )
    return _Objects(*(jnp.stack([leaf] * n_frames) for leaf in row))


def _pose_only_cfg(**overrides):
    cfg = {
        "USE_LEGACY_EGO_OBS": False,
        "USE_EGO_SIZE_FEATURES": False,
        "USE_EGO_VELOCITY_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": False,
        "USE_EGO_WEAPON_FEATURES": False,
        "EGO_DISTANCE_CLIP_MAX_PX": None,
        "EGO_ANGLE_BINS": None,
    }
    cfg.update(overrides)
    configure_ego_obs_from_config(cfg)
    return cfg


def teardown_function():
    configure_ego_obs_from_config({})


def test_triangular_bins_partition_of_unity_and_overlap():
    n = 8
    theta = jnp.linspace(-jnp.pi, jnp.pi, 64, endpoint=False)
    bins = triangular_angle_bins(theta, n)
    np.testing.assert_allclose(np.array(bins.sum(axis=-1)), 1.0, atol=1e-5)
    # Bin 0 peaks at θ=0 (screen-right).
    at_east = triangular_angle_bins(jnp.array(0.0), n)
    assert float(at_east[0]) == pytest.approx(1.0)
    assert float(at_east[1]) == pytest.approx(0.0)
    # Midpoint 22.5° → adjacent bins 0.5.
    mid = triangular_angle_bins(jnp.array(jnp.pi / 8.0), n)
    assert float(mid[0]) == pytest.approx(0.5)
    assert float(mid[1]) == pytest.approx(0.5)
    # Wrap: ±π is west, bin 4 for n=8.
    west = triangular_angle_bins(jnp.array(jnp.pi), n)
    assert int(np.argmax(np.array(west))) == 4
    west_n = triangular_angle_bins(jnp.array(-jnp.pi), n)
    np.testing.assert_allclose(np.array(west), np.array(west_n), atol=1e-5)
    # Screen-up (θ=−π/2) is bin 6.
    up = triangular_angle_bins(jnp.array(-jnp.pi / 2.0), n)
    assert int(np.argmax(np.array(up))) == 6


def test_angle_bins_replace_sincos_in_channel_count():
    _pose_only_cfg()
    assert ego_angle_bins() is None
    assert ego_pose_channels_per_frame() == 3
    assert ego_obs_channels_per_frame() == 3
    _pose_only_cfg(EGO_ANGLE_BINS=8)
    assert ego_angle_bins() == 8
    assert ego_pose_channels_per_frame() == 9
    assert ego_obs_channels_per_frame() == 9
    assert skill_ego_flat_dim(1, stack_size=4) == 36


def test_clip_50_normalizes_by_cap_not_screen():
    set_ego_distance_clip_max_pixels(50.0)
    assert float(distance_feature_from_pixels(jnp.array(25.0))) == pytest.approx(0.5)
    assert float(distance_feature_from_pixels(jnp.array(50.0))) == pytest.approx(1.0)
    assert float(distance_feature_from_pixels(jnp.array(200.0))) == pytest.approx(1.0)
    set_ego_distance_clip_max_pixels(None)


def test_get_ego_obs_bins_follow_world_bearing_not_facing():
    """θ bins are world bearing; facing (ψ) must not rotate them."""
    _pose_only_cfg(EGO_ANGLE_BINS=8)
    n_frames = 2
    # Target to the right of the player, same row.
    objs = _stacked_objects(28.0, 3.0, n_frames=n_frames)
    right_face = get_ego_obs(
        _FakeRawObs(n_frames, ori=90.0), objs, jnp.int32(0)
    ).reshape(n_frames, -1)
    left_face = get_ego_obs(
        _FakeRawObs(n_frames, ori=270.0), objs, jnp.int32(0)
    ).reshape(n_frames, -1)
    # pose: distance + 8 bins. Peak bin must match and ignore facing.
    assert int(np.argmax(np.array(right_face[0, 1:]))) == 0
    np.testing.assert_allclose(
        np.array(right_face[:, 1:]), np.array(left_face[:, 1:]), atol=1e-5
    )


def test_set_ego_angle_bins_rejects_one():
    with pytest.raises(ValueError):
        set_ego_angle_bins(1)
    set_ego_angle_bins(0)
    assert ego_angle_bins() is None


def test_sincos_snap_finite_alphabet_and_channel_count():
    from concept_based_learning.observations.egocentric import (
        quantize_angle_sincos,
        set_ego_angle_sincos_snap,
    )

    _pose_only_cfg(EGO_ANGLE_BINS=32, EGO_ANGLE_SINCOS_SNAP=True)
    assert ego_pose_channels_per_frame() == 3  # dist + sin + cos
    n_frames = 2
    objs = _stacked_objects(28.0, 3.0, n_frames=n_frames)
    obs = get_ego_obs(_FakeRawObs(n_frames), objs, jnp.int32(0)).reshape(n_frames, -1)
    sin_c = obs[:, 1:3]
    uniq = {tuple(np.round(row, 4)) for row in np.array(sin_c)}
    assert len(uniq) <= 32
    # Sector centres lie on the unit circle.
    norms = np.linalg.norm(np.array(sin_c), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    set_ego_angle_sincos_snap(False)
