"""Facing-frame restrained pose: finite L/C/R × near/far alphabet."""

from __future__ import annotations

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest

from concept_based_learning.environments.datatypes import LABEL_COLLECT
from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
    ego_pose_channels_per_frame,
    get_ego_obs,
    restrained_lcr_nearfar_channels,
    restrained_quadrant_nearfar_channels,
    set_ego_restrained_pose,
)
from concept_based_learning.observations.pov_facing_frame import (
    configure_pov_facing_frame_from_config,
)
from concept_based_learning.observations.pov_relative import (
    configure_pov_relative_from_config,
)

_Objects = namedtuple(
    "_Objects", ["x", "y", "width", "height", "active", "semantic_label"]
)


class _FakePlayer:
    def __init__(self, n_frames, *, x=80.0, y=100.0, w=8.0, h=8.0, ori=90.0):
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


def _facing_cfg(**overrides):
    cfg = {
        "USE_POV_RELATIVE_MAPPER": True,
        "USE_POV_FIXED_BILATERAL": False,
        "USE_LEGACY_EGO_OBS": False,
        "USE_POV_FACING_FRAME": True,
        "ENV_NAME": "seaquest",
        "USE_EGO_SIZE_FEATURES": False,
        "USE_EGO_VELOCITY_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": False,
        "USE_EGO_WEAPON_FEATURES": False,
        "EGO_POSE_XY": False,
        "EGO_RESTRAINED_POSE": "lcr_nearfar",
        "EGO_RESTRAINED_LATERAL_BAND_PX": 8.0,
        "EGO_RESTRAINED_NEAR_FAR_PX": 50.0,
    }
    cfg.update(overrides)
    configure_pov_relative_from_config(cfg)
    configure_pov_facing_frame_from_config(cfg)
    configure_ego_obs_from_config(cfg)
    return cfg


def teardown_function():
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_pov_facing_frame_from_config({"USE_POV_FACING_FRAME": False})
    configure_ego_obs_from_config({})


def test_lcr_nearfar_one_hot():
    ch = restrained_lcr_nearfar_channels(
        jnp.float32(12.0),
        jnp.float32(30.0),
        lateral_band_px=8.0,
        near_far_px=50.0,
    )
    assert len(ch) == 5
    np.testing.assert_array_equal(np.array(ch), [0, 0, 1, 1, 0])


def test_quadrant_nearfar_ahead():
    ch = restrained_quadrant_nearfar_channels(
        jnp.float32(0.0),
        jnp.float32(80.0),
        near_far_px=50.0,
    )
    assert len(ch) == 6
    np.testing.assert_array_equal(np.array(ch), [1, 0, 0, 0, 0, 1])


def test_configure_clears_continuous_pose():
    _facing_cfg()
    assert ego_pose_channels_per_frame() == 5
    assert ego_obs_channels_per_frame() == 5


def test_get_ego_obs_lcr_nearfar_shape():
    _facing_cfg()
    raw = _FakeRawObs(2, x=80.0, y=100.0, ori=90.0)
    # target 12px right, 30px ahead → right + near
    objs = _stacked_objects(92.0, 130.0, n_frames=2)
    ego = get_ego_obs(raw, objs, target_idx=0)
    n_frames = 2
    ch = ego_pose_channels_per_frame()
    assert ego.shape == (n_frames * ch,)
    frame0 = ego[:ch]
    np.testing.assert_allclose(np.array(frame0), [0, 0, 1, 1, 0], atol=1e-5)


def test_quadrant_mode_channel_count():
    _facing_cfg(EGO_RESTRAINED_POSE="quadrant_nearfar")
    assert ego_pose_channels_per_frame() == 6


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="EGO_RESTRAINED_POSE"):
        set_ego_restrained_pose("invalid")
