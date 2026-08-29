"""Tests for facing-locked FPS ego frame (view = player facing)."""

import jax.numpy as jnp
import pytest
from jaxatari.environment import JAXAtariAction as Atari

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_angle_channels_per_frame,
    ego_distance_channels_per_frame,
    ego_obs_channels_per_frame,
)
from concept_based_learning.observations.pov_facing_frame import (
    configure_pov_facing_frame_from_config,
    facing_frame_phi_rad,
    using_ego_pose_xy,
    using_pov_facing_frame,
    world_delta_to_facing_frame,
)
from concept_based_learning.observations.pov_relative import (
    configure_pov_relative_from_config,
    make_facing_relative_mappers,
)


FACING_BASE_CFG = {
    "USE_POV_RELATIVE_MAPPER": True,
    "USE_POV_FIXED_BILATERAL": False,
    "USE_POV_FACING_FRAME": True,
    "EGO_POSE_XY": False,
    "USE_EGO_SIZE_FEATURES": False,
    "USE_EGO_VELOCITY_FEATURES": False,
    "USE_EGO_BLOCKING_FEATURES": True,
    "USE_EGO_WEAPON_FEATURES": False,
    "ENV_NAME": "seaquest",
    "EGO_DISTANCE_CLIP_MAX_PX": 50,
}


def _configure_facing(**overrides):
    cfg = {**FACING_BASE_CFG, **overrides}
    configure_pov_relative_from_config(cfg)
    configure_pov_facing_frame_from_config(cfg)
    configure_ego_obs_from_config(cfg)
    return cfg


def _teardown():
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_pov_facing_frame_from_config({"USE_POV_FACING_FRAME": False})
    configure_ego_obs_from_config({})


def test_flag_requires_pov_mapper():
    with pytest.raises(ValueError, match="requires USE_POV_RELATIVE_MAPPER"):
        configure_pov_facing_frame_from_config({"USE_POV_FACING_FRAME": True})


def test_incompatible_with_fixed_bilateral():
    with pytest.raises(ValueError, match="incompatible with USE_POV_FIXED_BILATERAL"):
        configure_pov_facing_frame_from_config(
            {**FACING_BASE_CFG, "USE_POV_FIXED_BILATERAL": True}
        )


def test_incompatible_with_turn_primitives():
    with pytest.raises(ValueError, match="incompatible with USE_POV_TURN_PRIMITIVES"):
        configure_pov_facing_frame_from_config(
            {**FACING_BASE_CFG, "USE_POV_TURN_PRIMITIVES": True}
        )


def test_facing_right_forward_is_plus_x():
    fwd, lat = world_delta_to_facing_frame(
        jnp.float32(30.0), jnp.float32(10.0), jnp.float32(0.0)
    )
    assert float(fwd) == pytest.approx(30.0, abs=1e-5)
    assert float(lat) == pytest.approx(10.0, abs=1e-5)


def test_facing_left_mirrors_forward():
    fwd, lat = world_delta_to_facing_frame(
        jnp.float32(30.0), jnp.float32(10.0), jnp.float32(180.0)
    )
    assert float(fwd) == pytest.approx(-30.0, abs=1e-5)
    assert float(lat) == pytest.approx(-10.0, abs=1e-4)


def test_target_behind_has_negative_fwd():
    # Facing right; target to the left → behind.
    fwd, lat = world_delta_to_facing_frame(
        jnp.float32(-20.0), jnp.float32(0.0), jnp.float32(0.0)
    )
    assert float(fwd) < 0.0
    phi = facing_frame_phi_rad(fwd, lat)
    assert abs(float(phi)) == pytest.approx(jnp.pi, abs=1e-4)


def test_mapper_follows_sprite_facing_not_target():
    """FPS mapper must ignore target side (unlike fixed bilateral)."""
    _configure_facing()
    mapper, _ = make_facing_relative_mappers("seaquest")

    class _Player:
        x = jnp.array([50.0])
        y = jnp.array([80.0])
        width = jnp.array([8.0])
        height = jnp.array([8.0])
        orientation = jnp.array([180.0])  # sprite faces left

    class _Raw:
        player = _Player()

    class _Objs:
        x = jnp.array([[90.0]])  # target to the right
        y = jnp.array([[80.0]])
        width = jnp.array([[8.0]])
        height = jnp.array([[8.0]])

    mapped = int(mapper(jnp.int32(2), _Raw(), jnp.int32(0), _Objs()))  # FORWARD
    # Facing left → FORWARD maps to LEFT game action, even though target is right.
    assert mapped == int(Atari.LEFT)
    _teardown()


def test_polar_channel_count_psibins8():
    _configure_facing(EGO_PSI_ANGLE_BINS=8, EGO_POSE_XY=False)
    assert using_pov_facing_frame()
    assert not using_ego_pose_xy()
    assert ego_distance_channels_per_frame() == 1
    assert ego_angle_channels_per_frame() == 8
    with_bins = ego_obs_channels_per_frame()
    configure_ego_obs_from_config({**FACING_BASE_CFG, "EGO_PSI_ANGLE_BINS": None})
    with_sincos = ego_obs_channels_per_frame()
    assert with_bins == with_sincos + 6
    _teardown()


def test_xy_channel_count_no_angle():
    _configure_facing(EGO_POSE_XY=True, EGO_PSI_ANGLE_BINS=None)
    assert using_ego_pose_xy()
    assert ego_distance_channels_per_frame() == 2
    assert ego_angle_channels_per_frame() == 0
    # dist(2) + los(1) = 3 without weapon
    assert ego_obs_channels_per_frame() == 3
    _teardown()
