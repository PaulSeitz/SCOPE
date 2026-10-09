"""Tests for fixed bilateral POV (virtual sector selection + forward-hemisphere φ)."""

import jax.numpy as jnp
import pytest
from jaxtari.environment import JAXAtariAction as Atari

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
)
from concept_based_learning.observations.pov_fixed_bilateral import (
    configure_pov_fixed_bilateral_from_config,
    forward_hemisphere_angle_bins,
    forward_hemisphere_phi_rad,
    target_sector_idx,
    using_pov_fixed_bilateral,
)
from concept_based_learning.observations.pov_relative import (
    configure_pov_relative_from_config,
    make_facing_relative_mappers,
)


FIXED2_BASE_CFG = {
    "USE_POV_RELATIVE_MAPPER": True,
    "USE_POV_FIXED_BILATERAL": True,
    "USE_EGO_VELOCITY_FEATURES": False,
    "USE_EGO_BLOCKING_FEATURES": True,
    "USE_EGO_WEAPON_FEATURES": False,
    "ENV_NAME": "seaquest",
    "EGO_DISTANCE_CLIP_MAX_PX": 50,
}


def _configure_fixed2(**overrides):
    cfg = {**FIXED2_BASE_CFG, **overrides}
    configure_pov_relative_from_config(cfg)
    configure_pov_fixed_bilateral_from_config(cfg)
    configure_ego_obs_from_config(cfg)
    return cfg


def test_flag_requires_pov_mapper():
    with pytest.raises(ValueError, match="requires USE_POV_RELATIVE_MAPPER"):
        configure_pov_fixed_bilateral_from_config({"USE_POV_FIXED_BILATERAL": True})


def test_incompatible_with_velocity():
    with pytest.raises(ValueError, match="USE_EGO_VELOCITY_FEATURES"):
        configure_pov_fixed_bilateral_from_config(
            {**FIXED2_BASE_CFG, "USE_EGO_VELOCITY_FEATURES": True}
        )


def test_horizontal_sector_selection():
    # Target to the right → sector 0 (+x forward).
    assert int(target_sector_idx(jnp.float32(10.0), jnp.float32(0.0), axis="horizontal")) == 0
    # Target to the left → sector 1 (−x forward).
    assert int(target_sector_idx(jnp.float32(-10.0), jnp.float32(0.0), axis="horizontal")) == 1


def test_mirror_targets_share_phi():
    dx, dy = jnp.float32(30.0), jnp.float32(10.0)
    s0 = target_sector_idx(dx, dy, axis="horizontal")
    phi0 = forward_hemisphere_phi_rad(dx, dy, s0, axis="horizontal")
    s1 = target_sector_idx(-dx, dy, axis="horizontal")
    phi1 = forward_hemisphere_phi_rad(-dx, dy, s1, axis="horizontal")
    assert int(s0) == 0
    assert int(s1) == 1
    assert float(phi0) == pytest.approx(float(phi1), abs=1e-5)


def test_forward_hemisphere_phi_within_bounds():
    for dx, dy in [(20.0, 15.0), (20.0, -15.0), (-20.0, 5.0), (-20.0, -5.0)]:
        dx_j = jnp.float32(dx)
        dy_j = jnp.float32(dy)
        sector = target_sector_idx(dx_j, dy_j, axis="horizontal")
        phi = forward_hemisphere_phi_rad(dx_j, dy_j, sector, axis="horizontal")
        assert abs(float(phi)) <= jnp.pi / 2.0 + 1e-5


def test_mapper_uses_target_sector_not_sprite_facing():
    _configure_fixed2()
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
        x = jnp.array([[90.0]])  # target to the right of player
        y = jnp.array([[80.0]])
        width = jnp.array([[8.0]])
        height = jnp.array([[8.0]])

    raw = _Raw()
    objs = _Objs()
    ego_forward = jnp.int32(2)  # FORWARD
    mapped = int(mapper(ego_forward, raw, jnp.int32(0), objs))
    # Virtual sector 0 (+x forward) even though sprite faces left.
    assert mapped == int(Atari.RIGHT)
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_pov_fixed_bilateral_from_config({"USE_POV_FIXED_BILATERAL": False})
    configure_ego_obs_from_config({})


def test_fixed2_psibins8_channel_count():
    _configure_fixed2(EGO_PSI_ANGLE_BINS=8)
    assert using_pov_fixed_bilateral()
    with_bins = ego_obs_channels_per_frame()
    configure_ego_obs_from_config({**FIXED2_BASE_CFG, "EGO_PSI_ANGLE_BINS": None})
    with_sincos = ego_obs_channels_per_frame()
    assert with_bins == with_sincos + 6
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_pov_fixed_bilateral_from_config({"USE_POV_FIXED_BILATERAL": False})
    configure_ego_obs_from_config({})


def test_vertical_axis_override():
    configure_pov_relative_from_config(
        {**FIXED2_BASE_CFG, "POV_BILATERAL_AXIS": "vertical"}
    )
    configure_pov_fixed_bilateral_from_config(
        {**FIXED2_BASE_CFG, "POV_BILATERAL_AXIS": "vertical"}
    )
    from concept_based_learning.observations.pov_fixed_bilateral import active_bilateral_axis

    assert active_bilateral_axis() == "vertical"
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_pov_fixed_bilateral_from_config({"USE_POV_FIXED_BILATERAL": False})


def test_vertical_mapper_forward_is_up():
    configure_pov_relative_from_config(
        {**FIXED2_BASE_CFG, "POV_BILATERAL_AXIS": "vertical"}
    )
    configure_pov_fixed_bilateral_from_config(
        {**FIXED2_BASE_CFG, "POV_BILATERAL_AXIS": "vertical"}
    )
    mapper, _ = make_facing_relative_mappers("seaquest")

    class _Player:
        x = jnp.array([50.0])
        y = jnp.array([80.0])
        width = jnp.array([8.0])
        height = jnp.array([8.0])
        orientation = jnp.array([0.0])

    class _Raw:
        player = _Player()

    class _Objs:
        x = jnp.array([[50.0]])
        y = jnp.array([[40.0]])  # target above player
        width = jnp.array([[8.0]])
        height = jnp.array([[8.0]])

    mapped = int(mapper(jnp.int32(2), _Raw(), jnp.int32(0), _Objs()))
    assert mapped == int(Atari.UP)
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_pov_fixed_bilateral_from_config({"USE_POV_FIXED_BILATERAL": False})
