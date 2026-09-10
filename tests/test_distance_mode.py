"""EGO_DISTANCE_MODE: xy / delta_xy / delta_xy_sign."""

import jax.numpy as jnp
import pytest

from concept_based_learning.observations.egocentric import (
    body_frame_delta_xy_features,
    configure_ego_obs_from_config,
    ego_distance_channels_per_frame,
    ego_distance_mode,
    set_ego_distance_mode,
)


def test_delta_xy_zero_pads_oldest_and_scales():
    fwd = jnp.array([10.0, 14.0, 14.0, 6.0], dtype=jnp.float32)
    lat = jnp.array([0.0, -4.0, 4.0, 4.0], dtype=jnp.float32)
    d_fwd, d_lat = body_frame_delta_xy_features(fwd, lat, sign_only=False)
    assert float(d_fwd[0]) == pytest.approx(0.0)
    assert float(d_lat[0]) == pytest.approx(0.0)
    # Δfwd = +4 → 4/32
    assert float(d_fwd[1]) == pytest.approx(4.0 / 32.0)
    assert float(d_lat[1]) == pytest.approx(-4.0 / 32.0)
    assert float(d_fwd[3]) == pytest.approx(-8.0 / 32.0)


def test_delta_xy_sign_deadzone():
    fwd = jnp.array([0.0, 0.2, 3.0, 2.6], dtype=jnp.float32)
    lat = jnp.array([0.0, -0.1, -2.0, -2.0], dtype=jnp.float32)
    s_fwd, s_lat = body_frame_delta_xy_features(
        fwd, lat, sign_only=True, sign_eps_px=0.5
    )
    assert [float(x) for x in s_fwd] == [0.0, 0.0, 1.0, 0.0]
    assert [float(x) for x in s_lat] == [0.0, 0.0, -1.0, 0.0]


def test_configure_distance_mode_xy():
    configure_ego_obs_from_config(
        {
            "USE_POV_RELATIVE_MAPPER": True,
            "USE_POV_FACING_FRAME": True,
            "EGO_DISTANCE_CLIP_MAX_PX": 50,
            "EGO_DISTANCE_MODE": "xy",
            "EGO_PSI_ANGLE_BINS": 8,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": False,
            "USE_EGO_BLOCKING_FEATURES": False,
            "USE_EGO_WEAPON_FEATURES": False,
        }
    )
    assert ego_distance_mode() == "xy"
    assert ego_distance_channels_per_frame() == 2


def test_configure_distance_mode_rejects_pose_xy_combo():
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_ego_obs_from_config(
            {
                "USE_POV_RELATIVE_MAPPER": True,
                "USE_POV_FACING_FRAME": True,
                "EGO_POSE_XY": True,
                "EGO_DISTANCE_MODE": "delta_xy",
                "USE_EGO_SIZE_FEATURES": False,
                "USE_EGO_VELOCITY_FEATURES": False,
                "USE_EGO_WEAPON_FEATURES": False,
            }
        )
    # Clear sticky facing-frame globals left by the failed configure.
    configure_ego_obs_from_config(
        {
            "USE_POV_RELATIVE_MAPPER": True,
            "USE_POV_FACING_FRAME": True,
            "EGO_POSE_XY": False,
            "EGO_DISTANCE_MODE": "scalar",
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": False,
            "USE_EGO_WEAPON_FEATURES": False,
        }
    )
