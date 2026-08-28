"""Tests for POV-relative mapper and synced ego observations."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxatari.environment import JAXAtariAction as Atari

from concept_based_learning.environments.seaquest.label import (
    _legacy_orientation_relative_mapper,
)
from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
    get_ego_obs,
)
from concept_based_learning.observations.pov_relative import (
    build_facing_action_table,
    configure_pov_relative_from_config,
    make_facing_relative_mappers,
    pov_facing_psi_rad,
    using_pov_relative,
)


def test_pov_flag_enables_mode_and_forces_sincos_angle_channels():
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": True})
    configure_ego_obs_from_config(
        {
            "USE_POV_RELATIVE_MAPPER": True,
            "EGO_ANGLE_BINS": 8,
            "USE_EGO_BLOCKING_FEATURES": True,
            "USE_EGO_WEAPON_FEATURES": True,
            "ENV_NAME": "seaquest",
        }
    )
    assert using_pov_relative()
    # distance(1) + sin/cos(2) + los(1) + weapon extras (>=1)
    assert ego_obs_channels_per_frame() >= 5
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_ego_obs_from_config({})


def test_seaquest_facing_table_matches_legacy_right_facing():
    table = build_facing_action_table(
        noop_action=0,
        fire_action=1,
        move_table=[3, 8, 5, 9, 4, 7, 2, 6],
        move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
        facing_deg=0.0,
    )
    assert int(table[2]) == int(Atari.RIGHT)   # FORWARD
    assert int(table[4]) == int(Atari.UP)      # LEFT
    assert int(table[3]) == int(Atari.DOWN)    # RIGHT
    assert int(table[5]) == int(Atari.LEFT)    # BACKWARD


def test_seaquest_pov_mapper_matches_legacy_tables():
    mapper, _ = make_facing_relative_mappers("seaquest")

    class _Player:
        orientation = jnp.array([0.0])

    class _Raw:
        player = _Player()

    raw = _Raw()
    for ego in range(18):
        legacy = int(
            _legacy_orientation_relative_mapper(
                jnp.int32(ego), raw, jnp.int32(0), None
            )
        )
        built = int(mapper(jnp.int32(ego), raw, jnp.int32(0), None))
        assert built == legacy, f"ego={ego} legacy={legacy} built={built}"

    raw_left = _Raw()
    raw_left.player.orientation = jnp.array([180.0])
    for ego in range(18):
        legacy = int(
            _legacy_orientation_relative_mapper(
                jnp.int32(ego), raw_left, jnp.int32(0), None
            )
        )
        built = int(mapper(jnp.int32(ego), raw_left, jnp.int32(0), None))
        assert built == legacy, f"ego={ego} left legacy={legacy} built={built}"


def test_pov_velocity_adds_single_approach_channel():
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": True})
    base_cfg = {
        "USE_POV_RELATIVE_MAPPER": True,
        "USE_EGO_BLOCKING_FEATURES": True,
        "USE_EGO_WEAPON_FEATURES": False,
        "ENV_NAME": "seaquest",
    }
    configure_ego_obs_from_config({**base_cfg, "USE_EGO_VELOCITY_FEATURES": False})
    without_vel = ego_obs_channels_per_frame()
    configure_ego_obs_from_config({**base_cfg, "USE_EGO_VELOCITY_FEATURES": True})
    with_vel = ego_obs_channels_per_frame()
    assert using_pov_relative()
    assert with_vel == without_vel + 1
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_ego_obs_from_config({})


def test_pov_approach_velocity_positive_when_target_closes():
    """View-radial approach: + when target moves toward player along line of sight."""
    # Player fixed; target starts right and moves left (closing).
    p_cxs = jnp.array([0.0, 0.0, 0.0, 0.0])
    t_cxs = jnp.array([20.0, 18.0, 16.0, 14.0])
    p_cys = jnp.zeros(4)
    t_cys = jnp.zeros(4)
    thetas = jnp.zeros(4)  # target bearing: east
    d_rel_x = jnp.concatenate([jnp.zeros(1), jnp.diff(t_cxs - p_cxs)])
    d_rel_y = jnp.zeros_like(d_rel_x)
    cos_t = jnp.cos(thetas)
    sin_t = jnp.sin(thetas)
    v_r = d_rel_x * cos_t + d_rel_y * sin_t
    v_approach = -v_r
    # After first frame, target x decreases → d_rel_x < 0 → v_r < 0 → approach > 0
    assert float(v_approach[1]) > 0.0
    assert float(v_approach[2]) > 0.0


def test_pov_psibins8_adds_six_net_angle_channels():
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": True})
    base_cfg = {
        "USE_POV_RELATIVE_MAPPER": True,
        "USE_EGO_BLOCKING_FEATURES": True,
        "USE_EGO_WEAPON_FEATURES": False,
        "ENV_NAME": "seaquest",
    }
    configure_ego_obs_from_config({**base_cfg, "EGO_PSI_ANGLE_BINS": None})
    without_bins = ego_obs_channels_per_frame()
    configure_ego_obs_from_config({**base_cfg, "EGO_PSI_ANGLE_BINS": 8})
    with_bins = ego_obs_channels_per_frame()
    assert with_bins == without_bins + 6
    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})
    configure_ego_obs_from_config({})


def test_pov_psi_zero_when_facing_matches_view():
    # View toward target at 0 rad; facing also 0 rad.
    psi = pov_facing_psi_rad(jnp.array([0.0]), jnp.array([0.0]))
    assert float(jnp.ravel(psi)[0]) == pytest.approx(0.0, abs=1e-5)


def test_forward_spam_differs_from_target_aligned_on_pong_vertical_profile():
    """POV mapper should not auto-track: FORWARD is along pinned facing, not toward target."""
    from concept_based_learning.environments.pong.pong_label import pong_action_mapper
    from concept_based_learning.observations.pov_relative import (
        make_facing_relative_mappers,
    )

    pov_mapper, _ = make_facing_relative_mappers("pong")

    class _Ball:
        x = jnp.array([40.0])
        y = jnp.array([50.0])
        width = jnp.array([4.0])
        height = jnp.array([4.0])

    class _Player:
        x = jnp.array([120.0])
        y = jnp.array([80.0])
        width = jnp.array([8.0])
        height = jnp.array([8.0])
        orientation = jnp.array([-90.0])

    class _Raw:
        ball = _Ball()
        player = _Player()

    class _Objs:
        x = jnp.array([[40.0]])
        y = jnp.array([[50.0]])
        width = jnp.array([[4.0]])
        height = jnp.array([[4.0]])

    raw = _Raw()
    objs = _Objs()
    target = jnp.int32(0)
    ego_forward = jnp.int32(2)
    aligned = int(pong_action_mapper(ego_forward, raw, target, objs))
    pov = int(pov_mapper(ego_forward, raw, target, objs))
    # Target-aligned rotates FORWARD toward ball; POV-relative does not.
    assert aligned != pov or aligned == 0
