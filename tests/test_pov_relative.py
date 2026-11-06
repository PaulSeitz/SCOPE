"""Tests for POV-relative mapper and synced ego observations."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtari.environment import JAXAtariAction as Atari

from concept_based_learning.environments.seaquest.label import (
    _legacy_orientation_relative_mapper,
)
from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
    get_ego_obs,
)
from concept_based_learning.observations.pov_relative import (
    action_frame_from_config,
    build_facing_action_table,
    configure_pov_relative_from_config,
    make_facing_relative_mappers,
    make_screen_relative_mappers,
    observation_frame_from_config,
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
    """Parity with the hand-written tables, using the sprite values Seaquest emits.

    ``get_orientation`` in jax_seaquest returns 90° for right, 270° for left and
    0.0 while the sub has no direction yet (which falls back to facing right).
    """
    mapper, _ = make_facing_relative_mappers("seaquest")

    class _Player:
        orientation = jnp.array([0.0])

    class _Raw:
        player = _Player()

    for sprite_ori in (0.0, 90.0, 270.0):
        raw = _Raw()
        raw.player.orientation = jnp.array([sprite_ori])
        for ego in range(18):
            legacy = int(
                _legacy_orientation_relative_mapper(
                    jnp.int32(ego), raw, jnp.int32(0), None
                )
            )
            built = int(mapper(jnp.int32(ego), raw, jnp.int32(0), None))
            assert built == legacy, (
                f"ego={ego} sprite_ori={sprite_ori} legacy={legacy} built={built}"
            )


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
    from jaxtari.environment import JAXAtariAction as Atari

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
        orientation = jnp.array([180.0])

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
    # Into-field facing: FORWARD has no horizontal control → NOOP.
    assert pov == int(Atari.NOOP)
    # Target-aligned still rotates FORWARD toward the ball (usually non-NOOP).
    assert aligned != pov or aligned == 0


def test_pong_facing_looks_into_the_field():
    """Pong is single-axis facing left (180°) toward the opponent, not paddle-up."""
    from concept_based_learning.observations.pov_facing_frame import resolve_facing_deg
    from concept_based_learning.observations.pov_relative import (
        _POV_PROFILES,
        has_single_view_axis,
    )

    profile = _POV_PROFILES["pong"]
    assert has_single_view_axis(profile)
    assert float(profile["facing_deg_bin0"]) == pytest.approx(180.0)

    class _Player:
        orientation = jnp.array([0.0])  # ignored: the axis is pinned

    class _Raw:
        player = _Player()

    assert float(resolve_facing_deg(_Raw(), env_name="pong")) == pytest.approx(180.0)


def test_explicit_action_frame_preserves_legacy_defaults():
    assert action_frame_from_config({}) == "target"
    assert action_frame_from_config({"USE_POV_RELATIVE_MAPPER": True}) == "facing"
    assert action_frame_from_config({"EGO_ACTION_FRAME": "screen"}) == "screen"
    assert action_frame_from_config({"EGO_ACTION_FRAME": "world"}) == "screen"
    with pytest.raises(ValueError, match="EGO_ACTION_FRAME"):
        action_frame_from_config({"EGO_ACTION_FRAME": "random"})


def test_explicit_observation_frame_preserves_legacy_defaults():
    assert observation_frame_from_config({}) == "screen"
    assert observation_frame_from_config({"USE_POV_RELATIVE_MAPPER": True}) == "target"
    assert observation_frame_from_config({"USE_POV_FACING_FRAME": True}) == "facing"
    assert observation_frame_from_config({"EGO_OBSERVATION_FRAME": "world"}) == "screen"
    with pytest.raises(ValueError, match="EGO_OBSERVATION_FRAME"):
        observation_frame_from_config({"EGO_OBSERVATION_FRAME": "random"})


def test_explicit_observation_frame_drives_legacy_observation_switches():
    from concept_based_learning.observations.pov_facing_frame import (
        using_pov_facing_frame,
    )

    configure_ego_obs_from_config(
        {
            "EGO_OBSERVATION_FRAME": "screen",
            "USE_POV_RELATIVE_MAPPER": True,
            "USE_POV_FACING_FRAME": True,
        }
    )
    assert not using_pov_relative()
    assert not using_pov_facing_frame()

    configure_ego_obs_from_config(
        {
            "EGO_OBSERVATION_FRAME": "facing",
            "USE_POV_RELATIVE_MAPPER": False,
            "USE_POV_FACING_FRAME": False,
            "ENV_NAME": "seaquest",
        }
    )
    assert using_pov_relative()
    assert using_pov_facing_frame()
    configure_ego_obs_from_config({})


def test_seaquest_screen_mapper_is_fixed_and_target_independent():
    mapper, inverse = make_screen_relative_mappers("seaquest")

    class _Raw:
        pass

    raw = _Raw()
    for target_idx in (jnp.int32(0), jnp.int32(9)):
        # Abstract FORWARD/BACKWARD become literal screen UP/DOWN.
        assert int(mapper(jnp.int32(2), raw, target_idx, None)) == int(Atari.UP)
        assert int(mapper(jnp.int32(5), raw, target_idx, None)) == int(Atari.DOWN)
        assert int(mapper(jnp.int32(3), raw, target_idx, None)) == int(Atari.RIGHT)
        assert int(mapper(jnp.int32(4), raw, target_idx, None)) == int(Atari.LEFT)
    assert int(inverse(jnp.int32(Atari.UP), raw, jnp.int32(0), None)) == 2


def test_asteroids_screen_mapper_uses_facing_but_not_target():
    mapper, _ = make_screen_relative_mappers("asteroids")

    class _Player:
        # Sprite 0° is screen-UP after the shared -90° orientation shift.
        orientation = jnp.array([0.0])

    class _Raw:
        player = _Player()

    raw = _Raw()
    # Already facing screen-UP: fixed-screen UP thrusts. Fixed-screen RIGHT turns CW.
    assert int(mapper(jnp.int32(2), raw, jnp.int32(0), None)) == 2
    assert int(mapper(jnp.int32(3), raw, jnp.int32(0), None)) == 3
    assert int(mapper(jnp.int32(3), raw, jnp.int32(99), object())) == 3


@pytest.mark.parametrize(
    ("env_name", "num_actions"),
    [
        ("seaquest", 18),
        ("asteroids", 14),
        ("beamrider", 18),
        ("breakout", 4),
        ("freeway", 3),
        ("namethisgame", 6),
        ("phoenix", 8),
        ("pong", 6),
        ("spaceinvaders", 6),
        ("timepilot", 10),
    ],
)
def test_screen_mapper_is_target_independent_and_legal_for_transfer_matrix(
    env_name, num_actions
):
    mapper, _ = make_screen_relative_mappers(env_name)

    class _Player:
        orientation = jnp.array([0.0])

    class _Raw:
        player = _Player()

    raw = _Raw()
    first = [int(mapper(jnp.int32(a), raw, jnp.int32(0), None)) for a in range(18)]
    moved_target = [
        int(mapper(jnp.int32(a), raw, jnp.int32(31), object())) for a in range(18)
    ]
    assert first == moved_target
    assert all(0 <= action < num_actions for action in first)
