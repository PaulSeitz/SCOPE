"""Tests for the persistent-heading turn primitives (USE_POV_TURN_PRIMITIVES)."""

import jax.numpy as jnp
import pytest
from jaxatari.environment import JAXAtariAction as Atari

from concept_based_learning.environments.heuristics import initial_policy_state
from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
)
from concept_based_learning.observations.pov_turn import (
    TURN_EGO_ACTION_DIM,
    TURN_FIRE,
    TURN_FORWARD,
    TURN_FORWARD_FIRE,
    TURN_LEFT,
    TURN_NOOP,
    TURN_RIGHT,
    configure_pov_turn_from_config,
    heading_after,
    initial_heading_deg,
    make_turn_legal_mask_fn,
    make_turn_primitive_mappers,
    turn_step_deg,
    using_pov_turn_primitives,
    wrap_deg,
)


SEAQUEST_TURN_CFG = {
    "USE_POV_RELATIVE_MAPPER": True,
    "USE_POV_TURN_PRIMITIVES": True,
    "ENV_NAME": "seaquest",
}


@pytest.fixture(autouse=True)
def _reset_turn_globals():
    yield
    configure_ego_obs_from_config({})
    configure_pov_turn_from_config({})


def test_flag_requires_pov_relative_mapper():
    with pytest.raises(ValueError, match="requires USE_POV_RELATIVE_MAPPER"):
        configure_pov_turn_from_config({"USE_POV_TURN_PRIMITIVES": True})


def test_flag_rejects_relabel_all_objects():
    with pytest.raises(ValueError, match="RELABEL_ALL_OBJECTS"):
        configure_pov_turn_from_config(
            {**SEAQUEST_TURN_CFG, "RELABEL_ALL_OBJECTS": True}
        )


def test_seaquest_initial_heading_is_profile_bin0_facing():
    configure_pov_turn_from_config(SEAQUEST_TURN_CFG)
    assert using_pov_turn_primitives()
    assert initial_heading_deg() == pytest.approx(0.0)
    assert turn_step_deg() == pytest.approx(45.0)


def test_pong_initial_heading_points_into_field():
    """Pong player is on the right; facing 180° looks left into the court."""
    configure_pov_turn_from_config({**SEAQUEST_TURN_CFG, "ENV_NAME": "pong"})
    assert initial_heading_deg() == pytest.approx(180.0)


def test_policy_state_seeds_heading_from_config():
    configure_pov_turn_from_config({**SEAQUEST_TURN_CFG, "ENV_NAME": "pong"})
    state = initial_policy_state(-1)
    assert float(state.heading_deg) == pytest.approx(180.0)


def test_turns_integrate_over_time_and_wrap():
    configure_pov_turn_from_config(SEAQUEST_TURN_CFG)
    heading = jnp.float32(0.0)
    for _ in range(3):
        heading = heading_after(heading, jnp.int32(TURN_RIGHT))
    assert float(heading) == pytest.approx(135.0)
    # Crossing the seam stays in [-180, 180): 135 + 45 wraps to -180, not 180.
    heading = heading_after(heading, jnp.int32(TURN_RIGHT))
    assert float(heading) == pytest.approx(-180.0)
    heading = heading_after(heading, jnp.int32(TURN_RIGHT))
    assert float(wrap_deg(heading)) == pytest.approx(-135.0)
    # Eight right turns from 0° return to 0°.
    heading = jnp.float32(0.0)
    for _ in range(8):
        heading = heading_after(heading, jnp.int32(TURN_RIGHT))
    assert float(heading) == pytest.approx(0.0)


def test_non_turn_primitives_leave_heading_untouched():
    configure_pov_turn_from_config(SEAQUEST_TURN_CFG)
    for ego in (TURN_NOOP, TURN_FIRE, TURN_FORWARD, TURN_FORWARD_FIRE):
        assert float(heading_after(jnp.float32(30.0), jnp.int32(ego))) == pytest.approx(
            30.0
        )


def test_turn_left_and_right_are_symmetric():
    configure_pov_turn_from_config(SEAQUEST_TURN_CFG)
    left = float(heading_after(jnp.float32(0.0), jnp.int32(TURN_LEFT)))
    right = float(heading_after(jnp.float32(0.0), jnp.int32(TURN_RIGHT)))
    assert left == pytest.approx(-right)
    assert right > 0.0  # screen-Y-down: TURN_RIGHT is clockwise


def test_configurable_turn_step():
    configure_pov_turn_from_config({**SEAQUEST_TURN_CFG, "POV_TURN_STEP_DEG": 22.5})
    assert float(heading_after(jnp.float32(0.0), jnp.int32(TURN_RIGHT))) == pytest.approx(
        22.5
    )


def test_seaquest_mapper_moves_along_heading():
    configure_pov_turn_from_config(SEAQUEST_TURN_CFG)
    mapper, _ = make_turn_primitive_mappers("seaquest")
    cases = {
        0.0: Atari.RIGHT,
        90.0: Atari.DOWN,
        180.0: Atari.LEFT,
        -90.0: Atari.UP,
    }
    for heading, expected in cases.items():
        got = int(
            mapper(jnp.int32(TURN_FORWARD), jnp.float32(heading), None, None, None)
        )
        assert got == int(expected), f"heading={heading} got={got}"


def test_seaquest_mapper_stationary_and_fire_primitives():
    configure_pov_turn_from_config(SEAQUEST_TURN_CFG)
    mapper, _ = make_turn_primitive_mappers("seaquest")
    heading = jnp.float32(0.0)
    assert int(mapper(jnp.int32(TURN_NOOP), heading, None, None, None)) == int(
        Atari.NOOP
    )
    assert int(mapper(jnp.int32(TURN_FIRE), heading, None, None, None)) == int(
        Atari.FIRE
    )
    assert int(mapper(jnp.int32(TURN_FORWARD_FIRE), heading, None, None, None)) == int(
        Atari.RIGHTFIRE
    )


def test_turning_primitive_also_moves_along_the_new_heading():
    """TURN_* is not a free frame: it rotates and commits to the new direction."""
    configure_pov_turn_from_config(SEAQUEST_TURN_CFG)
    mapper, _ = make_turn_primitive_mappers("seaquest")
    heading = jnp.float32(0.0)  # facing right
    new_heading = heading_after(heading, jnp.int32(TURN_RIGHT))
    assert float(new_heading) == pytest.approx(45.0)
    # 45° in screen-Y-down is down-right.
    assert int(mapper(jnp.int32(TURN_RIGHT), new_heading, None, None, None)) == int(
        Atari.DOWNRIGHT
    )


def test_mapper_ignores_target_and_object_geometry():
    """Target-agnosticism is the transfer property we must not lose."""
    configure_pov_turn_from_config(SEAQUEST_TURN_CFG)
    mapper, _ = make_turn_primitive_mappers("seaquest")
    heading = jnp.float32(0.0)
    baseline = int(mapper(jnp.int32(TURN_FORWARD), heading, None, None, None))
    for target in (-1, 0, 7, 34):
        got = int(
            mapper(
                jnp.int32(TURN_FORWARD),
                heading,
                "raw-obs-sentinel",
                jnp.int32(target),
                "objects-sentinel",
            )
        )
        assert got == baseline


def test_pong_legal_mask_rejects_into_field_heading():
    """Pong's move table maps into-field headings to NOOP; the mask must catch it."""
    configure_pov_turn_from_config({**SEAQUEST_TURN_CFG, "ENV_NAME": "pong"})
    mapper, _ = make_turn_primitive_mappers("pong")
    legal = make_turn_legal_mask_fn(mapper, noop_action=0, fire_action=1)

    # Default facing into the court (180°): FORWARD is NOOP (no horizontal control).
    mask_into = legal(jnp.float32(180.0))
    assert not bool(mask_into[TURN_FORWARD])
    assert bool(mask_into[TURN_NOOP])
    assert bool(mask_into[TURN_FIRE])

    # Heading up: FORWARD moves the paddle vertically, so it is legal.
    mask_up = legal(jnp.float32(-90.0))
    assert bool(mask_up[TURN_FORWARD])
    assert mask_into.shape == (TURN_EGO_ACTION_DIM,)

def test_heading_replaces_sprite_orientation_in_psi():
    """ψ must come from the commanded heading, not the sprite, or the loop is open."""
    from concept_based_learning.observations.egocentric import get_ego_obs

    configure_ego_obs_from_config(
        {
            **SEAQUEST_TURN_CFG,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": False,
            "USE_EGO_BLOCKING_FEATURES": False,
            "USE_EGO_WEAPON_FEATURES": False,
        }
    )
    raw, objects = _seaquest_stub()
    target = jnp.int32(0)

    # Sprite orientation is pinned at 0° in the stub; only the heading varies.
    obs_a = get_ego_obs(raw, objects, target, jnp.float32(0.0))
    obs_b = get_ego_obs(raw, objects, target, jnp.float32(90.0))
    assert not bool(jnp.allclose(obs_a, obs_b))


def test_missing_heading_raises_under_turn_primitives():
    from concept_based_learning.observations.egocentric import get_ego_obs

    configure_ego_obs_from_config(SEAQUEST_TURN_CFG)
    raw, objects = _seaquest_stub()
    with pytest.raises(ValueError, match="without heading_deg"):
        get_ego_obs(raw, objects, jnp.int32(0))


def test_turn_primitives_do_not_change_ego_channel_count():
    """The action space shrinks; the observation layout must be untouched."""
    base = {
        "USE_POV_RELATIVE_MAPPER": True,
        "ENV_NAME": "seaquest",
        "EGO_PSI_ANGLE_BINS": 8,
        "USE_EGO_VELOCITY_FEATURES": True,
    }
    configure_ego_obs_from_config(base)
    without_turn = ego_obs_channels_per_frame()
    configure_ego_obs_from_config({**base, "USE_POV_TURN_PRIMITIVES": True})
    assert ego_obs_channels_per_frame() == without_turn


def _seaquest_stub():
    stack = 4

    class _Player:
        x = jnp.full((stack,), 40.0)
        y = jnp.full((stack,), 100.0)
        width = jnp.full((stack,), 8.0)
        height = jnp.full((stack,), 8.0)
        orientation = jnp.zeros((stack,))

    class _Raw:
        player = _Player()

    class _Objects:
        x = jnp.full((stack, 1), 80.0)
        y = jnp.full((stack, 1), 100.0)
        width = jnp.full((stack, 1), 8.0)
        height = jnp.full((stack, 1), 8.0)
        active = jnp.ones((stack, 1))
        semantic_label = jnp.zeros((stack, 1), dtype=jnp.int32)

    return _Raw(), _Objects()
