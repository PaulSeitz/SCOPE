"""The observation frame and the action-mapper frame must never drift apart.

A skill trained on POV/FPS observations learns an association between a target's
body-frame bearing and an ego action. That association is only transferable if
the angle used to *build* the observation is the same angle the mapper uses to
*resolve* the ego action into a game action. These tests pin that invariant, plus
the direction semantics spelled out for Seaquest (two view axes) and Pong (one).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from concept_based_learning.observations.action_space import (
    EGO_MOVE_DELTAS_DEG,
    _default_get_player,
)
from concept_based_learning.observations.pov_facing_frame import (
    facing_frame_phi_rad,
    resolve_facing_deg,
    world_delta_to_facing_frame,
)
from concept_based_learning.observations.pov_relative import (
    _MOVE_SPECS,
    _POV_PROFILES,
    _SPRITE_ORIENTATION_OFFSET_DEG,
    build_facing_action_table,
    has_single_view_axis,
    view_axes_deg,
)

POV_ENVS = sorted(e for e in _MOVE_SPECS if e in _POV_PROFILES)

# JAXAtari sprite orientation values (0 deg = up, 90 = right, 270 = left).
SPRITE_RIGHT = 90.0
SPRITE_LEFT = 270.0
SPRITE_UNSET = 0.0


def _raw_obs(env: str, sprite_ori: float):
    """Minimal stacked raw observation carrying just a player orientation."""

    class _Player:
        orientation = jnp.array([sprite_ori], dtype=jnp.float32)

    obs = type("_Raw", (), {})()
    for attr in ("player", "chicken", "skier"):
        setattr(obs, attr, _Player())
    return obs


def _wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _ego_for_bearing(phi_deg: float) -> int:
    deltas = np.asarray(EGO_MOVE_DELTAS_DEG[2:10], dtype=np.float64)
    return 2 + int(np.argmin([abs(_wrap180(d - phi_deg)) for d in deltas]))


def _body_pose(dx: float, dy: float, facing_deg: float) -> tuple[float, float, float]:
    fwd, lat = world_delta_to_facing_frame(
        jnp.float32(dx), jnp.float32(dy), jnp.float32(facing_deg)
    )
    return float(fwd), float(lat), math.degrees(float(facing_frame_phi_rad(fwd, lat)))


@pytest.mark.parametrize("env", POV_ENVS)
@pytest.mark.parametrize("sprite_ori", [SPRITE_UNSET, SPRITE_RIGHT, SPRITE_LEFT])
def test_obs_facing_matches_a_declared_view_axis(env, sprite_ori):
    facing = float(resolve_facing_deg(_raw_obs(env, sprite_ori), env_name=env))
    axis0, axis1 = view_axes_deg(_POV_PROFILES[env])
    assert any(
        abs(_wrap180(facing - axis)) < 1e-3 for axis in (axis0, axis1)
    ), f"{env}: obs facing {facing} is not one of the declared axes {(axis0, axis1)}"


@pytest.mark.parametrize("env", POV_ENVS)
@pytest.mark.parametrize("sprite_ori", [SPRITE_UNSET, SPRITE_RIGHT, SPRITE_LEFT])
def test_obs_facing_equals_mapper_facing(env, sprite_ori):
    """The headline invariant: same angle for the observation and the mapper table."""
    from concept_based_learning.observations.pov_relative import _facing_bin_idx

    profile = _POV_PROFILES[env]
    getter = profile.get("player_getter") or _default_get_player
    raw = _raw_obs(env, sprite_ori)

    obs_facing = float(resolve_facing_deg(raw, env_name=env))
    bin_idx = int(_facing_bin_idx(raw, profile, getter))
    mapper_facing = view_axes_deg(profile)[bin_idx]

    assert abs(_wrap180(obs_facing - mapper_facing)) < 1e-3, (
        f"{env}: observation built at {obs_facing} deg but mapper table keyed on "
        f"{mapper_facing} deg (sprite orientation {sprite_ori})"
    )


@pytest.mark.parametrize("env", POV_ENVS)
def test_single_axis_envs_ignore_sprite_orientation(env):
    if not has_single_view_axis(_POV_PROFILES[env]):
        pytest.skip(f"{env} declares two view axes")
    facings = {
        float(resolve_facing_deg(_raw_obs(env, ori), env_name=env))
        for ori in (SPRITE_UNSET, SPRITE_RIGHT, SPRITE_LEFT, 180.0)
    }
    assert len(facings) == 1, f"{env} is single-axis but facing varied: {facings}"


@pytest.mark.parametrize("env", POV_ENVS)
def test_two_axis_envs_follow_sprite_orientation(env):
    """Sprite 90 deg means facing right, 270 means left, in screen-angle terms."""
    if has_single_view_axis(_POV_PROFILES[env]):
        pytest.skip(f"{env} declares one view axis")
    right = float(resolve_facing_deg(_raw_obs(env, SPRITE_RIGHT), env_name=env))
    left = float(resolve_facing_deg(_raw_obs(env, SPRITE_LEFT), env_name=env))
    assert abs(_wrap180(right - (SPRITE_RIGHT + _SPRITE_ORIENTATION_OFFSET_DEG))) < 1e-3
    assert abs(_wrap180(left - (SPRITE_LEFT + _SPRITE_ORIENTATION_OFFSET_DEG))) < 1e-3
    assert abs(_wrap180(right - left)) > 90.0


@pytest.mark.parametrize("env", POV_ENVS)
@pytest.mark.parametrize("sprite_ori", [SPRITE_RIGHT, SPRITE_LEFT])
def test_mirrored_targets_give_mirrored_bearings(env, sprite_ori):
    """A target mirrored across the body's forward axis must mirror phi."""
    facing = float(resolve_facing_deg(_raw_obs(env, sprite_ori), env_name=env))
    frad = math.radians(facing)
    fwd_x, fwd_y = math.cos(frad), math.sin(frad)
    right_x, right_y = math.cos(frad + math.pi / 2), math.sin(frad + math.pi / 2)
    ahead, side = 40.0, 20.0

    _, lat_r, phi_r = _body_pose(
        fwd_x * ahead + right_x * side, fwd_y * ahead + right_y * side, facing
    )
    _, lat_l, phi_l = _body_pose(
        fwd_x * ahead - right_x * side, fwd_y * ahead - right_y * side, facing
    )
    assert lat_r > 0.0 > lat_l, f"{env}: +lat must point to the body's right"
    assert phi_r == pytest.approx(-phi_l, abs=1e-3)


def test_seaquest_two_axes_match_the_pov_model():
    """Facing right: below is my right -> DOWN; above is my left -> UP; ahead -> RIGHT."""
    from jaxatari.environment import JAXAtariAction as Atari

    noop, fire, move, mfire = _MOVE_SPECS["seaquest"]
    for sprite_ori, ahead_action, expected in (
        (SPRITE_RIGHT, Atari.RIGHT, {"below": Atari.DOWN, "above": Atari.UP}),
        (SPRITE_LEFT, Atari.LEFT, {"below": Atari.DOWN, "above": Atari.UP}),
    ):
        facing = float(resolve_facing_deg(_raw_obs("seaquest", sprite_ori), env_name="seaquest"))
        table = build_facing_action_table(
            noop_action=noop, fire_action=fire, move_table=move,
            move_fire_table=mfire, facing_deg=facing,
        )
        assert int(table[2]) == int(ahead_action)  # FORWARD points along the hull
        for where, dy in (("below", 40.0), ("above", -40.0)):
            _, _, phi = _body_pose(0.0, dy, facing)
            ego = _ego_for_bearing(phi)
            assert int(table[ego]) == int(expected[where]), (
                f"seaquest facing {facing}: target {where} -> ego {ego} -> "
                f"{int(table[ego])}, expected {int(expected[where])}"
            )


def test_pong_single_axis_maps_lateral_to_paddle_travel():
    """Ball above is my right -> paddle up; ball below is my left -> paddle down.

    JAXAtari Pong's ACTION_SET is [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
    and its ``_player_step`` treats ALE RIGHT as up, so emitted index 2 raises the
    paddle and index 3 lowers it.
    """
    facing = float(resolve_facing_deg(_raw_obs("pong", SPRITE_UNSET), env_name="pong"))
    assert facing == pytest.approx(180.0)

    noop, fire, move, mfire = _MOVE_SPECS["pong"]
    table = build_facing_action_table(
        noop_action=noop, fire_action=fire, move_table=move,
        move_fire_table=mfire, facing_deg=facing,
    )
    paddle_up, paddle_down = 2, 3

    _, lat_above, phi_above = _body_pose(0.0, -40.0, facing)
    _, lat_below, phi_below = _body_pose(0.0, 40.0, facing)
    assert lat_above > 0.0, "ball above the paddle is to the body's right"
    assert lat_below < 0.0, "ball below the paddle is to the body's left"
    assert int(table[_ego_for_bearing(phi_above)]) == paddle_up
    assert int(table[_ego_for_bearing(phi_below)]) == paddle_down
    # Looking into the court, FORWARD has no controllable axis.
    assert int(table[2]) == noop
    # Fire into the field is plain FIRE, not a lateral paddle nudge.
    assert int(table[10]) == fire
    assert int(table[11]) == 4  # RIGHTFIRE (paddle up + fire)
    assert int(table[12]) == 5  # LEFTFIRE (paddle down + fire)
    assert int(table[13]) == fire
