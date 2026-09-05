"""Hit-corridor + closest-points AABB geometry for ego pose."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from concept_based_learning.observations.egocentric import (
    aabb_closest_delta_px,
    aabb_surface_gap_px,
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
    hit_corridor_features,
    set_ego_aabb_geom,
    set_ego_hit_corridor,
)


def test_closest_delta_axis_aligned_and_overlap():
    # Player [0,10]x[0,10], target [20,30]x[0,10] → pure +x gap of 10.
    dx, dy, dist = aabb_closest_delta_px(
        0, 0, 10, 10, 20, 0, 10, 10, y_is_bottom_edge=False
    )
    assert float(dx) == pytest.approx(10.0)
    assert float(dy) == pytest.approx(0.0)
    assert float(dist) == pytest.approx(10.0)

    # Diagonal separation → corner vector.
    dx, dy, dist = aabb_closest_delta_px(
        0, 0, 10, 10, 20, 20, 10, 10, y_is_bottom_edge=False
    )
    assert float(dx) == pytest.approx(10.0)
    assert float(dy) == pytest.approx(10.0)
    assert float(dist) == pytest.approx((200.0) ** 0.5)

    # Overlap → zero.
    dx, dy, dist = aabb_closest_delta_px(
        0, 0, 10, 10, 5, 5, 10, 10, y_is_bottom_edge=False
    )
    assert float(dist) == pytest.approx(0.0)


def test_closest_points_differs_from_centre_ray_on_diagonal():
    # Tall player, small target diagonally offset: centre-ray gap ≠ closest gap.
    ray = float(
        aabb_surface_gap_px(0, 0, 8, 20, 30, 40, 4, 4, y_is_bottom_edge=False)
    )
    _dx, _dy, close = aabb_closest_delta_px(
        0, 0, 8, 20, 30, 40, 4, 4, y_is_bottom_edge=False
    )
    assert abs(ray - float(close)) > 1.0


def test_hit_corridor_scales_with_body_not_pixels():
    # Facing +x (0°): lateral is +y. half_sum = 8+2 = 10.
    lat_rel, inside = hit_corridor_features(
        lat_px=jnp.float32(4.0),
        facing_deg=jnp.float32(0.0),
        p_w=jnp.float32(4.0),
        p_h=jnp.float32(16.0),
        t_w=jnp.float32(2.0),
        t_h=jnp.float32(4.0),
        sat=4.0,
    )
    assert float(inside) == pytest.approx(0.6)  # 1 - 4/10
    assert float(lat_rel) == pytest.approx((4.0 / 10.0) / 4.0)

    # Same relative miss with 2× boxes → same features.
    lat_rel2, inside2 = hit_corridor_features(
        lat_px=jnp.float32(8.0),
        facing_deg=jnp.float32(0.0),
        p_w=jnp.float32(8.0),
        p_h=jnp.float32(32.0),
        t_w=jnp.float32(4.0),
        t_h=jnp.float32(8.0),
        sat=4.0,
    )
    assert float(lat_rel2) == pytest.approx(float(lat_rel))
    assert float(inside2) == pytest.approx(float(inside))


def test_hit_corridor_adds_two_channels():
    base = {
        "USE_POV_RELATIVE_MAPPER": True,
        "USE_POV_FACING_FRAME": True,
        "USE_LEGACY_EGO_OBS": False,
        "USE_EGO_SIZE_FEATURES": False,
        "USE_EGO_VELOCITY_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": False,
        "USE_EGO_WEAPON_FEATURES": False,
        "EGO_DISTANCE_CLIP_MAX_PX": 50,
        "EGO_PSI_ANGLE_BINS": 8,
        "USE_EGO_HIT_CORRIDOR": False,
        "EGO_AABB_GEOM": "centre_ray",
        "ENV_NAME": "seaquest",
    }
    configure_ego_obs_from_config(base)
    n0 = ego_obs_channels_per_frame()
    configure_ego_obs_from_config({**base, "USE_EGO_HIT_CORRIDOR": True})
    assert ego_obs_channels_per_frame() == n0 + 2
    set_ego_hit_corridor(False)
    set_ego_aabb_geom("centre_ray")
