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


def test_centres_geom_is_euclidean():
    set_ego_aabb_geom("centres")
    from concept_based_learning.observations.egocentric import aabb_geom_distance_px

    # centres  (5,5) and (15,5) → 10; surface gap would be smaller
    d = float(
        aabb_geom_distance_px(0, 0, 10, 10, 10, 0, 10, 10, y_is_bottom_edge=False)
    )
    assert d == pytest.approx(10.0)
    set_ego_aabb_geom("centre_ray")

def test_body_gaps_forward_and_lateral():
    import math
    from concept_based_learning.observations.egocentric import aabb_body_frame_gaps_px

    set_ego_aabb_geom("body_gaps")
    # Facing +x. Player 16×16 centre (8,8). Target 8×10 centre (58,36) → sep (50,28).
    d_fwd, d_lat, dist, phi = aabb_body_frame_gaps_px(
        0, 0, 16, 16,
        54, 31, 8, 10,
        facing_deg=0.0,
        y_is_bottom_edge=False,
    )
    assert float(d_fwd) == pytest.approx(38.0, abs=0.05)
    assert float(d_lat) == pytest.approx(15.0, abs=0.05)
    assert float(dist) == pytest.approx(math.hypot(38.0, 15.0), abs=0.1)
    assert float(phi) == pytest.approx(math.atan2(15.0, 38.0), abs=0.05)

    # Same centres, larger boxes → smaller contact dist
    _a, _b, dist2, _c = aabb_body_frame_gaps_px(
        -4, -4, 24, 24,
        50, 29, 16, 14,
        facing_deg=0.0,
        y_is_bottom_edge=False,
    )
    assert float(dist2) < float(dist)
    set_ego_aabb_geom("centre_ray")


def test_body_gaps_in_corridor_phi_near_zero():
    from concept_based_learning.observations.egocentric import aabb_body_frame_gaps_px

    # sep_fwd=40, sep_lat=4; halves_fwd=12, halves_lat=13 → gap_lat=-9 (overlap)
    _df, _dl, dist, phi = aabb_body_frame_gaps_px(
        0, 0, 16, 16,
        44, 7, 8, 10,
        facing_deg=0.0,
        y_is_bottom_edge=False,
    )
    assert float(dist) == pytest.approx(28.0, abs=0.2)
    assert abs(float(phi)) < 0.15


def test_body_gaps_geom_distance_matches():
    from concept_based_learning.observations.egocentric import (
        aabb_body_frame_gaps_px,
        aabb_geom_distance_px,
    )

    set_ego_aabb_geom("body_gaps")
    _df, _dl, dist, _phi = aabb_body_frame_gaps_px(
        0, 0, 16, 16, 54, 31, 8, 10, 0.0, y_is_bottom_edge=False
    )
    d2 = float(
        aabb_geom_distance_px(
            0, 0, 16, 16, 54, 31, 8, 10, y_is_bottom_edge=False, facing_deg=0.0
        )
    )
    assert d2 == pytest.approx(float(dist), abs=1e-5)
    set_ego_aabb_geom("centre_ray")
