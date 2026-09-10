"""Fixed distance tiers and nested threshold flags."""

import pytest
import jax.numpy as jnp

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    distance_features_from_pixels,
    ego_distance_channels_per_frame,
    exclusive_distance_band_onehots,
    set_ego_distance_bins,
    set_ego_distance_bin_centers_px,
    set_ego_distance_clip_max_pixels,
    set_ego_distance_flag_thresholds_px,
    set_ego_distance_hard_edges_px,
    set_ego_distance_mode,
    triangular_distance_bins,
)


def test_tier4_soft_bins_peak_near_center():
    set_ego_distance_hard_edges_px(None)
    set_ego_distance_flag_thresholds_px(None)
    set_ego_distance_clip_max_pixels(None)
    set_ego_distance_bins(4)
    set_ego_distance_bin_centers_px([15, 50, 120, 200])
    centers = jnp.array([15, 50, 120, 200], dtype=jnp.float32)
    feats = triangular_distance_bins(jnp.array([50.0]), centers)
    assert int(jnp.argmax(feats[0])) == 1


def test_hard_edges_exclusive_onehots():
    configure_ego_obs_from_config(
        {
            "USE_POV_RELATIVE_MAPPER": True,
            "USE_POV_FACING_FRAME": True,
            "EGO_POSE_XY": False,
            "EGO_DISTANCE_MODE": "scalar",
            "EGO_DISTANCE_HARD_EDGES_PX": [12.0, 40.0],
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": False,
            "USE_EGO_WEAPON_FEATURES": False,
        }
    )
    d = jnp.array([5.0, 12.0, 25.0, 40.0, 80.0])
    chans = distance_features_from_pixels(d)
    assert len(chans) == 3
    assert ego_distance_channels_per_frame() == 3
    stacked = jnp.stack(chans, axis=-1)
    # near / mid / far; edges are inclusive on the upper band (d >= edge).
    assert stacked[0].tolist() == [1.0, 0.0, 0.0]
    assert stacked[1].tolist() == [0.0, 1.0, 0.0]
    assert stacked[2].tolist() == [0.0, 1.0, 0.0]
    assert stacked[3].tolist() == [0.0, 0.0, 1.0]
    assert stacked[4].tolist() == [0.0, 0.0, 1.0]
    assert jnp.allclose(stacked.sum(axis=-1), 1.0)
    raw = exclusive_distance_band_onehots(
        jnp.array([25.0]), jnp.array([12.0, 40.0], dtype=jnp.float32)
    )
    assert raw[0].tolist() == [0.0, 1.0, 0.0]


def test_configure_hard_edges_clears_flags_and_clip():
    configure_ego_obs_from_config(
        {
            "EGO_DISTANCE_HARD_EDGES_PX": [12, 40],
            "EGO_DISTANCE_FLAG_THRESHOLDS_PX": [15, 50],
            "EGO_DISTANCE_BINS": 8,
            "EGO_DISTANCE_CLIP_MAX_PX": 50,
        }
    )
    assert ego_distance_channels_per_frame() == 3
    chans = distance_features_from_pixels(jnp.array([20.0]))
    assert [float(c[0]) for c in chans] == [0.0, 1.0, 0.0]


def test_flags3_nested_indicators():
    set_ego_distance_hard_edges_px(None)
    set_ego_distance_bins(None)
    set_ego_distance_clip_max_pixels(None)
    set_ego_distance_flag_thresholds_px([15, 50, 120])
    d = jnp.array([10.0, 40.0, 100.0])
    chans = distance_features_from_pixels(d)
    assert len(chans) == 3
    assert ego_distance_channels_per_frame() == 3
    stacked = jnp.stack(chans, axis=-1)
    assert stacked[0].tolist() == [1.0, 1.0, 1.0]
    assert stacked[1].tolist() == [0.0, 1.0, 1.0]
    assert stacked[2].tolist() == [0.0, 0.0, 1.0]


def test_configure_flags_clears_bins():
    configure_ego_obs_from_config(
        {
            "EGO_DISTANCE_FLAG_THRESHOLDS_PX": [15, 50],
            "EGO_DISTANCE_BINS": 4,
            "EGO_DISTANCE_CLIP_MAX_PX": 50,
        }
    )
    assert ego_distance_channels_per_frame() == 2
    chans = distance_features_from_pixels(jnp.array([20.0]))
    assert len(chans) == 2
    assert float(chans[0][0]) == 0.0
    assert float(chans[1][0]) == 1.0


def test_clip_cap_override_ignores_global():
    set_ego_distance_hard_edges_px(None)
    set_ego_distance_flag_thresholds_px(None)
    set_ego_distance_bins(None)
    set_ego_distance_clip_max_pixels(50.0)
    d = jnp.array([25.0, 100.0])
    # Global clip50: 25→0.5, 100→1.0
    base = distance_features_from_pixels(d)[0]
    assert float(base[0]) == pytest.approx(0.5)
    assert float(base[1]) == pytest.approx(1.0)
    # Override clip10: 25→1.0, 100→1.0
    ov = distance_features_from_pixels(d, clip_cap=10.0)[0]
    assert float(ov[0]) == pytest.approx(1.0)
    assert float(ov[1]) == pytest.approx(1.0)
    # Override clip80: 25→0.3125, 100→1.0
    ov80 = distance_features_from_pixels(d, clip_cap=80.0)[0]
    assert float(ov80[0]) == pytest.approx(25.0 / 80.0)
    assert float(ov80[1]) == pytest.approx(1.0)


def test_configure_clip_choices_keeps_fallback_clip():
    from concept_based_learning.observations.egocentric import (
        ego_distance_clip_choices_px,
        ego_distance_clip_max_pixels,
    )

    configure_ego_obs_from_config(
        {
            "EGO_DISTANCE_CLIP_MAX_PX": 50,
            "EGO_DISTANCE_CLIP_CHOICES_PX": [10, 20, 30, 50, 80],
        }
    )
    assert ego_distance_clip_max_pixels() == 50.0
    assert ego_distance_clip_choices_px() == (10.0, 20.0, 30.0, 50.0, 80.0)
    # Hard edges clear choices.
    configure_ego_obs_from_config(
        {
            "EGO_DISTANCE_HARD_EDGES_PX": [12, 40],
            "EGO_DISTANCE_CLIP_CHOICES_PX": [10, 50],
        }
    )
    assert ego_distance_clip_choices_px() is None
