"""Fixed distance tiers and nested threshold flags."""

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
    set_ego_distance_bins(None)
    set_ego_distance_clip_max_pixels(None)
    set_ego_distance_flag_thresholds_px(None)
    set_ego_distance_hard_edges_px([12.0, 40.0])
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
