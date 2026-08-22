"""Fixed distance tiers and nested threshold flags."""

import jax.numpy as jnp

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    distance_features_from_pixels,
    ego_distance_channels_per_frame,
    set_ego_distance_bins,
    set_ego_distance_bin_centers_px,
    set_ego_distance_clip_max_pixels,
    set_ego_distance_flag_thresholds_px,
    triangular_distance_bins,
)


def test_tier4_soft_bins_peak_near_center():
    set_ego_distance_flag_thresholds_px(None)
    set_ego_distance_clip_max_pixels(None)
    set_ego_distance_bins(4)
    set_ego_distance_bin_centers_px([15, 50, 120, 200])
    centers = jnp.array([15, 50, 120, 200], dtype=jnp.float32)
    feats = triangular_distance_bins(jnp.array([50.0]), centers)
    assert int(jnp.argmax(feats[0])) == 1


def test_flags3_nested_indicators():
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
