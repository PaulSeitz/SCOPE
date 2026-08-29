"""Visualizer POV mode CLI overrides."""

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
)
from concept_based_learning.observations.pov_fixed_bilateral import (
    active_bilateral_axis,
    using_pov_fixed_bilateral,
)
from concept_based_learning.pqn_visualizer import apply_visualizer_config_overrides
from concept_based_learning.visualizer_debug_panel import compute_fixed_bilateral_viz


def test_apply_visualizer_fixed_bilateral_mode():
    cfg = apply_visualizer_config_overrides(
        {
            "ENV_NAME": "seaquest",
            "USE_POV_RELATIVE_MAPPER": False,
            "USE_EGO_BLOCKING_FEATURES": False,
            "USE_EGO_WEAPON_FEATURES": False,
        },
        pov_mode="fixed-bilateral",
        pov_bilateral_axis="horizontal",
        ego_phi_bins=8,
    )
    configure_ego_obs_from_config(cfg)
    assert cfg["USE_POV_FIXED_BILATERAL"] is True
    assert cfg["USE_EGO_VELOCITY_FEATURES"] is False
    assert using_pov_fixed_bilateral()
    assert active_bilateral_axis() == "horizontal"
    assert ego_obs_channels_per_frame() >= 9


def test_apply_visualizer_relative_mode_clears_fixed_bilateral():
    cfg = apply_visualizer_config_overrides(
        {"USE_POV_FIXED_BILATERAL": True, "USE_POV_RELATIVE_MAPPER": True},
        pov_mode="relative",
    )
    configure_ego_obs_from_config(cfg)
    assert cfg["USE_POV_FIXED_BILATERAL"] is False
    assert cfg.get("USE_POV_FACING_FRAME") is False
    assert not using_pov_fixed_bilateral()


def test_apply_visualizer_facing_mode():
    cfg = apply_visualizer_config_overrides(
        {
            "ENV_NAME": "seaquest",
            "USE_EGO_BLOCKING_FEATURES": True,
            "USE_EGO_WEAPON_FEATURES": False,
            "USE_EGO_SIZE_FEATURES": False,
            "EGO_POSE_XY": False,
        },
        pov_mode="facing",
        ego_phi_bins=8,
    )
    configure_ego_obs_from_config(cfg)
    from concept_based_learning.observations.pov_facing_frame import (
        using_pov_facing_frame,
    )

    assert cfg["USE_POV_FACING_FRAME"] is True
    assert cfg["USE_POV_FIXED_BILATERAL"] is False
    assert using_pov_facing_frame()
    assert ego_obs_channels_per_frame() >= 9


def test_compute_fixed_bilateral_viz_horizontal():
    configure_ego_obs_from_config(
        {
            "ENV_NAME": "seaquest",
            "USE_POV_RELATIVE_MAPPER": True,
            "USE_POV_FIXED_BILATERAL": True,
            "USE_EGO_VELOCITY_FEATURES": False,
        }
    )
    sector, phi, facing, axis = compute_fixed_bilateral_viz(
        "seaquest",
        player_xy=(80.0, 100.0),
        target_xy=(120.0, 90.0),
    )
    assert sector == 0
    assert axis == "horizontal"
    assert facing == 0.0
    assert phi is not None
