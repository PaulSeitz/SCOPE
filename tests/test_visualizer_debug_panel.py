"""Tests for live visualizer debug panel helpers."""

import numpy as np

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
)
from concept_based_learning.visualizer_debug_panel import (
    DETAIL_PANEL_MIN_HEIGHT,
    PANEL_WIDTH,
    LiveVizSnapshot,
    compose_display,
    ego_channel_labels,
    estimate_detail_panel_height,
    render_detail_panel,
)


def test_ego_channel_labels_pov_psibins8():
    configure_ego_obs_from_config(
        {
            "USE_POV_RELATIVE_MAPPER": True,
            "EGO_PSI_ANGLE_BINS": 8,
            "USE_EGO_VELOCITY_FEATURES": True,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_BLOCKING_FEATURES": False,
        }
    )
    labels = ego_channel_labels()
    assert labels[0] == "dist"
    assert labels[1:9] == [f"psi{i}" for i in range(8)]
    assert labels[9] == "v_approach"
    assert len(labels) == ego_obs_channels_per_frame()


def test_render_detail_panel_shape():
    configure_ego_obs_from_config({"USE_POV_RELATIVE_MAPPER": True, "EGO_PSI_ANGLE_BINS": 8})
    channels = ego_obs_channels_per_frame()
    snap = LiveVizSnapshot(
        frame_rgb=np.zeros((210, 160, 3), dtype=np.uint8),
        overlay_text="test",
        target_obj_idx=1,
        no_target_idx=0,
        global_action=3,
        active_skill_idx=0,
        chosen_ego=2,
        distance=42.0,
        angle_rad=0.5,
        ego_obs=np.linspace(0, 1, channels * 4, dtype=np.float32),
        q_vals_active=np.array([0.1, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        skill_name="DESTROY",
        ego_name="FORWARD",
        heading_deg=None,
        facing_deg=0.0,
        player_xy=(80.0, 100.0),
        target_xy=(120.0, 90.0),
        frame_idx=3,
    )
    panel = render_detail_panel(snap, ego_action_names={2: "FORWARD"}, stack_size=4)
    needed = estimate_detail_panel_height(snap, stack_size=4, ego_action_names={2: "FORWARD"})
    assert panel.shape == (needed, PANEL_WIDTH, 3)
    assert needed >= DETAIL_PANEL_MIN_HEIGHT
    assert panel.dtype == np.uint8


def test_compose_display_with_panel():
    game = np.full((210, 160, 3), 128, dtype=np.uint8)
    panel = np.full((260, PANEL_WIDTH, 3), 64, dtype=np.uint8)
    canvas = compose_display(game, scale=2, panel_rgb=panel, overlay="hello")
    assert canvas.shape == (420, 160 * 2 + PANEL_WIDTH, 3)

    tall_panel = np.full((500, PANEL_WIDTH, 3), 64, dtype=np.uint8)
    canvas_tall = compose_display(game, scale=2, panel_rgb=tall_panel, overlay=None)
    assert canvas_tall.shape == (500, 160 * 2 + PANEL_WIDTH, 3)


def test_apply_pov_overlay_non_contiguous_frame():
    from concept_based_learning.visualizer_debug_panel import apply_pov_overlay

    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    frame = np.transpose(frame, (1, 0, 2))  # non-standard layout from CHW round-trip
    frame = np.transpose(frame, (1, 0, 2))
    snap = LiveVizSnapshot(
        frame_rgb=frame,
        overlay_text="",
        target_obj_idx=1,
        no_target_idx=0,
        global_action=0,
        active_skill_idx=0,
        chosen_ego=0,
        distance=10.0,
        angle_rad=0.0,
        ego_obs=None,
        q_vals_active=None,
        skill_name="X",
        ego_name="NOOP",
        heading_deg=None,
        facing_deg=0.0,
        player_xy=(80.0, 100.0),
        target_xy=(120.0, 90.0),
        frame_idx=0,
    )
    out = apply_pov_overlay(frame, snap)
    assert out.shape == frame.shape
    assert out.flags["C_CONTIGUOUS"]
