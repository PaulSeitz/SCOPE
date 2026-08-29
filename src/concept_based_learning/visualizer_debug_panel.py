"""Debug overlays and side panel for live PQN visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import jax.numpy as jnp

from concept_based_learning.observations.action_space import EGO_MOVE_DELTAS_DEG
from concept_based_learning.observations.egocentric import (
    ego_angle_channels_per_frame,
    ego_distance_channels_per_frame,
    ego_obs_channels_per_frame,
    ego_psi_angle_bins,
    skill_ego_extra_channels_per_frame,
    using_legacy_ego_obs,
)
from concept_based_learning.observations.pov_relative import (
    _POV_PROFILES,
    _player_orientation_deg,
    using_pov_relative,
)
from concept_based_learning.observations.pov_fixed_bilateral import (
    active_bilateral_axis,
    bilateral_facing_degrees,
    forward_hemisphere_phi_rad,
    target_sector_idx,
    using_pov_fixed_bilateral,
)
from concept_based_learning.observations.pov_facing_frame import (
    facing_frame_phi_rad,
    resolve_facing_deg,
    using_ego_pose_xy,
    using_pov_facing_frame,
    world_delta_to_facing_frame,
)

PANEL_WIDTH = 320


@dataclass
class LiveVizSnapshot:
    """One rendered live-viz frame plus policy debug tensors."""

    frame_rgb: np.ndarray
    overlay_text: str
    target_obj_idx: int
    no_target_idx: int
    global_action: int
    active_skill_idx: int
    chosen_ego: int
    distance: float
    angle_rad: float
    ego_obs: np.ndarray | None
    q_vals_active: np.ndarray | None
    skill_name: str
    ego_name: str
    heading_deg: float | None
    facing_deg: float | None
    player_xy: tuple[float, float]
    target_xy: tuple[float, float] | None
    frame_idx: int
    sector_idx: int | None = None
    phi_rad: float | None = None
    virtual_facing_deg: float | None = None
    bilateral_axis: str | None = None
    fwd_px: float | None = None
    lat_px: float | None = None


def ego_channel_labels() -> list[str]:
    """Human-readable labels for one frame of ego features (current runtime config)."""
    if using_legacy_ego_obs():
        labels = ["dx", "dy", "sin(o)", "cos(o)"]
        from concept_based_learning.observations.egocentric import (
            _USE_EGO_SIZE_FEATURES,
            _USE_EGO_VELOCITY_FEATURES,
            _USE_EGO_BLOCKING_FEATURES,
        )

        if _USE_EGO_VELOCITY_FEATURES:
            labels.extend(["dvx", "dvy"])
        if _USE_EGO_SIZE_FEATURES:
            labels.extend(["t_w", "t_h", "p_w", "p_h"])
        if _USE_EGO_BLOCKING_FEATURES:
            labels.append("blocked")
    else:
        dist_n = ego_distance_channels_per_frame()
        if using_pov_facing_frame() and using_ego_pose_xy() and dist_n == 2:
            labels = ["fwd", "lat"]
        elif dist_n == 1:
            labels = ["dist"]
        else:
            labels = [f"dist{i}" for i in range(dist_n)]

        ang_n = ego_angle_channels_per_frame()
        if ang_n > 0:
            if using_pov_facing_frame() and ego_psi_angle_bins():
                labels.extend([f"phi{i}" for i in range(ang_n)])
            elif using_pov_facing_frame():
                labels.extend(["sin(phi)", "cos(phi)"])
            elif using_pov_fixed_bilateral() and ego_psi_angle_bins():
                labels.extend([f"phi{i}" for i in range(ang_n)])
            elif using_pov_fixed_bilateral():
                labels.extend(["sin(phi)", "cos(phi)"])
            elif using_pov_relative() and ego_psi_angle_bins():
                labels.extend([f"psi{i}" for i in range(ang_n)])
            elif ang_n == 2:
                labels.extend(["sin(psi)", "cos(psi)"])
            else:
                labels.extend([f"theta{i}" for i in range(ang_n)])

        from concept_based_learning.observations.egocentric import (
            _USE_EGO_SIZE_FEATURES,
            _USE_EGO_VELOCITY_FEATURES,
            _USE_EGO_BLOCKING_FEATURES,
        )

        if _USE_EGO_VELOCITY_FEATURES:
            labels.append("v_approach" if using_pov_relative() else "v_rad")
            if not using_pov_relative():
                labels.append("v_tan")
        if _USE_EGO_SIZE_FEATURES:
            labels.extend(["t_w", "t_h"])
        if _USE_EGO_BLOCKING_FEATURES:
            labels.append("blocked")

    extra = skill_ego_extra_channels_per_frame()
    if extra:
        labels.extend([f"extra{i}" for i in range(extra)])
    return labels


def player_facing_deg(raw_obs, env_name: str) -> float:
    """Body-facing angle in screen-Y-down degrees."""
    game_key = str(env_name).lower()
    profile = _POV_PROFILES.get(game_key, {"facing_deg_bin0": 0.0, "facing_deg_bin1": 180.0})
    getter = profile.get("player_getter")
    if getter is None:
        from concept_based_learning.observations.action_space import _default_get_player

        getter = _default_get_player
    return float(_player_orientation_deg(raw_obs, profile, getter))


def _deg_to_screen_vec(deg: float, length: float) -> tuple[float, float]:
    """Screen coords: +x right, +y down."""
    rad = np.deg2rad(deg)
    return length * np.cos(rad), length * np.sin(rad)


def _blend_overlay(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    out = base.astype(np.float32)
    mask = np.any(overlay > 0, axis=-1)
    a = np.float32(alpha)
    out[mask] = out[mask] * (1.0 - a) + overlay[mask].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def virtual_mapper_facing_deg(env_name: str, sector_idx: int) -> float:
    """Mapper table facing for the active virtual bilateral sector."""
    profile = _POV_PROFILES.get(str(env_name).lower(), {})
    facing0, facing1 = bilateral_facing_degrees(profile, axis=active_bilateral_axis())
    return float(facing0 if int(sector_idx) == 0 else facing1)


def mapper_bin_facing_deg(raw_obs, env_name: str) -> float:
    """Body-table facing (0/180 or ±90) selected by sprite/synthetic orientation bin."""
    profile = _POV_PROFILES.get(
        str(env_name).lower(), {"facing_deg_bin0": 0.0, "facing_deg_bin1": 180.0}
    )
    getter = profile.get("player_getter")
    if getter is None:
        from concept_based_learning.observations.action_space import _default_get_player

        getter = _default_get_player
    ori = float(_player_orientation_deg(raw_obs, profile, getter))
    facing0 = float(profile.get("facing_deg_bin0", 0.0))
    facing1 = float(profile.get("facing_deg_bin1", 180.0))
    return facing1 if ori >= 180.0 else facing0


def compute_fixed_bilateral_viz(
    env_name: str,
    player_xy: tuple[float, float],
    target_xy: tuple[float, float] | None,
) -> tuple[int | None, float | None, float | None, str | None]:
    """Return (sector_idx, phi_rad, virtual_facing_deg, axis) for overlays."""
    if target_xy is None:
        return None, None, None, None
    px, py = player_xy
    tx, ty = target_xy
    dx = float(tx - px)
    dy = float(ty - py)
    axis = active_bilateral_axis()
    sector = int(target_sector_idx(jnp.float32(dx), jnp.float32(dy), axis=axis))
    phi = float(
        forward_hemisphere_phi_rad(
            jnp.float32(dx),
            jnp.float32(dy),
            jnp.int32(sector),
            axis=axis,
        )
    )
    facing = virtual_mapper_facing_deg(env_name, sector)
    return sector, phi, facing, axis


def compute_facing_frame_viz(
    raw_obs,
    env_name: str,
    player_xy: tuple[float, float],
    target_xy: tuple[float, float] | None,
) -> tuple[float, float | None, float | None, float | None]:
    """Return (facing_deg, phi_rad, fwd_px, lat_px) for FPS overlays.

    ``facing_deg`` is the mapper body-table angle (matches FORWARD), not raw
    sprite convention angles that can look like up/down on Seaquest.
    """
    facing = mapper_bin_facing_deg(raw_obs, env_name)
    if target_xy is None:
        return facing, None, None, None
    px, py = player_xy
    tx, ty = target_xy
    dx = float(tx - px)
    dy = float(ty - py)
    # Obs may use resolve_facing_deg (raw/synthetic); rotate with mapper facing so
    # the panel matches the FORWARD axis the action tables use.
    fwd, lat = world_delta_to_facing_frame(
        jnp.float32(dx), jnp.float32(dy), jnp.float32(facing)
    )
    phi = float(facing_frame_phi_rad(fwd, lat))
    return facing, phi, float(fwd), float(lat)


def apply_pov_overlay(
    frame_rgb: np.ndarray,
    snapshot: LiveVizSnapshot,
    *,
    show_fov_wedge: bool = True,
) -> np.ndarray:
    """Draw view direction, POV wedge, body facing, and chosen move on the game frame."""
    import cv2

    frame = np.ascontiguousarray(frame_rgb, dtype=np.uint8).copy()
    h, w = frame.shape[:2]
    px, py = snapshot.player_xy
    cx, cy = int(round(px)), int(round(py))
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    if snapshot.target_xy is not None and snapshot.target_obj_idx != snapshot.no_target_idx:
        tx, ty = snapshot.target_xy
        cv2.line(
            frame,
            (cx, cy),
            (int(round(tx)), int(round(ty))),
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        if show_fov_wedge and using_pov_relative() and not using_pov_fixed_bilateral():
            view_deg = float(np.rad2deg(snapshot.angle_rad))
            wedge = np.zeros_like(frame)
            cv2.ellipse(
                wedge,
                (cx, cy),
                (48, 48),
                0.0,
                view_deg - 90.0,
                view_deg + 90.0,
                (0, 160, 200),
                -1,
                cv2.LINE_AA,
            )
            frame = _blend_overlay(frame, wedge, 0.22)
            frame = np.ascontiguousarray(frame)
        elif show_fov_wedge and using_pov_fixed_bilateral():
            forward_deg = snapshot.virtual_facing_deg
            if forward_deg is not None:
                wedge = np.zeros_like(frame)
                cv2.ellipse(
                    wedge,
                    (cx, cy),
                    (48, 48),
                    0.0,
                    float(forward_deg) - 90.0,
                    float(forward_deg) + 90.0,
                    (0, 160, 200),
                    -1,
                    cv2.LINE_AA,
                )
                frame = _blend_overlay(frame, wedge, 0.22)
                frame = np.ascontiguousarray(frame)

    if using_pov_fixed_bilateral() and snapshot.virtual_facing_deg is not None:
        facing = snapshot.virtual_facing_deg
    else:
        facing = snapshot.heading_deg if snapshot.heading_deg is not None else snapshot.facing_deg
    if facing is not None:
        dx, dy = _deg_to_screen_vec(float(facing), 36.0)
        cv2.arrowedLine(
            frame,
            (cx, cy),
            (int(cx + dx), int(cy + dy)),
            (255, 210, 0),
            2,
            tipLength=0.35,
            line_type=cv2.LINE_AA,
        )

    if snapshot.chosen_ego >= 2 and facing is not None:
        move_delta = float(EGO_MOVE_DELTAS_DEG[snapshot.chosen_ego])
        move_deg = float(facing) + move_delta
        dx, dy = _deg_to_screen_vec(move_deg, 28.0)
        cv2.arrowedLine(
            frame,
            (cx, cy),
            (int(cx + dx), int(cy + dy)),
            (255, 80, 255),
            1,
            tipLength=0.3,
            line_type=cv2.LINE_AA,
        )

    return frame


def _draw_text(canvas, text: str, x: int, y: int, color=(220, 220, 220), scale=0.38):
    import cv2

    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_compass(canvas: np.ndarray, snapshot: LiveVizSnapshot, x0: int, y0: int, size: int) -> int:
    """Mini compass; returns y cursor after drawing."""
    import cv2

    cx, cy = x0 + size // 2, y0 + size // 2
    cv2.circle(canvas, (cx, cy), size // 2 - 2, (60, 60, 60), 1, cv2.LINE_AA)
    _draw_text(canvas, "compass", x0, y0 - 4, scale=0.34)

    if using_pov_fixed_bilateral() and snapshot.virtual_facing_deg is not None:
        forward_deg = float(snapshot.virtual_facing_deg)
        dx, dy = _deg_to_screen_vec(forward_deg, size * 0.34)
        cv2.line(
            canvas,
            (cx, cy),
            (int(cx + dx), int(cy + dy)),
            (0, 160, 200),
            1,
            cv2.LINE_AA,
        )
    elif snapshot.target_xy is not None:
        view_deg = float(np.rad2deg(snapshot.angle_rad))
        dx, dy = _deg_to_screen_vec(view_deg, size * 0.38)
        cv2.line(
            canvas,
            (cx, cy),
            (int(cx + dx), int(cy + dy)),
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    if using_pov_fixed_bilateral() and snapshot.virtual_facing_deg is not None:
        facing = snapshot.virtual_facing_deg
    else:
        facing = snapshot.heading_deg if snapshot.heading_deg is not None else snapshot.facing_deg
    if facing is not None:
        dx, dy = _deg_to_screen_vec(float(facing), size * 0.32)
        cv2.arrowedLine(
            canvas,
            (cx, cy),
            (int(cx + dx), int(cy + dy)),
            (255, 210, 0),
            1,
            tipLength=0.35,
            line_type=cv2.LINE_AA,
        )

    legend_y = y0 + size + 14
    if using_pov_fixed_bilateral():
        _draw_text(
            canvas,
            "cyan=target  wedge=sector forward  yellow=virt facing",
            x0,
            legend_y,
            scale=0.28,
        )
    else:
        _draw_text(
            canvas,
            "cyan=view  yellow=body  magenta=ego move",
            x0,
            legend_y,
            scale=0.28,
        )
    return legend_y + 10


def _draw_ego_heatmap(
    canvas: np.ndarray,
    ego_obs: np.ndarray,
    labels: list[str],
    x0: int,
    y0: int,
    width: int,
    max_height: int,
) -> int:
    """Stacked ego channels heatmap; returns y after block."""
    import cv2

    channels_per = ego_obs_channels_per_frame()
    if channels_per <= 0 or ego_obs.size == 0:
        _draw_text(canvas, "ego obs: n/a", x0, y0 + 12)
        return y0 + 24

    stack_size = max(1, int(ego_obs.size // channels_per))
    mat = ego_obs.reshape(stack_size, channels_per)
    row_h = max(6, min(14, max_height // max(stack_size, 1)))
    col_w = max(8, (width - 8) // max(channels_per, 1))
    block_h = row_h * stack_size + 18
    _draw_text(canvas, f"ego stack ({stack_size}x{channels_per})", x0, y0 + 10, scale=0.36)

    flat = mat.reshape(-1)
    vmin, vmax = float(np.min(flat)), float(np.max(flat))
    span = max(vmax - vmin, 1e-6)

    for r in range(stack_size):
        for c in range(channels_per):
            val = float(mat[r, c])
            norm = (val - vmin) / span
            color = (int(40 + 180 * norm), int(40 + 120 * (1 - norm)), int(80 + 80 * norm))
            x = x0 + c * col_w
            y = y0 + 16 + r * row_h
            cv2.rectangle(canvas, (x, y), (x + col_w - 1, y + row_h - 1), color, -1)

    label_y = y0 + 16 + stack_size * row_h + 4
    for c, lab in enumerate(labels[:channels_per]):
        short = lab if len(lab) <= 7 else lab[:6] + "."
        _draw_text(canvas, short, x0 + c * col_w, label_y, scale=0.28, color=(160, 160, 160))
    return label_y + 14


def _draw_q_bars(
    canvas: np.ndarray,
    q_vals: np.ndarray,
    chosen_ego: int,
    ego_action_names: dict[int, str],
    x0: int,
    y0: int,
    width: int,
    max_height: int,
) -> int:
    import cv2

    _draw_text(canvas, "Q (active skill)", x0, y0 + 10, scale=0.36)
    n = int(q_vals.shape[0])
    if n == 0:
        return y0 + 24

    finite = q_vals[np.isfinite(q_vals)]
    q_min = float(np.min(finite)) if finite.size else 0.0
    q_max = float(np.max(finite)) if finite.size else 1.0
    span = max(q_max - q_min, 1e-6)

    bar_h = max(8, min(12, (max_height - 20) // max(n, 1)))
    label_w = 72
    bar_w = width - label_w - 12
    y = y0 + 18
    for i in range(n):
        val = float(q_vals[i])
        if not np.isfinite(val):
            continue
        frac = (val - q_min) / span
        bx = x0 + label_w
        by = y + i * (bar_h + 2)
        color = (80, 220, 120) if i == chosen_ego else (100, 100, 180)
        cv2.rectangle(canvas, (bx, by), (bx + int(bar_w * frac), by + bar_h - 1), color, -1)
        name = ego_action_names.get(i, str(i))
        if len(name) > 10:
            name = name[:9] + "."
        _draw_text(canvas, name, x0, by + bar_h - 2, scale=0.30)
        _draw_text(canvas, f"{val:.2f}", bx + bar_w + 4, by + bar_h - 2, scale=0.28, color=(180, 180, 180))

    return y + n * (bar_h + 2) + 6


def blank_detail_panel(height: int) -> np.ndarray:
    """Placeholder panel reserved in the layout before detail mode is toggled on."""
    import cv2

    panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)
    _draw_text(panel, "detail panel", 8, height // 2 - 20, scale=0.38, color=(120, 120, 130))
    _draw_text(panel, "press M to show", 8, height // 2, scale=0.34, color=(100, 100, 110))
    _draw_text(panel, "ego / Q / POV", 8, height // 2 + 18, scale=0.34, color=(100, 100, 110))
    return panel


def render_detail_panel(
    snapshot: LiveVizSnapshot,
    *,
    height: int,
    ego_action_names: dict[int, str],
    stack_size: int,
) -> np.ndarray:
    """Render the right-hand debug panel as HxWx3 RGB uint8."""
    panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)

    x = 8
    y = 16
    _draw_text(panel, f"frame {snapshot.frame_idx}", x, y, scale=0.42)
    y += 18
    _draw_text(panel, f"target={snapshot.target_obj_idx} skill={snapshot.skill_name}", x, y, scale=0.34)
    y += 16
    _draw_text(
        panel,
        f"ego={snapshot.ego_name} game={snapshot.global_action}",
        x,
        y,
        scale=0.34,
    )
    y += 16

    if not np.isnan(snapshot.distance):
        if using_pov_fixed_bilateral() and snapshot.phi_rad is not None:
            sector_txt = (
                f" sector={snapshot.sector_idx}"
                if snapshot.sector_idx is not None
                else ""
            )
            axis_txt = (
                f" axis={snapshot.bilateral_axis}"
                if snapshot.bilateral_axis
                else ""
            )
            face_txt = (
                f" virt_face={snapshot.virtual_facing_deg:.0f}deg"
                if snapshot.virtual_facing_deg is not None
                else ""
            )
            _draw_text(
                panel,
                f"d={snapshot.distance:.1f}px phi={np.rad2deg(snapshot.phi_rad):.0f}deg"
                f"{sector_txt}{axis_txt}{face_txt}",
                x,
                y,
                scale=0.32,
            )
        else:
            psi_deg = np.nan
            facing = (
                snapshot.heading_deg
                if snapshot.heading_deg is not None
                else snapshot.facing_deg
            )
            if facing is not None:
                view_deg = float(np.rad2deg(snapshot.angle_rad))
                psi_deg = ((float(facing) - view_deg + 180.0) % 360.0) - 180.0
            psi_txt = f" psi={psi_deg:.1f}deg" if np.isfinite(psi_deg) else ""
            head_txt = (
                f" head={snapshot.heading_deg:.0f}deg"
                if snapshot.heading_deg is not None
                else ""
            )
            face_txt = (
                f" face={snapshot.facing_deg:.0f}deg"
                if snapshot.facing_deg is not None
                else ""
            )
            _draw_text(
                panel,
                f"d={snapshot.distance:.1f}px view={np.rad2deg(snapshot.angle_rad):.0f}deg"
                f"{psi_txt}{head_txt}{face_txt}",
                x,
                y,
                scale=0.32,
            )
        y += 18

    y = _draw_compass(panel, snapshot, x, y, size=88)
    y += 4

    labels = ego_channel_labels()
    if snapshot.ego_obs is not None:
        y = _draw_ego_heatmap(
            panel,
            np.asarray(snapshot.ego_obs, dtype=np.float32),
            labels,
            x,
            y,
            PANEL_WIDTH - 16,
            max(60, height // 4),
        )
    y += 6

    if snapshot.q_vals_active is not None:
        _draw_q_bars(
            panel,
            np.asarray(snapshot.q_vals_active, dtype=np.float32),
            snapshot.chosen_ego,
            ego_action_names,
            x,
            y,
            PANEL_WIDTH - 16,
            max(80, height - y - 8),
        )

    return panel


def compose_display(
    game_rgb: np.ndarray,
    *,
    scale: int,
    panel_rgb: np.ndarray,
    overlay: str | None,
) -> np.ndarray:
    """Scale game frame and append the debug panel slot horizontally."""
    import cv2

    game_rgb = np.ascontiguousarray(game_rgb, dtype=np.uint8)
    h, w = game_rgb.shape[:2]
    game_scaled = cv2.resize(
        game_rgb,
        (w * scale, h * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    panel_scaled = cv2.resize(
        np.ascontiguousarray(panel_rgb, dtype=np.uint8),
        (PANEL_WIDTH, h * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.concatenate([game_scaled, panel_scaled], axis=1)

    if overlay:
        cv2.putText(
            canvas,
            overlay[:120],
            (8, canvas.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
