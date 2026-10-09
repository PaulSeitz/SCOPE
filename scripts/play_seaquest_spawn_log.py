#!/usr/bin/env python3
"""
Human-play Seaquest on ALE and/or JAXAtari while logging diver spawn behaviour.

Controls
--------
  Arrows / WASD     move
  Space             fire (combine with arrows for FIRE+dir)
  R                 reset current backend
  Tab               switch ALE ↔ JAXAtari (disabled with --jax-only)
  O                 toggle oxygen cheat (force full O₂ every frame)
  P                 pause
  N                 single-step while paused
  [ / ]             slower / faster (FPS)
  Q / Esc           quit (flushes logs + optional ALE compare)

Logs under ``--out-dir`` (default ``tmp/seaquest_spawn_play/<timestamp>/``):
  events.jsonl / spawns.jsonl (divers) / enemies.jsonl / summary.txt / meta.json
  events also store per-lane enemy/diver x and frame Δx for speed compare
  vs_ale.json       written on quit if ``--compare-ale`` is set (or dual-run)

Examples
--------
  # Fresh dual run (ALE then Tab → JAX) with spawn + movement logging
  python scripts/play_seaquest_spawn_log.py --start ale

  # JAX-only vs an existing dual log that already has ALE (+ enemies + movement)
  python scripts/play_seaquest_spawn_log.py --jax-only \\
      --compare-ale tmp/seaquest_spawn_play/<dual_run>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pygame


# ---------------------------------------------------------------------------
# Shared action ids (ALE / JAXAtari ACTION_SET)
# ---------------------------------------------------------------------------
ACTION_NOOP = 0
ACTION_FIRE = 1
ACTION_UP = 2
ACTION_RIGHT = 3
ACTION_LEFT = 4
ACTION_DOWN = 5
ACTION_UPRIGHT = 6
ACTION_UPLEFT = 7
ACTION_DOWNRIGHT = 8
ACTION_DOWNLEFT = 9
ACTION_UPFIRE = 10
ACTION_RIGHTFIRE = 11
ACTION_LEFTFIRE = 12
ACTION_DOWNFIRE = 13
ACTION_UPRIGHTFIRE = 14
ACTION_UPLEFTFIRE = 15
ACTION_DOWNRIGHTFIRE = 16
ACTION_DOWNLEFTFIRE = 17

ALE_OXYGEN_RAM = 102
ALE_OXYGEN_FULL = 64
ALE_DIVER_X_RAM = (74, 73, 72, 71)

# Set only after a fresh dual run that includes enemies.jsonl (old logs lack it).
DEFAULT_ALE_BASELINE: Path | None = None


def _keys_to_action(keys) -> int:
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    if up and left:
        return ACTION_UPLEFTFIRE if fire else ACTION_UPLEFT
    if up and right:
        return ACTION_UPRIGHTFIRE if fire else ACTION_UPRIGHT
    if down and left:
        return ACTION_DOWNLEFTFIRE if fire else ACTION_DOWNLEFT
    if down and right:
        return ACTION_DOWNRIGHTFIRE if fire else ACTION_DOWNRIGHT
    if up:
        return ACTION_UPFIRE if fire else ACTION_UP
    if down:
        return ACTION_DOWNFIRE if fire else ACTION_DOWN
    if left:
        return ACTION_LEFTFIRE if fire else ACTION_LEFT
    if right:
        return ACTION_RIGHTFIRE if fire else ACTION_RIGHT
    if fire:
        return ACTION_FIRE
    return ACTION_NOOP


def _ale_is_missile_lane(lane_from_bottom: int, ram: np.ndarray) -> bool:
    i = 3 - lane_from_bottom
    return 3 < (int(ram[89 + i]) % 8) < 7


def _ale_snapshot(ram: np.ndarray) -> dict[str, Any]:
    divers = []
    for lane in range(4):
        x = int(ram[ALE_DIVER_X_RAM[lane]])
        missile = _ale_is_missile_lane(lane, ram)
        active_diver = (not missile) and (0 < x < 160)
        divers.append(
            {
                "lane": lane,
                "x": x,
                "active_diver": bool(active_diver),
                "is_missile_slot": bool(missile),
            }
        )
    return {
        "player_x": int(ram[70]),
        "player_y": int(ram[97]),
        "oxygen": int(ram[102]),
        "lives": int(ram[59]),
        # RAM $3D: increments on each surface (drives enemy/diver speed).
        "difficulty": int(ram[61]),
        "level": int(ram[61]),
        "divers_collected": int(ram[62]),
        "top_enemy_enabled": int(ram[60]),
        "enemy_x": [int(v) for v in ram[30:34]],
        "enemy_flags_89_93": [int(v) for v in ram[89:93]],
        "diver_missile_x_71_75": [int(v) for v in ram[71:75]],
        "surface_sub_x": int(ram[118]),
        "divers": divers,
        "diver_active": [bool(d["active_diver"]) for d in divers],
        # Lane enemies: RAM 30..33 holds x; treat on-screen if 0 < x < 160.
        "enemy_active": [bool(0 < int(ram[30 + i]) < 160) for i in range(4)],
        "enemy_x": [float(ram[30 + i]) if 0 < int(ram[30 + i]) < 160 else None for i in range(4)],
        "diver_x": [float(d["x"]) if d["active_diver"] else None for d in divers],
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# Max |Δx| treated as continuous motion of the same entity.
# Seaquest steps 0 or 1 px/frame; anything larger is a track break (new spawn /
# slot swap / representative-x hop), not speed.
MAX_CONTINUOUS_ABS_DX = 2


def _sane_dx(
    prev_x: float | None,
    cur_x: float | None,
    *,
    max_abs: float = MAX_CONTINUOUS_ABS_DX,
) -> float | None:
    if prev_x is None or cur_x is None:
        return None
    dx = float(cur_x) - float(prev_x)
    if abs(dx) > max_abs:
        return None
    return dx


def _lane_dx(
    prev_xs: list[float | None] | None,
    cur_xs: list[float | None] | None,
) -> list[float | None]:
    """Per-lane Δx with continuity filter (rejects track breaks)."""
    if prev_xs is None or cur_xs is None:
        return [None] * 4
    return [_sane_dx(p, c) for p, c in zip(prev_xs, cur_xs)]


def _jax_enemy_slots(shark: np.ndarray, sub: np.ndarray) -> list[list[dict[str, Any]]]:
    """4 lanes × 3 slots: continuous identities for movement tracking."""
    shark_l = shark.reshape(4, 3, 3)
    sub_l = sub.reshape(4, 3, 3)
    out: list[list[dict[str, Any]]] = []
    for lane in range(4):
        lane_slots: list[dict[str, Any]] = []
        for slot in range(3):
            if shark_l[lane, slot, 2] != 0:
                lane_slots.append(
                    {
                        "kind": "shark",
                        "x": float(shark_l[lane, slot, 0]),
                        "dir": float(shark_l[lane, slot, 2]),
                        "active": True,
                    }
                )
            elif sub_l[lane, slot, 2] != 0:
                lane_slots.append(
                    {
                        "kind": "sub",
                        "x": float(sub_l[lane, slot, 0]),
                        "dir": float(sub_l[lane, slot, 2]),
                        "active": True,
                    }
                )
            else:
                lane_slots.append(
                    {"kind": None, "x": None, "dir": None, "active": False}
                )
        out.append(lane_slots)
    return out


def _slot_motion_dxs(
    prev_slots: list[list[dict[str, Any]]] | None,
    cur_slots: list[list[dict[str, Any]]],
) -> tuple[list[list[float | None]], list[float], int]:
    """Per-slot Δx requiring same kind + sane |Δx|.

    Returns (slot_dx[lane][slot], flat continuous samples, n_rejected).
    """
    slot_dx: list[list[float | None]] = []
    samples: list[float] = []
    rejected = 0
    for lane in range(4):
        row: list[float | None] = []
        for slot in range(3):
            cur = cur_slots[lane][slot]
            if not cur.get("active"):
                row.append(None)
                continue
            if prev_slots is None:
                row.append(None)
                continue
            prev = prev_slots[lane][slot]
            if not prev.get("active") or prev.get("kind") != cur.get("kind"):
                # New entity in this slot — not a motion sample.
                row.append(None)
                continue
            raw = float(cur["x"]) - float(prev["x"])
            if abs(raw) > MAX_CONTINUOUS_ABS_DX:
                rejected += 1
                row.append(None)
                continue
            row.append(raw)
            samples.append(raw)
        slot_dx.append(row)
    return slot_dx, samples, rejected


def _busy_segment_lengths(active_series: list[list[bool]], n_lanes: int = 4) -> list[float]:
    """Contiguous True-run lengths across lanes (frames a lane stays occupied)."""
    lengths: list[float] = []
    for lane in range(n_lanes):
        run = 0
        for frame_active in active_series:
            on = bool(frame_active[lane]) if lane < len(frame_active) else False
            if on:
                run += 1
            elif run:
                lengths.append(float(run))
                run = 0
        if run:
            lengths.append(float(run))
    return lengths


def _movement_stats_from_events(ev: list[dict[str, Any]]) -> dict[str, Any]:
    """Speed / dwell from continuous per-entity Δx only (track breaks dropped).

    Prefer ``enemy_motion_dxs`` (slot-continuous samples). Fall back to filtered
    ``enemy_dx``. Global averages still mix difficulty stages — use by_difficulty.
    """
    enemy_abs_when_moving: list[float] = []
    enemy_abs_when_active: list[float] = []
    diver_abs_when_moving: list[float] = []
    diver_abs_when_active: list[float] = []
    enemy_motion_samples = 0
    enemy_moving_samples = 0
    diver_active_frames = 0
    diver_moving_frames = 0
    slot_counts: list[float] = []
    rejected_track_breaks = 0
    abs_dx_hist: dict[str, int] = {"0": 0, "1": 0, "other": 0}

    enemy_active_series: list[list[bool]] = []
    diver_active_series: list[list[bool]] = []
    by_diff: dict[int, dict[str, Any]] = {}

    def _bucket(diff: int | None) -> dict[str, Any]:
        key = int(diff) if diff is not None else -1
        if key not in by_diff:
            by_diff[key] = {
                "frames": 0,
                "enemy_abs_moving": [],
                "enemy_abs_active": [],
                "enemy_motion_samples": 0,
                "enemy_moving_samples": 0,
                "diver_abs_moving": [],
                "enemy_active_series": [],
                "slot_counts": [],
                "abs_dx_hist": {"0": 0, "1": 0, "other": 0},
            }
        return by_diff[key]

    def _consume_enemy_dx(dx: float, b: dict[str, Any]) -> None:
        nonlocal enemy_motion_samples, enemy_moving_samples
        adx = abs(float(dx))
        enemy_motion_samples += 1
        b["enemy_motion_samples"] += 1
        enemy_abs_when_active.append(adx)
        b["enemy_abs_active"].append(adx)
        if adx <= 0.5:
            abs_dx_hist["0"] += 1
            b["abs_dx_hist"]["0"] += 1
        elif adx <= 1.5:
            abs_dx_hist["1"] += 1
            b["abs_dx_hist"]["1"] += 1
        else:
            abs_dx_hist["other"] += 1
            b["abs_dx_hist"]["other"] += 1
        if adx > 0:
            enemy_moving_samples += 1
            b["enemy_moving_samples"] += 1
            enemy_abs_when_moving.append(adx)
            b["enemy_abs_moving"].append(adx)

    for e in ev:
        ea = e.get("enemy_active") or [False] * 4
        da = e.get("diver_active") or [False] * 4
        enemy_active_series.append([bool(v) for v in ea])
        diver_active_series.append([bool(v) for v in da])
        diff = e.get("difficulty")
        if diff is None:
            diff = e.get("level")
        b = _bucket(diff if diff is not None else None)
        b["frames"] += 1
        b["enemy_active_series"].append([bool(v) for v in ea])
        rejected_track_breaks += int(e.get("enemy_dx_rejected") or 0)

        motion = e.get("enemy_motion_dxs")
        if motion is not None:
            for dx in motion:
                _consume_enemy_dx(float(dx), b)
        else:
            # Legacy events: use lane enemy_dx but drop insane values.
            edx = e.get("enemy_dx") or [None] * 4
            for lane in range(4):
                if not (lane < len(ea) and ea[lane]):
                    continue
                dx = edx[lane] if lane < len(edx) else None
                if dx is None:
                    continue
                if abs(float(dx)) > MAX_CONTINUOUS_ABS_DX:
                    rejected_track_breaks += 1
                    continue
                _consume_enemy_dx(float(dx), b)

        ddx = e.get("diver_dx") or [None] * 4
        for lane in range(4):
            if da[lane] if lane < len(da) else False:
                diver_active_frames += 1
                dx = ddx[lane] if lane < len(ddx) else None
                if dx is None:
                    continue
                if abs(float(dx)) > MAX_CONTINUOUS_ABS_DX:
                    continue
                adx = abs(float(dx))
                diver_abs_when_active.append(adx)
                if adx > 0:
                    diver_moving_frames += 1
                    diver_abs_when_moving.append(adx)
                    b["diver_abs_moving"].append(adx)

        esc = e.get("enemy_slot_count")
        if esc:
            for n, on in zip(esc, ea):
                if on:
                    slot_counts.append(float(n))
                    b["slot_counts"].append(float(n))

    enemy_busy = _busy_segment_lengths(enemy_active_series)
    diver_busy = _busy_segment_lengths(diver_active_series)

    def _mean(xs: list[float]) -> float | None:
        return float(statistics.mean(xs)) if xs else None

    def _median(xs: list[float]) -> float | None:
        return float(statistics.median(xs)) if xs else None

    by_difficulty: dict[str, Any] = {}
    for dkey, b in sorted(by_diff.items(), key=lambda kv: kv[0]):
        busy = _busy_segment_lengths(b["enemy_active_series"])
        ms = b["enemy_motion_samples"]
        by_difficulty[str(dkey)] = {
            "frames": b["frames"],
            "enemy_motion_samples": ms,
            "enemy_mean_abs_dx_when_moving": _mean(b["enemy_abs_moving"]),
            "enemy_mean_abs_dx_when_active": _mean(b["enemy_abs_active"]),
            "enemy_move_frac_of_active": (
                b["enemy_moving_samples"] / ms if ms else None
            ),
            "enemy_abs_dx_hist": dict(b["abs_dx_hist"]),
            "enemy_mean_busy_frames": _mean(busy),
            "enemy_mean_slots_when_active": _mean(b["slot_counts"]),
            "diver_mean_abs_dx_when_moving": _mean(b["diver_abs_moving"]),
        }

    return {
        "enemy_mean_abs_dx_when_moving": _mean(enemy_abs_when_moving),
        "enemy_mean_abs_dx_when_active": _mean(enemy_abs_when_active),
        "enemy_median_abs_dx_when_moving": _median(enemy_abs_when_moving),
        "enemy_move_frac_of_active": (
            enemy_moving_samples / enemy_motion_samples if enemy_motion_samples else None
        ),
        "enemy_motion_samples": enemy_motion_samples,
        "enemy_abs_dx_hist": dict(abs_dx_hist),
        "enemy_dx_track_breaks_rejected": rejected_track_breaks,
        "enemy_mean_busy_frames": _mean(enemy_busy),
        "enemy_median_busy_frames": _median(enemy_busy),
        "enemy_n_busy_segments": len(enemy_busy),
        "enemy_mean_slots_when_active": _mean(slot_counts),
        "diver_mean_abs_dx_when_moving": _mean(diver_abs_when_moving),
        "diver_mean_abs_dx_when_active": _mean(diver_abs_when_active),
        "diver_move_frac_of_active": (
            diver_moving_frames / diver_active_frames if diver_active_frames else None
        ),
        "diver_mean_busy_frames": _mean(diver_busy),
        "diver_median_busy_frames": _median(diver_busy),
        "diver_n_busy_segments": len(diver_busy),
        "by_difficulty": by_difficulty,
        "speed_note": (
            "Speed uses continuous per-entity Δx only (|Δx|<=2, same slot/kind). "
            "Track breaks are rejected. Prefer by_difficulty + enemy_abs_dx_hist; "
            "ignore MIXED_STAGE means for calibration."
        ),
    }


def backend_spawn_stats(
    events: list[dict[str, Any]],
    spawns: list[dict[str, Any]],
    backend: str,
    *,
    enemy_spawns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ev = [e for e in events if e.get("backend") == backend]
    sp = [s for s in spawns if s.get("backend") == backend]
    esp = [s for s in (enemy_spawns or []) if s.get("backend") == backend]
    n_frames = len(ev)
    onsets = [int(s["frame"]) for s in sp]
    wave_frames = sorted(set(onsets))
    per_lane: dict[int, list[int]] = {i: [] for i in range(4)}
    for s in sp:
        per_lane[int(s["lane"])].append(int(s["frame"]))
    lane_ivs: list[float] = []
    lane_stats = {}
    for lane, fs in per_lane.items():
        fs = sorted(fs)
        iv = [b - a for a, b in zip(fs, fs[1:])]
        lane_ivs.extend(iv)
        lane_stats[lane] = {
            "n": len(fs),
            "mean_iv": float(statistics.mean(iv)) if iv else None,
            "median_iv": float(statistics.median(iv)) if iv else None,
            "first": fs[0] if fs else None,
        }
    occ = 0
    eocc = 0
    for e in ev:
        occ += sum(1 for a in e.get("diver_active", []) if a)
        eocc += sum(1 for a in e.get("enemy_active", []) if a)

    enemy_onsets = [int(s["frame"]) for s in esp]
    enemy_per_lane: dict[int, list[int]] = {i: [] for i in range(4)}
    for s in esp:
        enemy_per_lane[int(s["lane"])].append(int(s["frame"]))
    enemy_lane_ivs: list[float] = []
    enemy_lane_stats = {}
    for lane, fs in enemy_per_lane.items():
        fs = sorted(fs)
        iv = [b - a for a, b in zip(fs, fs[1:])]
        enemy_lane_ivs.extend(iv)
        enemy_lane_stats[lane] = {
            "n": len(fs),
            "mean_iv": float(statistics.mean(iv)) if iv else None,
            "median_iv": float(statistics.median(iv)) if iv else None,
            "first": fs[0] if fs else None,
        }

    movement = _movement_stats_from_events(ev)

    return {
        "backend": backend,
        "frames": n_frames,
        "lane_spawns": len(sp),
        "wave_events": len(wave_frames),
        "spawns_per_1k": (1000.0 * len(sp) / n_frames) if n_frames else 0.0,
        "waves_per_1k": (1000.0 * len(wave_frames) / n_frames) if n_frames else 0.0,
        "occupancy_frac": (occ / (n_frames * 4)) if n_frames else 0.0,
        "mean_active_lanes": (occ / n_frames) if n_frames else 0.0,
        "mean_per_lane_interval": float(statistics.mean(lane_ivs)) if lane_ivs else None,
        "first_spawn_frame": min(onsets) if onsets else None,
        "lane_stats": lane_stats,
        "enemy_lane_spawns": len(esp),
        "enemy_spawns_per_1k": (1000.0 * len(esp) / n_frames) if n_frames else 0.0,
        "enemy_occupancy_frac": (eocc / (n_frames * 4)) if n_frames else 0.0,
        "mean_active_enemy_lanes": (eocc / n_frames) if n_frames else 0.0,
        "mean_enemy_per_lane_interval": (
            float(statistics.mean(enemy_lane_ivs)) if enemy_lane_ivs else None
        ),
        "first_enemy_spawn_frame": min(enemy_onsets) if enemy_onsets else None,
        "enemy_lane_stats": enemy_lane_stats,
        "max_episode_return": max((float(e.get("episode_return") or 0) for e in ev), default=0.0),
        **movement,
    }


def compare_spawn_stats(ale: dict[str, Any], jax: dict[str, Any]) -> dict[str, Any]:
    def ratio(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or b == 0:
            return None
        return float(a) / float(b)

    ratios = {
        "spawns_per_1k": ratio(jax["spawns_per_1k"], ale["spawns_per_1k"]),
        "occupancy": ratio(jax["occupancy_frac"], ale["occupancy_frac"]),
        "mean_active_lanes": ratio(jax["mean_active_lanes"], ale["mean_active_lanes"]),
        "mean_per_lane_interval": ratio(
            jax["mean_per_lane_interval"], ale["mean_per_lane_interval"]
        ),
        "enemy_spawns_per_1k": ratio(
            jax["enemy_spawns_per_1k"], ale["enemy_spawns_per_1k"]
        ),
        "enemy_occupancy": ratio(
            jax["enemy_occupancy_frac"], ale["enemy_occupancy_frac"]
        ),
        "mean_enemy_per_lane_interval": ratio(
            jax["mean_enemy_per_lane_interval"], ale["mean_enemy_per_lane_interval"]
        ),
        # Global speed ratios are mixed-stage — kept only as coarse flags.
        "enemy_speed_when_moving_MIXED_STAGE": ratio(
            jax.get("enemy_mean_abs_dx_when_moving"),
            ale.get("enemy_mean_abs_dx_when_moving"),
        ),
        "enemy_speed_when_active_MIXED_STAGE": ratio(
            jax.get("enemy_mean_abs_dx_when_active"),
            ale.get("enemy_mean_abs_dx_when_active"),
        ),
        "enemy_busy_frames": ratio(
            jax.get("enemy_mean_busy_frames"), ale.get("enemy_mean_busy_frames")
        ),
        "diver_speed_when_moving_MIXED_STAGE": ratio(
            jax.get("diver_mean_abs_dx_when_moving"),
            ale.get("diver_mean_abs_dx_when_moving"),
        ),
        "diver_busy_frames": ratio(
            jax.get("diver_mean_busy_frames"), ale.get("diver_mean_busy_frames")
        ),
    }

    # Matched-difficulty speed compare (only stages present in both logs).
    ale_bd = ale.get("by_difficulty") or {}
    jax_bd = jax.get("by_difficulty") or {}
    matched: dict[str, Any] = {}
    speed_active_ratios: list[float] = []
    move_frac_ratios: list[float] = []
    for dkey in sorted(set(ale_bd) & set(jax_bd), key=lambda x: int(x) if x.lstrip("-").isdigit() else 0):
        a, j = ale_bd[dkey], jax_bd[dkey]
        ra = ratio(j.get("enemy_mean_abs_dx_when_active"), a.get("enemy_mean_abs_dx_when_active"))
        rf = ratio(j.get("enemy_move_frac_of_active"), a.get("enemy_move_frac_of_active"))
        matched[dkey] = {
            "ale_frames": a.get("frames"),
            "jax_frames": j.get("frames"),
            "enemy_speed_active_jax_over_ale": ra,
            "enemy_move_frac_jax_over_ale": rf,
            "ale_move_frac": a.get("enemy_move_frac_of_active"),
            "jax_move_frac": j.get("enemy_move_frac_of_active"),
            "ale_speed_active": a.get("enemy_mean_abs_dx_when_active"),
            "jax_speed_active": j.get("enemy_mean_abs_dx_when_active"),
        }
        if ra is not None:
            speed_active_ratios.append(ra)
        if rf is not None:
            move_frac_ratios.append(rf)
    ratios["enemy_speed_active_matched_difficulty_mean"] = (
        float(statistics.mean(speed_active_ratios)) if speed_active_ratios else None
    )
    ratios["enemy_move_frac_matched_difficulty_mean"] = (
        float(statistics.mean(move_frac_ratios)) if move_frac_ratios else None
    )

    return {
        "ale": ale,
        "jax": jax,
        "ratios_jax_over_ale": ratios,
        "speed_by_difficulty_matched": matched,
        "speed_note": (
            "Do not use MIXED_STAGE speed ratios for calibration. "
            "Use speed_by_difficulty_matched / enemy_*_matched_difficulty_mean."
        ),
    }


def stats_from_log_dir(log_dir: Path, backend: str) -> dict[str, Any]:
    events = _load_jsonl(log_dir / "events.jsonl")
    spawns = _load_jsonl(log_dir / "spawns.jsonl")
    enemies = _load_jsonl(log_dir / "enemies.jsonl")
    return backend_spawn_stats(events, spawns, backend, enemy_spawns=enemies)


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_compare(cmp: dict[str, Any]) -> None:
    ale, jax = cmp["ale"], cmp["jax"]
    r = cmp["ratios_jax_over_ale"]
    print("\n=== JAX vs ALE spawn density ===")
    print(
        f"ALE  frames={ale['frames']}  divers/1k={ale['spawns_per_1k']:.2f}  "
        f"diver_occ={ale['occupancy_frac']:.3f}  diver_iv={ale['mean_per_lane_interval']}  "
        f"enemies/1k={ale['enemy_spawns_per_1k']:.2f}  enemy_occ={ale['enemy_occupancy_frac']:.3f}  "
        f"enemy_iv={ale['mean_enemy_per_lane_interval']}"
    )
    print(
        f"JAX  frames={jax['frames']}  divers/1k={jax['spawns_per_1k']:.2f}  "
        f"diver_occ={jax['occupancy_frac']:.3f}  diver_iv={jax['mean_per_lane_interval']}  "
        f"enemies/1k={jax['enemy_spawns_per_1k']:.2f}  enemy_occ={jax['enemy_occupancy_frac']:.3f}  "
        f"enemy_iv={jax['mean_enemy_per_lane_interval']}"
    )
    print(
        f"ratios JAX/ALE: divers/1k={r['spawns_per_1k']}  diver_occ={r['occupancy']}  "
        f"enemies/1k={r['enemy_spawns_per_1k']}  enemy_occ={r['enemy_occupancy']}"
    )
    print("\n=== JAX vs ALE movement (matched difficulty, continuous tracks only) ===")
    print(
        "NOTE: |Δx| from same slot/kind only; |Δx|>2 rejected as track breaks. "
        "Speed rises each surface — compare per difficulty, prefer |dx| hist."
    )
    print(
        f"  ALE |dx| hist={ale.get('enemy_abs_dx_hist')}  "
        f"rejected={ale.get('enemy_dx_track_breaks_rejected')}  "
        f"samples={ale.get('enemy_motion_samples')}"
    )
    print(
        f"  JAX |dx| hist={jax.get('enemy_abs_dx_hist')}  "
        f"rejected={jax.get('enemy_dx_track_breaks_rejected')}  "
        f"samples={jax.get('enemy_motion_samples')}"
    )
    matched = cmp.get("speed_by_difficulty_matched") or {}
    if not matched:
        print("  (no overlapping difficulty stages with movement samples)")
    else:
        print(
            f"  matched-stage mean move_frac ratio={r.get('enemy_move_frac_matched_difficulty_mean')}  "
            f"(speed_active means are secondary)"
        )
        for dkey, row in list(matched.items())[:12]:
            print(
                f"  diff={dkey}: ALE move_frac={_fmt(row.get('ale_move_frac'))} "
                f"| JAX move_frac={_fmt(row.get('jax_move_frac'))} "
                f"| ratio_frac={_fmt(row.get('enemy_move_frac_jax_over_ale'))} "
                f"(frames A/J {row.get('ale_frames')}/{row.get('jax_frames')})"
            )
        if len(matched) > 12:
            print(f"  ... {len(matched) - 12} more stages in vs_ale.json")
    print(
        f"\n(coarse dwell, may mix stages) enemy_busy ratio={r.get('enemy_busy_frames')}  "
        f"diver_busy ratio={r.get('diver_busy_frames')}"
    )


def _jax_snapshot(state) -> dict[str, Any]:
    diver = np.asarray(state.diver_positions)
    shark = np.asarray(state.shark_positions)
    sub = np.asarray(state.sub_positions)
    spawn = state.spawn_state
    divers = []
    for lane in range(4):
        x, y, d = (float(diver[lane, 0]), float(diver[lane, 1]), float(diver[lane, 2]))
        divers.append(
            {
                "lane": lane,
                "x": x,
                "y": y,
                "dir": d,
                "active_diver": bool(d != 0),
            }
        )
    enemy_slots = _jax_enemy_slots(shark, sub)
    enemy_active = []
    enemies = []
    for lane in range(4):
        xs = [s["x"] for s in enemy_slots[lane] if s["active"]]
        active = bool(xs)
        enemy_active.append(active)
        kinds = {s["kind"] for s in enemy_slots[lane] if s["active"]}
        enemies.append(
            {
                "lane": lane,
                "active_enemy": active,
                "has_shark": "shark" in kinds,
                "has_sub": "sub" in kinds,
                "xs": xs,
                "x": (min(xs) if xs else None),
                "slots": enemy_slots[lane],
            }
        )
    return {
        "player_x": int(state.player_x),
        "player_y": int(state.player_y),
        "oxygen": int(state.oxygen),
        "lives": int(state.lives),
        "score": int(state.score),
        "divers_collected": int(state.divers_collected),
        "successful_rescues": int(state.successful_rescues),
        "step_counter": int(state.step_counter),
        "difficulty": int(spawn.difficulty),
        "spawn_timers": [int(v) for v in np.asarray(spawn.spawn_timers)],
        "diver_array": [int(v) for v in np.asarray(spawn.diver_array)],
        "lane_directions": [int(v) for v in np.asarray(spawn.lane_directions)],
        "prev_sub": [bool(v) for v in np.asarray(spawn.prev_sub)],
        "survived": [int(v) for v in np.asarray(spawn.survived).reshape(-1)],
        "shark_positions": shark.tolist(),
        "sub_positions": sub.tolist(),
        "diver_positions": diver.tolist(),
        "divers": divers,
        "diver_active": [bool(d["active_diver"]) for d in divers],
        "enemies": enemies,
        "enemy_active": enemy_active,
        "enemy_x": [e["x"] for e in enemies],
        "enemy_slots": enemy_slots,
        "diver_x": [float(d["x"]) if d["active_diver"] else None for d in divers],
        "enemy_slot_count": [len(e["xs"]) for e in enemies],
    }


class AleBackend:
    name = "ale"

    def __init__(self, seed: int, sticky: float):
        import ale_py
        from ale_py import ALEInterface

        self.ale = ALEInterface()
        self.ale.setInt("random_seed", int(seed))
        self.ale.setFloat("repeat_action_probability", float(sticky))
        self.ale.loadROM(ale_py.roms.get_rom_path("seaquest"))
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.ale.reset_game()
        # Start episode
        for _ in range(40):
            self.ale.act(ACTION_FIRE)
            if int(self.ale.getRAM()[ALE_OXYGEN_RAM]) > 0:
                break
        self.frame = 0
        self.episode_return = 0.0
        self._prev_active = np.zeros(4, dtype=bool)
        self._prev_enemy = np.zeros(4, dtype=bool)
        self._prev_enemy_x: list[float | None] = [None] * 4
        self._prev_diver_x: list[float | None] = [None] * 4

    def step(self, action: int, oxygen_cheat: bool) -> tuple[np.ndarray, float, bool, dict]:
        if oxygen_cheat:
            self.ale.setRAM(ALE_OXYGEN_RAM, ALE_OXYGEN_FULL)
        reward = float(self.ale.act(int(action)))
        if oxygen_cheat:
            self.ale.setRAM(ALE_OXYGEN_RAM, ALE_OXYGEN_FULL)
        self.episode_return += reward
        self.frame += 1
        done = bool(self.ale.game_over())
        ram = self.ale.getRAM()
        snap = _ale_snapshot(ram)
        active = np.asarray(snap["diver_active"], dtype=bool)
        newly = np.logical_and(active, np.logical_not(self._prev_active))
        self._prev_active = active
        enemy = np.asarray(snap["enemy_active"], dtype=bool)
        enemy_newly = np.logical_and(enemy, np.logical_not(self._prev_enemy))
        self._prev_enemy = enemy

        # ALE: one x/lane. Continuity filter drops track breaks (|Δx|>2).
        enemy_dx: list[float | None] = []
        motion_dxs: list[float] = []
        rejected = 0
        for lane in range(4):
            prev_x = self._prev_enemy_x[lane]
            cur_x = snap["enemy_x"][lane]
            if prev_x is not None and cur_x is not None:
                raw = float(cur_x) - float(prev_x)
                if abs(raw) > MAX_CONTINUOUS_ABS_DX:
                    rejected += 1
                    enemy_dx.append(None)
                else:
                    enemy_dx.append(raw)
                    motion_dxs.append(raw)
            else:
                enemy_dx.append(None)
        diver_dx = _lane_dx(self._prev_diver_x, snap["diver_x"])
        self._prev_enemy_x = list(snap["enemy_x"])
        self._prev_diver_x = list(snap["diver_x"])
        rgb = self.ale.getScreenRGB()
        info = {
            "snapshot": snap,
            "spawn_onsets": [int(i) for i in np.where(newly)[0]],
            "enemy_spawn_onsets": [int(i) for i in np.where(enemy_newly)[0]],
            "enemy_dx": enemy_dx,
            "enemy_motion_dxs": motion_dxs,
            "enemy_dx_rejected": rejected,
            "diver_dx": diver_dx,
            "reward": reward,
            "done": done,
        }
        return rgb, reward, done, info


class JaxBackend:
    name = "jax"

    def __init__(self, seed: int):
        import jax
        import jax.numpy as jnp
        import jaxtari

        self._jax = jax
        self._jnp = jnp
        self.env = jaxtari.make("seaquest")
        self._rng = jax.random.PRNGKey(int(seed))
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self._rng, key = self._jax.random.split(self._rng)
        _obs, self.state = self.env.reset(key)
        self.frame = 0
        self.episode_return = 0.0
        self._prev_active = np.zeros(4, dtype=bool)
        self._prev_enemy = np.zeros(4, dtype=bool)
        self._prev_enemy_x: list[float | None] = [None] * 4
        self._prev_diver_x: list[float | None] = [None] * 4
        self._prev_enemy_slots: list[list[dict[str, Any]]] | None = None

    def step(self, action: int, oxygen_cheat: bool) -> tuple[np.ndarray, float, bool, dict]:
        jnp = self._jnp
        if oxygen_cheat:
            self.state = self.state.replace(oxygen=jnp.array(64, dtype=jnp.int32))
        _obs, self.state, reward, done, _info = self.env.step(self.state, int(action))
        if oxygen_cheat:
            self.state = self.state.replace(oxygen=jnp.array(64, dtype=jnp.int32))
        reward_f = float(reward)
        self.episode_return += reward_f
        self.frame += 1
        snap = _jax_snapshot(self.state)
        active = np.asarray(snap["diver_active"], dtype=bool)
        newly = np.logical_and(active, np.logical_not(self._prev_active))
        self._prev_active = active
        enemy = np.asarray(snap["enemy_active"], dtype=bool)
        enemy_newly = np.logical_and(enemy, np.logical_not(self._prev_enemy))
        self._prev_enemy = enemy

        slot_dx, motion_dxs, rejected = _slot_motion_dxs(
            self._prev_enemy_slots, snap["enemy_slots"]
        )
        # Lane summary: mean of continuous slot dxs this frame (HUD only).
        enemy_dx: list[float | None] = []
        for lane in range(4):
            vals = [d for d in slot_dx[lane] if d is not None]
            enemy_dx.append(float(statistics.mean(vals)) if vals else None)
        diver_dx = _lane_dx(self._prev_diver_x, snap["diver_x"])
        self._prev_enemy_x = list(snap["enemy_x"])
        self._prev_diver_x = list(snap["diver_x"])
        self._prev_enemy_slots = snap["enemy_slots"]
        rgb = np.asarray(self.env.render(self.state), dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        info = {
            "snapshot": snap,
            "spawn_onsets": [int(i) for i in np.where(newly)[0]],
            "enemy_spawn_onsets": [int(i) for i in np.where(enemy_newly)[0]],
            "enemy_dx": enemy_dx,
            "enemy_slot_dx": slot_dx,
            "enemy_motion_dxs": motion_dxs,
            "enemy_dx_rejected": rejected,
            "diver_dx": diver_dx,
            "reward": reward_f,
            "done": bool(done),
        }
        if bool(done):
            self._prev_active = np.zeros(4, dtype=bool)
            self._prev_enemy = np.zeros(4, dtype=bool)
            self._prev_enemy_x = [None] * 4
            self._prev_diver_x = [None] * 4
            self._prev_enemy_slots = None
        return rgb, reward_f, bool(done), info


class SpawnPlaySession:
    def __init__(
        self,
        out_dir: Path,
        start: str,
        seed: int,
        sticky: float,
        oxygen_cheat: bool,
        scale: int,
        fps: int,
        *,
        jax_only: bool = False,
        compare_ale: Path | None = None,
    ):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_f = (out_dir / "events.jsonl").open("w", encoding="utf-8")
        self.spawns_f = (out_dir / "spawns.jsonl").open("w", encoding="utf-8")
        self.enemies_f = (out_dir / "enemies.jsonl").open("w", encoding="utf-8")
        self.summary_path = out_dir / "summary.txt"
        self.oxygen_cheat = oxygen_cheat
        self.scale = scale
        self.fps = fps
        self.paused = False
        self.seed = seed
        self.sticky = sticky
        self.jax_only = bool(jax_only)
        self.compare_ale = Path(compare_ale) if compare_ale else None
        self.ale_baseline_stats: dict[str, Any] | None = None
        if self.compare_ale is not None:
            self.ale_baseline_stats = stats_from_log_dir(self.compare_ale, "ale")
            print(
                f"ALE baseline: {self.compare_ale}  "
                f"spawns/1k={self.ale_baseline_stats['spawns_per_1k']:.2f}  "
                f"occ={self.ale_baseline_stats['occupancy_frac']:.3f}",
                flush=True,
            )

        if self.jax_only:
            self.backends = {"jax": JaxBackend(seed)}
            self.backend_name = "jax"
        else:
            self.backends = {
                "ale": AleBackend(seed, sticky),
                "jax": JaxBackend(seed),
            }
            self.backend_name = "ale" if start == "ale" else "jax"
        self.backend = self.backends[self.backend_name]

        self.totals = {
            "ale": {
                "frames": 0,
                "lane_spawns": 0,
                "enemy_lane_spawns": 0,
                "episodes": 0,
                "return": 0.0,
                "enemy_abs_dx_sum": 0.0,
                "enemy_abs_dx_n": 0,
            },
            "jax": {
                "frames": 0,
                "lane_spawns": 0,
                "enemy_lane_spawns": 0,
                "episodes": 0,
                "return": 0.0,
                "enemy_abs_dx_sum": 0.0,
                "enemy_abs_dx_n": 0,
            },
        }
        meta = {
            "started": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "ale_sticky": sticky,
            "start": "jax" if self.jax_only else start,
            "jax_only": self.jax_only,
            "compare_ale": str(self.compare_ale) if self.compare_ale else None,
            "oxygen_cheat_default": oxygen_cheat,
            "controls": "arrows/WASD move, space fire, Tab switch (if dual), O oxygen cheat, R reset",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._write_summary_header()

        pygame.init()
        self.screen = pygame.display.set_mode((160 * scale + 360, 210 * scale))
        caption = "Seaquest spawn logger — JAX only" if self.jax_only else "Seaquest spawn logger — Tab switches ALE/JAX"
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_sm = pygame.font.SysFont("monospace", 12)
        self.clock = pygame.time.Clock()

    def _write_summary_header(self) -> None:
        lines = [
            "Seaquest spawn play log",
            f"out={self.out_dir}",
            f"jax_only={self.jax_only}",
        ]
        if self.compare_ale:
            lines.append(f"compare_ale={self.compare_ale}")
        lines.append("diver onsets → spawns.jsonl ; enemy onsets → enemies.jsonl")
        lines.append("events.jsonl also logs enemy_x/diver_x and per-frame Δx for speed compare")
        lines.append("")
        self.summary_path.write_text("\n".join(lines) + "\n")

    def _append_summary(self, line: str) -> None:
        with self.summary_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def switch_backend(self) -> None:
        if self.jax_only:
            print("Tab ignored (--jax-only)", flush=True)
            return
        other = "jax" if self.backend_name == "ale" else "ale"
        self._append_summary(
            f"[{self.backend_name}] end episode return={self.backend.episode_return:.0f} "
            f"frames={self.backend.frame}"
        )
        self.totals[self.backend_name]["episodes"] += 1
        self.totals[self.backend_name]["return"] += self.backend.episode_return
        self.backend_name = other
        self.backend = self.backends[other]
        self.backend.reset()
        self._append_summary(f"--- switched to {other} ---")
        print(f"Switched to {other}", flush=True)

    def reset_current(self) -> None:
        self._append_summary(
            f"[{self.backend_name}] manual reset return={self.backend.episode_return:.0f}"
        )
        self.totals[self.backend_name]["episodes"] += 1
        self.totals[self.backend_name]["return"] += self.backend.episode_return
        self.backend.reset()

    def _log_step(self, action: int, info: dict) -> None:
        snap = info["snapshot"]
        active = snap["diver_active"]
        enemy_active = snap.get("enemy_active", [False] * 4)
        enemy_dx = info.get("enemy_dx") or [None] * 4
        diver_dx = info.get("diver_dx") or [None] * 4
        rec = {
            "t": time.time(),
            "backend": self.backend_name,
            "frame": self.backend.frame,
            "action": action,
            "reward": info["reward"],
            "done": info["done"],
            "oxygen_cheat": self.oxygen_cheat,
            "episode_return": self.backend.episode_return,
            "diver_active": active,
            "enemy_active": enemy_active,
            "enemy_x": snap.get("enemy_x"),
            "diver_x": snap.get("diver_x"),
            "enemy_dx": enemy_dx,
            "diver_dx": diver_dx,
            "enemy_motion_dxs": info.get("enemy_motion_dxs") or [],
            "enemy_slot_dx": info.get("enemy_slot_dx"),
            "enemy_dx_rejected": int(info.get("enemy_dx_rejected") or 0),
            "enemy_slot_count": snap.get("enemy_slot_count"),
            "player": {"x": snap.get("player_x"), "y": snap.get("player_y")},
            "oxygen": snap.get("oxygen"),
            "divers_collected": snap.get("divers_collected"),
            "spawn_timers": snap.get("spawn_timers"),
            "difficulty": snap.get("difficulty", snap.get("level")),
            "successful_rescues": snap.get("successful_rescues"),
        }
        self.events_f.write(json.dumps(rec) + "\n")

        for dx in info.get("enemy_motion_dxs") or []:
            if abs(float(dx)) > 0:
                self.totals[self.backend_name]["enemy_abs_dx_sum"] += abs(float(dx))
                self.totals[self.backend_name]["enemy_abs_dx_n"] += 1

        onsets = info["spawn_onsets"]
        if onsets:
            for lane in onsets:
                spawn_rec = {
                    "t": time.time(),
                    "backend": self.backend_name,
                    "frame": self.backend.frame,
                    "kind": "diver",
                    "lane": lane,
                    "action": action,
                    "oxygen_cheat": self.oxygen_cheat,
                    "episode_return": self.backend.episode_return,
                    "snapshot": snap,
                }
                self.spawns_f.write(json.dumps(spawn_rec) + "\n")
                self.totals[self.backend_name]["lane_spawns"] += 1
                diver = snap["divers"][lane]
                self._append_summary(
                    f"[{self.backend_name}] frame={self.backend.frame} "
                    f"DIVER lane={lane} diver={diver}"
                )
                print(
                    f"DIVER [{self.backend_name}] f={self.backend.frame} lane={lane} {diver}",
                    flush=True,
                )

        enemy_onsets = info.get("enemy_spawn_onsets") or []
        if enemy_onsets:
            for lane in enemy_onsets:
                enemy_rec = {
                    "t": time.time(),
                    "backend": self.backend_name,
                    "frame": self.backend.frame,
                    "kind": "enemy",
                    "lane": lane,
                    "action": action,
                    "oxygen_cheat": self.oxygen_cheat,
                    "episode_return": self.backend.episode_return,
                    "snapshot": snap,
                }
                self.enemies_f.write(json.dumps(enemy_rec) + "\n")
                self.totals[self.backend_name]["enemy_lane_spawns"] += 1
                detail = None
                if "enemies" in snap:
                    detail = snap["enemies"][lane]
                elif "enemy_x" in snap:
                    detail = {"x": snap["enemy_x"][lane], "flags": snap.get("enemy_flags_89_93")}
                self._append_summary(
                    f"[{self.backend_name}] frame={self.backend.frame} "
                    f"ENEMY lane={lane} {detail}"
                )
                print(
                    f"ENEMY [{self.backend_name}] f={self.backend.frame} lane={lane} {detail}",
                    flush=True,
                )

        self.totals[self.backend_name]["frames"] += 1
        if self.backend.frame % 50 == 0:
            self.events_f.flush()
            self.spawns_f.flush()
            self.enemies_f.flush()

    def _draw(self, rgb: np.ndarray) -> None:
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (160 * self.scale, 210 * self.scale))
        self.screen.fill((18, 18, 22))
        self.screen.blit(surf, (0, 0))

        tot = self.totals[self.backend_name]
        jax_tot = self.totals["jax"]
        live_speed = (
            (tot["enemy_abs_dx_sum"] / tot["enemy_abs_dx_n"])
            if tot["enemy_abs_dx_n"]
            else 0.0
        )
        lines = [
            f"backend: {self.backend_name.upper()}"
            + ("  (jax-only)" if self.jax_only else "   Tab=switch"),
            f"frame: {self.backend.frame}   return: {self.backend.episode_return:.0f}",
            f"O2 cheat: {'ON' if self.oxygen_cheat else 'off'} (O)",
            f"divers: {tot['lane_spawns']}   enemies: {tot['enemy_lane_spawns']}",
            f"enemy |dx| moving: {live_speed:.3f} px/f",
            f"fps target: {self.fps}   paused: {self.paused}",
            "",
            f"JAX live: f={jax_tot['frames']} D={jax_tot['lane_spawns']} E={jax_tot['enemy_lane_spawns']}",
        ]
        if self.ale_baseline_stats is not None:
            ab = self.ale_baseline_stats
            live_d = (
                (1000.0 * jax_tot["lane_spawns"] / jax_tot["frames"])
                if jax_tot["frames"]
                else 0.0
            )
            live_e = (
                (1000.0 * jax_tot["enemy_lane_spawns"] / jax_tot["frames"])
                if jax_tot["frames"]
                else 0.0
            )
            lines += [
                f"ALE base D/1k={ab['spawns_per_1k']:.2f} E/1k={ab.get('enemy_spawns_per_1k', 0):.2f}",
                f"JAX live D/1k={live_d:.2f} E/1k={live_e:.2f}",
            ]
            if ab.get("enemy_mean_abs_dx_when_moving") is not None:
                lines.append(
                    f"ALE |dx|={ab['enemy_mean_abs_dx_when_moving']:.3f}  "
                    f"busy={ab.get('enemy_mean_busy_frames')}"
                )
        if not self.jax_only:
            at = self.totals["ale"]
            ale_speed = (
                (at["enemy_abs_dx_sum"] / at["enemy_abs_dx_n"])
                if at["enemy_abs_dx_n"]
                else 0.0
            )
            lines += [
                "",
                f"ALE totals: f={at['frames']} D={at['lane_spawns']} E={at['enemy_lane_spawns']}",
                f"ALE |dx| moving: {ale_speed:.3f} px/f",
            ]
        lines += [
            "",
            "Arrows/WASD + Space",
            "R reset  P pause  N step",
            "[ ] speed   Q quit",
        ]
        x0 = 160 * self.scale + 12
        y = 12
        for line in lines:
            img = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(img, (x0, y))
            y += 18

        # Live diver strip from last step stored on self
        snap = getattr(self, "_last_snap", None)
        if snap is not None:
            y += 8
            self.screen.blit(self.font.render("divers:", True, (180, 220, 255)), (x0, y))
            y += 18
            for d in snap["divers"]:
                mark = "*" if d.get("active_diver") else "."
                if "y" in d:
                    txt = f"  L{d['lane']} {mark} x={d.get('x')} y={d.get('y')} dir={d.get('dir')}"
                else:
                    txt = f"  L{d['lane']} {mark} x={d.get('x')} missile={d.get('is_missile_slot')}"
                img = self.font_sm.render(txt, True, (200, 200, 200))
                self.screen.blit(img, (x0, y))
                y += 16

        pygame.display.flip()

    def run(self) -> None:
        running = True
        step_once = False
        # Prime a frame
        rgb, _, _, info = self.backend.step(ACTION_NOOP, self.oxygen_cheat)
        self._last_snap = info["snapshot"]
        self._log_step(ACTION_NOOP, info)
        self._draw(rgb)

        while running:
            step_once = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif ev.key == pygame.K_TAB:
                        self.switch_backend()
                        rgb, _, _, info = self.backend.step(ACTION_NOOP, self.oxygen_cheat)
                        self._last_snap = info["snapshot"]
                        self._log_step(ACTION_NOOP, info)
                    elif ev.key == pygame.K_r:
                        self.reset_current()
                        rgb, _, _, info = self.backend.step(ACTION_NOOP, self.oxygen_cheat)
                        self._last_snap = info["snapshot"]
                        self._log_step(ACTION_NOOP, info)
                    elif ev.key == pygame.K_o:
                        self.oxygen_cheat = not self.oxygen_cheat
                        print(f"oxygen_cheat={self.oxygen_cheat}", flush=True)
                    elif ev.key == pygame.K_p:
                        self.paused = not self.paused
                    elif ev.key == pygame.K_n:
                        step_once = True
                    elif ev.key == pygame.K_LEFTBRACKET:
                        self.fps = max(5, self.fps - 5)
                    elif ev.key == pygame.K_RIGHTBRACKET:
                        self.fps = min(120, self.fps + 5)

            keys = pygame.key.get_pressed()
            do_step = (not self.paused) or step_once
            if do_step:
                action = _keys_to_action(keys)
                rgb, _rew, done, info = self.backend.step(action, self.oxygen_cheat)
                self._last_snap = info["snapshot"]
                self._log_step(action, info)
                if done:
                    self._append_summary(
                        f"[{self.backend_name}] GAME OVER return={self.backend.episode_return:.0f}"
                    )
                    self.totals[self.backend_name]["episodes"] += 1
                    self.totals[self.backend_name]["return"] += self.backend.episode_return
                    self.backend.reset()
                    print(f"[{self.backend_name}] game over — reset", flush=True)

            self._draw(rgb)
            self.clock.tick(self.fps)

        self.close()

    def close(self) -> None:
        self.events_f.flush()
        self.spawns_f.flush()
        self.enemies_f.flush()
        self.events_f.close()
        self.spawns_f.close()
        self.enemies_f.close()
        summary = {
            "totals": self.totals,
            "oxygen_cheat_final": self.oxygen_cheat,
        }
        (self.out_dir / "final_totals.json").write_text(json.dumps(summary, indent=2))
        self._append_summary("=== FINAL ===")
        self._append_summary(json.dumps(summary, indent=2))

        if self.compare_ale is not None:
            ale = self.ale_baseline_stats or stats_from_log_dir(self.compare_ale, "ale")
            jax = stats_from_log_dir(self.out_dir, "jax")
            cmp = compare_spawn_stats(ale, jax)
            (self.out_dir / "vs_ale.json").write_text(json.dumps(cmp, indent=2))
            print_compare(cmp)
            self._append_summary(json.dumps(cmp["ratios_jax_over_ale"], indent=2))
        elif not self.jax_only and self.totals["ale"]["frames"] and self.totals["jax"]["frames"]:
            # Dual session: compare ALE vs JAX from this same log dir.
            cmp = compare_spawn_stats(
                stats_from_log_dir(self.out_dir, "ale"),
                stats_from_log_dir(self.out_dir, "jax"),
            )
            (self.out_dir / "vs_ale.json").write_text(json.dumps(cmp, indent=2))
            print_compare(cmp)
            self._append_summary(json.dumps(cmp["ratios_jax_over_ale"], indent=2))

        pygame.quit()
        print(f"Logs written to {self.out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", choices=("ale", "jax"), default="ale")
    ap.add_argument("--jax-only", action="store_true", help="Only run JAXAtari (no ALE / no Tab)")
    ap.add_argument(
        "--compare-ale",
        type=Path,
        default=None,
        help="Existing ALE play log dir to compare against on quit (needs enemies.jsonl)",
    )
    ap.add_argument(
        "--compare-jax",
        type=Path,
        default=None,
        help="With --compare-only: JAX log dir to compare",
    )
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip play; only print/write compare of --compare-ale vs --compare-jax",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ale-sticky", type=float, default=0.0)
    ap.add_argument("--oxygen-cheat", action="store_true")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: tmp/seaquest_spawn_play/<timestamp>/",
    )
    args = ap.parse_args()

    if args.compare_only:
        ale_dir = args.compare_ale or DEFAULT_ALE_BASELINE
        jax_dir = args.compare_jax
        if jax_dir is None:
            ap.error("--compare-only requires --compare-jax")
        cmp = compare_spawn_stats(
            stats_from_log_dir(ale_dir, "ale"),
            stats_from_log_dir(jax_dir, "jax"),
        )
        print_compare(cmp)
        out = jax_dir / "vs_ale.json"
        out.write_text(json.dumps(cmp, indent=2))
        print(f"Wrote {out}")
        return

    compare_ale = args.compare_ale
    if args.jax_only and compare_ale is None and DEFAULT_ALE_BASELINE is not None and DEFAULT_ALE_BASELINE.is_dir():
        compare_ale = DEFAULT_ALE_BASELINE

    out = args.out_dir or Path("tmp/seaquest_spawn_play") / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    print(f"Logging to {out}", flush=True)
    if args.jax_only:
        print("JAX-only play; will compare to ALE baseline on quit.", flush=True)
    else:
        print("Play; Tab switches backends.", flush=True)
    session = SpawnPlaySession(
        out_dir=out,
        start="jax" if args.jax_only else args.start,
        seed=args.seed,
        sticky=args.ale_sticky,
        oxygen_cheat=args.oxygen_cheat,
        scale=args.scale,
        fps=args.fps,
        jax_only=args.jax_only,
        compare_ale=compare_ale,
    )
    session.run()


if __name__ == "__main__":
    main()
