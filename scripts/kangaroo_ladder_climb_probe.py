#!/usr/bin/env python3
"""Ladder-only Kangaroo climb probe: greedy FORWARD toward heuristic waypoints.

Uses JAXAtari mods to strip enemies / hazards so the only task is climbing
L0 → L1 → L2 by always taking ego FORWARD toward the current plan target.

Example:
  python scripts/kangaroo_ladder_climb_probe.py
  python scripts/kangaroo_ladder_climb_probe.py --max-frames 4000 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concept_based_learning.environments.datatypes import (  # noqa: E402
    configure_label_routing_from_config,
)
from concept_based_learning.environments.env_manager import EnvManager  # noqa: E402
from concept_based_learning.environments.heuristics import (  # noqa: E402
    initial_policy_state,
)
from concept_based_learning.environments.kangaroo import kangaroo_label as kg_label  # noqa: E402
from concept_based_learning.observations.egocentric import (  # noqa: E402
    configure_ego_obs_from_config,
)
from concept_based_learning.pqn_visualizer import (  # noqa: E402
    create_eval_env,
    load_base_config,
)

# Ego FORWARD = directly toward current target (action_space index 2).
EGO_FORWARD = 2

SLOT_HINT = {
    **{i: f"L{i - 1}_BOT" for i in range(1, 21)},
    **{i: f"L{i - 21}_TOP" for i in range(21, 41)},
    45: "CHILD",
}

# Ladder-only: no monkeys / coconuts / fruit / bell; lock L1.
LADDER_ONLY_MODS = [
    "no_danger",
    "no_thrown_coconut",
    "no_fruit",
    "no_bell",
    "first_level_only",
]

# Plan stages in order (must match kangaroo_policy).
PLAN_SLOTS = (1, 21, 2, 22, 3, 23)
# L1 top platform y=28; standing player.y = 28 - PLAYER_HEIGHT(24) = 4.
STANDING_ON_TOP_Y_MAX = 8.0
L2_TOP_SLOT = 23


def _player_xy(raw):
    px = float(jax.device_get(raw.player.x[-1] + raw.player.width[-1] / 2.0))
    py = float(jax.device_get(raw.player.y[-1] + raw.player.height[-1] / 2.0))
    x = float(jax.device_get(raw.player.x[-1]))
    y = float(jax.device_get(raw.player.y[-1]))
    w = float(jax.device_get(raw.player.width[-1]))
    h = float(jax.device_get(raw.player.height[-1]))
    return px, py, x, y, w, h


def _obj_xy(objects, idx: int):
    ox = float(jax.device_get(objects.x[-1, idx] + objects.width[-1, idx] / 2.0))
    oy = float(jax.device_get(objects.y[-1, idx] + objects.height[-1, idx] / 2.0))
    active = int(jax.device_get(objects.active[-1, idx]))
    w = float(jax.device_get(objects.width[-1, idx]))
    h = float(jax.device_get(objects.height[-1, idx]))
    return ox, oy, active, w, h


def _climbing_flag(raw) -> int:
    st = raw.player.state
    val = st[-1] if jnp.ndim(st) > 0 else st
    return int(jax.device_get(val))


def _ladder_overlap_x(player_x: float, player_w: float, lad_x: float, lad_w: float) -> float:
    return max(0.0, min(player_x + player_w, lad_x + lad_w) - max(player_x, lad_x))


def run_probe(*, max_frames: int, seed: int, verbose: bool) -> dict:
    base = load_base_config(experiment="feature_eval/kangaroo_ladder_only_probe")
    base["WANDB_MODE"] = "disabled"
    base["ENV_NAME"] = "kangaroo"
    # Prefer YAML mods; allow CLI override via already-set GAME_MODS.
    if not base.get("GAME_MODS"):
        base["GAME_MODS"] = list(LADDER_ONLY_MODS)
        base["EVAL_GAME_MODS"] = list(LADDER_ONLY_MODS)
    mods = list(base["GAME_MODS"])
    configure_ego_obs_from_config(base)
    configure_label_routing_from_config(base)

    env, _ = create_eval_env(base)
    components = EnvManager().get_components("kangaroo")
    mapper = components.action_mapper
    heuristic = components.get_heuristic_with_state
    no_tgt = int(components.no_target_idx)

    rng = jax.random.PRNGKey(seed)
    obs, state = env.reset(rng)
    obs, state = jax.device_get((obs, state))
    policy_state = initial_policy_state(no_tgt)
    prev_target = no_tgt

    stage_events: list[str] = []
    stuck_frames = 0
    prev_py = None
    reached_top = False
    final_frame = max_frames - 1
    min_player_y = 1e9

    # Active L1 ladders from observation (first 3).
    raw0 = obs.raw_observation
    lad_x = np.asarray(jax.device_get(raw0.ladders.x[-1] if jnp.ndim(raw0.ladders.x) > 1 else raw0.ladders.x))
    lad_w = np.asarray(jax.device_get(raw0.ladders.width[-1] if jnp.ndim(raw0.ladders.width) > 1 else raw0.ladders.width))
    lad_y = np.asarray(jax.device_get(raw0.ladders.y[-1] if jnp.ndim(raw0.ladders.y) > 1 else raw0.ladders.y))
    lad_h = np.asarray(jax.device_get(raw0.ladders.height[-1] if jnp.ndim(raw0.ladders.height) > 1 else raw0.ladders.height))
    lad_act = np.asarray(
        jax.device_get(raw0.ladders.active[-1] if jnp.ndim(raw0.ladders.active) > 1 else raw0.ladders.active)
    )

    print("=== Kangaroo ladder climb probe ===")
    print(f"mods={mods}")
    print(
        f"WP constants: TOP_LIFT={kg_label._TOP_WAYPOINT_LIFT_PX} "
        f"BOT_Y_OFF={kg_label._BOTTOM_WAYPOINT_Y_OFFSET_PX} "
        f"WP_W={kg_label._PLAYER_STAND_WIDTH} WP_H={kg_label._LADDER_WAYPOINT_H}"
    )
    print(f"seed={seed} max_frames={max_frames}")
    for i in range(3):
        if float(lad_act[i]) > 0.5:
            print(
                f"  ladder{i}: x={lad_x[i]:.0f} y={lad_y[i]:.0f} "
                f"w={lad_w[i]:.0f} h={lad_h[i]:.0f}"
            )
    # Dump initial waypoint geometry for plan slots.
    for slot in PLAN_SLOTS:
        ox, oy, act, w, h = _obj_xy(obs.objects, slot)
        print(f"  slot {slot:2d} {SLOT_HINT.get(slot,'?'):8s}: "
              f"cx,cy=({ox:.1f},{oy:.1f}) size={w:.0f}x{h:.0f} active={act}")

    print(
        "frame  slot hint       dist  px,py -> tx,ty     pxy "
        "climb  lad_ovlp  game_a"
    )

    for t in range(max_frames):
        rng, step_key = jax.random.split(rng)
        tgt, policy_state = heuristic(
            obs.raw_observation,
            obs.objects,
            jnp.array(prev_target, dtype=jnp.int32),
            policy_state,
            step_key,
        )
        tgt = int(jnp.ravel(jax.device_get(tgt))[0])
        if tgt != prev_target and tgt != no_tgt:
            hint = SLOT_HINT.get(tgt, f"slot{tgt}")
            px, py, *_ = _player_xy(obs.raw_observation)
            stage_events.append(
                f"t={t} → {hint} (slot {tgt}) player=({px:.1f},{py:.1f})"
            )
            if verbose or True:
                # Ladder overlap at stage switch (critical for bot→top).
                _, _, x, y, w, h = _player_xy(obs.raw_observation)
                overlaps = [
                    _ladder_overlap_x(x, w, float(lad_x[i]), float(lad_w[i]))
                    for i in range(3)
                    if float(lad_act[i]) > 0.5
                ]
                max_ov = max(overlaps) if overlaps else 0.0
                climbable = max_ov >= 0.3 * min(w, 8.0)
                print(
                    f"  ** stage → {hint} at t={t}: player_x={x:.1f} "
                    f"max_ladder_x_overlap={max_ov:.1f} climbable≈{climbable}"
                )
        prev_target = tgt

        px, py, x, y, w, h = _player_xy(obs.raw_observation)
        min_player_y = min(min_player_y, y)
        climbing = _climbing_flag(obs.raw_observation)

        if tgt == no_tgt or tgt < 0:
            dist = float("nan")
            tx = ty = float("nan")
            game_a = 0
            max_ov = 0.0
        else:
            tx, ty, _act, _ow, _oh = _obj_xy(obs.objects, tgt)
            dist = float(np.hypot(px - tx, py - ty))
            game_a = int(
                jax.device_get(
                    mapper(
                        jnp.int32(EGO_FORWARD),
                        obs.raw_observation,
                        jnp.int32(tgt),
                        obs.objects,
                    )
                )
            )
            overlaps = [
                _ladder_overlap_x(x, w, float(lad_x[i]), float(lad_w[i]))
                for i in range(3)
                if float(lad_act[i]) > 0.5
            ]
            max_ov = max(overlaps) if overlaps else 0.0

        hint = SLOT_HINT.get(tgt, f"slot{tgt}" if tgt != no_tgt else "NONE")
        if verbose or t < 30 or t % 50 == 0 or (tgt in PLAN_SLOTS and dist < 20):
            print(
                f"{t:5d}  {tgt:3d} {hint:10s} {dist:5.1f}  "
                f"{px:5.1f},{py:5.1f}->{tx:5.1f},{ty:5.1f}  "
                f"({x:.0f},{y:.0f}) climb={climbing} ov={max_ov:.1f} a={game_a}"
            )

        # Stuck detection: target is a top WP, not climbing, not moving up.
        if tgt in (21, 22, 23) and climbing == 0:
            if prev_py is not None and abs(py - prev_py) < 0.5 and max_ov < 2.4:
                stuck_frames += 1
            else:
                stuck_frames = 0
        else:
            stuck_frames = 0
        prev_py = py

        if y <= STANDING_ON_TOP_Y_MAX and climbing == 0:
            reached_top = True
            final_frame = t
            print(f"=== SUCCESS: standing on top at frame {t} (player y={y:.1f}) ===")
            break

        # Past L2_TOP stage with no target left and already high up.
        if tgt == no_tgt and y <= STANDING_ON_TOP_Y_MAX:
            reached_top = True
            final_frame = t
            print(f"=== SUCCESS: plan done on top at frame {t} (player y={y:.1f}) ===")
            break

        if stuck_frames > 120:
            print(
                f"=== STUCK at frame {t}: target={hint} player=({x:.1f},{y:.1f}) "
                f"ov={max_ov:.1f} — cannot start climb ==="
            )
            final_frame = t
            break

        obs, state, _reward, done, _info = env.step(state, game_a)
        obs, state, done = jax.device_get((obs, state, done))
        if bool(done):
            print(f"--- episode done at frame {t} ---")
            break

    px, py, x, y, w, h = _player_xy(obs.raw_observation)
    result = {
        "reached_top": reached_top,
        "final_frame": final_frame,
        "final_player_xy": (x, y),
        "final_player_center": (px, py),
        "min_player_y": min_player_y,
        "stage_events": stage_events,
        "wp_w": float(kg_label._PLAYER_STAND_WIDTH),
        "bot_y_off": float(kg_label._BOTTOM_WAYPOINT_Y_OFFSET_PX),
        "top_lift": float(kg_label._TOP_WAYPOINT_LIFT_PX),
    }
    print("\n=== SUMMARY ===")
    print(f"reached_top={reached_top} frames={final_frame}")
    print(f"final player top-left=({x:.1f},{y:.1f}) center=({px:.1f},{py:.1f})")
    print(f"min player.y (highest on screen)={min_player_y:.1f}")
    print("stage transitions:")
    for ev in stage_events:
        print(f"  {ev}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-frames", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    run_probe(max_frames=args.max_frames, seed=args.seed, verbose=args.verbose)


if __name__ == "__main__":
    main()
