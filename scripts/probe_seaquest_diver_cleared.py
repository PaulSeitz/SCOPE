#!/usr/bin/env python3
"""
Cleared-enemy spawn density probe for JAXAtari Seaquest.

Each frame: if a lane has enemies, instantly clear them and reload that lane's
spawn timer with the diver-conditional reload (long only when diver_array==1).
Oxygen is forced full. Reports diver + enemy onset intervals for ALE compare.

Usage:
  python scripts/probe_seaquest_diver_cleared.py --frames 15000
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _lane_stats(onsets: list[tuple[int, int]]) -> tuple[dict, list[float]]:
    per_lane = {i: [] for i in range(4)}
    for f, lane in onsets:
        per_lane[lane].append(f)
    lane_stats = {}
    all_iv: list[float] = []
    for lane, fs in per_lane.items():
        iv = [b - a for a, b in zip(fs, fs[1:])]
        all_iv.extend(iv)
        lane_stats[lane] = {
            "n": len(fs),
            "mean_iv": float(statistics.mean(iv)) if iv else None,
            "median_iv": float(statistics.median(iv)) if iv else None,
            "first": fs[0] if fs else None,
        }
    return lane_stats, all_iv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=15_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--keep-diver-armed",
        action="store_true",
        help="After instant-collect, set diver_array=1 (old probe). Default: set 0 like real collect.",
    )
    ap.add_argument("--out", type=Path, default=Path("tmp/seaquest_diver_cleared_probe.json"))
    args = ap.parse_args()

    import jaxatari
    from jaxatari.environment import JAXAtariAction as Action
    from jaxatari.games.jax_seaquest import SeaquestConstants

    consts = SeaquestConstants()
    enemy_reload = int(consts.ENEMY_SPAWN_TIMER_RELOAD)
    diver_reload = int(consts.DIVER_WAVE_SPAWN_TIMER_RELOAD)
    trigger_v = int(consts.DIVER_SPAWN_TIMER_TRIGGER)
    print(
        f"ENEMY_SPAWN_TIMER_RELOAD={enemy_reload} "
        f"DIVER_WAVE_SPAWN_TIMER_RELOAD={diver_reload} "
        f"DIVER_SPAWN_TIMER_TRIGGER={trigger_v}",
        flush=True,
    )

    env = jaxatari.make("seaquest")
    # Underlying game for conditional reload helper
    game = getattr(env, "game", None) or getattr(getattr(env, "unwrapped", env), "game", None)
    action_set = np.asarray(env.ACTION_SET)
    up = int(np.where(action_set == int(Action.UP))[0][0])

    rng = jax.random.PRNGKey(args.seed)
    _obs, state = env.reset(rng)
    prev_d = np.zeros(4, dtype=bool)
    prev_e = np.zeros(4, dtype=bool)
    diver_onsets: list[tuple[int, int]] = []
    enemy_onsets: list[tuple[int, int]] = []
    occ_d = 0
    occ_e = 0

    for t in range(args.frames):
        state = state.replace(oxygen=jnp.array(64, dtype=jnp.int32))

        _obs, state, _r, done, _info = env.step(state, up)
        if bool(done):
            rng, key = jax.random.split(rng)
            _obs, state = env.reset(key)
            prev_d = np.zeros(4, dtype=bool)
            prev_e = np.zeros(4, dtype=bool)
            continue

        diver = np.asarray(state.diver_positions)
        cur_d = diver[:, 2] != 0
        occ_d += int(cur_d.sum())
        for lane in np.where(np.logical_and(cur_d, np.logical_not(prev_d)))[0]:
            diver_onsets.append((t, int(lane)))
        prev_d = cur_d.copy()

        shark = np.asarray(state.shark_positions).reshape(4, 3, 3)
        sub = np.asarray(state.sub_positions).reshape(4, 3, 3)
        cur_e = np.any(shark[:, :, 2] != 0, axis=1) | np.any(sub[:, :, 2] != 0, axis=1)
        occ_e += int(cur_e.sum())
        for lane in np.where(np.logical_and(cur_e, np.logical_not(prev_e)))[0]:
            enemy_onsets.append((t, int(lane)))
        prev_e = cur_e.copy()

        had = cur_e.copy()

        # Instant-collect divers so cadence reflects spawn timers, not swim time.
        diver = diver.copy()
        darr = np.asarray(state.spawn_state.diver_array).copy()
        collected = diver[:, 2] != 0
        if collected.any():
            diver[collected] = 0
            darr[collected] = 1 if args.keep_diver_armed else 0
            prev_d[collected] = False

        if had.any() or collected.any():
            timers = np.asarray(state.spawn_state.spawn_timers).copy()
            if had.any():
                if game is not None:
                    reloads = np.asarray(game.spawn_timer_reload(jnp.asarray(darr)))
                else:
                    reloads = np.where(darr == 1, diver_reload, enemy_reload)
                timers[had] = reloads[had]
            state = state.replace(
                shark_positions=jnp.zeros_like(state.shark_positions),
                sub_positions=jnp.zeros_like(state.sub_positions),
                enemy_missile_positions=jnp.zeros_like(state.enemy_missile_positions),
                diver_positions=jnp.asarray(diver, dtype=state.diver_positions.dtype),
                spawn_state=state.spawn_state.replace(
                    spawn_timers=jnp.asarray(timers, dtype=jnp.int32),
                    diver_array=jnp.asarray(darr, dtype=jnp.int32),
                    to_be_spawned=jnp.zeros_like(state.spawn_state.to_be_spawned),
                    survived=jnp.zeros_like(state.spawn_state.survived),
                ),
            )
            prev_e[:] = False

    d_stats, d_iv = _lane_stats(diver_onsets)
    e_stats, e_iv = _lane_stats(enemy_onsets)

    payload = {
        "frames": args.frames,
        "seed": args.seed,
        "keep_diver_armed": args.keep_diver_armed,
        "ENEMY_SPAWN_TIMER_RELOAD": enemy_reload,
        "DIVER_WAVE_SPAWN_TIMER_RELOAD": diver_reload,
        "DIVER_SPAWN_TIMER_TRIGGER": trigger_v,
        "divers": {
            "lane_spawns": len(diver_onsets),
            "spawns_per_1k": 1000.0 * len(diver_onsets) / args.frames,
            "occupancy_frac": occ_d / (args.frames * 4),
            "mean_per_lane_interval": float(statistics.mean(d_iv)) if d_iv else None,
            "lane_stats": d_stats,
        },
        "enemies": {
            "lane_spawns": len(enemy_onsets),
            "spawns_per_1k": 1000.0 * len(enemy_onsets) / args.frames,
            "occupancy_frac": occ_e / (args.frames * 4),
            "mean_per_lane_interval": float(statistics.mean(e_iv)) if e_iv else None,
            "lane_stats": e_stats,
        },
        "enemy_per_diver": (
            len(enemy_onsets) / len(diver_onsets) if diver_onsets else None
        ),
        "ale_play_targets": {
            "diver_mean_iv": 745.7,
            "enemy_mean_iv": 335.8,
            "enemy_per_diver": 2.16,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
