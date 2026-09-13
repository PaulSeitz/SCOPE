#!/usr/bin/env python3
"""
Compare early Seaquest diver spawn rates: ALE (RAM) vs JAXAtari.

Protocol (both):
  - Start the game (ALE: FIRE until oxygen rises; JAXAtari: native reset).
  - Stay at the surface (hold UP).
  - Force oxygen full every frame (ALE RAM 102 = 64; JAXAtari oxygen = 64).
  - Count diver presence / spawn onsets without collecting.

ALE diver detection follows OC_Atari/ocatari/ram/seaquest.py:
  lane i uses ram[74 - i]; diver if not missile-lane and 0 < x < 160.
  Oxygen: ram[102], full = 64.

Usage:
  python scripts/probe_seaquest_diver_spawns.py --frames 10000 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ALE_OXYGEN_RAM = 102
ALE_OXYGEN_FULL = 64
ALE_DIVER_X_RAM = (74, 73, 72, 71)  # lanes bottom → top


def _ale_is_missile_lane(lane_from_bottom: int, ram: np.ndarray) -> bool:
    """ocatari._is_submarine(3 - lane, ram) → missile shares diver X RAM."""
    i = 3 - lane_from_bottom
    return 3 < (int(ram[89 + i]) % 8) < 7


def _ale_diver_active(ram: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=bool)
    for lane in range(4):
        x = int(ram[ALE_DIVER_X_RAM[lane]])
        if _ale_is_missile_lane(lane, ram):
            continue
        out[lane] = 0 < x < 160
    return out


@dataclass
class SpawnStats:
    backend: str
    seeds: list[int]
    frames: int
    lane_spawns: list[int]
    wave_events: list[int]
    diver_frame_occupancy: list[int]
    first_wave_frame: list[int | None]
    wave_intervals: list[list[int]]
    mean_lane_spawns: float
    mean_wave_events: float
    mean_occupancy: float
    mean_occupancy_frac: float
    mean_first_wave: float | None
    mean_wave_interval: float | None


def _summarize(backend: str, seeds: list[int], frames: int, runs: list[dict[str, Any]]) -> SpawnStats:
    lane = [int(r["lane_spawns"]) for r in runs]
    waves = [int(r["wave_events"]) for r in runs]
    occ = [int(r["diver_frame_occupancy"]) for r in runs]
    firsts = [r["first_wave_frame"] for r in runs]
    intervals = [list(map(int, r["wave_intervals"])) for r in runs]
    flat_iv = [x for xs in intervals for x in xs]
    valid_first = [f for f in firsts if f is not None]
    return SpawnStats(
        backend=backend,
        seeds=list(seeds),
        frames=frames,
        lane_spawns=lane,
        wave_events=waves,
        diver_frame_occupancy=occ,
        first_wave_frame=firsts,
        wave_intervals=intervals,
        mean_lane_spawns=float(np.mean(lane)),
        mean_wave_events=float(np.mean(waves)),
        mean_occupancy=float(np.mean(occ)),
        mean_occupancy_frac=float(np.mean(occ)) / (frames * 4),
        mean_first_wave=float(np.mean(valid_first)) if valid_first else None,
        mean_wave_interval=float(np.mean(flat_iv)) if flat_iv else None,
    )


def _track_from_active_mask(active_seq: list[np.ndarray]) -> dict[str, Any]:
    """active_seq[t] = bool[4] diver presence."""
    prev = np.zeros(4, dtype=bool)
    lane_spawns = 0
    wave_frames: list[int] = []
    occupancy = 0
    spawns_per_lane = np.zeros(4, dtype=np.int32)
    for t, cur in enumerate(active_seq):
        cur = np.asarray(cur, dtype=bool)
        occupancy += int(cur.sum())
        newly = np.logical_and(cur, np.logical_not(prev))
        n_new = int(newly.sum())
        if n_new:
            lane_spawns += n_new
            spawns_per_lane += newly.astype(np.int32)
            wave_frames.append(t)
        prev = cur
    intervals = np.diff(wave_frames).tolist() if len(wave_frames) > 1 else []
    return {
        "lane_spawns": lane_spawns,
        "wave_events": len(wave_frames),
        "diver_frame_occupancy": occupancy,
        "first_wave_frame": wave_frames[0] if wave_frames else None,
        "wave_intervals": intervals,
        "spawns_per_lane": spawns_per_lane.tolist(),
        "wave_frames": wave_frames,
    }


def probe_ale(frames: int, seeds: list[int], sticky: float) -> tuple[SpawnStats, list[dict]]:
    import ale_py
    from ale_py import ALEInterface, Action

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        ale = ALEInterface()
        ale.setInt("random_seed", int(seed))
        ale.setFloat("repeat_action_probability", float(sticky))
        ale.loadROM(ale_py.roms.get_rom_path("seaquest"))
        ale.reset_game()

        # Start game (oxygen begins filling after FIRE).
        for _ in range(30):
            ale.act(Action.FIRE)
            if int(ale.getRAM()[ALE_OXYGEN_RAM]) > 0:
                break

        active_seq: list[np.ndarray] = []
        for _t in range(frames):
            ale.setRAM(ALE_OXYGEN_RAM, ALE_OXYGEN_FULL)
            ram = ale.getRAM()
            active_seq.append(_ale_diver_active(ram))
            ale.act(Action.UP)
            if ale.game_over():
                ale.reset_game()
                for _ in range(30):
                    ale.act(Action.FIRE)
                    if int(ale.getRAM()[ALE_OXYGEN_RAM]) > 0:
                        break

        stats = _track_from_active_mask(active_seq)
        stats["seed"] = seed
        runs.append(stats)
        print(
            f"[ALE] seed={seed} waves={stats['wave_events']} "
            f"lane_spawns={stats['lane_spawns']} "
            f"occ={stats['diver_frame_occupancy']} "
            f"first={stats['first_wave_frame']} "
            f"per_lane={stats['spawns_per_lane']}",
            flush=True,
        )
    return _summarize("ALE", seeds, frames, runs), runs


def probe_jaxatari(frames: int, seeds: list[int]) -> tuple[SpawnStats, list[dict]]:
    import jax
    import jax.numpy as jnp
    import jaxatari
    from jaxatari.environment import JAXAtariAction as Action

    env = jaxatari.make("seaquest")
    action_set = np.asarray(env.ACTION_SET)
    up_idxs = np.where(action_set == int(Action.UP))[0]
    if len(up_idxs) == 0:
        raise RuntimeError(f"UP not in ACTION_SET={action_set}")
    action_up = int(up_idxs[0])

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        rng = jax.random.PRNGKey(int(seed))
        _obs, state = env.reset(rng)
        active_seq: list[np.ndarray] = []
        for _t in range(frames):
            state = state.replace(oxygen=jnp.array(64, dtype=jnp.int32))
            diver_pos = np.asarray(state.diver_positions)
            active_seq.append(diver_pos[:, 2] != 0)
            _obs, state, _r, done, _info = env.step(state, action_up)
            if bool(done):
                rng, reset_key = jax.random.split(rng)
                _obs, state = env.reset(reset_key)

        stats = _track_from_active_mask(active_seq)
        stats["seed"] = seed
        runs.append(stats)
        print(
            f"[JAX] seed={seed} waves={stats['wave_events']} "
            f"lane_spawns={stats['lane_spawns']} "
            f"occ={stats['diver_frame_occupancy']} "
            f"first={stats['first_wave_frame']} "
            f"per_lane={stats['spawns_per_lane']}",
            flush=True,
        )
    return _summarize("JAXAtari", seeds, frames, runs), runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ale-sticky", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=Path("tmp/seaquest_diver_spawn_probe.json"))
    args = ap.parse_args()

    print(f"Probing {args.frames} frames × seeds={args.seeds}", flush=True)
    ale_stats, ale_runs = probe_ale(args.frames, args.seeds, sticky=args.ale_sticky)
    jax_stats, jax_runs = probe_jaxatari(args.frames, args.seeds)

    payload = {
        "frames": args.frames,
        "seeds": args.seeds,
        "ale_sticky": args.ale_sticky,
        "notes": {
            "oxygen_ram": ALE_OXYGEN_RAM,
            "oxygen_full": ALE_OXYGEN_FULL,
            "wave_event": "frame where ≥1 lane gains a new diver (onset)",
            "lane_spawn": "count of per-lane onsets (4 if all lanes appear together)",
            "occupancy": "sum over frames of (# active diver lanes)",
        },
        "ale": asdict(ale_stats),
        "jaxatari": asdict(jax_stats),
        "ale_runs": ale_runs,
        "jaxatari_runs": jax_runs,
        "ratios_jax_over_ale": {
            "lane_spawns": jax_stats.mean_lane_spawns / ale_stats.mean_lane_spawns
            if ale_stats.mean_lane_spawns
            else None,
            "wave_events": jax_stats.mean_wave_events / ale_stats.mean_wave_events
            if ale_stats.mean_wave_events
            else None,
            "occupancy": jax_stats.mean_occupancy / ale_stats.mean_occupancy
            if ale_stats.mean_occupancy
            else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Drop huge wave_frames lists from file for readability? Keep them — useful.
    args.out.write_text(json.dumps(payload, indent=2))

    print("\n=== Summary (means over seeds) ===")
    print(
        f"ALE      waves={ale_stats.mean_wave_events:.1f}  "
        f"lane_spawns={ale_stats.mean_lane_spawns:.1f}  "
        f"occ_frac={ale_stats.mean_occupancy_frac:.3f}  "
        f"first≈{ale_stats.mean_first_wave}  "
        f"interval≈{ale_stats.mean_wave_interval}"
    )
    print(
        f"JAXAtari waves={jax_stats.mean_wave_events:.1f}  "
        f"lane_spawns={jax_stats.mean_lane_spawns:.1f}  "
        f"occ_frac={jax_stats.mean_occupancy_frac:.3f}  "
        f"first≈{jax_stats.mean_first_wave}  "
        f"interval≈{jax_stats.mean_wave_interval}"
    )
    print("ratios JAX/ALE:", payload["ratios_jax_over_ale"])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
