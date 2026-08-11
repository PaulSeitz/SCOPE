"""Env-aligned episode metrics from JAXAtari game state + semantic events.

Greedy paper-eval currently discards ``info`` and only returns score. This module
aggregates task-relevant counters from:

1. Unwrapped base env state (score, lives, divers, rescues, sector, …)
2. Semantic ``was_killed`` / ``was_collected`` stamps (when obs has objects)
3. Simple reward-sign counts as a weak proxy when state fields are missing

Designed to be closed over at JIT compile time via ``make_metric_fns(env_name, ...)``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

from concept_based_learning.training.episode_stats import unwrap_base_env_state

# Stable column names for CSV / aggregation.
EPISODE_METRIC_KEYS: tuple[str, ...] = (
    "env_kills",
    "env_collects",
    "env_rescues",
    "env_surface_events",
    "env_divers_max",
    "env_divers_final",
    "env_score_final",
    "env_lives_final",
    "env_life_losses",
    "env_sector_final",
    "env_player_score",
    "env_enemy_score",
    "env_player_points",
    "env_enemy_points",
    "env_player_sets",
    "env_enemy_sets",
    "env_chicken_y_min",
    "env_positive_reward_steps",
    "env_steps",
)


def _f32(x: Any, default: float = 0.0) -> jnp.ndarray:
    if x is None:
        return jnp.float32(default)
    return jnp.asarray(x, dtype=jnp.float32).reshape(())


def _get(obj: Any, name: str, default: float = 0.0) -> jnp.ndarray:
    if obj is None or not hasattr(obj, name):
        return jnp.float32(default)
    return _f32(getattr(obj, name), default)


def _lives(gs: Any) -> jnp.ndarray:
    if hasattr(gs, "lives"):
        return _f32(gs.lives)
    if hasattr(gs, "player_lives"):
        return _f32(gs.player_lives)
    return jnp.float32(0.0)


def _score(gs: Any) -> jnp.ndarray:
    if hasattr(gs, "score"):
        return _f32(gs.score)
    if hasattr(gs, "player_score") and not hasattr(gs, "enemy_score"):
        return _f32(gs.player_score)
    return jnp.float32(0.0)


def _semantic_event_sums(obs: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sum was_killed / was_collected on the latest stack frame (exclude noop pad)."""
    objects = getattr(obs, "objects", None)
    if objects is None:
        return jnp.float32(0.0), jnp.float32(0.0)

    def _sum_flag(name: str) -> jnp.ndarray:
        if not hasattr(objects, name):
            return jnp.float32(0.0)
        arr = getattr(objects, name)
        if arr is None:
            return jnp.float32(0.0)
        arr = jnp.asarray(arr)
        if arr.ndim >= 2:
            arr = arr[-1]
        # Index 0 is the NOOP / empty pad in CBL semantic layouts.
        if arr.shape[0] > 1:
            arr = arr[1:]
        return jnp.sum(arr.astype(jnp.float32))

    return _sum_flag("was_killed"), _sum_flag("was_collected")


def init_episode_metrics() -> dict[str, jnp.ndarray]:
    zeros = {k: jnp.float32(0.0) for k in EPISODE_METRIC_KEYS}
    zeros["env_chicken_y_min"] = jnp.float32(1.0e6)
    return zeros


def metrics_to_python(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in EPISODE_METRIC_KEYS:
        v = metrics.get(k, 0.0)
        out[k] = float(v)
    # Sentinel → empty progress when chicken never observed.
    if out["env_chicken_y_min"] >= 1.0e6 - 1.0:
        out["env_chicken_y_min"] = float("nan")
    return out


def aggregate_episode_metrics(
    episode_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Mean/std over episodes for each metric key (``mean_env_*`` / ``std_env_*``)."""
    import numpy as np

    agg: dict[str, float] = {}
    if not episode_metrics:
        return agg
    for k in EPISODE_METRIC_KEYS:
        vals = np.asarray([float(m.get(k, float("nan"))) for m in episode_metrics], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            agg[f"mean_{k}"] = float("nan")
            agg[f"std_{k}"] = float("nan")
            continue
        agg[f"mean_{k}"] = float(finite.mean())
        agg[f"std_{k}"] = float(finite.std()) if finite.size > 1 else 0.0
    return agg


def make_metric_fns(
    env_name: str,
    *,
    has_semantic_obs: bool = True,
) -> tuple[Callable, Callable]:
    """
    Return ``(update, finalize)`` closed over ``env_name``.

    ``update(metrics, prev_wrapper_state, next_wrapper_state, next_obs, reward)``
    ``finalize(metrics, final_wrapper_state)``
    """
    env = str(env_name or "").lower().strip()

    def update(metrics, prev_state, next_state, next_obs, reward):
        prev_gs = unwrap_base_env_state(prev_state)
        next_gs = unwrap_base_env_state(next_state)

        kills = jnp.float32(0.0)
        collects = jnp.float32(0.0)
        if has_semantic_obs:
            kills, collects = _semantic_event_sums(next_obs)

        life_loss = jnp.maximum(_lives(prev_gs) - _lives(next_gs), jnp.float32(0.0))
        pos_r = jnp.where(jnp.asarray(reward, dtype=jnp.float32) > 0, jnp.float32(1.0), jnp.float32(0.0))

        metrics = {
            **metrics,
            "env_kills": metrics["env_kills"] + kills,
            "env_collects": metrics["env_collects"] + collects,
            "env_life_losses": metrics["env_life_losses"] + life_loss,
            "env_positive_reward_steps": metrics["env_positive_reward_steps"] + pos_r,
            "env_steps": metrics["env_steps"] + jnp.float32(1.0),
            "env_score_final": _score(next_gs),
            "env_lives_final": _lives(next_gs),
        }

        if "seaquest" in env:
            divers = _get(next_gs, "divers_collected")
            rescues = _get(next_gs, "successful_rescues")
            prev_rescues = _get(prev_gs, "successful_rescues")
            rescue_delta = jnp.maximum(rescues - prev_rescues, jnp.float32(0.0))
            just_surfaced = _get(next_gs, "just_surfaced")
            surface_evt = jnp.where(just_surfaced > 0, jnp.float32(1.0), jnp.float32(0.0))
            metrics = {
                **metrics,
                "env_divers_max": jnp.maximum(metrics["env_divers_max"], divers),
                "env_divers_final": divers,
                "env_rescues": metrics["env_rescues"] + rescue_delta,
                "env_surface_events": metrics["env_surface_events"] + surface_evt,
            }
        elif "beamrider" in env:
            metrics = {
                **metrics,
                "env_sector_final": _get(next_gs, "sector"),
            }
        elif "freeway" in env:
            chicken_y = _get(next_gs, "chicken_y", default=1.0e6)
            metrics = {
                **metrics,
                "env_chicken_y_min": jnp.minimum(metrics["env_chicken_y_min"], chicken_y),
            }
        elif "pong" in env:
            metrics = {
                **metrics,
                "env_player_score": _get(next_gs, "player_score"),
                "env_enemy_score": _get(next_gs, "enemy_score"),
            }
        elif "tennis" in env:
            # Tennis nests points/sets under ``game_state``.
            gs = getattr(next_gs, "game_state", next_gs)
            metrics = {
                **metrics,
                "env_player_points": _get(gs, "player_score"),
                "env_enemy_points": _get(gs, "enemy_score"),
                "env_player_sets": _get(gs, "player_game_score"),
                "env_enemy_sets": _get(gs, "enemy_game_score"),
                "env_player_score": _get(next_gs, "player_points"),
                "env_enemy_score": _get(next_gs, "enemy_points"),
            }
        elif "asterix" in env:
            # Score/lives already tracked; kills/collects come from semantic stamps.
            pass

        return metrics

    def finalize(metrics, final_state):
        gs = unwrap_base_env_state(final_state)
        metrics = {
            **metrics,
            "env_score_final": _score(gs),
            "env_lives_final": _lives(gs),
        }
        if "seaquest" in env:
            metrics = {
                **metrics,
                "env_divers_final": _get(gs, "divers_collected"),
                # Prefer cumulative rescue increments; fall back to final counter.
                "env_rescues": jnp.maximum(
                    metrics["env_rescues"], _get(gs, "successful_rescues")
                ),
            }
        elif "beamrider" in env:
            metrics = {**metrics, "env_sector_final": _get(gs, "sector")}
        elif "freeway" in env:
            metrics = {
                **metrics,
                "env_chicken_y_min": jnp.minimum(
                    metrics["env_chicken_y_min"], _get(gs, "chicken_y", default=1.0e6)
                ),
            }
        elif "pong" in env:
            metrics = {
                **metrics,
                "env_player_score": _get(gs, "player_score"),
                "env_enemy_score": _get(gs, "enemy_score"),
            }
        elif "tennis" in env:
            nested = getattr(gs, "game_state", gs)
            metrics = {
                **metrics,
                "env_player_points": _get(nested, "player_score"),
                "env_enemy_points": _get(nested, "enemy_score"),
                "env_player_sets": _get(nested, "player_game_score"),
                "env_enemy_sets": _get(nested, "enemy_game_score"),
            }
        return metrics

    return update, finalize


# ---------------------------------------------------------------------------
# Batched training helpers (NUM_ENVS × K metric matrix)
# ---------------------------------------------------------------------------

NUM_EPISODE_METRICS = len(EPISODE_METRIC_KEYS)
_METRIC_IDX = {k: i for i, k in enumerate(EPISODE_METRIC_KEYS)}


def init_metric_matrix(num_envs: int) -> jnp.ndarray:
    """``[num_envs, K]`` running / returned episode metric buffer."""
    m = jnp.zeros((num_envs, NUM_EPISODE_METRICS), dtype=jnp.float32)
    return m.at[:, _METRIC_IDX["env_chicken_y_min"]].set(1.0e6)


def make_vmapped_metric_update(
    env_name: str,
    *,
    has_semantic_obs: bool = True,
):
    """Return ``update(running, prev_state, next_state, next_obs, reward)`` vmapped over envs."""
    update_one, _ = make_metric_fns(env_name, has_semantic_obs=has_semantic_obs)

    def _row(row, prev_state, next_state, next_obs, reward):
        metrics = {k: row[i] for i, k in enumerate(EPISODE_METRIC_KEYS)}
        metrics = update_one(metrics, prev_state, next_state, next_obs, reward)
        return jnp.stack([metrics[k] for k in EPISODE_METRIC_KEYS]).astype(jnp.float32)

    return jax.vmap(_row)


def latch_and_reset_metric_matrix(
    running: jnp.ndarray,
    returned: jnp.ndarray,
    done: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """On ``done``, copy running → returned and reset running rows."""
    done_b = done.astype(jnp.bool_)[:, None]
    returned = jnp.where(done_b, running, returned)
    reset = init_metric_matrix(running.shape[0])
    running = jnp.where(done_b, reset, running)
    return running, returned


def metric_matrix_to_log_dict(
    returned: jnp.ndarray,
    *,
    prefix: str = "env/",
) -> dict[str, jnp.ndarray]:
    """Mean over env slots of last completed full-game metrics."""
    out: dict[str, jnp.ndarray] = {}
    for i, key in enumerate(EPISODE_METRIC_KEYS):
        short = key[4:] if key.startswith("env_") else key
        vals = returned[:, i]
        if key == "env_chicken_y_min":
            finite = jnp.where(vals < 1.0e6 - 1.0, vals, jnp.nan)
            out[f"{prefix}{short}"] = jnp.nanmean(finite)
        else:
            out[f"{prefix}{short}"] = jnp.mean(vals)
    return out


def append_training_metrics_csv(
    path: str | Path,
    log_dict: dict[str, Any],
) -> None:
    """Append one training-update row (scalar metrics) next to checkpoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "env/step",
        "update_steps",
        "charts/avg_episode_return",
        "charts/avg_game_return",
        *[
            f"env/{(k[4:] if k.startswith('env_') else k)}"
            for k in EPISODE_METRIC_KEYS
        ],
    ]
    row = {k: log_dict[k] for k in fieldnames if isinstance(log_dict.get(k), (int, float))}
    if not row:
        return

    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
