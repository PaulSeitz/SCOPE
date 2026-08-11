"""
RTPT progress tracking for cluster training (nvidia-smi sync, ETA).

Training loops run inside ``jax.lax.scan``; call :func:`rtpt_step_callback` from
``jax.debug.callback`` once per update (see train_skills / train_tandem / train_meta_agent).
Start tracking from ``main.single_run`` via :func:`start_rtpt`` before ``block_until_ready``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_RTPT: Any = None
_STARTED: bool = False
_NUM_SEEDS: int = 1
_SEED_ACCUM: int = 0
_LOGICAL_UPDATES: int = 0
_HOST_UNITS: int = 0
_STEP_EVERY: int = 1


def num_updates_from_timesteps(
    total_timesteps: int,
    *,
    num_steps: int,
    num_envs: int,
) -> int:
    divisor = max(1, int(num_steps) * int(num_envs))
    return int(total_timesteps) // divisor


def max_training_updates(config: Dict[str, Any]) -> int:
    """PPO-style update count for the full run (curriculum phases sum to TOTAL_TIMESTEPS)."""
    return num_updates_from_timesteps(
        int(config.get("TOTAL_TIMESTEPS", 0)),
        num_steps=int(config.get("NUM_STEPS", 128)),
        num_envs=int(config.get("NUM_ENVS", 512)),
    )


def is_rtpt_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("RTPT_ENABLED", True))


def rtpt_step_every(config: Dict[str, Any]) -> int:
    """Host sync frequency: one RTPT step every N training updates (not env frames)."""
    return max(1, int(config.get("RTPT_STEP_EVERY", 1)))


def _experiment_name(config: Dict[str, Any]) -> str:
    custom = config.get("RTPT_EXPERIMENT_NAME")
    if custom is not None and str(custom).strip():
        return str(custom)
    alg = str(config.get("ALG_NAME", "train"))
    env = str(config.get("ENV_NAME", "env"))
    mode = str(config.get("TRAIN_MODE", ""))
    if mode:
        return f"{alg}_{env}_{mode}"
    return f"{alg}_{env}"


def start_rtpt_with_total_units(total_units: int, config: Dict[str, Any]) -> bool:
    """
    Start RTPT when progress is counted in explicit host-side units (e.g. eval episodes).

    ``max_iterations`` becomes ``ceil(total_units / RTPT_STEP_EVERY)``.
    Pair with :func:`rtpt_host_step` once per completed unit.
    """
    cfg = dict(config)
    cfg["TOTAL_TIMESTEPS"] = max(1, int(total_units))
    cfg["NUM_STEPS"] = 1
    cfg["NUM_ENVS"] = 1
    return start_rtpt(cfg)


def start_rtpt(config: Dict[str, Any]) -> bool:
    """Create and start RTPT for this run. Safe to call once per process."""
    global _RTPT, _STARTED, _NUM_SEEDS, _SEED_ACCUM, _LOGICAL_UPDATES, _HOST_UNITS, _STEP_EVERY

    if not is_rtpt_enabled(config):
        return False
    if _STARTED:
        return _RTPT is not None

    try:
        from rtpt import RTPT
    except ImportError:
        return False

    logical_updates = max_training_updates(config)
    if logical_updates <= 0:
        return False

    _STEP_EVERY = rtpt_step_every(config)
    # RTPT bar counts host steps; each step() = STEP_EVERY JAX updates.
    max_iterations = (logical_updates + _STEP_EVERY - 1) // _STEP_EVERY

    _NUM_SEEDS = max(1, int(config.get("NUM_SEEDS", 1)))
    _SEED_ACCUM = 0
    _LOGICAL_UPDATES = 0
    _HOST_UNITS = 0
    _RTPT = RTPT(
        name_initials=str(config.get("RTPT_NAME_INITIALS", "PS")),
        experiment_name=_experiment_name(config),
        max_iterations=max_iterations,
    )
    _RTPT.start()
    _STARTED = True
    return True


def rtpt_host_step() -> None:
    """Advance RTPT after one host-side unit (e.g. a completed eval episode)."""
    global _RTPT, _HOST_UNITS, _STEP_EVERY

    if _RTPT is None:
        return

    _HOST_UNITS += 1
    if _HOST_UNITS % _STEP_EVERY == 0:
        _RTPT.step()


def rtpt_step_callback(*_ignored) -> None:
    """Count completed training updates; call RTPT.step() every RTPT_STEP_EVERY updates.

    Accepts optional ignored args so callers can pass a JAX value into
    ``jax.debug.callback`` (a no-arg callback can be hoisted / run once).
    """
    global _RTPT, _NUM_SEEDS, _SEED_ACCUM, _LOGICAL_UPDATES, _STEP_EVERY

    if _RTPT is None:
        return

    _SEED_ACCUM += 1
    if _SEED_ACCUM < _NUM_SEEDS:
        return
    _SEED_ACCUM = 0

    _LOGICAL_UPDATES += 1
    if _LOGICAL_UPDATES % _STEP_EVERY == 0:
        _RTPT.step()


def reset_rtpt() -> None:
    """Clear module state (tests or successive runs in one interpreter)."""
    global _RTPT, _STARTED, _NUM_SEEDS, _SEED_ACCUM, _LOGICAL_UPDATES, _HOST_UNITS, _STEP_EVERY
    _RTPT = None
    _STARTED = False
    _NUM_SEEDS = 1
    _SEED_ACCUM = 0
    _LOGICAL_UPDATES = 0
    _HOST_UNITS = 0
    _STEP_EVERY = 1
