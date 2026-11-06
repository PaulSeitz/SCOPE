"""
RTPT progress tracking for cluster training (nvidia-smi sync, ETA).

Training loops run inside ``jax.lax.scan``; call :func:`rtpt_step_callback` from
``jax.debug.callback`` once per update (see train_skills / train_tandem / train_meta_agent).
Start tracking from ``main.single_run`` via :func:`start_rtpt`` before ``block_until_ready``.

Paper-eval suites mix ego (fast) and pixel (slow) jobs plus evals that do not
emit training updates. Use :func:`set_rtpt_job_weight` so one pixel update
consumes several RTPT units, and :func:`rtpt_add_units` during eval.
"""

from __future__ import annotations

from typing import Any, Dict

_RTPT: Any = None
_STARTED: bool = False
_NUM_SEEDS: int = 1
_SEED_ACCUM: int = 0
_LOGICAL_UPDATES: int = 0
_HOST_UNITS: int = 0
_STEP_EVERY: int = 1
_JOB_WEIGHT: int = 1

# Calibrated on A6000 paper_eval: ego 25M ~5–45 min, pixel 25M ~10–11 h (~10–13×).
RTPT_WEIGHT_EGO = 1
RTPT_WEIGHT_OC = 2
RTPT_WEIGHT_PIXEL = 10
# One parallel eval wave (all mods at once) ≈ this many unweighted ego-train updates.
# Serial single-mod evals still multiply by the mod count in run_paper_eval.
RTPT_EVAL_UNITS_PER_MOD = 25


def rtpt_weight_for_input_mode(mode: str, *, object_centric: bool | None = None) -> int:
    m = str(mode or "").lower().strip()
    if m.startswith("pixel") or m in {"pixel_embed", "pixel_embed_cond"}:
        return RTPT_WEIGHT_PIXEL
    if m.startswith("oc") or m in {"oc_embed", "object_embed", "object_embedding"}:
        return RTPT_WEIGHT_OC
    if object_centric is False:
        return RTPT_WEIGHT_PIXEL
    if object_centric is True and not m:
        return RTPT_WEIGHT_OC
    return RTPT_WEIGHT_EGO


def rtpt_weight_for_config(config: Dict[str, Any], *, kind: str = "") -> int:
    """Wall-time weight for one training update of this job (ego=1, oc=2, pixel=10)."""
    kind = str(kind or config.get("PAPER_EVAL_KIND") or "").lower().strip()
    if kind == "flat_pqn":
        oc = config.get("PAPER_EVAL_OBJECT_CENTRIC", config.get("OBJECT_CENTRIC", True))
        return RTPT_WEIGHT_OC if bool(oc) else RTPT_WEIGHT_PIXEL

    skill = str(config.get("SKILL_INPUT_MODE", "") or "")
    meta = str(config.get("META_INPUT_MODE", "") or "")
    return max(
        rtpt_weight_for_input_mode(skill),
        rtpt_weight_for_input_mode(meta),
    )


def set_rtpt_job_weight(weight: int) -> None:
    """How many RTPT units one training update is worth for the current job."""
    global _JOB_WEIGHT
    _JOB_WEIGHT = max(1, int(weight))


def rtpt_job_weight() -> int:
    return _JOB_WEIGHT


def rtpt_add_units(n: int) -> None:
    """Advance the bar by ``n`` units (eval mods, weighted train leftovers)."""
    global _RTPT, _LOGICAL_UPDATES, _STEP_EVERY
    if _RTPT is None or n <= 0:
        return
    _LOGICAL_UPDATES += int(n)
    while _LOGICAL_UPDATES >= _STEP_EVERY:
        _LOGICAL_UPDATES -= _STEP_EVERY
        _RTPT.step()


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
    """Host sync frequency: one RTPT step every N training *units* (weighted)."""
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
    # RTPT bar counts host steps; each step() = STEP_EVERY weighted units.
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
    rtpt_add_units(1)


def rtpt_step_callback(*_ignored) -> None:
    """Count completed training updates; call RTPT.step() every RTPT_STEP_EVERY units.

    Accepts optional ignored args so callers can pass a JAX value into
    ``jax.debug.callback`` (a no-arg callback can be hoisted / run once).
    """
    global _RTPT, _NUM_SEEDS, _SEED_ACCUM

    if _RTPT is None:
        return

    _SEED_ACCUM += 1
    if _SEED_ACCUM < _NUM_SEEDS:
        return
    _SEED_ACCUM = 0

    rtpt_add_units(_JOB_WEIGHT)


def reset_rtpt() -> None:
    """Clear module state (tests or successive runs in one interpreter)."""
    global _RTPT, _STARTED, _NUM_SEEDS, _SEED_ACCUM, _LOGICAL_UPDATES, _HOST_UNITS, _STEP_EVERY
    global _JOB_WEIGHT
    _RTPT = None
    _STARTED = False
    _NUM_SEEDS = 1
    _SEED_ACCUM = 0
    _LOGICAL_UPDATES = 0
    _HOST_UNITS = 0
    _STEP_EVERY = 1
    _JOB_WEIGHT = 1
