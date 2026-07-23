"""Shared wandb-facing metric helpers for skill outcomes and env info."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np


def resolve_skill_names(config: Mapping[str, Any], num_skills: int) -> list[str]:
    """Human-readable skill names for wandb keys (e.g. DESTROY, COLLECT)."""
    from concept_based_learning.environments.skill_rewards import SKILL_REWARD_ORDER

    raw = config.get("SKILL_NAMES")
    if raw is None or len(raw) != num_skills:
        return [
            SKILL_REWARD_ORDER[i].upper()
            if i < len(SKILL_REWARD_ORDER)
            else f"skill_{i}"
            for i in range(num_skills)
        ]
    return [str(name) for name in raw]


def info_log_metrics(infos: Any) -> dict[str, jnp.ndarray]:
    """
    Flatten env ``info`` pytrees into scalar ``info/<field>`` means.

    Supports flax ``struct.dataclass`` infos (SeaquestInfo, FreewayInfo, …)
    and plain dicts. Non-numeric leaves are skipped.
    """
    if infos is None:
        return {}

    if isinstance(infos, dict):
        items = infos.items()
    elif hasattr(infos, "__dataclass_fields__"):
        items = ((k, getattr(infos, k)) for k in infos.__dataclass_fields__)
    else:
        return {}

    metrics: dict[str, jnp.ndarray] = {}
    for key, value in items:
        try:
            arr = jnp.asarray(value)
        except TypeError:
            continue
        if arr.size == 0:
            continue
        if not (jnp.issubdtype(arr.dtype, jnp.number) or arr.dtype == jnp.bool_):
            continue
        metrics[f"info/{key}"] = jnp.mean(arr.astype(jnp.float32))
    return metrics


def player_progress_metrics(
    raw_observation: Any,
    *,
    screen_height: float = 210.0,
) -> dict[str, jnp.ndarray]:
    """Log player vertical progress for climb-style envs (Kangaroo, …).

    Atari y grows downward, so smaller ``player_y`` means higher on screen.
    ``altitude = screen_height - y`` is logged so higher values = climbed further.

    Accepts unbatched ``[stack]``, batched ``[N, stack]``, or rollout
    ``[T, N, stack]`` trees (as in tandem transitions).
    """
    from concept_based_learning.observations.egocentric import _get_player

    try:
        player = _get_player(raw_observation)
    except AttributeError:
        return {}
    y = jnp.asarray(player.y, dtype=jnp.float32)
    # Frame stack is the trailing axis.
    y_last = y[..., -1] if y.ndim >= 1 else y
    altitude = jnp.asarray(screen_height, dtype=jnp.float32) - y_last
    metrics = {
        "env/player_y_mean": jnp.mean(y_last),
        "env/player_altitude_mean": jnp.mean(altitude),
        "env/player_altitude_max": jnp.max(altitude),
    }
    # Rollout peak per env then average (axis 0 = time).
    if altitude.ndim >= 2:
        metrics["env/player_altitude_peak_mean"] = jnp.mean(jnp.max(altitude, axis=0))
    return metrics


def _recover_pure_skill_rewards(
    skill_rewards: jnp.ndarray,
    env_reward: jnp.ndarray | None,
    skill_reward_lambda: float,
) -> jnp.ndarray:
    """Undo env-reward blending so reward sign reflects skill success/failure."""
    lam = float(skill_reward_lambda)
    if env_reward is None or lam <= 1e-8:
        return skill_rewards
    if lam >= 1.0 - 1e-8:
        # Pure env reward — cannot recover skill outcome polarity.
        return skill_rewards
    env_b = env_reward[..., None]
    return (skill_rewards - lam * env_b) / (1.0 - lam)


def selected_skill_outcome_metrics(
    active_skill: jnp.ndarray,
    skill_done: jnp.ndarray,
    skill_rewards: jnp.ndarray,
    skill_names: Sequence[str],
    *,
    valid_mask: jnp.ndarray | None = None,
    env_reward: jnp.ndarray | None = None,
    skill_reward_lambda: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """
    Per-skill success/failure rates for the *selected* skill.

    A step counts as a selected-skill success when the active skill terminates
    with a positive skill reward, and as a failure when it terminates with a
    negative skill reward. Rates are normalized by how often each skill was
    selected (valid meta targets only), so meta misalignment shows up as high
    ``selected_frac`` with low ``selected_success_rate``.
    """
    from concept_based_learning.environments.datatypes import project_skill_outcomes

    num_skills = len(skill_names)
    pure_rewards = _recover_pure_skill_rewards(
        skill_rewards, env_reward, skill_reward_lambda
    )
    # Reward extractors always emit the full skill vocabulary; project onto the
    # configured NUM_SKILLS heads (folds NAVIGATE→COLLECT / JUMP_OVER→AVOID).
    pure_rewards, skill_done = project_skill_outcomes(
        pure_rewards, skill_done, num_skills
    )

    if valid_mask is None:
        valid_mask = jnp.ones(active_skill.shape, dtype=jnp.bool_)

    selected = jnp.logical_and(
        valid_mask[..., None],
        active_skill[..., None] == jnp.arange(num_skills, dtype=active_skill.dtype),
    )
    terminal = jnp.logical_and(selected, skill_done)
    success = jnp.logical_and(terminal, pure_rewards > 0)
    failure = jnp.logical_and(terminal, pure_rewards < 0)

    selected_count = jnp.sum(selected.astype(jnp.float32), axis=(0, 1))
    success_count = jnp.sum(success.astype(jnp.float32), axis=(0, 1))
    failure_count = jnp.sum(failure.astype(jnp.float32), axis=(0, 1))
    total_steps = jnp.asarray(active_skill.size, dtype=jnp.float32)
    eps = jnp.asarray(1e-8, dtype=jnp.float32)

    metrics: dict[str, jnp.ndarray] = {
        "skill/selected_success_frac": jnp.mean(
            jnp.any(success, axis=-1).astype(jnp.float32)
        ),
        "skill/selected_failure_frac": jnp.mean(
            jnp.any(failure, axis=-1).astype(jnp.float32)
        ),
    }
    for i, name in enumerate(skill_names):
        prefix = f"skill_{name}"
        sel = selected_count[i]
        metrics[f"{prefix}/selected_frac"] = sel / (total_steps + eps)
        metrics[f"{prefix}/selected_success_rate"] = success_count[i] / (sel + eps)
        metrics[f"{prefix}/selected_failure_rate"] = failure_count[i] / (sel + eps)
        metrics[f"{prefix}/selected_success_count"] = success_count[i]
        metrics[f"{prefix}/selected_failure_count"] = failure_count[i]
    return metrics


def flag_combo_label(combo_id: int, flag_names: Sequence[str], n_levels: Sequence[int]) -> str:
    """Human-readable combo label, e.g. ``los_blocked=1|weapon_state=-1``."""
    parts = []
    remaining = int(combo_id)
    for name, levels in zip(flag_names, n_levels):
        digit = remaining % int(levels)
        remaining //= int(levels)
        # Ternary weapon_state is stored as {0,1,2} ↔ {-1,0,+1}.
        if name == "weapon_state" and int(levels) == 3:
            parts.append(f"{name}={digit - 1}")
        else:
            parts.append(f"{name}={digit}")
    return "|".join(parts) if parts else "none"


def extract_flag_combo_ids(
    ego_obs: jnp.ndarray,
) -> tuple[jnp.ndarray, list[str], list[int]]:
    """
    Pack latest-frame discrete ego flags into a mixed-radix combination index.

    ``ego_obs`` may be shaped ``[..., ego_dim]`` (flattened frame stack). Returns
    combo ids with the same leading shape, ordered flag names (digit 0 first),
    and per-flag alphabet sizes. When no discrete flags are configured, returns
    zeros and empty lists.
    """
    from concept_based_learning.observations.egocentric import (
        discrete_ego_flag_layout,
        ego_obs_channels_per_frame,
    )

    layout = discrete_ego_flag_layout()
    flag_names = [name for name, _, _, _ in layout]
    n_levels = [levels for _, _, levels, _ in layout]
    if not layout:
        return jnp.zeros(ego_obs.shape[:-1], dtype=jnp.int32), flag_names, n_levels

    channels_per_frame = int(ego_obs_channels_per_frame())
    last_frame = ego_obs[..., -channels_per_frame:]
    indices = jnp.asarray([idx for _, idx, _, _ in layout], dtype=jnp.int32)
    offsets = jnp.asarray([off for _, _, _, off in layout], dtype=jnp.float32)
    levels = jnp.asarray(n_levels, dtype=jnp.int32)
    flag_vals = last_frame[..., indices]
    # Map raw channel → discrete level in [0, n_levels).
    digits = jnp.round(flag_vals + offsets).astype(jnp.int32)
    digits = jnp.maximum(digits, 0)
    digits = jnp.minimum(digits, levels - 1)
    # Mixed-radix pack: digit0 + digit1*L0 + digit2*L0*L1 + ...
    strides = []
    stride = 1
    for lv in n_levels:
        strides.append(stride)
        stride *= int(lv)
    stride_arr = jnp.asarray(strides, dtype=jnp.int32)
    combo_ids = jnp.sum(digits * stride_arr, axis=-1).astype(jnp.int32)
    return combo_ids, flag_names, n_levels


def flag_combination_occupancy_metrics(
    ego_obs: jnp.ndarray,
    active_skill: jnp.ndarray,
    skill_names: Sequence[str],
    *,
    valid_mask: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """
    Occupancy histogram of discrete ego flag combinations, per skill and overall.

    For thesis coverage claims: ``explored_frac`` is the fraction of the
    joint cells (product of per-flag ``n_levels``) with at least one visit in
    this batch. Counts are logged under ``*/flag_combo_counts`` for wandb.
    """
    combo_ids, flag_names, n_levels = extract_flag_combo_ids(ego_obs)
    n_flags = len(flag_names)
    if n_flags == 0:
        return {}

    num_combos = 1
    for lv in n_levels:
        num_combos *= int(lv)
    if valid_mask is None:
        valid_mask = jnp.ones(active_skill.shape, dtype=jnp.bool_)

    flat_ids = combo_ids.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    flat_skill = active_skill.reshape(-1)
    eps = jnp.asarray(1e-8, dtype=jnp.float32)

    def _bincount(mask: jnp.ndarray) -> jnp.ndarray:
        # Scatter into fixed bins (jit-friendly; avoids host-side bincount length).
        one_hot = jnp.arange(num_combos, dtype=jnp.int32)[None, :] == flat_ids[:, None]
        return jnp.sum(
            jnp.logical_and(mask[:, None], one_hot).astype(jnp.float32), axis=0
        )

    overall_counts = _bincount(flat_valid)
    overall_total = jnp.sum(overall_counts) + eps
    metrics: dict[str, jnp.ndarray] = {
        "skill/flag_combo_counts": overall_counts,
        "skill/flag_combo_explored_frac": jnp.mean(
            (overall_counts > 0).astype(jnp.float32)
        ),
        "skill/flag_combo_n_flags": jnp.asarray(float(n_flags), dtype=jnp.float32),
        "skill/flag_combo_n_combos": jnp.asarray(float(num_combos), dtype=jnp.float32),
    }
    for c in range(num_combos):
        label = flag_combo_label(c, flag_names, n_levels)
        metrics[f"skill/flag_combo_frac/{label}"] = overall_counts[c] / overall_total

    for i, name in enumerate(skill_names):
        prefix = f"skill_{name}"
        skill_mask = jnp.logical_and(flat_valid, flat_skill == i)
        counts = _bincount(skill_mask)
        total = jnp.sum(counts) + eps
        metrics[f"{prefix}/flag_combo_counts"] = counts
        metrics[f"{prefix}/flag_combo_explored_frac"] = jnp.mean(
            (counts > 0).astype(jnp.float32)
        )
        for c in range(num_combos):
            label = flag_combo_label(c, flag_names, n_levels)
            metrics[f"{prefix}/flag_combo_frac/{label}"] = counts[c] / total

    return metrics


# Host-side cumulative occupancy across updates (for "fully explored" thesis plots).
_CUM_FLAG_COMBO_COUNTS: dict[str, np.ndarray] = {}


def reset_flag_combo_cumulative() -> None:
    """Clear cumulative flag-combo counters (call at run start)."""
    _CUM_FLAG_COMBO_COUNTS.clear()


def accumulate_flag_combo_counts(
    metrics: Mapping[str, Any],
) -> dict[str, float | np.ndarray]:
    """
    Fold batch ``*/flag_combo_counts`` into cumulative totals and return extras.

    Returns scalar ``*/flag_combo_cum_explored_frac`` plus ``*/flag_combo_cum_counts``
    arrays for wandb histogram logging.
    """
    extras: dict[str, float | np.ndarray] = {}
    for key, value in metrics.items():
        if not str(key).endswith("/flag_combo_counts"):
            continue
        counts = np.asarray(value, dtype=np.float64)
        if counts.ndim == 2:
            # Multi-seed: sum across seeds.
            counts = counts.sum(axis=0)
        counts = counts.reshape(-1)
        prefix = str(key)[: -len("/flag_combo_counts")]
        prev = _CUM_FLAG_COMBO_COUNTS.get(prefix)
        if prev is None or prev.shape != counts.shape:
            cum = counts.copy()
        else:
            cum = prev + counts
        _CUM_FLAG_COMBO_COUNTS[prefix] = cum
        extras[f"{prefix}/flag_combo_cum_counts"] = cum.astype(np.float32)
        extras[f"{prefix}/flag_combo_cum_explored_frac"] = float(np.mean(cum > 0))
    return extras


def metrics_to_wandb_log_dict(
    metrics: Mapping[str, Any],
    *,
    accumulate_flag_combos: bool = True,
) -> dict[str, Any]:
    """
    Convert JAX/numpy training metrics into a wandb-friendly log dict.

    Scalar leaves become floats. ``*/flag_combo_counts`` (and cumulative twins)
    become ``wandb.Histogram`` bar charts. Optionally accumulates flag occupancy
    across calls for ``*/flag_combo_cum_*`` thesis coverage curves.
    """
    try:
        import wandb
    except ImportError:
        wandb = None  # type: ignore

    log_dict: dict[str, Any] = {}
    for key, value in metrics.items():
        value_np = np.asarray(value)
        key_s = str(key)
        if key_s.endswith("/flag_combo_counts") or key_s.endswith(
            "/flag_combo_cum_counts"
        ):
            counts = value_np
            if counts.ndim == 2:
                counts = counts.sum(axis=0)
            counts = counts.reshape(-1).astype(np.float64)
            if wandb is not None:
                edges = np.arange(counts.size + 1, dtype=np.float64)
                log_dict[key_s.replace("_counts", "_hist")] = wandb.Histogram(
                    np_histogram=(counts, edges)
                )
            # Keep a scalar mean count for CSV/backends that drop histograms.
            log_dict[f"{key_s}_mean"] = float(counts.mean()) if counts.size else 0.0
            continue
        if value_np.size == 1:
            log_dict[key_s] = float(value_np.ravel()[0])
        else:
            log_dict[key_s] = float(value_np.mean())

    if accumulate_flag_combos:
        cum = accumulate_flag_combo_counts(metrics)
        for key, value in cum.items():
            if str(key).endswith("/flag_combo_cum_counts"):
                counts = np.asarray(value, dtype=np.float64).reshape(-1)
                if wandb is not None:
                    edges = np.arange(counts.size + 1, dtype=np.float64)
                    log_dict[str(key).replace("_counts", "_hist")] = wandb.Histogram(
                        np_histogram=(counts, edges)
                    )
                log_dict[f"{key}_mean"] = float(counts.mean()) if counts.size else 0.0
            else:
                log_dict[str(key)] = float(value)

    return log_dict
