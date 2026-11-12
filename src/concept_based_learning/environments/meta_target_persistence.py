"""Config-driven meta-policy target persistence (post-selection override)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_JUMP_OVER,
    LABEL_NAVIGATE,
)
from concept_based_learning.environments.meta_target_mask import (
    is_meta_fire_target,
    is_meta_noop_target,
)

PERSIST_UNTIL_DONE = "until_done"
PERSIST_MISSION_AVOID = "mission_avoid"
PERSIST_FIXED_STEPS = "fixed_steps"
PERSIST_MODES = (PERSIST_UNTIL_DONE, PERSIST_MISSION_AVOID, PERSIST_FIXED_STEPS)


def meta_target_persist_mode(config: Optional[dict]) -> Optional[str]:
    if not config:
        return None
    mode = config.get("META_TARGET_PERSIST_MODE")
    if mode is None or mode == "" or mode == "none":
        return None
    mode = str(mode)
    if mode not in PERSIST_MODES:
        raise ValueError(
            f"Unknown META_TARGET_PERSIST_MODE={mode!r}; "
            f"expected one of {PERSIST_MODES} or null."
        )
    return mode


def meta_target_persist_enabled(config: Optional[dict]) -> bool:
    return meta_target_persist_mode(config) is not None


def meta_avoid_resume_mission(config: Optional[dict]) -> bool:
    return bool(config and config.get("META_AVOID_RESUME_MISSION", False))


def meta_target_locked_input_enabled(config: Optional[dict]) -> bool:
    return bool(config and config.get("META_TARGET_LOCKED_INPUT", False))


def meta_q_stickiness_enabled(config: Optional[dict]) -> bool:
    return bool(config and config.get("META_Q_STICKINESS_ENABLED", False))


def meta_q_stickiness_threshold(config: Optional[dict]) -> float:
    if not config:
        return 0.0
    return float(config.get("META_Q_STICKINESS_THRESHOLD", 0.0))


def _current_target_q(
    meta_q_vals: jnp.ndarray, target_idxs: jnp.ndarray
) -> jnp.ndarray:
    safe = jnp.clip(target_idxs, 0, meta_q_vals.shape[-1] - 1)
    return jnp.take_along_axis(meta_q_vals, safe[:, None], axis=1).squeeze(1)


def _last_target_still_targetable(
    active_masks: jnp.ndarray, target_idxs: jnp.ndarray
) -> jnp.ndarray:
    safe = jnp.clip(target_idxs, 0, active_masks.shape[-1] - 1)
    return jnp.take_along_axis(active_masks, safe[:, None], axis=1).squeeze(1)


def apply_meta_q_stickiness(
    proposed_targets: jnp.ndarray,
    last_targets: jnp.ndarray,
    objects: Any,
    meta_q_vals: jnp.ndarray,
    active_masks: jnp.ndarray,
    config: Optional[dict],
    *,
    no_target_idx: int,
    max_objects: int,
) -> jnp.ndarray:
    """
    Learned-meta stickiness: keep ``last_targets`` while its Q-value stays at/above
    ``META_Q_STICKINESS_THRESHOLD``; otherwise allow ``proposed_targets``.

    Independent from ``META_TARGET_PERSIST_MODE`` and compatible with
    ``META_SWITCH_PENALTY`` (switches are still penalized once committed).
    """
    if not meta_q_stickiness_enabled(config):
        return proposed_targets

    threshold = jnp.asarray(
        meta_q_stickiness_threshold(config), dtype=meta_q_vals.dtype
    )
    valid_last = is_valid_persist_target(
        last_targets, config, no_target_idx=no_target_idx, max_objects=max_objects
    )
    disappeared = target_disappeared(objects, last_targets)
    still_targetable = _last_target_still_targetable(active_masks, last_targets)
    current_q = _current_target_q(meta_q_vals, last_targets)

    keep_sticky = jnp.logical_and(
        valid_last,
        jnp.logical_and(
            jnp.logical_not(disappeared),
            jnp.logical_and(still_targetable, current_q >= threshold),
        ),
    )
    return jnp.where(keep_sticky, last_targets, proposed_targets).astype(jnp.int32)


def commit_meta_targets(
    proposed_targets: jnp.ndarray,
    last_targets: jnp.ndarray,
    objects: Any,
    meta_q_vals: jnp.ndarray,
    active_masks: jnp.ndarray,
    config: Optional[dict],
    *,
    no_target_idx: int,
    max_objects: int,
    persist_steps_remaining: jnp.ndarray,
    suspended_mission_idxs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply learned Q stickiness, then optional heuristic persistence."""
    targets = apply_meta_q_stickiness(
        proposed_targets,
        last_targets,
        objects,
        meta_q_vals,
        active_masks,
        config,
        no_target_idx=no_target_idx,
        max_objects=max_objects,
    )
    if meta_target_persist_enabled(config):
        return apply_meta_target_persistence(
            targets,
            last_targets,
            objects,
            meta_q_vals,
            active_masks,
            config,
            no_target_idx=no_target_idx,
            max_objects=max_objects,
            persist_steps_remaining=persist_steps_remaining,
            suspended_mission_idxs=suspended_mission_idxs,
        )
    return targets, persist_steps_remaining, suspended_mission_idxs


def _object_field_latest(values: jnp.ndarray) -> jnp.ndarray:
    """
    Latest object-frame field per env.

    Supports:
      - (stack, max_objects) — single env (eval / inside vmap)
      - (num_envs, stack, max_objects) — batched tandem rollout
    """
    if values.ndim == 2:
        return values[-1]
    if values.ndim == 3:
        return values[:, -1, :]
    if values.ndim == 1:
        return values
    return values


def _gather_object_field(objects: Any, field: str, target_idxs: jnp.ndarray) -> jnp.ndarray:
    """Read ``field`` at ``target_idxs`` from the latest object frame."""
    values = getattr(objects, field)
    latest = _object_field_latest(values)
    if latest.ndim == 1:
        return latest[target_idxs]
    safe = jnp.clip(target_idxs, 0, latest.shape[-1] - 1)
    return jnp.take_along_axis(latest, safe[:, None], axis=1).squeeze(1)


def is_valid_persist_target(
    target_idxs: jnp.ndarray,
    config: Optional[dict],
    *,
    no_target_idx: int,
    max_objects: int,
) -> jnp.ndarray:
    """True for real game-object slots that can be locked."""
    in_range = jnp.logical_and(target_idxs >= 0, target_idxs < max_objects)
    not_sentinel = target_idxs != no_target_idx
    special = jnp.logical_or(
        is_meta_noop_target(target_idxs, config),
        is_meta_fire_target(target_idxs, config),
    )
    return jnp.logical_and(jnp.logical_and(in_range, not_sentinel), jnp.logical_not(special))


def target_disappeared(objects: Any, target_idxs: jnp.ndarray) -> jnp.ndarray:
    """
    Lenient cross-env disappearance signal.

    Release persistence when the object is inactive or a terminal success flag
    fires (``was_killed`` / ``was_collected``).
    """
    active = _gather_object_field(objects, "active", target_idxs)
    inactive = active <= 0.5
    was_killed = _gather_object_field(objects, "was_killed", target_idxs) > 0
    was_collected = _gather_object_field(objects, "was_collected", target_idxs) > 0
    return jnp.logical_or(inactive, jnp.logical_or(was_killed, was_collected))


def _labels_at(objects: Any, target_idxs: jnp.ndarray) -> jnp.ndarray:
    return _gather_object_field(objects, "semantic_label", target_idxs)


def _is_mission_target(objects: Any, target_idxs: jnp.ndarray) -> jnp.ndarray:
    labels = _labels_at(objects, target_idxs)
    return jnp.logical_or(
        labels == LABEL_COLLECT,
        jnp.logical_or(
            labels == LABEL_DESTROY,
            jnp.logical_or(labels == LABEL_NAVIGATE, labels == LABEL_JUMP_OVER),
        ),
    )


def _on_mission(
    last_targets: jnp.ndarray,
    objects: Any,
    config: Optional[dict],
    *,
    no_target_idx: int,
    max_objects: int,
) -> jnp.ndarray:
    valid_last = is_valid_persist_target(
        last_targets, config, no_target_idx=no_target_idx, max_objects=max_objects
    )
    disappeared = target_disappeared(objects, last_targets)
    return jnp.logical_and(
        valid_last,
        jnp.logical_and(
            jnp.logical_not(disappeared),
            _is_mission_target(objects, last_targets),
        ),
    )


def _suspended_mission_valid(
    suspended_mission_idxs: jnp.ndarray,
    objects: Any,
    config: Optional[dict],
    *,
    no_target_idx: int,
    max_objects: int,
) -> jnp.ndarray:
    valid = is_valid_persist_target(
        suspended_mission_idxs,
        config,
        no_target_idx=no_target_idx,
        max_objects=max_objects,
    )
    disappeared = target_disappeared(objects, suspended_mission_idxs)
    return jnp.logical_and(valid, jnp.logical_not(disappeared))


def compute_meta_target_locked(
    last_targets: jnp.ndarray,
    suspended_mission_idxs: jnp.ndarray,
    objects: Any,
    config: Optional[dict],
    *,
    no_target_idx: int,
    max_objects: int,
    persist_steps_remaining: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-env flag (0/1) appended to meta-policy input when ``META_TARGET_LOCKED_INPUT``.

    True when persistence will override free target selection on this step.
    """
    if not meta_target_locked_input_enabled(config):
        return jnp.zeros_like(last_targets, dtype=jnp.float32)

    mode = meta_target_persist_mode(config)
    if mode is None and not meta_q_stickiness_enabled(config):
        return jnp.zeros_like(last_targets, dtype=jnp.float32)

    valid_last = is_valid_persist_target(
        last_targets, config, no_target_idx=no_target_idx, max_objects=max_objects
    )
    disappeared = target_disappeared(objects, last_targets)

    on_detour = jnp.zeros_like(last_targets, dtype=bool)
    if mode == PERSIST_MISSION_AVOID and meta_avoid_resume_mission(config):
        on_detour = _suspended_mission_valid(
            suspended_mission_idxs,
            objects,
            config,
            no_target_idx=no_target_idx,
            max_objects=max_objects,
        )

    if mode is None:
        locked = jnp.zeros_like(last_targets, dtype=bool)
    elif mode == PERSIST_UNTIL_DONE:
        locked = jnp.logical_and(valid_last, jnp.logical_not(disappeared))
    elif mode == PERSIST_MISSION_AVOID:
        locked = jnp.logical_and(_on_mission(
            last_targets,
            objects,
            config,
            no_target_idx=no_target_idx,
            max_objects=max_objects,
        ), jnp.logical_not(on_detour))
    elif mode == PERSIST_FIXED_STEPS:
        locked = jnp.logical_and(
            persist_steps_remaining > 0,
            jnp.logical_and(valid_last, jnp.logical_not(disappeared)),
        )
    else:
        locked = jnp.zeros_like(last_targets, dtype=bool)

    if meta_q_stickiness_enabled(config):
        q_locked = jnp.logical_and(valid_last, jnp.logical_not(disappeared))
        locked = jnp.logical_or(locked, q_locked)

    return locked.astype(jnp.float32)


def _best_avoid_q(
    meta_q_vals: jnp.ndarray,
    active_masks: jnp.ndarray,
    objects: Any,
) -> jnp.ndarray:
    labels = _object_field_latest(objects.semantic_label)
    avoid_mask = jnp.logical_and(active_masks, labels == LABEL_AVOID)
    masked_q = jnp.where(avoid_mask, meta_q_vals, -jnp.inf)
    return jnp.max(masked_q, axis=-1)


def _best_avoid_interrupt(
    meta_q_vals: jnp.ndarray,
    active_masks: jnp.ndarray,
    objects: Any,
    *,
    current_target_idxs: jnp.ndarray,
    margin: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (should_interrupt, best_avoid_idx) using meta Q on AVOID slots only."""
    labels = _object_field_latest(objects.semantic_label)
    avoid_mask = jnp.logical_and(active_masks, labels == LABEL_AVOID)
    masked_q = jnp.where(avoid_mask, meta_q_vals, -jnp.inf)
    best_avoid_q = jnp.max(masked_q, axis=-1)
    best_avoid_idx = jnp.argmax(masked_q, axis=-1).astype(jnp.int32)
    has_avoid = jnp.any(avoid_mask, axis=-1)

    safe_current = jnp.clip(current_target_idxs, 0, meta_q_vals.shape[-1] - 1)
    current_q = jnp.take_along_axis(
        meta_q_vals, safe_current[:, None], axis=1
    ).squeeze(1)
    interrupt = jnp.logical_and(
        has_avoid,
        best_avoid_q > (current_q + jnp.asarray(margin, dtype=meta_q_vals.dtype)),
    )
    return interrupt, best_avoid_idx


def _should_resume_mission(
    meta_q_vals: jnp.ndarray,
    active_masks: jnp.ndarray,
    objects: Any,
    suspended_mission_idxs: jnp.ndarray,
    *,
    margin: float,
    no_target_idx: int,
    max_objects: int,
    config: Optional[dict],
) -> jnp.ndarray:
    """Resume when AVOID urgency drops below the suspended mission Q + margin."""
    suspended_valid = _suspended_mission_valid(
        suspended_mission_idxs,
        objects,
        config,
        no_target_idx=no_target_idx,
        max_objects=max_objects,
    )
    best_avoid_q = _best_avoid_q(meta_q_vals, active_masks, objects)
    safe_suspended = jnp.clip(
        suspended_mission_idxs, 0, meta_q_vals.shape[-1] - 1
    )
    suspended_q = jnp.take_along_axis(
        meta_q_vals, safe_suspended[:, None], axis=1
    ).squeeze(1)
    return jnp.logical_and(
        suspended_valid,
        best_avoid_q <= (suspended_q + jnp.asarray(margin, dtype=meta_q_vals.dtype)),
    )


def apply_meta_target_persistence(
    proposed_targets: jnp.ndarray,
    last_targets: jnp.ndarray,
    objects: Any,
    meta_q_vals: jnp.ndarray,
    active_masks: jnp.ndarray,
    config: Optional[dict],
    *,
    no_target_idx: int,
    max_objects: int,
    persist_steps_remaining: jnp.ndarray,
    suspended_mission_idxs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Override ``proposed_targets`` according to ``META_TARGET_PERSIST_MODE``.

    Returns ``(committed_targets, updated_persist_steps_remaining, suspended_mission_idxs)``.
    """
    mode = meta_target_persist_mode(config)
    if mode is None:
        return proposed_targets, persist_steps_remaining, suspended_mission_idxs

    valid_last = is_valid_persist_target(
        last_targets, config, no_target_idx=no_target_idx, max_objects=max_objects
    )
    disappeared = target_disappeared(objects, last_targets)
    on_mission = _on_mission(
        last_targets,
        objects,
        config,
        no_target_idx=no_target_idx,
        max_objects=max_objects,
    )

    if mode == PERSIST_UNTIL_DONE:
        keep = jnp.logical_and(valid_last, jnp.logical_not(disappeared))
        committed = jnp.where(keep, last_targets, proposed_targets).astype(jnp.int32)
        new_steps = jnp.zeros_like(persist_steps_remaining)
        cleared_suspended = jnp.full_like(suspended_mission_idxs, no_target_idx)
        return committed, new_steps, cleared_suspended

    if mode == PERSIST_MISSION_AVOID:
        margin = float(config.get("META_AVOID_INTERRUPT_MARGIN", 0.0))
        resume_enabled = meta_avoid_resume_mission(config)
        on_detour = jnp.zeros_like(last_targets, dtype=bool)
        should_resume = jnp.zeros_like(last_targets, dtype=bool)
        if resume_enabled:
            on_detour = _suspended_mission_valid(
                suspended_mission_idxs,
                objects,
                config,
                no_target_idx=no_target_idx,
                max_objects=max_objects,
            )
            should_resume = _should_resume_mission(
                meta_q_vals,
                active_masks,
                objects,
                suspended_mission_idxs,
                margin=margin,
                no_target_idx=no_target_idx,
                max_objects=max_objects,
                config=config,
            )

        interrupt, best_avoid_idx = _best_avoid_interrupt(
            meta_q_vals,
            active_masks,
            objects,
            current_target_idxs=last_targets,
            margin=margin,
        )
        keep_mission = jnp.logical_and(on_mission, jnp.logical_not(interrupt))

        mission_committed = jnp.where(
            keep_mission,
            last_targets,
            jnp.where(interrupt, best_avoid_idx, proposed_targets),
        ).astype(jnp.int32)
        detour_committed = jnp.where(
            should_resume,
            suspended_mission_idxs,
            proposed_targets,
        ).astype(jnp.int32)
        committed = jnp.where(on_detour, detour_committed, mission_committed).astype(
            jnp.int32
        )

        new_suspended = suspended_mission_idxs
        if resume_enabled:
            new_suspended = jnp.where(
                jnp.logical_and(
                    jnp.logical_and(jnp.logical_not(on_detour), on_mission),
                    interrupt,
                ),
                last_targets,
                new_suspended,
            )
            new_suspended = jnp.where(
                should_resume,
                jnp.full_like(new_suspended, no_target_idx),
                new_suspended,
            )
            adopted_new_mission = jnp.logical_and(
                on_detour,
                jnp.logical_and(
                    jnp.logical_not(should_resume),
                    _is_mission_target(objects, detour_committed),
                ),
            )
            new_suspended = jnp.where(
                adopted_new_mission,
                jnp.full_like(new_suspended, no_target_idx),
                new_suspended,
            )

        return committed, jnp.zeros_like(persist_steps_remaining), new_suspended

    if mode == PERSIST_FIXED_STEPS:
        persist_steps = int(config.get("META_TARGET_PERSIST_STEPS", 300))
        steps_init = jnp.full_like(persist_steps_remaining, persist_steps, dtype=jnp.int32)
        still_locked = jnp.logical_and(
            persist_steps_remaining > 0,
            jnp.logical_and(valid_last, jnp.logical_not(disappeared)),
        )
        committed = jnp.where(still_locked, last_targets, proposed_targets).astype(jnp.int32)
        adopted_lock = jnp.logical_and(
            jnp.logical_not(still_locked),
            is_valid_persist_target(
                committed, config, no_target_idx=no_target_idx, max_objects=max_objects
            ),
        )
        new_steps = jnp.where(
            still_locked,
            jnp.maximum(0, persist_steps_remaining - 1),
            jnp.where(adopted_lock, steps_init, jnp.zeros_like(persist_steps_remaining)),
        )
        cleared_suspended = jnp.full_like(suspended_mission_idxs, no_target_idx)
        return committed, new_steps, cleared_suspended

    return proposed_targets, persist_steps_remaining, suspended_mission_idxs
