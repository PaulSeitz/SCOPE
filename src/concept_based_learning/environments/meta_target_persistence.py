"""Config-driven meta-policy target persistence (post-selection override)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Override ``proposed_targets`` according to ``META_TARGET_PERSIST_MODE``.

    Returns ``(committed_targets, updated_persist_steps_remaining)``.
    """
    mode = meta_target_persist_mode(config)
    if mode is None:
        return proposed_targets, persist_steps_remaining

    valid_last = is_valid_persist_target(
        last_targets, config, no_target_idx=no_target_idx, max_objects=max_objects
    )
    disappeared = target_disappeared(objects, last_targets)
    last_labels = _labels_at(objects, last_targets)
    on_mission = jnp.logical_and(
        valid_last,
        jnp.logical_and(
            jnp.logical_not(disappeared),
            jnp.logical_or(
                last_labels == LABEL_COLLECT,
                last_labels == LABEL_DESTROY,
            ),
        ),
    )

    if mode == PERSIST_UNTIL_DONE:
        keep = jnp.logical_and(valid_last, jnp.logical_not(disappeared))
        committed = jnp.where(keep, last_targets, proposed_targets).astype(jnp.int32)
        new_steps = jnp.zeros_like(persist_steps_remaining)
        return committed, new_steps

    if mode == PERSIST_MISSION_AVOID:
        margin = float(config.get("META_AVOID_INTERRUPT_MARGIN", 0.0))
        interrupt, best_avoid_idx = _best_avoid_interrupt(
            meta_q_vals,
            active_masks,
            objects,
            current_target_idxs=last_targets,
            margin=margin,
        )
        keep_mission = jnp.logical_and(on_mission, jnp.logical_not(interrupt))
        committed = jnp.where(
            keep_mission,
            last_targets,
            jnp.where(interrupt, best_avoid_idx, proposed_targets),
        ).astype(jnp.int32)
        return committed, jnp.zeros_like(persist_steps_remaining)

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
        return committed, new_steps

    return proposed_targets, persist_steps_remaining
