"""Bootstrap cut masks for skill Q(λ) / TD targets at meta target switches."""

from __future__ import annotations

import jax.numpy as jnp

# Cut every relabel slot when the committed meta target switches (default).
SKILL_TARGET_CUT_GLOBAL = "global"
# Cut only slots for the old or new committed target at a switch.
SKILL_TARGET_CUT_INVOLVED = "involved"
# No meta-switch cuts; multi-step credit ends at skill_done / episode done only (HER lifetime).
SKILL_TARGET_CUT_LIFETIME = "lifetime"

DEFAULT_SKILL_TARGET_CUT = SKILL_TARGET_CUT_GLOBAL

_ALIASES = {
    "global": SKILL_TARGET_CUT_GLOBAL,
    "all": SKILL_TARGET_CUT_GLOBAL,
    "involved": SKILL_TARGET_CUT_INVOLVED,
    "involved_only": SKILL_TARGET_CUT_INVOLVED,
    "lifetime": SKILL_TARGET_CUT_LIFETIME,
    "her": SKILL_TARGET_CUT_LIFETIME,
    "her_lifetime": SKILL_TARGET_CUT_LIFETIME,
    "none": SKILL_TARGET_CUT_LIFETIME,
    "off": SKILL_TARGET_CUT_LIFETIME,
}


def normalize_skill_target_cut(mode: str) -> str:
    key = str(mode).lower().strip()
    if key not in _ALIASES:
        allowed = sorted({v for v in _ALIASES.values()})
        raise ValueError(
            f"Unknown SKILL_TARGET_CUT={mode!r}. "
            f"Use one of: {', '.join(allowed)} "
            f"(aliases: {', '.join(sorted(_ALIASES))})."
        )
    return _ALIASES[key]


def skill_target_switch_cut(
    target_idx,
    num_objects: int,
    mode: str,
):
    """
    Per-edge bootstrap cut mask for relabel-all skill training.

    Returns float32 array [T-1, E, O] where 1.0 means zero bootstrap across
    step t -> t+1 for object slot O.

    Modes:
    - ``global``: cut all slots at every committed target switch.
    - ``involved``: cut only slots equal to the old or new committed target.
    - ``lifetime``: no meta-switch cuts (HER lifetime; ``skill_done`` only).
    """
    mode = normalize_skill_target_cut(mode)
    committed_prev = target_idx[:-1]
    committed_next = target_idx[1:]
    switched = (committed_prev != committed_next).astype(jnp.float32)

    if mode == SKILL_TARGET_CUT_LIFETIME:
        return jnp.zeros(
            (*switched.shape, num_objects),
            dtype=jnp.float32,
        )

    if mode == SKILL_TARGET_CUT_INVOLVED:
        obj_ids = jnp.arange(num_objects, dtype=target_idx.dtype)[None, None, :]
        involved = jnp.logical_or(
            committed_prev[:, :, None] == obj_ids,
            committed_next[:, :, None] == obj_ids,
        )
        return (switched[:, :, None] * involved.astype(jnp.float32)).astype(jnp.float32)

    return jnp.broadcast_to(
        switched[:, :, None],
        (*switched.shape, num_objects),
    ).astype(jnp.float32)


def skill_target_switch_cut_padded(target_idx, num_objects: int, mode: str):
    """``skill_target_switch_cut`` with a zero row for rollout step T-1. Shape [T, E, O]."""
    cut = skill_target_switch_cut(target_idx, num_objects, mode)
    pad = jnp.zeros((1, cut.shape[1], cut.shape[2]), dtype=cut.dtype)
    return jnp.concatenate([cut, pad], axis=0)


def skill_target_switch_cut_env(target_idx, mode: str):
    """
    Per-edge cut for single-target (non-relabel) skill paths. Shape [T-1, E].

    ``involved`` matches ``global`` here because only the committed target trace
    is trained.
    """
    mode = normalize_skill_target_cut(mode)
    switched = (target_idx[:-1] != target_idx[1:]).astype(jnp.float32)
    if mode == SKILL_TARGET_CUT_LIFETIME:
        return jnp.zeros_like(switched)
    return switched


def skill_target_switch_cut_env_padded(target_idx, mode: str):
    """``skill_target_switch_cut_env`` padded to shape [T, E]."""
    cut = skill_target_switch_cut_env(target_idx, mode)
    pad = jnp.zeros((1, cut.shape[1]), dtype=cut.dtype)
    return jnp.concatenate([cut, pad], axis=0)
