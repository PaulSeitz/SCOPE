"""Q-decomposition composition primitives.

Pure-JAX building blocks for composing per-slot ego-Q vectors into a single
global game-action Q vector, plus helpers for slot selection and inverting
the executed global action back to per-slot ego actions.

All functions operate on a single environment (no leading num_envs axis) and
are designed to be jit/vmap-friendly. The training loops wrap them with an
outer vmap over the num_envs axis.

Slot axis convention: ``K`` is the maximum number of objects composed per
frame (config: ``COMPOSITION_K``). Inactive slots are masked via
``slot_active`` and contribute zero weight after the masked softmax.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)


# ---------------------------------------------------------------------------
# Scatter: ego Q -> global game-action Q
# ---------------------------------------------------------------------------

def ego_q_to_global_q(
    ego_q: jnp.ndarray,
    raw_obs,
    target_idx: jnp.ndarray,
    objects,
    action_mapper: Callable,
    num_game_actions: int,
) -> jnp.ndarray:
    """Project an ego Q vector into the global game-action space.

    Multiple ego actions can collapse to the same game action (e.g. FORWARD
    and RIGHT when the target is directly to the right). We take the max
    across the preimage of each game action — it preserves "the best ego
    action that resolves to this game action".

    Game actions in the image of no ego action get -inf so they cannot win
    an argmax and contribute nothing to a sum after softmax weighting.

    Parameters
    ----------
    ego_q : float32[EGO_ACTION_DIM]
    raw_obs, target_idx, objects : passed through to ``action_mapper``
    action_mapper : signature ``(ego_action, raw_obs, target_idx, objects) -> game_action``
    num_game_actions : size of the global action space

    Returns
    -------
    global_q : float32[num_game_actions]
    """
    ego_dim = ego_q.shape[-1]
    ego_actions = jnp.arange(ego_dim, dtype=jnp.int32)

    game_action_per_ego = jax.vmap(
        lambda e: action_mapper(e, raw_obs, target_idx, objects)
    )(ego_actions)

    neg_inf = jnp.full((num_game_actions,), -jnp.inf, dtype=ego_q.dtype)
    # segment_max with an initial -inf baseline: for each game action g, take
    # max of ego_q over ego actions whose mapper output equals g.
    global_q = jax.ops.segment_max(
        ego_q,
        game_action_per_ego,
        num_segments=num_game_actions,
    )
    # segment_max fills empty segments with the smallest representable value
    # for the dtype, which is *not* -inf on all platforms; force -inf where
    # no ego action mapped to g.
    has_any = jax.ops.segment_max(
        jnp.ones_like(ego_q),
        game_action_per_ego,
        num_segments=num_game_actions,
    ) > 0
    global_q = jnp.where(has_any, global_q, neg_inf)
    return global_q


# ---------------------------------------------------------------------------
# Slot selection
# ---------------------------------------------------------------------------

# Order of label slots in per_label mode: must match label_to_skill_idx
# (skill 0 = DESTROY, 1 = COLLECT, 2 = AVOID), so slot k uses skill k.
_PER_LABEL_LABELS = jnp.array([LABEL_DESTROY, LABEL_COLLECT, LABEL_AVOID], dtype=jnp.int32)


def _label_to_skill_idx(label: jnp.ndarray) -> jnp.ndarray:
    """Match the local convention in train_skills.py / train_tandem.py."""
    return jnp.clip(label.astype(jnp.int32) - 1, 0, 2)


def select_slots_per_label(
    meta_q: jnp.ndarray,
    active_mask: jnp.ndarray,
    semantic_labels: jnp.ndarray,
    no_target_idx: int,
    K: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One slot per affordance class. K must equal 3 here.

    Slot 0 = best DESTROY object, slot 1 = best COLLECT, slot 2 = best AVOID.
    Each slot's target is the argmax of meta_q restricted to that label class
    among active objects; if no active object exists for that class, the slot
    is inactive and its target_idx is ``no_target_idx``.

    Parameters
    ----------
    meta_q : float32[max_objects]
    active_mask : bool[max_objects]
    semantic_labels : int32[max_objects] (taken from objects.semantic_label[-1])
    no_target_idx : int — sentinel for "no valid target"
    K : must be 3.

    Returns
    -------
    slot_target_idxs : int32[K]
    slot_active      : bool[K]
    slot_skill_idx   : int32[K]
    """
    assert K == 3, "per_label slot mode requires K=3"

    def one_label(label):
        label_mask = (semantic_labels == label) & active_mask
        masked_q = jnp.where(label_mask, meta_q, -jnp.inf)
        idx = jnp.argmax(masked_q).astype(jnp.int32)
        active = jnp.any(label_mask)
        target = jnp.where(active, idx, jnp.int32(no_target_idx))
        return target, active

    slot_target_idxs, slot_active = jax.vmap(one_label)(_PER_LABEL_LABELS)
    # By construction, slot k uses skill k.
    slot_skill_idx = jnp.arange(K, dtype=jnp.int32)
    return slot_target_idxs, slot_active, slot_skill_idx


def select_slots_topk(
    meta_q: jnp.ndarray,
    active_mask: jnp.ndarray,
    semantic_labels: jnp.ndarray,
    no_target_idx: int,
    K: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Top-K objects by meta-Q regardless of label.

    Returns
    -------
    slot_target_idxs : int32[K]
    slot_active      : bool[K]
    slot_skill_idx   : int32[K]  (derived from each chosen object's label)
    """
    masked_q = jnp.where(active_mask, meta_q, -jnp.inf)
    top_vals, top_idxs = jax.lax.top_k(masked_q, K)
    slot_active = jnp.isfinite(top_vals)
    slot_target_idxs = jnp.where(
        slot_active, top_idxs.astype(jnp.int32), jnp.int32(no_target_idx)
    )
    chosen_labels = semantic_labels[slot_target_idxs]
    slot_skill_idx = _label_to_skill_idx(chosen_labels)
    return slot_target_idxs, slot_active, slot_skill_idx


def select_slots(
    meta_q: jnp.ndarray,
    active_mask: jnp.ndarray,
    semantic_labels: jnp.ndarray,
    no_target_idx: int,
    mode: str,
    K: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Dispatch to per_label or topk based on a static ``mode`` string."""
    if mode == "per_label":
        return select_slots_per_label(meta_q, active_mask, semantic_labels, no_target_idx, K)
    if mode == "topk":
        return select_slots_topk(meta_q, active_mask, semantic_labels, no_target_idx, K)
    raise ValueError(f"Unknown COMPOSITION_SLOT_MODE: {mode!r}")


def slots_from_heuristic_target(
    heuristic_target_idx: jnp.ndarray,
    active_mask: jnp.ndarray,
    semantic_labels: jnp.ndarray,
    player_x: jnp.ndarray,
    player_y: jnp.ndarray,
    obj_x: jnp.ndarray,
    obj_y: jnp.ndarray,
    no_target_idx: int,
    mode: str,
    K: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build K-shaped slot data from a single heuristic target.

    For ``per_label`` mode (K=3): slot k is the **nearest active object** of
    label class k, with a preference override — if the heuristic's chosen
    target matches that label class, use it instead. Inactive label classes
    yield an inactive slot.

    For ``topk`` mode: slot 0 = heuristic target, remaining slots are the next
    K-1 nearest active objects (regardless of label). Inactive slots for envs
    with fewer than K active objects.

    All inputs are for a single env (no leading num_envs axis). Caller should
    vmap over envs as usual. Coordinates are pixel-space; only relative
    distance ordering matters.
    """
    if mode == "per_label":
        assert K == 3, "per_label requires K=3"
        # Distance from player to each object.
        d2 = (obj_x - player_x) ** 2 + (obj_y - player_y) ** 2

        def best_for_label(label):
            label_active = jnp.logical_and(active_mask, semantic_labels == label)
            masked_d2 = jnp.where(label_active, d2, jnp.inf)
            idx = jnp.argmin(masked_d2).astype(jnp.int32)
            any_lbl = jnp.any(label_active)
            tgt = jnp.where(any_lbl, idx, jnp.int32(no_target_idx))
            return tgt, any_lbl

        slot_targets, slot_active = jax.vmap(best_for_label)(_PER_LABEL_LABELS)

        # If the heuristic's chosen target matches a label class, prefer it
        # over the nearest-object pick for that slot.
        h_idx = heuristic_target_idx
        h_active = jnp.logical_and(
            h_idx != no_target_idx,
            jnp.logical_and(h_idx >= 0, h_idx < active_mask.shape[-1]),
        )
        safe_h = jnp.where(h_active, h_idx, 0)
        h_label = jnp.where(h_active, semantic_labels[safe_h], jnp.int32(-1))

        def maybe_override(k, current_target, current_active, label_k):
            match = jnp.logical_and(h_active, h_label == label_k)
            return (
                jnp.where(match, h_idx, current_target),
                jnp.where(match, True, current_active),
            )
        slot_targets, slot_active = jax.vmap(maybe_override)(
            jnp.arange(K, dtype=jnp.int32),
            slot_targets,
            slot_active,
            _PER_LABEL_LABELS,
        )
        slot_skill_idx = jnp.arange(K, dtype=jnp.int32)
        return slot_targets, slot_active, slot_skill_idx

    if mode == "topk":
        # Slot 0 = heuristic, then K-1 nearest others.
        d2 = (obj_x - player_x) ** 2 + (obj_y - player_y) ** 2
        h_idx = heuristic_target_idx
        h_active = jnp.logical_and(
            h_idx != no_target_idx,
            jnp.logical_and(h_idx >= 0, h_idx < active_mask.shape[-1]),
        )
        # Mask the heuristic-chosen index out of the remaining pool.
        safe_h = jnp.where(h_active, h_idx, 0)
        remaining_mask = active_mask.at[safe_h].set(
            jnp.where(h_active, False, active_mask[safe_h])
        )
        masked_d2 = jnp.where(remaining_mask, d2, jnp.inf)
        # Sort ascending by distance → take K-1 closest.
        sort_idx = jnp.argsort(masked_d2)[: K - 1].astype(jnp.int32)
        sort_active = jnp.isfinite(jnp.take(masked_d2, sort_idx))

        slot_targets = jnp.concatenate(
            [
                jnp.where(h_active, h_idx, jnp.int32(no_target_idx))[None],
                jnp.where(sort_active, sort_idx, jnp.int32(no_target_idx)),
            ]
        )
        slot_active = jnp.concatenate([h_active[None], sort_active])
        safe_targets = jnp.where(slot_active, slot_targets, 0)
        slot_skill_idx = _label_to_skill_idx(semantic_labels[safe_targets])
        return slot_targets, slot_active, slot_skill_idx

    raise ValueError(f"Unknown COMPOSITION_SLOT_MODE: {mode!r}")


def select_slots_eps(
    rng,
    meta_q: jnp.ndarray,
    active_mask: jnp.ndarray,
    semantic_labels: jnp.ndarray,
    no_target_idx: int,
    mode: str,
    K: int,
    eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ε-greedy slot selection. Each slot is independently perturbed: with
    probability ε its target is replaced by a random object drawn from that
    slot's eligibility set (label class in per_label mode, all active objects
    in topk mode). Inactive slots stay inactive.

    Parameters
    ----------
    rng : single PRNGKey for this env-step (split internally per slot).
    eps : scalar in [0, 1].

    Returns same shapes as select_slots.
    """
    greedy_target_idxs, greedy_active, greedy_skill_idx = select_slots(
        meta_q, active_mask, semantic_labels, no_target_idx, mode, K
    )

    # Build per-slot eligibility masks for the random branch.
    if mode == "per_label":
        slot_labels = _PER_LABEL_LABELS
        # eligibility[k] = active_mask & (semantic_labels == slot_labels[k])
        eligibility = jax.vmap(
            lambda lbl: jnp.logical_and(active_mask, semantic_labels == lbl)
        )(slot_labels)  # [K, max_objects]
    elif mode == "topk":
        # All K slots draw from the same active set.
        eligibility = jnp.broadcast_to(active_mask, (K, active_mask.shape[-1]))
    else:
        raise ValueError(f"Unknown COMPOSITION_SLOT_MODE: {mode!r}")

    rngs = jax.random.split(rng, K)

    def perturb_slot(rng_slot, k, greedy_idx, greedy_act, greedy_skill):
        rng_e, rng_a = jax.random.split(rng_slot)
        elig = eligibility[k]
        any_elig = jnp.any(elig)
        # Random valid index under elig mask. If no eligible object, fall back
        # to greedy (which will already be inactive for that slot).
        random_logits = jnp.where(elig, 0.0, -jnp.inf)
        random_idx = jax.random.categorical(rng_a, random_logits).astype(jnp.int32)
        u = jax.random.uniform(rng_e, ())
        do_explore = jnp.logical_and(u < eps, any_elig)
        new_idx = jnp.where(do_explore, random_idx, greedy_idx)
        new_active = jnp.where(do_explore, True, greedy_act)
        # In per_label mode skill_idx is the slot index; in topk it follows the
        # chosen object's label.
        if mode == "per_label":
            new_skill = greedy_skill  # = k
        else:
            new_skill = _label_to_skill_idx(semantic_labels[new_idx])
        return new_idx, new_active, new_skill

    new_targets, new_active, new_skills = jax.vmap(perturb_slot)(
        rngs,
        jnp.arange(K, dtype=jnp.int32),
        greedy_target_idxs,
        greedy_active,
        greedy_skill_idx,
    )
    return new_targets, new_active, new_skills


# ---------------------------------------------------------------------------
# Composition: weighted sum of per-slot global Qs
# ---------------------------------------------------------------------------

def _masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray, tau: float) -> jnp.ndarray:
    """Softmax over masked entries. Inactive entries get weight 0.

    If *all* entries are inactive, returns all-zeros (so the composed Q is the
    zero vector; callers should detect "no active slots" upstream and fall
    back to NOOP).
    """
    very_negative = jnp.full_like(logits, -1e30)
    safe_logits = jnp.where(mask, logits / tau, very_negative)
    # Stable softmax over active entries.
    m = jnp.max(jnp.where(mask, safe_logits, very_negative))
    # If no entry is active, m is -1e30; exp(0) below for inactive becomes 0 via mask.
    e = jnp.where(mask, jnp.exp(safe_logits - m), 0.0)
    z = jnp.sum(e)
    weights = jnp.where(z > 0, e / z, jnp.zeros_like(e))
    return weights


def _zscore(q: jnp.ndarray) -> jnp.ndarray:
    """Z-score along the last axis, masking -inf entries.

    -inf entries (unreachable game actions) keep -inf after normalization so
    they remain non-selectable.
    """
    finite_mask = jnp.isfinite(q)
    # Use only finite entries to compute stats.
    safe_q = jnp.where(finite_mask, q, 0.0)
    n = jnp.sum(finite_mask).astype(q.dtype)
    mean = jnp.where(n > 0, jnp.sum(safe_q) / jnp.maximum(n, 1.0), 0.0)
    var = jnp.where(
        n > 0,
        jnp.sum(jnp.where(finite_mask, (safe_q - mean) ** 2, 0.0)) / jnp.maximum(n, 1.0),
        1.0,
    )
    std = jnp.sqrt(var) + 1e-6
    normed = (q - mean) / std
    return jnp.where(finite_mask, normed, q)


def compose_global_q(
    per_slot_global_q: jnp.ndarray,  # [K, num_game_actions]
    per_slot_meta_q: jnp.ndarray,    # [K]
    slot_active: jnp.ndarray,        # [K] bool
    tau: float,
    normalize: str,
    weight_threshold: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compose per-slot global Qs into a single global Q vector.

    Parameters
    ----------
    weight_threshold : if > 0, slots whose softmax weight falls below this
        threshold are dropped (weight set to 0) and the surviving weights are
        renormalized. Gives the meta a hard "ignore this slot" mechanism: by
        pushing a slot's meta-Q far below the others, its softmax weight drops
        below the threshold and that slot is excluded from the composition.
        Useful with large K (e.g. K=max_objects in topk mode) so the meta can
        concentrate attention on a small effective number of objects.

    Returns
    -------
    composed_q : float32[num_game_actions]
    weights    : float32[K] — the softmax weights actually applied (post-
                 threshold and renormalized).
    """
    if normalize == "zscore":
        slot_q = jax.vmap(_zscore)(per_slot_global_q)
    elif normalize == "none":
        slot_q = per_slot_global_q
    else:
        raise ValueError(f"Unknown COMPOSITION_NORMALIZE_SKILL_Q: {normalize!r}")

    weights = _masked_softmax(per_slot_meta_q, slot_active, tau)

    # Optional threshold-and-renormalize: zero out low-weight slots, renormalize
    # surviving weights. If *all* weights are below threshold, all weights stay
    # zero and the composed Q ends up all -inf — callers should detect this
    # (e.g. by checking weights.sum() == 0 or via slot_active) and fall back to
    # a NOOP. Semantically this corresponds to "the meta decided no object is
    # worth attending to this frame."
    if weight_threshold > 0.0:
        survivor_mask = weights >= weight_threshold
        kept = jnp.where(survivor_mask, weights, 0.0)
        kept_sum = jnp.sum(kept)
        # If anything survives, renormalize. Otherwise leave weights at zero
        # (signalling "all dropped" to the caller).
        weights = jnp.where(kept_sum > 0, kept / jnp.maximum(kept_sum, 1e-12), kept)
        # Update slot_active to reflect dropped slots for downstream reachability check.
        slot_active = jnp.logical_and(slot_active, survivor_mask)

    # Inactive slots: zero out their contribution. Unreachable game actions
    # (-inf entries) need to be cleared too — otherwise -inf * 0 = nan.
    # Strategy: treat -inf entries as 0 *within active slots* and rely on the
    # union mask below to mark unreachable composed entries as -inf.
    slot_finite = jnp.isfinite(slot_q)
    safe_slot_q = jnp.where(slot_finite, slot_q, 0.0)
    weighted = weights[:, None] * safe_slot_q  # [K, A]
    composed = jnp.sum(weighted, axis=0)        # [A]

    # A composed game action is reachable iff at least one *active* (and not
    # threshold-dropped) slot could reach it.
    reachable_per_slot = slot_finite & slot_active[:, None]
    any_reachable = jnp.any(reachable_per_slot, axis=0)
    composed = jnp.where(any_reachable, composed, -jnp.inf)
    return composed, weights


# ---------------------------------------------------------------------------
# Inverse: executed global action -> per-slot ego action
# ---------------------------------------------------------------------------

def invert_global_to_ego_action(
    executed_global_action: jnp.ndarray,  # int32 scalar
    slot_ego_q: jnp.ndarray,              # [K, EGO_ACTION_DIM]
    slot_target_idxs: jnp.ndarray,        # [K]
    slot_active: jnp.ndarray,             # [K]
    raw_obs,
    objects,
    action_mapper: Callable,
) -> jnp.ndarray:
    """For each slot, recover the ego action that resolves to the executed game action.

    There may be multiple such ego actions (segment_max collisions); we
    tie-break by ego-Q magnitude, which keeps the inverse consistent with the
    scatter direction. Slots that cannot reach the executed action (or are
    inactive) fall back to NOOP (ego action 0).

    Returns
    -------
    slot_ego_actions : int32[K]
    """
    ego_dim = slot_ego_q.shape[-1]
    ego_actions = jnp.arange(ego_dim, dtype=jnp.int32)

    def one_slot(target_idx, active, ego_q):
        mapped = jax.vmap(
            lambda e: action_mapper(e, raw_obs, target_idx, objects)
        )(ego_actions)
        matches = mapped == executed_global_action
        # Use ego-Q with non-matches set to -inf; argmax picks the strongest
        # ego action that resolves to the executed game action.
        masked = jnp.where(matches, ego_q, -jnp.inf)
        any_match = jnp.any(matches)
        chosen = jnp.argmax(masked).astype(jnp.int32)
        # If slot inactive or no ego action matches, fall back to NOOP.
        return jnp.where(active & any_match, chosen, jnp.int32(0))

    return jax.vmap(one_slot)(slot_target_idxs, slot_active, slot_ego_q)
