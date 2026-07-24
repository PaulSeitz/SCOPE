"""
Heuristic meta-policy for Seaquest.

Restored layered-distance targeting (pre-gameplay-plan refactor), with a
small tweak: DESTROY and AVOID share one threat tier.

- Threat tier: nearest of AVOID ∪ DESTROY (closer = more urgent).
- Else sticky / nearest COLLECT (sticky DESTROY/COLLECT while active and
  no threat on screen).
- Close AVOID/DESTROY within ``CLOSE_DANGER_RANGE`` overrides, but never
  interrupts an active surfacing commit (low O₂ / full divers).
- Surface waypoint (slot 34, LABEL_NAVIGATE) when oxygen is low *underwater*,
  while refilling at the surface with divers on board, or when divers ≥ 6.
  (Does not force surface during life-start oxygen fill at y≈46.)

Object slots after meta-NOOP:
  divers 1–4, sharks 5–16, subs 17–28, surface sub 29,
  missiles 30–33, surface waypoint 34.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)
from concept_based_learning.environments.manual_policy import register_predicate
from concept_based_learning.environments.manual_policy.state import (
    NO_TARGET_IDX,
    PolicyState,
)

# Last game-object slot (index 0 is reserved meta NOOP).
SURFACE_DUMMY_IDX = 34
# Real surface submarine in the flattened object list (after meta NOOP).
SURFACE_SUB_IDX = 29

OXYGEN_LOW_THRESHOLD = 15.0
CRITICAL_OXYGEN_THRESHOLD = 5.0  # retained for callers / future use
SURFACE_SUB_NEAR_RANGE = 40.0  # retained for callers / future use
CLOSE_DANGER_RANGE = 25.0
DANGER_RELEASE_RANGE = 40.0  # retained for callers / future use
X_AXIS_WEIGHT = 1.0  # NOTE: deactivated for now (weight == 1 → Euclidean)
# Past the Seaquest breathing band (y in [47, 52]); matches jax_seaquest.
PLAYER_UNDERWATER_Y = 52.0


def _surface_required_from_raw(raw_observation):
    """True when surfacing is required (low O2 underwater, refill, or six divers).

    Life start sits at y≈46 while oxygen fills 0→64. Treating that as
    ``oxygen_low`` would force the surface NAVIGATE waypoint every life and
    trap skills on an already-reached target. Low O2 only forces surface once
    underwater, or while refilling at the surface with divers on board.
    """
    last = jax.tree_util.tree_map(
        lambda x: x[..., -1] if jnp.ndim(x) > 0 else x, raw_observation
    )
    oxygen_low = last.oxygen_level < OXYGEN_LOW_THRESHOLD
    full_divers = last.collected_divers >= 6
    py = last.player.y
    underwater = py > PLAYER_UNDERWATER_Y
    oxygen_low_underwater = jnp.logical_and(oxygen_low, underwater)
    # Stay on the WP while refilling after a real rescue (divers still held).
    refilling = jnp.logical_and(
        jnp.logical_not(underwater),
        jnp.logical_and(oxygen_low, last.collected_divers > 0),
    )
    return jnp.logical_or(
        full_divers,
        jnp.logical_or(oxygen_low_underwater, refilling),
    )


def _surface_needed_pred(raw_observation, objects, ctx):
    del objects, ctx
    return _surface_required_from_raw(raw_observation)


register_predicate("seaquest.surface_needed", _surface_needed_pred)


def apply_meta_mask_overrides(raw_obs, active_masks, config, objects=None):
    del objects
    if active_masks.shape[-1] <= SURFACE_DUMMY_IDX:
        return active_masks

    always_active = bool(config.get("ALWAYS_ACTIVE_SURFACE_TARGET", False))
    force_surface = _surface_required_from_raw(raw_obs)
    if active_masks.ndim == 1:
        force_surface = jnp.reshape(force_surface, ())
        if always_active:
            return active_masks.at[SURFACE_DUMMY_IDX].set(True)
        return active_masks.at[SURFACE_DUMMY_IDX].set(
            jnp.logical_and(active_masks[SURFACE_DUMMY_IDX], force_surface)
        )

    if always_active:
        return active_masks.at[:, SURFACE_DUMMY_IDX].set(True)
    return active_masks.at[:, SURFACE_DUMMY_IDX].set(
        jnp.logical_and(active_masks[:, SURFACE_DUMMY_IDX], force_surface)
    )


def get_forced_target_override(raw_obs, current_targets, config, rng=None):
    if current_targets.ndim == 0:
        return current_targets

    force_surface = _surface_required_from_raw(raw_obs)

    surface_pick_prob = float(config.get("SURFACE_PICK_PROB", 0.0))
    if rng is not None and surface_pick_prob > 0.0:
        rng_surface = jax.random.split(rng, current_targets.shape[0])
        pick_surface = jax.vmap(lambda r: jax.random.uniform(r, ()) < surface_pick_prob)(
            rng_surface
        )
        pick_surface = jnp.logical_and(force_surface, pick_surface)
        current_targets = jnp.where(
            pick_surface,
            jnp.array(SURFACE_DUMMY_IDX, dtype=current_targets.dtype),
            current_targets,
        )

    force_override = bool(config.get("FORCE_SURFACE_OVERRIDE", False))
    if force_override:
        current_targets = jnp.where(
            force_surface,
            jnp.array(SURFACE_DUMMY_IDX, dtype=current_targets.dtype),
            current_targets,
        )
    return current_targets


def _select_target(raw_observation, objects, prev_target_idx):
    """Core threat-by-distance then COLLECT selection (+ surfacing)."""
    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    active_mask = objects.active[-1] > 0.5
    idxs = jnp.arange(active_mask.shape[0])

    # Exclude the surface-only waypoint from normal scoring.
    normal_active_mask = jnp.logical_and(active_mask, idxs != SURFACE_DUMMY_IDX)

    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)

    dx = px_center - ox_center
    dy = py_center - oy_center
    distances = jnp.sqrt(dx ** 2 + dy ** 2)
    weighted_distances = jnp.sqrt((X_AXIS_WEIGHT * dx) ** 2 + dy ** 2)

    labels = objects.semantic_label[-1]

    avoid_mask = jnp.logical_and(normal_active_mask, labels == LABEL_AVOID)
    destroy_mask = jnp.logical_and(normal_active_mask, labels == LABEL_DESTROY)
    collect_mask = jnp.logical_and(normal_active_mask, labels == LABEL_COLLECT)

    # Same tier: pick whichever threat (AVOID or DESTROY) is closer.
    threat_mask = jnp.logical_or(avoid_mask, destroy_mask)
    threat_dist = jnp.where(threat_mask, weighted_distances, jnp.inf)
    best_threat_idx = jnp.argmin(threat_dist).astype(jnp.int32)
    any_threat = jnp.any(threat_mask)

    collect_dist = jnp.where(collect_mask, weighted_distances, jnp.inf)
    best_collect_idx = jnp.argmin(collect_dist).astype(jnp.int32)

    # Sticky mission: keep prev DESTROY/COLLECT while it stays active.
    prev_valid = jnp.logical_and(
        prev_target_idx >= 0,
        prev_target_idx < active_mask.shape[0],
    )
    prev_active = jnp.where(prev_valid, normal_active_mask[prev_target_idx], False)
    prev_label = jnp.where(prev_valid, labels[prev_target_idx], -1)
    prev_is_mission = jnp.logical_or(
        prev_label == LABEL_DESTROY, prev_label == LABEL_COLLECT
    )
    keep_prev = jnp.logical_and(prev_active, prev_is_mission)

    # Threats always re-pick nearest; stickiness only when the screen is clear.
    collect_or_sticky = jnp.where(keep_prev, prev_target_idx, best_collect_idx)
    best_normal_idx = jnp.where(any_threat, best_threat_idx, collect_or_sticky)

    oxygen = raw_observation.oxygen_level[-1]
    oxygen_low = oxygen < OXYGEN_LOW_THRESHOLD
    collected_divers = raw_observation.collected_divers[-1]
    have_full_divers = collected_divers >= 6
    py = raw_observation.player.y[-1]
    underwater = py > PLAYER_UNDERWATER_Y
    oxygen_low_underwater = jnp.logical_and(oxygen_low, underwater)
    refilling = jnp.logical_and(
        jnp.logical_not(underwater),
        jnp.logical_and(oxygen_low, collected_divers > 0),
    )
    force_surface = jnp.logical_or(
        have_full_divers,
        jnp.logical_or(oxygen_low_underwater, refilling),
    )

    is_threat = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)
    is_close_danger = jnp.logical_and(distances <= CLOSE_DANGER_RANGE, is_threat)
    is_close_danger = jnp.logical_and(is_close_danger, normal_active_mask)

    # While surfacing, ignore the surface sub as a "danger" interrupt.
    is_surface_sub = idxs == SURFACE_SUB_IDX
    ignore_danger = jnp.logical_and(force_surface, is_surface_sub)
    valid_close_danger = jnp.logical_and(
        is_close_danger, jnp.logical_not(ignore_danger)
    )
    have_valid_danger = jnp.any(valid_close_danger)
    danger_dist = jnp.where(valid_close_danger, distances, jnp.inf)
    best_danger_idx = jnp.argmin(danger_dist).astype(jnp.int32)

    target_idx = best_normal_idx
    target_idx = jnp.where(
        force_surface,
        jnp.array(SURFACE_DUMMY_IDX, dtype=jnp.int32),
        target_idx,
    )

    # Close danger overrides — but never interrupt an active surfacing commit.
    suppress_danger = force_surface
    target_idx = jnp.where(
        jnp.logical_and(have_valid_danger, jnp.logical_not(suppress_danger)),
        best_danger_idx,
        target_idx,
    )

    any_active = jnp.any(normal_active_mask)
    any_targetable = jnp.logical_or(any_active, force_surface)
    target_idx = jnp.where(
        any_targetable,
        target_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )
    return target_idx.astype(jnp.int32)


def get_heuristic_target_with_state(raw_observation, objects, prev_target_idx, state, rng):
    del rng
    # Prefer PolicyState mission stickiness; fall back to prev_target_idx.
    sticky = jnp.asarray(state.mission_target, dtype=jnp.int32)
    sticky = jnp.where(sticky == NO_TARGET_IDX, prev_target_idx, sticky)
    target = _select_target(raw_observation, objects, sticky)
    # Persist DESTROY/COLLECT only (AVOID is reactive; surface WP is re-forced).
    labels = objects.semantic_label[-1]
    tgt_clip = jnp.clip(target, 0, labels.shape[0] - 1)
    tgt_label = jnp.where(target >= 0, labels[tgt_clip], -1)
    is_mission = jnp.logical_or(
        tgt_label == LABEL_DESTROY, tgt_label == LABEL_COLLECT
    )
    new_mission = jnp.where(is_mission, target, jnp.int32(NO_TARGET_IDX))
    return target, state.replace(mission_target=new_mission)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    """Stateless wrapper: stickiness via ``prev_target_idx`` only."""
    del rng
    return _select_target(raw_observation, objects, prev_target_idx)
