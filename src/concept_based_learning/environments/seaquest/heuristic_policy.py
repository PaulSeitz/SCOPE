"""
Heuristic meta-policy for Seaquest.

Simple nearest-object targeting:
- Always pick the nearest active game object (Euclidean, object centers).
- Surface sub (slot 29) is ignored unless the player is near the surface and
  within ``SURFACE_SUB_NEAR_RANGE``.
- Surface waypoint (slot 34) when oxygen is low *underwater*, while refilling
  at the surface with divers on board, or when divers ≥ 6 — but only if the
  surface sub is inactive / left of the player, or oxygen is critical.
  While forced to surface, a danger within ``SURFACE_DANGER_CLOSE_RANGE``
  overrides the waypoint (nearest AVOID/DESTROY).

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
CRITICAL_OXYGEN_THRESHOLD = 5.0
SURFACE_SUB_NEAR_RANGE = 15.0
# While forced to surface, AVOID/DESTROY within this Euclidean radius interrupt.
# Overridable via config ``SURFACE_DANGER_CLOSE_RANGE`` (see resolve_heuristic_*).
SURFACE_DANGER_CLOSE_RANGE = 25.0
DANGER_RELEASE_RANGE = 40.0  # retained for callers / future use
# Past the Seaquest breathing band (y in [47, 52]); matches jax_seaquest.
PLAYER_UNDERWATER_Y = 52.0
# Same band used by ascent-block / near-surface checks in label.py.
SURFACE_Y_THRESHOLD = 52.0


def _latest_leaf(x):
    """Take the last stack frame from an array leaf (or return scalars)."""
    x = jnp.asarray(x)
    return x[..., -1] if x.ndim > 0 else x


def _latest_object_frame(field):
    """Latest stack frame: ``[B, stack, ...]`` → ``[B, ...]``, ``[stack, ...]`` → ``[...]``."""
    field = jnp.asarray(field)
    if field.ndim >= 3:
        return field[:, -1, ...]
    if field.ndim == 2:
        return field[-1]
    return field


def _surface_approach_clear(raw_observation, objects=None):
    """True if surfacing is safe enough: sub gone/left, or O₂ critical."""
    oxygen = _latest_leaf(raw_observation.oxygen_level)
    oxygen_critical = oxygen < CRITICAL_OXYGEN_THRESHOLD
    player = raw_observation.player
    px = _latest_leaf(player.x) + _latest_leaf(player.width) / 2.0

    if objects is not None:
        ox = _latest_object_frame(objects.x)
        ow = _latest_object_frame(objects.width)
        active = _latest_object_frame(objects.active) > 0.5
        sub_active = active[..., SURFACE_SUB_IDX]
        sub_cx = ox[..., SURFACE_SUB_IDX] + ow[..., SURFACE_SUB_IDX] / 2.0
    else:
        # enemies layout: [12 sharks, 12 subs, 1 surface sub]
        enemies = raw_observation.enemies
        ex = _latest_object_frame(enemies.x)
        ew = _latest_object_frame(enemies.width)
        ea = _latest_object_frame(enemies.active)
        sub_active = ea[..., 24] > 0.5
        sub_cx = ex[..., 24] + ew[..., 24] / 2.0

    sub_left_or_gone = jnp.logical_or(jnp.logical_not(sub_active), sub_cx < px)
    return jnp.logical_or(oxygen_critical, sub_left_or_gone)


def _surface_required_from_raw(raw_observation, objects=None):
    """True when surfacing is required and the approach is clear enough.

    Life start sits at y≈46 while oxygen fills 0→64. Treating that as
    ``oxygen_low`` would force the surface NAVIGATE waypoint every life and
    trap skills on an already-reached target. Low O2 only forces surface once
    underwater, or while refilling at the surface with divers on board.

    Surface WP is gated on surface-sub left/gone unless oxygen is critical.
    """
    oxygen = _latest_leaf(raw_observation.oxygen_level)
    collected_divers = _latest_leaf(raw_observation.collected_divers)
    py = _latest_leaf(raw_observation.player.y)
    oxygen_low = oxygen < OXYGEN_LOW_THRESHOLD
    full_divers = collected_divers >= 6
    underwater = py > PLAYER_UNDERWATER_Y
    oxygen_low_underwater = jnp.logical_and(oxygen_low, underwater)
    # Stay on the WP while refilling after a real rescue (divers still held).
    refilling = jnp.logical_and(
        jnp.logical_not(underwater),
        jnp.logical_and(oxygen_low, collected_divers > 0),
    )
    want_surface = jnp.logical_or(
        full_divers,
        jnp.logical_or(oxygen_low_underwater, refilling),
    )
    return jnp.logical_and(want_surface, _surface_approach_clear(raw_observation, objects))


def _surface_needed_pred(raw_observation, objects, ctx):
    del ctx
    return _surface_required_from_raw(raw_observation, objects)


register_predicate("seaquest.surface_needed", _surface_needed_pred)


def apply_meta_mask_overrides(raw_obs, active_masks, config, objects=None):
    if active_masks.shape[-1] <= SURFACE_DUMMY_IDX:
        return active_masks

    always_active = bool(config.get("ALWAYS_ACTIVE_SURFACE_TARGET", False))
    force_surface = _surface_required_from_raw(raw_obs, objects)
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


def get_forced_target_override(raw_obs, current_targets, config, rng=None, objects=None):
    if current_targets.ndim == 0:
        return current_targets

    force_surface = _surface_required_from_raw(raw_obs, objects)

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


def _select_target(raw_observation, objects, prev_target_idx, *, danger_close_range=None):
    """Nearest object; surface WP with danger-close override."""
    del prev_target_idx
    if danger_close_range is None:
        danger_close_range = SURFACE_DANGER_CLOSE_RANGE
    danger_close_range = jnp.asarray(danger_close_range, dtype=jnp.float32)

    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    active_mask = objects.active[-1] > 0.5
    idxs = jnp.arange(active_mask.shape[0])
    labels = objects.semantic_label[-1]

    px_center = px + (raw_observation.player.width[-1] / 2.0)
    py_center = py + (raw_observation.player.height[-1] / 2.0)
    ox_center = ox + (objects.width[-1] / 2.0)
    oy_center = oy + (objects.height[-1] / 2.0)
    distances = jnp.sqrt((px_center - ox_center) ** 2 + (py_center - oy_center) ** 2)

    # Exclude surface waypoint from normal nearest scoring.
    selectable = jnp.logical_and(active_mask, idxs != SURFACE_DUMMY_IDX)

    # Surface sub: only when near surface and within a tight range.
    is_surface_sub = idxs == SURFACE_SUB_IDX
    near_surface = py <= SURFACE_Y_THRESHOLD
    surface_sub_eligible = jnp.logical_and(
        is_surface_sub,
        jnp.logical_and(distances <= SURFACE_SUB_NEAR_RANGE, near_surface),
    )
    selectable = jnp.logical_and(
        selectable,
        jnp.logical_or(jnp.logical_not(is_surface_sub), surface_sub_eligible),
    )

    nearest_dist = jnp.where(selectable, distances, jnp.inf)
    best_idx = jnp.argmin(nearest_dist).astype(jnp.int32)
    any_selectable = jnp.any(selectable)

    # Danger-close while surfacing: nearest AVOID/DESTROY within configured range.
    # range <= 0 disables the override (always keep surface WP when forced).
    is_danger = jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_DESTROY)
    danger_close = jnp.logical_and(
        selectable,
        jnp.logical_and(
            is_danger,
            jnp.logical_and(danger_close_range > 0.0, distances <= danger_close_range),
        ),
    )
    danger_dist = jnp.where(danger_close, distances, jnp.inf)
    best_danger_idx = jnp.argmin(danger_dist).astype(jnp.int32)
    any_danger_close = jnp.any(danger_close)

    force_surface = _surface_required_from_raw(raw_observation, objects)

    target_idx = jnp.where(
        any_selectable,
        best_idx,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    )
    # Prefer surface WP when required, unless a nearby danger overrides.
    target_idx = jnp.where(
        force_surface,
        jnp.where(
            any_danger_close,
            best_danger_idx,
            jnp.array(SURFACE_DUMMY_IDX, dtype=jnp.int32),
        ),
        target_idx,
    )
    return target_idx.astype(jnp.int32)


def get_heuristic_target_with_state(
    raw_observation,
    objects,
    prev_target_idx,
    state,
    rng,
    *,
    danger_close_range=None,
):
    del rng
    target = _select_target(
        raw_observation,
        objects,
        prev_target_idx,
        danger_close_range=danger_close_range,
    )
    # Persist DESTROY/COLLECT only (AVOID is reactive; surface WP is re-forced).
    labels = objects.semantic_label[-1]
    tgt_clip = jnp.clip(target, 0, labels.shape[0] - 1)
    tgt_label = jnp.where(target >= 0, labels[tgt_clip], -1)
    is_mission = jnp.logical_or(
        tgt_label == LABEL_DESTROY, tgt_label == LABEL_COLLECT
    )
    new_mission = jnp.where(is_mission, target, jnp.int32(NO_TARGET_IDX))
    return target, state.replace(mission_target=new_mission)


def get_heuristic_target(
    raw_observation,
    objects,
    prev_target_idx,
    rng,
    *,
    danger_close_range=None,
):
    """Stateless wrapper."""
    del rng
    return _select_target(
        raw_observation,
        objects,
        prev_target_idx,
        danger_close_range=danger_close_range,
    )
