"""
Heuristic meta-policy for Seaquest.

Priority (low → high):
- Default: nearest COLLECT (diver), with sticky COLLECT/DESTROY while active
  and no close threat.
- Close threats interrupt inside an axis-aligned ellipse: ``|dy| ≤ R`` and
  ``|dx| ≤ X_RANGE_SCALE·R``. Among elliptical threats, pick by **Euclidean**
  distance (nearest).
    * missiles (AVOID): ``AVOID_RANGE``, or ``AVOID_CLOSING_RANGE`` when
      closing on the player (frame-diff vx; range only — scoring is Euclidean).
    * sharks (DESTROY): ``CLOSE_DANGER_RANGE``.
    * underwater subs (DESTROY): ``SUB_DESTROY_RANGE``.
- Imminent-death sphere (``distances ≤ IMMINENT_DEATH_RANGE``): overrides
  everything except critical surfacing. Includes AVOID / DESTROY / COLLECT.
  Selection: priority AVOID > DESTROY > COLLECT, nearest within that class.
- Surface waypoint (slot 34) when oxygen is low *underwater*, while refilling
  at the surface with divers on board, or when divers ≥ 6 — but only if the
  surface sub is inactive / left of the player, or oxygen is critical.

Surface sub (slot 29, AVOID) is ignored unless the player is near the surface
and within ``SURFACE_SUB_NEAR_RANGE``.

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

# Slot ranges (inclusive) after meta NOOP.
SHARK_LO, SHARK_HI = 5, 16
SUB_LO, SUB_HI = 17, 28
MISSILE_LO, MISSILE_HI = 30, 33

OXYGEN_LOW_THRESHOLD = 15.0
CRITICAL_OXYGEN_THRESHOLD = 5.0
SURFACE_SUB_NEAR_RANGE = 15.0
# Sharks / default DESTROY interrupt — vertical half-axis of the ellipse.
CLOSE_DANGER_RANGE = 18.0
# Horizontal half-axis = X_RANGE_SCALE * vertical.
X_RANGE_SCALE = 1.5
# Underwater subs: engage from farther so DESTROY isn't forced nose-on.
SUB_DESTROY_RANGE = 42.0
# Enemy missiles.
AVOID_RANGE = 22.0
AVOID_CLOSING_RANGE = 40.0
AVOID_Y_BAND = 14.0  # closing missiles must share a loose y-band
# Spherical override: any AVOID/DESTROY/COLLECT inside this Euclidean radius.
IMMINENT_DEATH_RANGE = 12.0
# Lexicographic score: priority_rank * LARGE + distance (lower wins).
_PRIORITY_DISTANCE_SCALE = 1.0e4
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


def _in_interrupt_ellipse(abs_dx, abs_dy, range_y):
    """True when inside axis-aligned ellipse with vertical radius ``range_y``."""
    range_x = range_y * X_RANGE_SCALE
    return jnp.logical_and(abs_dy <= range_y, abs_dx <= range_x)


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


def _object_vx(objects):
    """Per-slot horizontal velocity from the frame stack.

    ``objects.x`` can be either ``[stack, slots]`` or ``[batch, stack, slots]``.
    Always difference along the stack axis (second-to-last), never across batch.
    """
    x = objects.x
    if x.ndim < 2:
        return jnp.zeros(x.shape[-1:], dtype=jnp.float32)
    if x.shape[-2] < 2:
        return jnp.zeros(x.shape[:-2] + (x.shape[-1],), dtype=jnp.float32)
    return (x[..., -1, :] - x[..., -2, :]).astype(jnp.float32)


def _priority_rank(labels):
    """AVOID=0, DESTROY=1, COLLECT=2, else large (ignored)."""
    return jnp.where(
        labels == LABEL_AVOID,
        0,
        jnp.where(
            labels == LABEL_DESTROY,
            1,
            jnp.where(labels == LABEL_COLLECT, 2, 99),
        ),
    )


def _select_target(raw_observation, objects, prev_target_idx):
    """Diver-first; elliptical Euclidean threats; imminent sphere override."""
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
    abs_dx = jnp.abs(dx)
    abs_dy = jnp.abs(dy)
    distances = jnp.sqrt(dx ** 2 + dy ** 2)

    labels = objects.semantic_label[-1]

    avoid_mask = jnp.logical_and(normal_active_mask, labels == LABEL_AVOID)
    destroy_mask = jnp.logical_and(normal_active_mask, labels == LABEL_DESTROY)
    collect_mask = jnp.logical_and(normal_active_mask, labels == LABEL_COLLECT)

    is_sub = jnp.logical_and(idxs >= SUB_LO, idxs <= SUB_HI)
    is_missile = jnp.logical_and(idxs >= MISSILE_LO, idxs <= MISSILE_HI)

    # Surface sub: only a threat when near surface and within a tight range.
    is_surface_sub = idxs == SURFACE_SUB_IDX
    near_surface = py <= SURFACE_Y_THRESHOLD
    surface_sub_close = jnp.logical_and(
        distances <= SURFACE_SUB_NEAR_RANGE,
        near_surface,
    )
    surface_sub_eligible = jnp.logical_and(is_surface_sub, surface_sub_close)
    avoid_mask = jnp.logical_and(
        avoid_mask,
        jnp.logical_or(jnp.logical_not(is_surface_sub), surface_sub_eligible),
    )

    destroy_aimable = destroy_mask

    # --- Imminent-death sphere (includes COLLECT) ---
    imminent_type = jnp.logical_or(
        avoid_mask,
        jnp.logical_or(destroy_aimable, collect_mask),
    )
    imminent_mask = jnp.logical_and(
        imminent_type,
        distances <= IMMINENT_DEATH_RANGE,
    )
    # AVOID > DESTROY > COLLECT, then nearest Euclidean within that class.
    imminent_score = (
        _priority_rank(labels).astype(jnp.float32) * _PRIORITY_DISTANCE_SCALE
        + distances
    )
    imminent_dist = jnp.where(imminent_mask, imminent_score, jnp.inf)
    best_imminent_idx = jnp.argmin(imminent_dist).astype(jnp.int32)
    any_imminent = jnp.any(imminent_mask)

    # --- Elliptical close threats (AVOID/DESTROY only); Euclidean nearest ---
    destroy_range_y = jnp.where(is_sub, SUB_DESTROY_RANGE, CLOSE_DANGER_RANGE)
    destroy_close = jnp.logical_and(
        destroy_aimable,
        _in_interrupt_ellipse(abs_dx, abs_dy, destroy_range_y),
    )

    vx = _object_vx(objects)
    closing = jnp.logical_or(
        jnp.logical_and(ox_center > px_center, vx < -0.5),
        jnp.logical_and(ox_center < px_center, vx > 0.5),
    )
    same_lane = abs_dy <= AVOID_Y_BAND
    closing_threat = jnp.logical_and(closing, same_lane)
    avoid_range_y = jnp.where(
        jnp.logical_and(is_missile, closing_threat),
        AVOID_CLOSING_RANGE,
        AVOID_RANGE,
    )
    avoid_close = jnp.logical_and(
        avoid_mask,
        _in_interrupt_ellipse(abs_dx, abs_dy, avoid_range_y),
    )

    close_threat_mask = jnp.logical_or(destroy_close, avoid_close)
    threat_dist = jnp.where(close_threat_mask, distances, jnp.inf)
    best_threat_idx = jnp.argmin(threat_dist).astype(jnp.int32)
    any_close_threat = jnp.any(close_threat_mask)

    collect_dist = jnp.where(collect_mask, distances, jnp.inf)
    best_collect_idx = jnp.argmin(collect_dist).astype(jnp.int32)
    any_collect = jnp.any(collect_mask)

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

    # Default: nearest diver (or sticky mission); elliptical threats interrupt.
    collect_or_sticky = jnp.where(keep_prev, prev_target_idx, best_collect_idx)
    best_normal_idx = jnp.where(any_close_threat, best_threat_idx, collect_or_sticky)
    # Fallback when nothing close / no diver: nearest aimable DESTROY or AVOID.
    fallback_threat = jnp.logical_or(destroy_aimable, avoid_mask)
    best_normal_idx = jnp.where(
        jnp.logical_or(any_close_threat, jnp.logical_or(keep_prev, any_collect)),
        best_normal_idx,
        jnp.where(
            jnp.any(fallback_threat),
            jnp.argmin(jnp.where(fallback_threat, distances, jnp.inf)).astype(
                jnp.int32
            ),
            best_normal_idx,
        ),
    )

    force_surface = _surface_required_from_raw(raw_observation, objects)

    target_idx = best_normal_idx
    target_idx = jnp.where(
        force_surface,
        jnp.array(SURFACE_DUMMY_IDX, dtype=jnp.int32),
        target_idx,
    )

    # Elliptical danger overrides non-surface picks. While surfacing, never interrupt.
    target_idx = jnp.where(
        jnp.logical_and(any_close_threat, jnp.logical_not(force_surface)),
        best_threat_idx,
        target_idx,
    )

    # Imminent sphere overrides elliptical / collect (still not while forced surfacing).
    target_idx = jnp.where(
        jnp.logical_and(any_imminent, jnp.logical_not(force_surface)),
        best_imminent_idx,
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
