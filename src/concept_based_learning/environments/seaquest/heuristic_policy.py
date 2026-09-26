"""
Heuristic meta-policy for Seaquest.

Simple nearest-object targeting, with two Seaquest-specific overrides:
- COLLECT (divers) only if that diver's horizontal lane is clear of DESTROY
  (no shark/sub with |Δy| ≤ ``DIVER_LANE_Y_PX``). Otherwise prefer sharks,
  even when they are farther than the diver, so lanes are cleared before
  diving in for a collect.
- Optional ``COLLECT_REQUIRES_NO_DESTROY``: if any selectable DESTROY exists,
  never target a diver (nearest shark/sub wins). The surface waypoint is
  still forced by index (not nearest-COLLECT) so this flag does not steal
  a required surface. Use this so COLLECT is not asked to fight.
- Optional ``COLLECT_EDGE_MARGIN_PX``: soft midfield COLLECT preference. When
  > 0 and any lane-clear diver has centre x in ``[margin, 160-margin]``,
  edge divers are skipped so COLLECT is not trained on edge-suicide rushes.
  If no midfield diver is available, edge divers remain selectable.
- Surface sub (slot 28) is ignored for *nearest* targeting unless the player
  is near the surface and within ``SURFACE_SUB_NEAR_RANGE``. It still
  interrupts as AVOID from any depth inside ``SURFACE_SUB_AVOID_RANGE_{X,Y}``.
- Surface waypoint (slot 33) when oxygen is low *underwater*, while refilling
  at the surface with divers on board, or when divers ≥ 6 — but only if the
  surface sub is inactive / left of the player, or oxygen is low.
  While forced to surface, a danger within ``SURFACE_DANGER_CLOSE_RANGE``
  overrides the waypoint (nearest AVOID, or DESTROY within the larger
  ``DESTROY_INTERRUPT_RANGE``). Anytime, a DESTROY within
  ``DESTROY_INTERRUPT_RANGE`` *or* a missile inside the AVOID ellipse
  (``AVOID_INTERRUPT_RANGE`` × ``AVOID_INTERRUPT_RANGE_Y``) can interrupt
  COLLECT / nearer non-combat targets; when **both** are valid, pick the
  **nearest** (no fixed AVOID-over-DESTROY or DESTROY-over-AVOID priority).
  The surface sub still interrupts anytime it is inside
  ``SURFACE_SUB_AVOID_RANGE_{X,Y}`` (contact dodge).
  When oxygen is low, those enemy interrupt radii shrink (and go to 0 at
  critical O₂) so the surface waypoint stays selected and the player goes up.

Object slots (meta-NOOP appended last; see ``meta_object_layout``):
  divers 0–3, sharks 4–15, subs 16–27, surface sub 28,
  missiles 29–32, surface waypoint 33, meta-NOOP 34.
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
from concept_based_learning.observations.egocentric import (
    aabb_closest_delta_px,
    aabb_surface_gap_px,
)

# Surface waypoint: last *game* slot before trailing meta-NOOP (max_objects=35).
SURFACE_DUMMY_IDX = 33
# Real surface submarine in the flattened game list (append-NOOP layout).
SURFACE_SUB_IDX = 28

OXYGEN_LOW_THRESHOLD = 15.0
CRITICAL_OXYGEN_THRESHOLD = 5.0
SURFACE_SUB_NEAR_RANGE = 5.0
# While forced to surface, AVOID within this surface-gap radius interrupt.
# Overridable via config ``SURFACE_DANGER_CLOSE_RANGE`` (see resolve_heuristic_*).
SURFACE_DANGER_CLOSE_RANGE = 25.0
# DESTROY (sharks/subs) interrupt COLLECT — and surface WP — at this larger
# AABB gap so FRAME_SKIP=4 cannot close a shark before the heuristic retargets.
# Overridable via ``DESTROY_INTERRUPT_RANGE``.
DESTROY_INTERRUPT_RANGE = 40.0
# AVOID (torpedoes) compete with DESTROY interrupts inside a wide-X / tight-Y
# ellipse of AABB edge-gaps. When both a shark/sub and a missile are in-band,
# pick the nearer — do not always prefer AVOID. X matches the old 55px
# same-lane band; Y is a corridor so a missile in another row does not lock
# AVOID. Overridable via ``AVOID_INTERRUPT_RANGE`` /
# ``AVOID_INTERRUPT_RANGE_Y``. ``<=0`` disables.
AVOID_INTERRUPT_RANGE = 55.0
AVOID_INTERRUPT_RANGE_Y = 18.0
# Surface sub: AVOID interrupt from any depth when this close (edge-gap ellipse).
# Distinct from ``SURFACE_SUB_NEAR_RANGE``, which still gates *nearest* targeting
# to the surface band so we do not hunt the patrol from mid-water. 5px edge-gap:
# the patrol is slow — only dodge at contact range so a shark in the upper lane
# is not ignored.
SURFACE_SUB_AVOID_RANGE_X = 5.0
SURFACE_SUB_AVOID_RANGE_Y = 5.0
# While oxygen is low the surface WP must win: shrink combat interrupts so a
# shark in the usual 40px band cannot keep the player underwater. Critical O₂
# disables interrupts entirely (must surface).
O2_LOW_ENEMY_INTERRUPT_RANGE = 12.0
O2_CRITICAL_ENEMY_INTERRUPT_RANGE = 0.0
# A diver's lane is blocked if a DESTROY shares this vertical band (centre Δy).
# Overridable via config ``DIVER_LANE_Y_PX``.
DIVER_LANE_Y_PX = 14.0
# Soft COLLECT curriculum: when ``COLLECT_EDGE_MARGIN_PX`` > 0, prefer divers
# whose centre x lies in ``[margin, SCREEN_W - margin]``. If any midfield
# lane-clear diver exists, edge divers are ignored; otherwise edge divers
# remain selectable (fall back). Meta risk triage stays in the heuristic;
# COLLECT skills should not learn "don't chase far/edge targets".
# Overridable via ``COLLECT_EDGE_MARGIN_PX`` (``<=0`` disables).
COLLECT_EDGE_MARGIN_PX = 0.0
SEAQUEST_SCREEN_W = 160.0
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
    """True if surfacing is allowed: sub gone/left, or oxygen already low.

    Low O₂ forces the surface WP even if the surface sub is still to the right;
    the waypoint itself sits on a left/mid/right slot away from the sub.
    """
    oxygen = _latest_leaf(raw_observation.oxygen_level)
    oxygen_low = oxygen < OXYGEN_LOW_THRESHOLD
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
    return jnp.logical_or(oxygen_low, sub_left_or_gone)


def _surface_required_from_raw(raw_observation, objects=None):
    """True when surfacing is required and the approach is clear enough.

    Life start sits at y≈46 while oxygen fills 0→64. Treating that as
    ``oxygen_low`` would force the surface COLLECT waypoint every life and
    trap skills on an already-reached target. Low O2 only forces surface once
    underwater, or while refilling at the surface with divers on board.

    Surface WP is gated on surface-sub left/gone unless oxygen is already low
    (must surface; the WP uses a left/mid/right slot away from the sub).
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


def _oxygen_rising_from_raw(raw_observation):
    """True when the oxygen meter increased between the last two stacked frames.

    Oxygen only rises while Seaquest is refilling (actions ignored). Independent
    of player y — the breathing band freezes O₂, drain decreases it.
    """
    o2 = jnp.asarray(raw_observation.oxygen_level, dtype=jnp.float32)
    if o2.ndim == 0 or o2.shape[-1] < 2:
        return jnp.bool_(False)
    return o2[..., -1] > o2[..., -2]


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


def _ellipse_in_range(gap_dx, gap_dy, range_x, range_y):
    """True when AABB edge-gaps lie inside the axis-aligned ellipse."""
    rx = jnp.asarray(range_x, dtype=jnp.float32)
    ry = jnp.asarray(range_y, dtype=jnp.float32)
    enabled = jnp.logical_and(rx > 0.0, ry > 0.0)
    rx = jnp.maximum(rx, jnp.float32(1e-6))
    ry = jnp.maximum(ry, jnp.float32(1e-6))
    radius = (gap_dx / rx) ** 2 + (gap_dy / ry) ** 2
    return jnp.logical_and(enabled, radius <= 1.0)


def _oxygen_scaled_interrupt_range(oxygen, full_range):
    """Shrink (or zero) an enemy interrupt radius when oxygen is low/critical."""
    oxygen = jnp.asarray(oxygen, dtype=jnp.float32)
    full_range = jnp.asarray(full_range, dtype=jnp.float32)
    low_cap = jnp.float32(O2_LOW_ENEMY_INTERRUPT_RANGE)
    crit_cap = jnp.float32(O2_CRITICAL_ENEMY_INTERRUPT_RANGE)
    return jnp.where(
        oxygen < jnp.float32(CRITICAL_OXYGEN_THRESHOLD),
        jnp.minimum(full_range, crit_cap),
        jnp.where(
            oxygen < jnp.float32(OXYGEN_LOW_THRESHOLD),
            jnp.minimum(full_range, low_cap),
            full_range,
        ),
    )


def _select_target(
    raw_observation,
    objects,
    prev_target_idx,
    *,
    danger_close_range=None,
    destroy_interrupt_range=None,
    avoid_interrupt_range=None,
    avoid_interrupt_range_y=None,
    surface_sub_avoid_range_x=None,
    surface_sub_avoid_range_y=None,
    surface_sub_near_range=None,
    diver_lane_y_px=None,
    collect_requires_no_destroy=False,
    collect_edge_margin_px=None,
):
    """Nearest object by AABB surface gap; lane-clear COLLECT; surface WP."""
    del prev_target_idx
    if danger_close_range is None:
        danger_close_range = SURFACE_DANGER_CLOSE_RANGE
    if destroy_interrupt_range is None:
        destroy_interrupt_range = DESTROY_INTERRUPT_RANGE
    if avoid_interrupt_range is None:
        avoid_interrupt_range = AVOID_INTERRUPT_RANGE
    if avoid_interrupt_range_y is None:
        avoid_interrupt_range_y = AVOID_INTERRUPT_RANGE_Y
    if surface_sub_avoid_range_x is None:
        surface_sub_avoid_range_x = SURFACE_SUB_AVOID_RANGE_X
    if surface_sub_avoid_range_y is None:
        surface_sub_avoid_range_y = SURFACE_SUB_AVOID_RANGE_Y
    if surface_sub_near_range is None:
        surface_sub_near_range = SURFACE_SUB_NEAR_RANGE
    if diver_lane_y_px is None:
        diver_lane_y_px = DIVER_LANE_Y_PX
    if collect_edge_margin_px is None:
        collect_edge_margin_px = COLLECT_EDGE_MARGIN_PX
    collect_requires_no_destroy = bool(collect_requires_no_destroy)
    danger_close_range = jnp.asarray(danger_close_range, dtype=jnp.float32)
    destroy_interrupt_range = jnp.asarray(destroy_interrupt_range, dtype=jnp.float32)
    avoid_interrupt_range = jnp.asarray(avoid_interrupt_range, dtype=jnp.float32)
    avoid_interrupt_range_y = jnp.asarray(avoid_interrupt_range_y, dtype=jnp.float32)
    surface_sub_avoid_range_x = jnp.asarray(
        surface_sub_avoid_range_x, dtype=jnp.float32
    )
    surface_sub_avoid_range_y = jnp.asarray(
        surface_sub_avoid_range_y, dtype=jnp.float32
    )
    surface_sub_near_range = jnp.asarray(surface_sub_near_range, dtype=jnp.float32)
    diver_lane_y_px = jnp.asarray(diver_lane_y_px, dtype=jnp.float32)
    collect_edge_margin_px = jnp.asarray(collect_edge_margin_px, dtype=jnp.float32)
    oxygen = _latest_leaf(raw_observation.oxygen_level)
    danger_close_range = _oxygen_scaled_interrupt_range(oxygen, danger_close_range)
    destroy_interrupt_range = _oxygen_scaled_interrupt_range(
        oxygen, destroy_interrupt_range
    )
    avoid_interrupt_range = _oxygen_scaled_interrupt_range(
        oxygen, avoid_interrupt_range
    )
    avoid_interrupt_range_y = _oxygen_scaled_interrupt_range(
        oxygen, avoid_interrupt_range_y
    )
    surface_sub_avoid_range_x = _oxygen_scaled_interrupt_range(
        oxygen, surface_sub_avoid_range_x
    )
    surface_sub_avoid_range_y = _oxygen_scaled_interrupt_range(
        oxygen, surface_sub_avoid_range_y
    )

    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]
    pw = raw_observation.player.width[-1]
    ph = raw_observation.player.height[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    ow = objects.width[-1]
    oh = objects.height[-1]
    active_mask = objects.active[-1] > 0.5
    idxs = jnp.arange(active_mask.shape[0])
    labels = objects.semantic_label[-1]

    # Same edge/surface-gap metric as skill ego distance features.
    distances = aabb_surface_gap_px(
        px, py, pw, ph, ox, oy, ow, oh, y_is_bottom_edge=False
    )
    gap_dx, gap_dy, _ = aabb_closest_delta_px(
        px, py, pw, ph, ox, oy, ow, oh, y_is_bottom_edge=False
    )

    # Exclude surface waypoint from normal nearest scoring.
    selectable = jnp.logical_and(active_mask, idxs != SURFACE_DUMMY_IDX)

    # Surface sub: only when near surface and within a tight range.
    is_surface_sub = idxs == SURFACE_SUB_IDX
    near_surface = py <= SURFACE_Y_THRESHOLD
    surface_sub_eligible = jnp.logical_and(
        is_surface_sub,
        jnp.logical_and(distances <= surface_sub_near_range, near_surface),
    )
    selectable = jnp.logical_and(
        selectable,
        jnp.logical_or(jnp.logical_not(is_surface_sub), surface_sub_eligible),
    )

    is_destroy = labels == LABEL_DESTROY
    is_avoid = labels == LABEL_AVOID
    is_collect = labels == LABEL_COLLECT
    # Lane-clear COLLECT: a diver is ineligible if any DESTROY shares its
    # horizontal row, so far sharks still win over a same-lane diver.
    cy = oy.astype(jnp.float32) + oh.astype(jnp.float32) / 2.0
    same_lane = jnp.logical_and(
        diver_lane_y_px > 0.0,
        jnp.abs(cy[:, None] - cy[None, :]) <= diver_lane_y_px,
    )
    destroy_j = jnp.logical_and(is_destroy, selectable)
    blocks = jnp.logical_and(same_lane, destroy_j[None, :])
    blocks = jnp.logical_and(blocks, idxs[:, None] != idxs[None, :])
    diver_blocked = jnp.logical_and(is_collect, jnp.any(blocks, axis=1))
    selectable_nearest = jnp.logical_and(selectable, jnp.logical_not(diver_blocked))

    # Soft midfield COLLECT: if any lane-clear midfield diver exists, drop edge
    # divers from nearest scoring so COLLECT is not trained on edge rushes.
    cx = ox.astype(jnp.float32) + ow.astype(jnp.float32) / 2.0
    in_midfield = jnp.logical_and(
        cx >= collect_edge_margin_px,
        cx <= (jnp.float32(SEAQUEST_SCREEN_W) - collect_edge_margin_px),
    )
    mid_collect = jnp.logical_and(
        selectable_nearest,
        jnp.logical_and(is_collect, in_midfield),
    )
    edge_collect = jnp.logical_and(
        is_collect,
        jnp.logical_not(in_midfield),
    )
    drop_edge_collect = jnp.logical_and(
        collect_edge_margin_px > 0.0,
        jnp.logical_and(jnp.any(mid_collect), edge_collect),
    )
    selectable_nearest = jnp.logical_and(
        selectable_nearest, jnp.logical_not(drop_edge_collect)
    )

    nearest_dist = jnp.where(selectable_nearest, distances, jnp.inf)
    best_idx = jnp.argmin(nearest_dist).astype(jnp.int32)
    any_selectable = jnp.any(selectable_nearest)
    # Empty scene → trailing meta-NOOP (appended last; never game slot 0).
    idle_noop_idx = jnp.asarray(active_mask.shape[0] - 1, dtype=jnp.int32)

    # Nearby sharks/subs steal COLLECT (and other nearer non-DESTROY) so
    # FRAME_SKIP cannot ram a shark the heuristic has not yet selected.
    destroy_close = jnp.logical_and(
        selectable,
        jnp.logical_and(
            is_destroy,
            jnp.logical_and(
                destroy_interrupt_range > 0.0,
                distances <= destroy_interrupt_range,
            ),
        ),
    )

    # Torpedoes: same-lane corridor. Surface sub: close ellipse at any depth.
    # DESTROY-in-band and missile-in-band compete by nearest gap (no label bias).
    is_missile_avoid = jnp.logical_and(is_avoid, jnp.logical_not(is_surface_sub))
    missile_interrupt = jnp.logical_and(
        selectable,
        jnp.logical_and(
            is_missile_avoid,
            _ellipse_in_range(
                gap_dx, gap_dy, avoid_interrupt_range, avoid_interrupt_range_y
            ),
        ),
    )
    surface_sub_interrupt = jnp.logical_and(
        active_mask,
        jnp.logical_and(
            is_surface_sub,
            _ellipse_in_range(
                gap_dx,
                gap_dy,
                surface_sub_avoid_range_x,
                surface_sub_avoid_range_y,
            ),
        ),
    )
    combat_interrupt = jnp.logical_or(destroy_close, missile_interrupt)
    combat_interrupt_dist = jnp.where(combat_interrupt, distances, jnp.inf)
    best_combat_idx = jnp.argmin(combat_interrupt_dist).astype(jnp.int32)
    any_combat_interrupt = jnp.any(combat_interrupt)
    surface_sub_interrupt_dist = jnp.where(surface_sub_interrupt, distances, jnp.inf)
    best_surface_sub_idx = jnp.argmin(surface_sub_interrupt_dist).astype(jnp.int32)
    any_surface_sub_interrupt = jnp.any(surface_sub_interrupt)
    # Surfacing override still considers both missile + surface-sub AVOID.
    avoid_interrupt = jnp.logical_or(missile_interrupt, surface_sub_interrupt)

    # Surfacing: AVOID at the tight range, DESTROY at the (larger) interrupt
    # range. The anytime AVOID band also interrupts WP so a 50px torpedo is
    # not overwritten by the surface waypoint.
    avoid_close = jnp.logical_and(
        selectable,
        jnp.logical_and(
            is_avoid,
            jnp.logical_and(danger_close_range > 0.0, distances <= danger_close_range),
        ),
    )
    surface_interrupt = jnp.logical_or(
        jnp.logical_or(avoid_close, destroy_close),
        avoid_interrupt,
    )
    interrupt_dist = jnp.where(surface_interrupt, distances, jnp.inf)
    best_interrupt_idx = jnp.argmin(interrupt_dist).astype(jnp.int32)
    any_surface_interrupt = jnp.any(surface_interrupt)

    force_surface = _surface_required_from_raw(raw_observation, objects)

    target_idx = jnp.where(
        any_selectable,
        best_idx,
        idle_noop_idx,
    )
    # Combat interrupt: nearest among in-band DESTROY and in-band missiles.
    target_idx = jnp.where(any_combat_interrupt, best_combat_idx, target_idx)
    # Surface-sub contact dodge may still override (ram risk).
    target_idx = jnp.where(
        any_surface_sub_interrupt, best_surface_sub_idx, target_idx
    )
    if collect_requires_no_destroy:
        # Any live shark/sub beats COLLECT so the collector is never the fighter.
        # Does not steal the surface waypoint: that is re-forced by index below.
        any_destroy = jnp.any(jnp.logical_and(is_destroy, selectable))
        destroy_any_dist = jnp.where(
            jnp.logical_and(is_destroy, selectable), distances, jnp.inf
        )
        nearest_destroy_idx = jnp.argmin(destroy_any_dist).astype(jnp.int32)
        tgt_clip = jnp.clip(target_idx, 0, labels.shape[0] - 1)
        is_collect_tgt = jnp.logical_and(
            target_idx >= 0, labels[tgt_clip] == LABEL_COLLECT
        )
        target_idx = jnp.where(
            jnp.logical_and(is_collect_tgt, any_destroy),
            nearest_destroy_idx,
            target_idx,
        )
    # Prefer surface WP when required, unless a nearby danger overrides.
    target_idx = jnp.where(
        force_surface,
        jnp.where(
            any_surface_interrupt,
            best_interrupt_idx,
            jnp.array(SURFACE_DUMMY_IDX, dtype=jnp.int32),
        ),
        target_idx,
    )
    # Oxygen only increases while the env pins the player to refill. Selecting
    # an object on those frames trains a target/action pair on ignored actions.
    target_idx = jnp.where(
        _oxygen_rising_from_raw(raw_observation),
        idle_noop_idx,
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
    destroy_interrupt_range=None,
    avoid_interrupt_range=None,
    avoid_interrupt_range_y=None,
    surface_sub_avoid_range_x=None,
    surface_sub_avoid_range_y=None,
    surface_sub_near_range=None,
    diver_lane_y_px=None,
    collect_requires_no_destroy=False,
    collect_edge_margin_px=None,
):
    del rng
    target = _select_target(
        raw_observation,
        objects,
        prev_target_idx,
        danger_close_range=danger_close_range,
        destroy_interrupt_range=destroy_interrupt_range,
        avoid_interrupt_range=avoid_interrupt_range,
        avoid_interrupt_range_y=avoid_interrupt_range_y,
        surface_sub_avoid_range_x=surface_sub_avoid_range_x,
        surface_sub_avoid_range_y=surface_sub_avoid_range_y,
        surface_sub_near_range=surface_sub_near_range,
        diver_lane_y_px=diver_lane_y_px,
        collect_requires_no_destroy=collect_requires_no_destroy,
        collect_edge_margin_px=collect_edge_margin_px,
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
    destroy_interrupt_range=None,
    avoid_interrupt_range=None,
    avoid_interrupt_range_y=None,
    surface_sub_avoid_range_x=None,
    surface_sub_avoid_range_y=None,
    surface_sub_near_range=None,
    diver_lane_y_px=None,
    collect_requires_no_destroy=False,
    collect_edge_margin_px=None,
):
    """Stateless wrapper."""
    del rng
    return _select_target(
        raw_observation,
        objects,
        prev_target_idx,
        danger_close_range=danger_close_range,
        destroy_interrupt_range=destroy_interrupt_range,
        avoid_interrupt_range=avoid_interrupt_range,
        avoid_interrupt_range_y=avoid_interrupt_range_y,
        surface_sub_avoid_range_x=surface_sub_avoid_range_x,
        surface_sub_avoid_range_y=surface_sub_avoid_range_y,
        surface_sub_near_range=surface_sub_near_range,
        diver_lane_y_px=diver_lane_y_px,
        collect_requires_no_destroy=collect_requires_no_destroy,
        collect_edge_margin_px=collect_edge_margin_px,
    )
