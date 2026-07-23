"""
Heuristic meta-policy for Seaquest — gameplay-shaped manual plan.

Rules (player-described):
  1. Prefer killing nearby enemies (shallow / close DESTROY).
  2. Soft bias toward the upper water column (stay near the top).
  3. Collect a nearby diver only if no enemy sits behind it.
  4. If a diver is guarded, kill the behind-enemy instead.
  5. Submarines are only targetable when moving away from the player
     (so we can get behind them); sharks are always fair game.
  6. Surface (NAVIGATE WP) when divers ≥ 6, but wait until the surface
     sub is left of us (safer approach) unless oxygen is already low —
     then surface immediately.
  7. AVOID only when a missile/surface-sub is closing on us *and* shares
     our y-band within a short x range (stricter — no early shot picks).
  8. Nearby danger can interrupt surfacing (except critical oxygen).
  9. Inside IMMEDIATE_RANGE, hard priority DESTROY > AVOID > COLLECT.

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
    LABEL_NAVIGATE,
)
from concept_based_learning.environments.manual_policy.context import build_frame_context
from concept_based_learning.environments.manual_policy import register_predicate
from concept_based_learning.environments.manual_policy.state import (
    NO_TARGET_IDX,
    PolicyState,
)

# ---------------------------------------------------------------------------
# Slot layout (stacked objects, index 0 = meta NOOP)
# ---------------------------------------------------------------------------
DIVER_LO, DIVER_HI = 1, 5          # [1, 5)
SHARK_LO, SHARK_HI = 5, 17         # [5, 17)
SUB_LO, SUB_HI = 17, 29            # [17, 29)
SURFACE_SUB_IDX = 29
MISSILE_LO, MISSILE_HI = 30, 34    # [30, 34)
SURFACE_DUMMY_IDX = 34

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
OXYGEN_LOW_THRESHOLD = 10.0  # was 15; surface later (~1/3 more of the tank)
CRITICAL_OXYGEN_THRESHOLD = 4.0
# Prefer engagements above this player/object y (stay near the top).
PREFERRED_MAX_Y = 100.0
DEPTH_PENALTY = 0.35  # added to score per px below PREFERRED_MAX_Y
# Diver "near me" and same-lane enemy checks.
DIVER_NEAR_RANGE = 55.0
LANE_Y_TOL = 14.0
# Sub "moving away" needs a clear horizontal velocity signal.
SUB_AWAY_VX_MIN = 0.5
# AVOID: missiles only when truly on a collision course (was triggering too early).
AVOID_Y_BAND = 8.0
AVOID_APPROACH_X = 28.0
# Really-near override: hard priority DESTROY > AVOID > COLLECT inside this radius.
IMMEDIATE_RANGE = 22.0
# Close danger that may interrupt surfacing (slightly looser than IMMEDIATE).
CLOSE_DANGER_RANGE = 28.0
CLOSE_DANGER_Y = 14.0


def _surface_required_from_raw(raw_observation):
    """True when surfacing is required (low O2 or six divers). Supports batched raw obs."""
    last = jax.tree_util.tree_map(
        lambda x: x[..., -1] if jnp.ndim(x) > 0 else x, raw_observation
    )
    return jnp.logical_or(
        last.oxygen_level < OXYGEN_LOW_THRESHOLD,
        last.collected_divers >= 6,
    )


def _oxygen_low(raw_observation):
    last = jax.tree_util.tree_map(
        lambda x: x[..., -1] if jnp.ndim(x) > 0 else x, raw_observation
    )
    return last.oxygen_level < OXYGEN_LOW_THRESHOLD


def _oxygen_critical(raw_observation):
    last = jax.tree_util.tree_map(
        lambda x: x[..., -1] if jnp.ndim(x) > 0 else x, raw_observation
    )
    return last.oxygen_level < CRITICAL_OXYGEN_THRESHOLD


def _divers_full(raw_observation):
    last = jax.tree_util.tree_map(
        lambda x: x[..., -1] if jnp.ndim(x) > 0 else x, raw_observation
    )
    return last.collected_divers >= 6


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


def _object_vx(objects):
    """Horizontal velocity from the last two frames (0 if stack too short)."""
    x = objects.x
    if int(x.shape[0]) < 2:
        return jnp.zeros(x.shape[-1], dtype=jnp.float32)
    raw = x[-1].astype(jnp.float32) - x[-2].astype(jnp.float32)
    # Zero out wrap / teleport noise.
    return jnp.where(jnp.abs(raw) > 20.0, 0.0, raw)


def _slot_range_mask(n, lo, hi):
    idxs = jnp.arange(n, dtype=jnp.int32)
    return jnp.logical_and(idxs >= lo, idxs < hi)


def _best_scored(mask, score):
    safe = jnp.where(mask, score, jnp.inf)
    idx = jnp.argmin(safe).astype(jnp.int32)
    return idx, jnp.any(mask)


def _depth_score(oy, dist):
    """Closer is better; deeper (larger y) is penalized → stay near the top."""
    depth = jnp.maximum(oy - PREFERRED_MAX_Y, 0.0)
    return dist + DEPTH_PENALTY * depth


def _seaquest_plan_target(raw_observation, objects, state: PolicyState):
    """Core target selection implementing rules 1–6 (before interrupts)."""
    ctx = build_frame_context(raw_observation, objects)
    n = ctx.n
    vx = _object_vx(objects)

    diver_m = jnp.logical_and(ctx.active, _slot_range_mask(n, DIVER_LO, DIVER_HI))
    shark_m = jnp.logical_and(ctx.active, _slot_range_mask(n, SHARK_LO, SHARK_HI))
    sub_m = jnp.logical_and(ctx.active, _slot_range_mask(n, SUB_LO, SUB_HI))

    # Rule 5: subs only when moving away from the player.
    sub_right = ctx.ox > ctx.px
    moving_away = jnp.logical_or(
        jnp.logical_and(sub_right, vx > SUB_AWAY_VX_MIN),
        jnp.logical_and(jnp.logical_not(sub_right), vx < -SUB_AWAY_VX_MIN),
    )
    sub_ok = jnp.logical_and(sub_m, moving_away)
    destroy_m = jnp.logical_or(shark_m, sub_ok)

    # Rule 3–4: divers near us; behind-enemy → DESTROY that enemy; else COLLECT.
    near_diver = jnp.logical_and(diver_m, ctx.distances <= DIVER_NEAR_RANGE)

    guard_idx = jnp.int32(NO_TARGET_IDX)
    guard_any = jnp.bool_(False)
    guard_dist = jnp.inf
    safe_collect_mask = near_diver
    for i in range(DIVER_LO, DIVER_HI):
        is_d = near_diver[i]
        d_x = ctx.ox[i]
        d_y = ctx.oy[i]
        d_abs = jnp.abs(d_x - ctx.px)
        same_lane = jnp.abs(ctx.oy - d_y) <= LANE_Y_TOL
        same_horz_side = (ctx.ox - ctx.px) * (d_x - ctx.px) > 0
        further = jnp.abs(ctx.ox - ctx.px) > d_abs + 2.0
        behind = jnp.logical_and(
            destroy_m,
            jnp.logical_and(same_lane, jnp.logical_and(same_horz_side, further)),
        )
        behind = jnp.logical_and(behind, is_d)
        b_idx, b_any = _best_scored(behind, ctx.distances)
        b_dist = jnp.where(b_any, ctx.distances[b_idx], jnp.inf)
        take = jnp.logical_and(b_any, b_dist < guard_dist)
        guard_idx = jnp.where(take, b_idx, guard_idx)
        guard_dist = jnp.where(take, b_dist, guard_dist)
        guard_any = jnp.logical_or(guard_any, b_any)
        safe_collect_mask = safe_collect_mask.at[i].set(
            jnp.logical_and(near_diver[i], jnp.logical_not(b_any))
        )

    safe_collect_idx, any_safe_collect = _best_scored(safe_collect_mask, ctx.distances)

    # Rule 1–2: nearest shallow destroy among eligible.
    destroy_score = _depth_score(ctx.oy, ctx.distances)
    destroy_idx, any_destroy = _best_scored(destroy_m, destroy_score)

    # Priority: guarded-diver enemy → safe collect → shallow destroy.
    fresh = jnp.int32(NO_TARGET_IDX)
    fresh = jnp.where(any_destroy, destroy_idx, fresh)
    fresh = jnp.where(any_safe_collect, safe_collect_idx, fresh)
    fresh = jnp.where(guard_any, guard_idx, fresh)

    # Rule 6: surfacing (needed before stickiness so we don't lock the WP forever).
    o2_low = _oxygen_low(raw_observation)
    o2_crit = _oxygen_critical(raw_observation)
    full = _divers_full(raw_observation)
    surf_active = jnp.where(SURFACE_SUB_IDX < n, ctx.active[SURFACE_SUB_IDX], False)
    surf_x = jnp.where(SURFACE_SUB_IDX < n, ctx.ox[SURFACE_SUB_IDX], ctx.px)
    sub_left_or_gone = jnp.logical_or(jnp.logical_not(surf_active), surf_x < ctx.px)
    want_surface = jnp.logical_or(
        o2_low,
        jnp.logical_and(full, sub_left_or_gone),
    )

    # Sticky DESTROY / COLLECT; NAVIGATE only while still surfacing.
    mission = state.mission_target
    valid = jnp.logical_and(mission >= 0, mission < n)
    m_active = jnp.where(valid, ctx.active[mission], False)
    m_label = jnp.where(valid, ctx.labels[mission], -1)
    m_clip = jnp.clip(mission, 0, n - 1)
    m_still_destroy = jnp.logical_and(m_label == LABEL_DESTROY, destroy_m[m_clip])
    m_still_collect = jnp.logical_and(m_label == LABEL_COLLECT, ctx.active[m_clip])
    m_still_nav = jnp.logical_and(m_label == LABEL_NAVIGATE, want_surface)
    keep = jnp.logical_and(
        m_active,
        jnp.logical_or(m_still_destroy, jnp.logical_or(m_still_collect, m_still_nav)),
    )
    # Drop a stuck diver mission if it just became guarded.
    stick_diver = jnp.logical_and(keep, m_label == LABEL_COLLECT)
    keep = jnp.logical_and(keep, jnp.logical_not(jnp.logical_and(stick_diver, guard_any)))

    target = jnp.where(keep, mission, fresh)
    target = jnp.where(want_surface, jnp.int32(SURFACE_DUMMY_IDX), target)

    new_state = state.replace(mission_target=target)
    return target, want_surface, o2_crit, new_state, ctx, vx


def _apply_seaquest_interrupts(target, want_surface, o2_crit, ctx, vx, state):
    """Rules 7–8 + immediate proximity priority DESTROY > AVOID > COLLECT."""
    n = ctx.n
    avoid_slots = jnp.logical_or(
        _slot_range_mask(n, MISSILE_LO, MISSILE_HI),
        jnp.arange(n) == SURFACE_SUB_IDX,
    )
    avoid_m = jnp.logical_and(ctx.active, avoid_slots)
    destroy_m = jnp.logical_and(ctx.active, ctx.labels == LABEL_DESTROY)
    collect_m = jnp.logical_and(ctx.active, ctx.labels == LABEL_COLLECT)

    # Rule 7: AVOID only if closing *and* sharing our y-band within short x.
    on_right = ctx.ox > ctx.px
    closing = jnp.logical_or(
        jnp.logical_and(on_right, vx < -SUB_AWAY_VX_MIN),
        jnp.logical_and(jnp.logical_not(on_right), vx > SUB_AWAY_VX_MIN),
    )
    avoid_threat = jnp.logical_and(
        avoid_m,
        jnp.logical_and(
            closing,
            jnp.logical_and(ctx.dy <= AVOID_Y_BAND, ctx.dx <= AVOID_APPROACH_X),
        ),
    )
    avoid_idx, any_avoid = _best_scored(avoid_threat, ctx.distances)

    # Surfacing interrupt: nearby DESTROY/AVOID (not divers / surface WP).
    danger_m = jnp.logical_or(destroy_m, avoid_m)
    in_danger = jnp.logical_and(
        danger_m,
        jnp.logical_and(ctx.dx <= CLOSE_DANGER_RANGE, ctx.dy <= CLOSE_DANGER_Y),
    )
    in_danger = jnp.logical_and(
        in_danger,
        jnp.logical_not(
            jnp.logical_and(want_surface, jnp.arange(n) == SURFACE_SUB_IDX)
        ),
    )
    danger_idx, any_danger = _best_scored(in_danger, ctx.distances)

    # Immediate proximity: whatever is right next to us, fixed label priority.
    in_imm = jnp.logical_and(
        ctx.active,
        jnp.logical_and(
            ctx.distances <= IMMEDIATE_RANGE,
            jnp.logical_or(
                destroy_m, jnp.logical_or(avoid_m, collect_m)
            ),
        ),
    )
    imm_destroy_idx, imm_destroy = _best_scored(
        jnp.logical_and(in_imm, destroy_m), ctx.distances
    )
    imm_avoid_idx, imm_avoid = _best_scored(
        jnp.logical_and(in_imm, avoid_m), ctx.distances
    )
    imm_collect_idx, imm_collect = _best_scored(
        jnp.logical_and(in_imm, collect_m), ctx.distances
    )
    imm_idx = jnp.int32(NO_TARGET_IDX)
    imm_idx = jnp.where(imm_collect, imm_collect_idx, imm_idx)
    imm_idx = jnp.where(imm_avoid, imm_avoid_idx, imm_idx)
    imm_idx = jnp.where(imm_destroy, imm_destroy_idx, imm_idx)
    any_imm = jnp.logical_or(imm_destroy, jnp.logical_or(imm_avoid, imm_collect))

    allow_danger = jnp.logical_or(jnp.logical_not(want_surface), jnp.logical_not(o2_crit))
    out = target
    out = jnp.where(any_avoid, avoid_idx, out)
    out = jnp.where(jnp.logical_and(any_danger, allow_danger), danger_idx, out)
    # Highest priority: really-near objects (DESTROY > AVOID > COLLECT).
    out = jnp.where(any_imm, imm_idx, out)

    mission = state.mission_target
    mission = jnp.where(
        jnp.logical_and(want_surface, out != SURFACE_DUMMY_IDX),
        jnp.int32(SURFACE_DUMMY_IDX),
        mission,
    )
    mission = jnp.where(
        jnp.logical_or(out == mission, want_surface),
        mission,
        out,
    )
    return out.astype(jnp.int32), state.replace(mission_target=mission)


def get_heuristic_target_with_state(raw_observation, objects, prev_target_idx, state, rng):
    del prev_target_idx, rng
    target, want_surface, o2_crit, state, ctx, vx = _seaquest_plan_target(
        raw_observation, objects, state
    )
    target, state = _apply_seaquest_interrupts(
        target, want_surface, o2_crit, ctx, vx, state
    )
    any_active = jnp.any(ctx.active)
    target = jnp.where(
        jnp.logical_or(any_active, want_surface),
        target,
        jnp.int32(NO_TARGET_IDX),
    )
    return target.astype(jnp.int32), state


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    state = PolicyState.zero(no_target_idx=NO_TARGET_IDX).replace(
        mission_target=jnp.asarray(prev_target_idx, dtype=jnp.int32)
    )
    target, _ = get_heuristic_target_with_state(
        raw_observation, objects, prev_target_idx, state, rng
    )
    return target
