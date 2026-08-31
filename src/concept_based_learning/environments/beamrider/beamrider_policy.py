"""Env-specific Beamrider heuristic meta-policy.

Player sits on the bottom beams (y≈164) and shoots upward. Targeting priority:

  1. AVOID (shots / chasing meteoroids / falling rocks / lane blockers /
     bouncer / shot rejuvenator debris) when in our lane or within
     CLOSE_AVOID_RANGE (AABB surface gap).
  2. DESTROY (UFOs / kamikaze / mothership) when on our beam ahead and within
     DESTROY_SHOOT_RANGE.
  3. COLLECT (coins / intact rejuvenator life) when no DESTROY sits in front
     of it on the same lane.
  4. Else nearest active DESTROY on a reachable beam (lanes 1–5; outer
     diagonals 0/6 are approach-only and ignored), else NO_TARGET.

All proximity checks use AABB surface gap (skill ego distance metric).
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

NO_TARGET_IDX = -1

# Horizontal "same beam / lane" tolerance (centre-x px).
LANE_X_THRESH = 10.0
# AVOID: react when in-lane or this close (AABB surface gap).
CLOSE_AVOID_RANGE = 30.0
# DESTROY: shoot when in-lane ahead or within this surface gap.
DESTROY_SHOOT_RANGE = 90.0
# JAXAtari beam geometry (7 diagonal lanes; 0/6 are unreachable approach rails).
_TOP_CLIP = 43.0
_TOP_LANES_X = jnp.array([38.0, 61.0, 71.0, 81.0, 91.0, 102.0, 123.0], dtype=jnp.float32)
_LANE_DX_OVER_DY = jnp.array([-0.5, -0.25, -0.13, 0.0, 0.13, 0.25, 0.5], dtype=jnp.float32)


def _on_unreachable_outer_beam(o_cx, o_cy):
    """True when object centre sits on diagonal beams 0 or 6 (player cannot engage)."""
    dy = o_cy - _TOP_CLIP
    lane_x_at_y = _TOP_LANES_X[None, :] + _LANE_DX_OVER_DY[None, :] * dy[:, None]
    lane_id = jnp.argmin(jnp.abs(lane_x_at_y - o_cx[:, None]), axis=1)
    return jnp.logical_or(lane_id == 0, lane_id == 6)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del prev_target_idx, rng

    distances, p_cx, p_cy, o_cx, o_cy = player_object_aabb_gaps(
        raw_observation, objects
    )
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    # Beamrider: player at bottom; "ahead / in front" = smaller y (toward top).
    ahead = o_cy < p_cy
    in_lane = jnp.abs(o_cx - p_cx) <= LANE_X_THRESH

    is_avoid = labels == LABEL_AVOID
    is_destroy = labels == LABEL_DESTROY
    is_collect = labels == LABEL_COLLECT
    on_outer_beam = _on_unreachable_outer_beam(o_cx, o_cy)
    reachable_beam = jnp.logical_not(on_outer_beam)

    # ------------------------------------------------------------------
    # 1. AVOID: in-lane threats or anything within CLOSE_AVOID_RANGE
    # ------------------------------------------------------------------
    avoid_threat = jnp.logical_and(
        jnp.logical_and(active, is_avoid),
        jnp.logical_or(in_lane, distances <= CLOSE_AVOID_RANGE),
    )
    have_avoid = jnp.any(avoid_threat)
    avoid_idx = jnp.argmin(jnp.where(avoid_threat, distances, jnp.inf)).astype(
        jnp.int32
    )

    # ------------------------------------------------------------------
    # 2. DESTROY: shootable enemies — same beam/lane ahead within laser reach
    # ------------------------------------------------------------------
    destroy_shootable = jnp.logical_and(
        jnp.logical_and(active, is_destroy),
        jnp.logical_and(
            reachable_beam,
            jnp.logical_and(
                jnp.logical_and(in_lane, ahead),
                distances <= DESTROY_SHOOT_RANGE,
            ),
        ),
    )
    have_destroy = jnp.any(destroy_shootable)
    destroy_idx = jnp.argmin(jnp.where(destroy_shootable, distances, jnp.inf)).astype(
        jnp.int32
    )

    # Fallback DESTROY: reachable beams only (ignore outer 0/6 approach rails).
    any_destroy = jnp.logical_and(
        jnp.logical_and(active, is_destroy),
        reachable_beam,
    )
    have_any_destroy = jnp.any(any_destroy)
    any_destroy_idx = jnp.argmin(jnp.where(any_destroy, distances, jnp.inf)).astype(
        jnp.int32
    )

    # ------------------------------------------------------------------
    # 3. COLLECT: only if no DESTROY sits between player and collectible
    #    on the same lane (enemy "in front of" the life/coin).
    # ------------------------------------------------------------------
    collect_mask = jnp.logical_and(active, is_collect)

    # For each collectible c: blocked if some DESTROY d shares c's lane and
    # lies between player and c on y (p_cy > d_cy >= c_cy when c is ahead).
    # Broadcast: destroy (N,) vs collect (N,) → (N_d, N_c) then any over d.
    d_cx = o_cx[:, None]
    d_cy = o_cy[:, None]
    c_cx = o_cx[None, :]
    c_cy = o_cy[None, :]
    d_active_destroy = jnp.logical_and(
        jnp.logical_and(active, is_destroy), reachable_beam
    )[:, None]
    same_lane_as_collect = jnp.abs(d_cx - c_cx) <= LANE_X_THRESH
    between = jnp.logical_and(d_cy < p_cy, d_cy >= c_cy)
    blocking = jnp.logical_and(
        d_active_destroy, jnp.logical_and(same_lane_as_collect, between)
    )
    collect_blocked = jnp.any(blocking, axis=0)
    collect_clear = jnp.logical_and(collect_mask, jnp.logical_not(collect_blocked))
    have_collect = jnp.any(collect_clear)
    collect_idx = jnp.argmin(jnp.where(collect_clear, distances, jnp.inf)).astype(
        jnp.int32
    )

    # ------------------------------------------------------------------
    # Priority cascade
    # ------------------------------------------------------------------
    target = jnp.array(NO_TARGET_IDX, dtype=jnp.int32)
    target = jnp.where(have_any_destroy, any_destroy_idx, target)
    target = jnp.where(have_collect, collect_idx, target)
    target = jnp.where(have_destroy, destroy_idx, target)
    target = jnp.where(have_avoid, avoid_idx, target)
    return target.astype(jnp.int32)
