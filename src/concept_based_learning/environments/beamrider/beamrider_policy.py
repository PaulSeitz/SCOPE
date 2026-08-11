"""Env-specific Beamrider heuristic meta-policy.

Player sits on the bottom beams (y≈164) and shoots upward. Targeting priority:

  1. AVOID (shots / falling rocks / lane blockers / bouncer) when in our lane
     or within CLOSE_AVOID_RANGE (AABB surface gap).
  2. DESTROY (UFOs / meteoroids / kamikaze / mothership) when on our beam
     ahead and within DESTROY_SHOOT_RANGE.
  3. COLLECT (coins / rejuvenator life) when no DESTROY sits in front of it
     on the same lane.
  4. Else nearest active DESTROY (to move toward a beam with enemies), else
     NO_TARGET.

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
            jnp.logical_and(in_lane, ahead),
            distances <= DESTROY_SHOOT_RANGE,
        ),
    )
    have_destroy = jnp.any(destroy_shootable)
    destroy_idx = jnp.argmin(jnp.where(destroy_shootable, distances, jnp.inf)).astype(
        jnp.int32
    )

    # Fallback DESTROY: any active enemy (when nothing is in shoot range yet).
    any_destroy = jnp.logical_and(active, is_destroy)
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
    d_active_destroy = jnp.logical_and(active, is_destroy)[:, None]
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
