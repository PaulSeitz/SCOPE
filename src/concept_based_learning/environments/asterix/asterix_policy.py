"""Env-specific Asterix heuristic meta-policy.

Asterix is an 8-lane horizontal scroller: enemies (AVOID) and collectibles
(COLLECT) share fixed lane y's. Targeting priority:

  1. AVOID when on our lane (or within CLOSE_AVOID_RANGE AABB surface gap),
     except off-lane enemies below the player (they only encourage dodging up).
  2. Else nearest COLLECT that is not blocked by an enemy on the same lane
     between the player and the item — ranked with a middle-lane bias so the
     agent does not climb toward top-lane items.
  3. Else nearest COLLECT (same middle-lane bias), else NO_TARGET.

Distances use AABB surface gap (skill ego metric).
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_COLLECT
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

NO_TARGET_IDX = -1

# Same-lane tolerance (centre-y px); stage spacing is 16.
LANE_Y_THRESH = 8.0
# AVOID: react when same-lane or this close (AABB surface gap).
CLOSE_AVOID_RANGE = 28.0
# COLLECT ranking: prefer middle lanes (JAXAtari lane centres ≈31..143, step 16).
PREFERRED_COLLECT_CY = 87.0  # between lanes 3–4 (player spawn band)
COLLECT_LANE_BIAS = 0.6  # rank penalty per px of |cy − preferred| (~10 px / lane)


def _collect_rank(distances, o_cy):
    """Distance plus middle-lane bias for COLLECT target selection."""
    lane_penalty = COLLECT_LANE_BIAS * jnp.abs(o_cy - PREFERRED_COLLECT_CY)
    return distances + lane_penalty


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del prev_target_idx, rng

    distances, p_cx, p_cy, o_cx, o_cy = player_object_aabb_gaps(
        raw_observation, objects
    )
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]

    same_lane = jnp.abs(o_cy - p_cy) <= LANE_Y_THRESH

    is_avoid = labels == LABEL_AVOID
    is_collect = labels == LABEL_COLLECT

    # ------------------------------------------------------------------
    # 1. AVOID: same-lane threats or close off-lane threats, but ignore
    #    off-lane enemies below us — reacting to them only dodges upward.
    # ------------------------------------------------------------------
    below_player = o_cy > p_cy + LANE_Y_THRESH
    off_lane_below = jnp.logical_and(jnp.logical_not(same_lane), below_player)
    close_avoid = jnp.logical_and(
        distances <= CLOSE_AVOID_RANGE,
        jnp.logical_not(off_lane_below),
    )
    avoid_threat = jnp.logical_and(
        jnp.logical_and(active, is_avoid),
        jnp.logical_or(same_lane, close_avoid),
    )
    have_avoid = jnp.any(avoid_threat)
    avoid_idx = jnp.argmin(jnp.where(avoid_threat, distances, jnp.inf)).astype(
        jnp.int32
    )

    # ------------------------------------------------------------------
    # 2. COLLECT: prefer items with a clear path on their lane
    # ------------------------------------------------------------------
    collect_mask = jnp.logical_and(active, is_collect)

    # Blocked if some AVOID shares the collectible's lane and lies between
    # player and item on x (same horizontal corridor).
    a_cx = o_cx[:, None]
    a_cy = o_cy[:, None]
    c_cx = o_cx[None, :]
    c_cy = o_cy[None, :]
    a_active_avoid = jnp.logical_and(active, is_avoid)[:, None]
    same_lane_as_collect = jnp.abs(a_cy - c_cy) <= LANE_Y_THRESH
    # Between on x: enemy x is between player centre and collectible centre.
    lo = jnp.minimum(p_cx, c_cx)
    hi = jnp.maximum(p_cx, c_cx)
    between_x = jnp.logical_and(a_cx >= lo, a_cx <= hi)
    blocking = jnp.logical_and(
        a_active_avoid, jnp.logical_and(same_lane_as_collect, between_x)
    )
    collect_blocked = jnp.any(blocking, axis=0)
    collect_clear = jnp.logical_and(collect_mask, jnp.logical_not(collect_blocked))
    collect_rank = _collect_rank(distances, o_cy)
    have_clear = jnp.any(collect_clear)
    clear_idx = jnp.argmin(jnp.where(collect_clear, collect_rank, jnp.inf)).astype(
        jnp.int32
    )

    have_any_collect = jnp.any(collect_mask)
    any_collect_idx = jnp.argmin(jnp.where(collect_mask, collect_rank, jnp.inf)).astype(
        jnp.int32
    )

    target = jnp.array(NO_TARGET_IDX, dtype=jnp.int32)
    target = jnp.where(have_any_collect, any_collect_idx, target)
    target = jnp.where(have_clear, clear_idx, target)
    target = jnp.where(have_avoid, avoid_idx, target)
    return target.astype(jnp.int32)
