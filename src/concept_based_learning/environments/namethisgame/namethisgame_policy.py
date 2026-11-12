"""Name This Game heuristic meta-policy.

Default: pick the lowest on-screen target (max screen-y) among DESTROY
(shark + tentacle tips).

Override: while the oxygen COLLECT line is active, stick on it — unless a shark
is in one of the bottom two lanes, or the oxygen bar is still above ~80%
(then fall back to lowest DESTROY).
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_COLLECT, LABEL_DESTROY
from concept_based_learning.environments.namethisgame.namethisgame_label import (
    BOTTOM_SHARK_LANES_Y,
)
from concept_based_learning.observations.egocentric import player_object_aabb_gaps

NO_TARGET_IDX = -1

# After meta-NOOP: shark=1, tentacles=2..9, oxygen=10.
_SHARK_SLOT = 1
_OXYGEN_SLOT = 10
_LANE_Y_SLACK = 4.0
# jax_namethisgame hud_bar_initial_px — full tank.
_OXYGEN_BAR_FULL_PX = 128.0
# Skip oxygen COLLECT while the bar is still comfortably full.
_OXYGEN_COLLECT_MAX_FRAC = 0.8


def _shark_in_bottom_lanes(objects) -> jnp.ndarray:
    """True when the shark is active in one of the bottom two lane y's."""
    active = objects.active[-1] > 0.5
    shark_active = active[_SHARK_SLOT]
    # jax_namethisgame stores shark_y as the lane top-left (matches shark_lanes_y).
    shark_y = objects.y[-1][_SHARK_SLOT].astype(jnp.float32)
    in_lane = jnp.bool_(False)
    for lane_y in BOTTOM_SHARK_LANES_Y:
        in_lane = jnp.logical_or(
            in_lane, jnp.abs(shark_y - float(lane_y)) <= _LANE_Y_SLACK
        )
    return jnp.logical_and(shark_active, in_lane)


def _oxygen_frac(raw_observation) -> jnp.ndarray:
    """Latest-frame oxygen bar fraction in [0, 1]."""
    oxy = jnp.asarray(raw_observation.oxygen_frames_remaining, dtype=jnp.float32)
    oxy_last = oxy[-1] if oxy.ndim > 0 else oxy
    return jnp.clip(oxy_last / jnp.float32(_OXYGEN_BAR_FULL_PX), 0.0, 1.0)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del rng
    distances, _, _, _, _ = player_object_aabb_gaps(raw_observation, objects)
    del distances
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]
    # Lowest on screen = largest centre-y (screen y grows downward).
    o_cy = objects.y[-1] + objects.height[-1] * 0.5

    is_destroy = labels == LABEL_DESTROY
    is_collect = labels == LABEL_COLLECT
    destroy_mask = jnp.logical_and(active, is_destroy)
    collect_mask = jnp.logical_and(active, is_collect)

    # Rank DESTROY by lowest (max y); inactive → -inf so they never win.
    destroy_rank = jnp.where(destroy_mask, o_cy, -jnp.inf)
    lowest_destroy = jnp.argmax(destroy_rank).astype(jnp.int32)
    have_destroy = jnp.any(destroy_mask)

    # Oxygen COLLECT: sticky while active (prefer prev if still the oxygen slot).
    oxygen_active = jnp.logical_and(
        active[_OXYGEN_SLOT], labels[_OXYGEN_SLOT] == LABEL_COLLECT
    )
    prev_is_oxygen = jnp.logical_and(
        prev_target_idx == _OXYGEN_SLOT,
        oxygen_active,
    )
    collect_idx = jnp.where(
        prev_is_oxygen,
        jnp.int32(_OXYGEN_SLOT),
        jnp.argmin(jnp.where(collect_mask, 0.0, jnp.inf)).astype(jnp.int32),
    )
    have_collect = jnp.any(collect_mask)

    shark_blocks_collect = _shark_in_bottom_lanes(objects)
    need_oxygen = _oxygen_frac(raw_observation) <= jnp.float32(_OXYGEN_COLLECT_MAX_FRAC)
    prefer_collect = jnp.logical_and(
        have_collect,
        jnp.logical_and(jnp.logical_not(shark_blocks_collect), need_oxygen),
    )

    target = jnp.int32(NO_TARGET_IDX)
    target = jnp.where(have_destroy, lowest_destroy, target)
    target = jnp.where(prefer_collect, collect_idx, target)
    return target.astype(jnp.int32)
