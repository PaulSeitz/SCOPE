"""Shared skill reward helpers aligned with the working Seaquest reward contract."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

NEAR_PLAYER_TOLERANCE = 4.0
AVOID_FAILURE_PENALTY = -10.0


def last_frame(tree):
    """Take the last frame from a possibly stacked pytree."""
    return jax.tree_util.tree_map(lambda x: x[-1] if jnp.ndim(x) > 0 else x, tree)


def _bbox_overlap(player, target) -> jnp.ndarray:
    p_left, p_right = player.x, player.x + player.width
    p_top, p_bottom = player.y, player.y + player.height
    t_left, t_right = target.x, target.x + target.width
    t_top, t_bottom = target.y, target.y + target.height
    horiz = jnp.logical_and(p_left <= t_right, p_right >= t_left)
    vert = jnp.logical_and(p_top <= t_bottom, p_bottom >= t_top)
    return jnp.logical_and(horiz, vert)


def _near_player_overlap(player, target, tolerance: float) -> jnp.ndarray:
    p_left, p_right = player.x, player.x + player.width
    p_top, p_bottom = player.y, player.y + player.height
    t_left, t_right = target.x, target.x + target.width
    t_top, t_bottom = target.y, target.y + target.height
    horizontal_overlap = jnp.logical_and(
        p_left - tolerance <= t_right, p_right + tolerance >= t_left
    )
    vertical_overlap = jnp.logical_and(
        p_top - tolerance <= t_bottom, p_bottom + tolerance >= t_top
    )
    return jnp.logical_and(horizontal_overlap, vertical_overlap)


def _at_screen_edge(
    target,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> jnp.ndarray:
    edge = jnp.array(False)
    if x_min is not None:
        edge = jnp.logical_or(edge, target.x <= x_min)
    if x_max is not None:
        edge = jnp.logical_or(edge, target.x >= x_max)
    if y_min is not None:
        edge = jnp.logical_or(edge, target.y <= y_min)
    if y_max is not None:
        edge = jnp.logical_or(edge, target.y >= y_max)
    return edge


def reward_transition_context(
    prev_raw_obs,
    prev_target,
    next_target,
    *,
    near_tolerance: float = NEAR_PLAYER_TOLERANCE,
    x_min: Optional[float] = 15.0,
    x_max: Optional[float] = 145.0,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    """Common per-transition flags used across environment reward extractors."""
    prev_player = last_frame(prev_raw_obs.player)
    prev_target = last_frame(prev_target)
    next_target = last_frame(next_target)

    was_active = prev_target.active == 1
    is_active = next_target.active == 1
    disappeared = jnp.logical_and(was_active, jnp.logical_not(is_active))
    hit_by_assigned = _bbox_overlap(prev_player, prev_target)
    near_player = _near_player_overlap(prev_player, prev_target, near_tolerance)
    at_screen_edge = _at_screen_edge(
        prev_target,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    return prev_player, prev_target, next_target, disappeared, hit_by_assigned, near_player, at_screen_edge


def pack_skill_rewards(
    *,
    destroy_success: jnp.ndarray,
    destroy_failure: jnp.ndarray,
    collect_success: jnp.ndarray,
    avoid_success: jnp.ndarray,
    avoid_failure: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pack per-skill rewards and terminal masks using the Seaquest contract:
    +1 sparse success, avoid failure -10, destroy failure marks done with 0 reward.
    """
    destroy_reward = jnp.where(destroy_success, 1.0, 0.0)
    collect_reward = jnp.where(collect_success, 1.0, 0.0)
    avoid_reward = jnp.where(
        avoid_success,
        1.0,
        jnp.where(avoid_failure, AVOID_FAILURE_PENALTY, 0.0),
    )

    destroy_done = jnp.logical_or(destroy_success, destroy_failure)
    collect_done = collect_success
    avoid_done = jnp.logical_or(avoid_success, avoid_failure)

    rewards = jnp.array(
        [destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32
    )
    skill_done = jnp.array([destroy_done, collect_done, avoid_done], dtype=jnp.bool_)
    return rewards, skill_done


def standard_avoid_outcomes(
    prev_target,
    *,
    disappeared: jnp.ndarray,
    hit_by_assigned: jnp.ndarray,
    life_lost: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Object-relative (hindsight relabeling): scored for the selected object
    # regardless of its true label, so every skill learns off-policy from every
    # transition. `prev_target` is kept for call-site compatibility.
    del prev_target
    avoid_success = jnp.logical_and(disappeared, jnp.logical_not(hit_by_assigned))
    avoid_failure = jnp.logical_and(life_lost > 0, hit_by_assigned)
    return avoid_success, avoid_failure


def standard_collect_success(
    prev_target,
    *,
    disappeared: jnp.ndarray,
    near_player: jnp.ndarray,
    collect_success: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    # Object-relative (hindsight relabeling): scored for the selected object
    # regardless of its true label. `prev_target` kept for call-site compat.
    del prev_target
    if collect_success is None:
        collect_success = jnp.logical_and(disappeared, near_player)
    return collect_success


def standard_destroy_outcomes(
    prev_target,
    *,
    disappeared: jnp.ndarray,
    hit_by_assigned: jnp.ndarray,
    at_screen_edge: jnp.ndarray,
    life_lost: jnp.ndarray,
    destroy_success: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Object-relative (hindsight relabeling): scored for the selected object
    # regardless of its true label. `prev_target` kept for call-site compat.
    del prev_target
    if destroy_success is None:
        destroy_success = jnp.logical_and(
            disappeared,
            jnp.logical_and(
                jnp.logical_not(hit_by_assigned), jnp.logical_not(at_screen_edge)
            ),
        )
    destroy_failure = jnp.logical_and(life_lost > 0, hit_by_assigned)
    return destroy_success, destroy_failure
