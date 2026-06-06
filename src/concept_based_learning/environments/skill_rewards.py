"""
Global skill reward contract — env-independent.

Reward magnitudes and trigger conditions:
  AVOID   : -1  on rising edge of AABB overlap (player enters object hitbox)
  COLLECT : +1  on rising edge of AABB overlap (player enters object hitbox)
  DESTROY : +1  when next_target.was_killed == 1 (set by env wrapper from raw state)
            -1  when the player dies while in contact with this specific object

All rewards fire on transition edges so sustained overlap cannot be farmed.
DESTROY relies on was_killed rather than position-based disappearance detection
because JAX-Atari zeros position for both torpedo kills and scroll-offs.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

# Unified reward magnitudes — single source of truth.
AVOID_FAILURE_REWARD   = -1.0
COLLECT_SUCCESS_REWARD =  1.0
DESTROY_SUCCESS_REWARD =  1.0
DESTROY_FAILURE_REWARD = -1.0
# Shrink interaction hitboxes for edge-triggered overlap rewards.
# This reduces reward farming from grazing at object borders.
OVERLAP_INSET_PX = 1


def last_frame(tree):
    """Take the last frame from a possibly stacked pytree."""
    return jax.tree_util.tree_map(lambda x: x[-1] if jnp.ndim(x) > 0 else x, tree)


def object_reward_eligible(prev_target, next_target) -> jnp.ndarray:
    """
    Whether to run per-object skill rewards for this transition.

    DESTROY success is signaled on ``next_target.was_killed`` while ``active`` is
    already 0, so filtering only on post-step ``active`` drops kill rewards.
    """
    prev = last_frame(prev_target)
    next_ = last_frame(next_target)
    prev_active = prev.active > 0
    next_active = next_.active > 0
    killed = next_.was_killed > 0
    return jnp.logical_or(prev_active, jnp.logical_or(next_active, killed))


def _agent_bbox(raw_obs):
    """
    Return the controllable-agent object from a raw env observation.

    Keep reward logic env-agnostic by resolving common names here instead of
    hardcoding per-env branches in reward computations.

    TODO: Move this resolution into env-specific hooks/components and pass a
    normalized agent bbox into the reward extractor.
    """
    for name in ("player", "chicken", "agent"):
        if hasattr(raw_obs, name):
            return getattr(raw_obs, name)
    raise AttributeError(
        "Raw observation does not expose a known controllable agent field "
        "(expected one of: player, chicken, agent)."
    )


def _bbox_overlap(player, target, inset_px: int = 0) -> jnp.ndarray:
    """Standard AABB overlap with optional inward target inset."""
    inset = jnp.array(inset_px, dtype=target.x.dtype)
    target_x_min = target.x + inset
    target_y_min = target.y + inset
    target_w = jnp.maximum(target.width - 2 * inset, 1)
    target_h = jnp.maximum(target.height - 2 * inset, 1)
    target_x_max = target_x_min + target_w
    target_y_max = target_y_min + target_h

    horiz = jnp.logical_and(
        player.x <= target_x_max,
        player.x + player.width >= target_x_min,
    )
    vert  = jnp.logical_and(
        player.y <= target_y_max,
        player.y + player.height >= target_y_min,
    )
    return jnp.logical_and(horiz, vert)


def reward_transition_context(prev_raw_obs, next_raw_obs, prev_target, next_target):
    """
    Reduce stacked observations to last-frame scalars and compute AABB overlaps.

    next_overlap uses prev_target's position when next_target just went inactive
    (active 1→0).  Some envs (e.g. JAXAtari Seaquest) zero the object position
    at the same step it is collected, making next_target.x/y = 0 and causing the
    overlap check to fail even though the player genuinely entered the hitbox.
    Falling back to prev_target position for the overlap test recovers that signal.

    Returns:
        prev_player  — last-frame player struct at t
        prev_target  — last-frame target struct at t
        next_target  — last-frame target struct at t+1
        prev_overlap — bool: AABB overlap at t
        next_overlap — bool: AABB overlap at t+1 (using prev_target pos if target just vanished)
    """
    prev_player = last_frame(_agent_bbox(prev_raw_obs))
    next_player = last_frame(_agent_bbox(next_raw_obs))
    prev_target = last_frame(prev_target)
    next_target = last_frame(next_target)

    prev_overlap = _bbox_overlap(prev_player, prev_target, inset_px=OVERLAP_INSET_PX)

    # Standard check: player at t+1 vs target at t+1.
    next_overlap_live = _bbox_overlap(next_player, next_target, inset_px=OVERLAP_INSET_PX)

    # Vanish check: target went active→inactive this step.
    # Some envs (e.g. JAXAtari Seaquest) zero the position at the same step the
    # collision is registered, so next_target.x/y = 0 and the overlap check
    # against next_target always fails.
    #
    # When can_go_offscreen=0 the only reason a target deactivates is that it was
    # collected/triggered — treat this as an unconditional overlap (rising-edge
    # fires regardless of which entity moved into which).  Scroll-off objects have
    # can_go_offscreen=1 and are excluded so we don't misfire on scroll-offs.
    just_deactivated = jnp.logical_and(prev_target.active > 0, next_target.active == 0)
    collected_in_place = jnp.logical_and(just_deactivated, prev_target.can_go_offscreen == 0)
    next_overlap = jnp.where(collected_in_place, jnp.array(True), next_overlap_live)

    return prev_player, prev_target, next_target, prev_overlap, next_overlap


def extract_skill_rewards(
    prev_raw_obs,
    next_raw_obs,
    prev_target,
    next_target,
    life_lost,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute rewards and done signals for all three skills against a single target.

    This is the only place reward values are computed. Env-specific files call
    this after extracting object positions/dimensions — no reward logic elsewhere.

    Args:
        prev_raw_obs : raw game observation at t (player + target at t)
        next_raw_obs : raw game observation at t+1 (player at t+1)
        prev_target  : LabeledObjectObservation at t  (stacked or last-frame)
        next_target  : LabeledObjectObservation at t+1 (stacked or last-frame)
        life_lost    : scalar int — number of lives lost this step

    Returns:
        rewards   : float32 [3] — [destroy, collect, avoid]
        skill_done: bool    [3] — [destroy, collect, avoid]
    """
    _prev_player, prev_target, next_target, prev_overlap, next_overlap = \
        reward_transition_context(prev_raw_obs, next_raw_obs, prev_target, next_target)

    # Rising edge: not overlapping at t, overlapping at t+1.
    overlap_rising_edge = jnp.logical_and(jnp.logical_not(prev_overlap), next_overlap)

    # ---- COLLECT ----
    collect_success = overlap_rising_edge
    collect_reward  = jnp.where(collect_success, COLLECT_SUCCESS_REWARD, 0.0)
    collect_done    = collect_success

    # ---- AVOID ----
    avoid_failure = overlap_rising_edge
    avoid_reward  = jnp.where(avoid_failure, AVOID_FAILURE_REWARD, 0.0)
    avoid_done    = avoid_failure

    # ---- DESTROY ----
    # was_killed is stamped on next_target by the env wrapper from raw game state —
    # the only reliable kill signal when the env zeros position for both kills and
    # scroll-offs (e.g. JAX-Atari Seaquest).
    destroy_success = next_target.was_killed == 1
    # Failure: player died while in contact with this specific object.
    # Guard: not if this step was a collect/avoid rising-edge (surfacing into the
    # surface waypoint while simultaneously losing oxygen would otherwise misfire).
    destroy_failure = jnp.logical_and(
        life_lost > 0,
        jnp.logical_and(prev_overlap, jnp.logical_not(overlap_rising_edge)),
    )
    destroy_reward = jnp.where(
        destroy_success,
        DESTROY_SUCCESS_REWARD,
        jnp.where(destroy_failure, DESTROY_FAILURE_REWARD, 0.0),
    )
    destroy_done = jnp.logical_or(destroy_success, destroy_failure)

    rewards    = jnp.array([destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32)
    skill_done = jnp.array([destroy_done,   collect_done,   avoid_done],   dtype=jnp.bool_)
    return rewards, skill_done
