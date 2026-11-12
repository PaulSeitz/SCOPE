"""
Global skill reward contract — env-independent.

Reward magnitudes and trigger conditions:
  AVOID    : +0.01 when AABB surface-gap to an active AVOID target *increases*
             (escape progress; same distance metric as ego obs). No per-frame
             survival farm.
             -1  on rising edge of AABB overlap / was_killed (contact fail).
             No sparse “cleared” success terminal by default (escape shaping only).
             AVOID_SPARSE_ONLY: no per-step escape shaping; +1 and skill-done on
             the first escape step (distance increased, no contact) — sparse
             “switched away safely”; −1 on contact / death still applies.
  COLLECT  : +1  on rising edge of AABB overlap, or next_target.was_collected == 1.
             Default dense term: +0.01 when AABB surface-gap to the (still active)
             target decreases (approach shaping; matches ego distance).
             Label-agnostic: contact/approach on a DESTROY enemy still trains COLLECT.
             COLLECT_LIVING_PENALTY: replace approach with −0.01 every frame the
             collectible is still active and not collected this step (same clock
             as DESTROY living tax). NAVIGATE is unchanged.
             COLLECT_SPARSE_ONLY: no approach shaping and no living tax (sparse
             +1 / death only). Wins over COLLECT_LIVING_PENALTY when both set.
  DESTROY  : +1  when next_target.was_killed == 1 (set by env wrapper from raw state).
             Optional per-run extras (config flags, default off):
             DESTROY_Y_ALIGN: +0.01 when |Δy| to the target decreases.
             DESTROY_STANDOFF: +0.01 backing out of melee (surface-gap < 20px and
               increasing); −0.01 closing further while already in melee.
             DESTROY_CONTACT_FAIL: −1 and skill-done on rising-edge AABB overlap
               (ram). Kill still wins if was_killed on the same step.
             DESTROY_LIVING_PENALTY: −0.01 every frame the target stays alive
               (both still active, no kill this step). Combined with contact-fail
               this is the “urgency” arm: ram loses, waiting/jiggling ticks down.
  NAVIGATE : +1  on rising-edge overlap / was_collected (arrive at waypoint);
             signed dense shaping on AABB surface-gap change while the waypoint
             stays active: +0.05 per 10px closer, −0.05 per 10px farther
             (``NAVIGATE_PROGRESS_REWARD`` / ``NAVIGATE_PROGRESS_UNIT_PX``).
             Idle (Δ=0) stays 0.
  JUMP_OVER: +1  when the relative horizontal offset (player center - target
             center) changes side between t and t+1 while the player is fully
             above the target at both frames and never touches it (covers both
             "player leaps over object" and "object passes underneath")
             -1  on rising edge of AABB overlap, or next_target.was_killed == 1

  All skills: success/failure as above. Additionally, AVOID and DESTROY receive
  ``SKILL_DEATH_PENALTY`` (-1) and a skill-done when ``life_lost > 0`` (overrides
  a same-step success). COLLECT / NAVIGATE / JUMP_OVER do **not** by default —
  dying while collecting is often caused by threats those skills cannot see.
  Set ``penalize_collect_death=True`` (config ``SKILL_DEATH_PENALIZE_COLLECT``)
  to also stamp the death penalty on COLLECT (ablation / pre-7b4348d behavior).
  ``COLLECT_DEATH_PRECEDENCE: success_first`` keeps a same-transition contact /
  collection success at +1 even when that transition also loses a life; deaths
  without COLLECT success remain -1.  The default ``death_first`` preserves the
  historical override.

  By default AVOID treats every ``was_killed`` event as failure.  This is useful
  in games that use the event for precise harmful contact, but it also turns a
  ranged DESTROY success into an off-policy AVOID failure.  ``avoid_label_only``
  consumes ``was_killed`` only for AVOID-labelled targets.  The reference
  recipe uses ``none``: destruction of the target is not an AVOID failure.
  Geometric contact remains a failure in every mode.

All rewards fire on transition edges so sustained overlap cannot be farmed.
DESTROY relies on was_killed rather than position-based disappearance detection
because JAX-Atari zeros position for both torpedo kills and scroll-offs.

**Off-policy multi-skill rewards:** each skill's mechanics run against the
*current target* regardless of ``semantic_label``. Labels select which skill the
heuristic executes; they do not gate reward. Touching a DESTROY enemy still
gives COLLECT +1 (contact) and, with ``DESTROY_CONTACT_FAIL``, DESTROY −1 (ram).
That is what lets Phoenix (no collectibles) teach a transferable COLLECT head.

The reward/done vectors are ordered by skill index (label - 1); see
SKILL_REGISTRY / SKILL_REWARD_ORDER.

Adding a new skill:
  1. Add a LABEL_* constant in datatypes.py (contiguous, before LABEL_META_FIRE)
     and bump LABEL_META_FIRE / NUM_LABELS.
  2. Write a reward function ``(ctx: RewardContext) -> (reward, done)`` below.
     Label gating is applied by the registry loop — the function only encodes
     the skill's success/failure mechanics.
  3. Append a SkillSpec to SKILL_REGISTRY at the position matching its label.
The reward vector width, name tables, and eligibility plumbing follow from the
registry; env label files and training code need no changes beyond assigning
the new label (and sizing NUM_SKILLS/SKILL_NAMES in configs that use it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_JUMP_OVER,
    LABEL_NAVIGATE,
)
from concept_based_learning.observations.egocentric import (
    _uses_bottom_edge_y,
    aabb_geom_distance_px,
)

# Unified reward magnitudes — single source of truth.
AVOID_ESCAPE_REWARD = 0.01  # distance-to-threat increased this step
AVOID_SURVIVAL_REWARD = AVOID_ESCAPE_REWARD  # back-compat alias (old survival farm removed)
AVOID_SUCCESS_REWARD = 1.0  # sparse AVOID: one-shot escape success
AVOID_FAILURE_REWARD = -1.0
COLLECT_SUCCESS_REWARD = 1.0
COLLECT_PROGRESS_REWARD = 0.01  # distance-to-collectible decreased this step
COLLECT_LIVING_PENALTY = -0.01  # per-frame tax while the collectible is uncollected
DESTROY_SUCCESS_REWARD = 1.0
DESTROY_FAILURE_REWARD = -1.0  # contact-fail ablation (DESTROY_CONTACT_FAIL)
DESTROY_Y_ALIGN_REWARD = 0.01  # |centre-Δy| decreased this step
DESTROY_STANDOFF_REWARD = 0.01  # centre-dist increased while inside melee radius
DESTROY_STANDOFF_PENALTY = -0.01  # centre-dist decreased while inside melee radius
DESTROY_STANDOFF_PX = 20.0  # melee radius (centre-to-centre)
DESTROY_LIVING_PENALTY = -0.01  # per-frame tax while the enemy still lives
NAVIGATE_SUCCESS_REWARD = 1.0
# Signed approach shaping: reward = REWARD * (prev_dist - next_dist) / UNIT_PX.
# +0.05 for closing 10px, −0.05 for opening 10px (binary ±0.01-on-any-Δ was too weak).
NAVIGATE_PROGRESS_REWARD = 0.05
NAVIGATE_PROGRESS_UNIT_PX = 10.0
JUMP_SUCCESS_REWARD = 1.0
JUMP_FAILURE_REWARD = -1.0
# Applied to AVOID / DESTROY on life loss (see extract_skill_rewards).
# COLLECT is optional via ``penalize_collect_death`` / SKILL_DEATH_PENALIZE_COLLECT.
# NAVIGATE / JUMP_OVER stay exempt.
SKILL_DEATH_PENALTY = -1.0

# Shrink interaction hitboxes for edge-triggered overlap rewards.
# This reduces reward farming from grazing at object borders.
OVERLAP_INSET_PX = 1


@dataclass(frozen=True)
class RewardContext:
    """Per-transition quantities shared by all skill reward functions.

    Player/target structs are last-frame (unstacked); overlaps use the inset
    AABB check, with next_overlap falling back to prev_target's position when
    the target was collected in place (see reward_transition_context).
    """

    prev_player: Any
    next_player: Any
    prev_target: Any
    next_target: Any
    prev_overlap: jnp.ndarray
    next_overlap: jnp.ndarray
    overlap_rising_edge: jnp.ndarray
    life_lost: jnp.ndarray
    avoid_escape_reward: float
    avoid_sparse_only: bool
    avoid_killed_event_avoid_label_only: bool
    destroy_y_align: bool
    destroy_standoff: bool
    destroy_contact_fail: bool
    destroy_living_penalty: bool
    destroy_standoff_px: float
    collect_living_penalty: bool
    collect_sparse_only: bool
    y_is_bottom_edge: bool = False
    facing_deg: float = 0.0
    avoid_killed_event_ignored: bool = False


@dataclass(frozen=True)
class SkillSpec:
    """One entry of the global skill vocabulary.

    ``label`` must equal registry position + 1 (skill_idx = label - 1, matching
    label_to_skill_idx in the training modules).  ``compute`` returns
    (reward, done) for a transition; the registry loop applies label gating.
    ``collect_in_place_fallback`` marks labels whose in-place target
    deactivation (can_go_offscreen=0) counts as overlap at t+1 — envs that zero
    object positions on the collection step need this to keep the AABB signal.
    """

    name: str
    label: int
    compute: Callable[[RewardContext], Tuple[jnp.ndarray, jnp.ndarray]]
    collect_in_place_fallback: bool = False


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
    collected = next_.was_collected > 0
    return jnp.logical_or(
        prev_active,
        jnp.logical_or(next_active, jnp.logical_or(killed, collected)),
    )


def _agent_bbox(raw_obs):
    """
    Return the controllable-agent object from a raw env observation.

    Keep reward logic env-agnostic by resolving common names here instead of
    hardcoding per-env branches in reward computations.

    TODO: Move this resolution into env-specific hooks/components and pass a
    normalized agent bbox into the reward extractor.
    """
    for name in ("player", "chicken", "agent", "skier"):
        if hasattr(raw_obs, name):
            return getattr(raw_obs, name)
    raise AttributeError(
        "Raw observation does not expose a known controllable agent field "
        "(expected one of: player, chicken, agent, skier)."
    )


def _bbox_overlap(
    player, target, inset_px: int = 0, *, y_is_bottom_edge: bool = False
) -> jnp.ndarray:
    """AABB overlap. ``y`` is the top edge unless ``y_is_bottom_edge``."""
    inset = jnp.array(inset_px, dtype=target.x.dtype)
    target_x_min = target.x + inset
    target_w = jnp.maximum(target.width - 2 * inset, 1)
    target_h = jnp.maximum(target.height - 2 * inset, 1)
    target_x_max = target_x_min + target_w
    if y_is_bottom_edge:
        # y is the bottom edge; the box extends upward.
        target_y_max = target.y - inset
        target_y_min = target_y_max - target_h
        player_y_min = player.y - player.height
        player_y_max = player.y
    else:
        target_y_min = target.y + inset
        target_y_max = target_y_min + target_h
        player_y_min = player.y
        player_y_max = player.y + player.height

    horiz = jnp.logical_and(
        player.x <= target_x_max,
        player.x + player.width >= target_x_min,
    )
    vert = jnp.logical_and(
        player_y_min <= target_y_max,
        player_y_max >= target_y_min,
    )
    return jnp.logical_and(horiz, vert)


def reward_transition_context(prev_raw_obs, next_raw_obs, prev_target, next_target):
    """
    Reduce stacked observations to last-frame scalars and compute AABB overlaps.

    A target that just became inactive is tested against its last real box,
    and only if the player actually overlaps that box. Clearing the slot to
    the origin is not contact. A collectible that leaves the screen without
    the player touching it is not contact either.

    Returns:
        prev_player  — last-frame player struct at t
        next_player  — last-frame player struct at t+1
        prev_target  — last-frame target struct at t
        next_target  — last-frame target struct at t+1
        prev_overlap — bool: AABB overlap at t
        next_overlap — bool: AABB overlap at t+1 (using prev_target pos if target just vanished)
    """
    prev_player = last_frame(_agent_bbox(prev_raw_obs))
    next_player = last_frame(_agent_bbox(next_raw_obs))
    prev_target = last_frame(prev_target)
    next_target = last_frame(next_target)

    y_is_bottom_edge = bool(_uses_bottom_edge_y(prev_raw_obs))
    prev_overlap = _bbox_overlap(
        prev_player, prev_target, inset_px=OVERLAP_INSET_PX,
        y_is_bottom_edge=y_is_bottom_edge,
    )

    # Live geometry only. A cleared inactive slot must not collide at (0, 0).
    next_overlap_live = jnp.logical_and(
        next_target.active > 0,
        _bbox_overlap(
            next_player, next_target, inset_px=OVERLAP_INSET_PX,
            y_is_bottom_edge=y_is_bottom_edge,
        ),
    )

    # Collection can zero the sprite in the same step the player touches it.
    # Recover that only when the player overlaps the last real box. A diver
    # that swims off screen does not.
    just_deactivated = jnp.logical_and(prev_target.active > 0, next_target.active == 0)
    is_collect_like = jnp.zeros_like(just_deactivated)
    for spec in SKILL_REGISTRY:
        if spec.collect_in_place_fallback:
            is_collect_like = jnp.logical_or(
                is_collect_like, prev_target.semantic_label == spec.label
            )
    touched_last_box = _bbox_overlap(
        next_player, prev_target, inset_px=OVERLAP_INSET_PX,
        y_is_bottom_edge=y_is_bottom_edge,
    )
    collected_in_place = jnp.logical_and(
        just_deactivated,
        jnp.logical_and(
            prev_target.can_go_offscreen == 0,
            jnp.logical_and(is_collect_like, touched_last_box),
        ),
    )
    next_overlap = jnp.where(collected_in_place, jnp.array(True), next_overlap_live)

    return prev_player, next_player, prev_target, next_target, prev_overlap, next_overlap


def explain_collect_contact(prev_raw_obs, next_raw_obs, prev_target, next_target):
    """
    Host-side breakdown of why COLLECT would (not) fire on this transition.

    Training still uses ``reward_transition_context`` + ``_collect_reward``.
    This mirrors that logic and names the success channel for diagnostics:
    ``was_collected``, ``overlap_rising_live``, ``collect_in_place``, or
    ``none``. A diver that deactivates without ``touched_last_box`` is
    ``missed_deactivation`` (the hypothesized mid-skip pickup miss).
    """
    (
        _prev_player,
        _next_player,
        prev_t,
        next_t,
        prev_overlap,
        next_overlap,
    ) = reward_transition_context(prev_raw_obs, next_raw_obs, prev_target, next_target)

    y_is_bottom_edge = bool(_uses_bottom_edge_y(prev_raw_obs))
    next_player = last_frame(_agent_bbox(next_raw_obs))
    next_overlap_live = bool(
        jax.device_get(
            jnp.logical_and(
                next_t.active > 0,
                _bbox_overlap(
                    next_player,
                    next_t,
                    inset_px=OVERLAP_INSET_PX,
                    y_is_bottom_edge=y_is_bottom_edge,
                ),
            )
        )
    )
    just_deactivated = bool(
        jax.device_get(
            jnp.logical_and(prev_t.active > 0, next_t.active == 0)
        )
    )
    touched_last_box = bool(
        jax.device_get(
            _bbox_overlap(
                next_player,
                prev_t,
                inset_px=OVERLAP_INSET_PX,
                y_is_bottom_edge=y_is_bottom_edge,
            )
        )
    )
    is_collect_like = False
    for spec in SKILL_REGISTRY:
        if spec.collect_in_place_fallback and int(
            jax.device_get(prev_t.semantic_label)
        ) == int(spec.label):
            is_collect_like = True
            break
    collected_in_place = bool(
        just_deactivated
        and int(jax.device_get(prev_t.can_go_offscreen)) == 0
        and is_collect_like
        and touched_last_box
    )
    prev_ov = bool(jax.device_get(prev_overlap))
    next_ov = bool(jax.device_get(next_overlap))
    overlap_rising = bool(next_ov and not prev_ov)
    was_collected = int(jax.device_get(next_t.was_collected)) > 0
    success = bool(overlap_rising or was_collected)

    if was_collected and overlap_rising:
        reason = "was_collected+overlap_rising"
    elif was_collected:
        reason = "was_collected"
    elif overlap_rising and collected_in_place:
        reason = "collect_in_place"
    elif overlap_rising and next_overlap_live:
        reason = "overlap_rising_live"
    elif overlap_rising:
        reason = "overlap_rising_other"
    elif just_deactivated and is_collect_like and not touched_last_box:
        reason = "missed_deactivation"
    else:
        reason = "none"

    return {
        "success": success,
        "reason": reason,
        "was_collected": was_collected,
        "overlap_rising": overlap_rising,
        "next_overlap_live": next_overlap_live,
        "collected_in_place": collected_in_place,
        "just_deactivated": just_deactivated,
        "touched_last_box": touched_last_box,
        "prev_overlap": prev_ov,
        "next_overlap": next_ov,
        "prev_active": int(jax.device_get(prev_t.active)),
        "next_active": int(jax.device_get(next_t.active)),
        "label": int(jax.device_get(prev_t.semantic_label)),
        "prev_xy": (
            float(jax.device_get(prev_t.x)),
            float(jax.device_get(prev_t.y)),
        ),
        "next_xy": (
            float(jax.device_get(next_t.x)),
            float(jax.device_get(next_t.y)),
        ),
    }


# ---------------------------------------------------------------------------
# Per-skill reward functions.
#
# Each takes a RewardContext and returns (reward, done) for one transition.
# Mechanics only — ``extract_skill_rewards`` applies every skill to the target
# (no label gate) so off-policy heads learn from the same contact/approach.
# ---------------------------------------------------------------------------


def _approach_distances(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Contact distances at t and t+1 (same ``EGO_AABB_GEOM`` metric as ego obs)."""
    y_bot = bool(ctx.y_is_bottom_edge)
    face = jnp.asarray(ctx.facing_deg, dtype=jnp.float32)
    prev_dist = aabb_geom_distance_px(
        ctx.prev_player.x,
        ctx.prev_player.y,
        ctx.prev_player.width,
        ctx.prev_player.height,
        ctx.prev_target.x,
        ctx.prev_target.y,
        ctx.prev_target.width,
        ctx.prev_target.height,
        y_is_bottom_edge=y_bot,
        facing_deg=face,
    )
    next_dist = aabb_geom_distance_px(
        ctx.next_player.x,
        ctx.next_player.y,
        ctx.next_player.width,
        ctx.next_player.height,
        ctx.next_target.x,
        ctx.next_target.y,
        ctx.next_target.width,
        ctx.next_target.height,
        y_is_bottom_edge=y_bot,
        facing_deg=face,
    )
    return prev_dist, next_dist


def _destroy_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # was_killed is stamped on next_target by the env wrapper from raw game state —
    # the only reliable kill signal when the env zeros position for both kills and
    # scroll-offs (e.g. JAX-Atari Seaquest).
    # Life-loss penalty is applied uniformly in extract_skill_rewards.
    success = ctx.next_target.was_killed == 1
    contact = jnp.logical_and(ctx.destroy_contact_fail, ctx.overlap_rising_edge)
    prev_dist, next_dist = _approach_distances(ctx)
    both_active = jnp.logical_and(
        ctx.prev_target.active > 0, ctx.next_target.active > 0
    )
    prev_dy = jnp.abs(
        (ctx.prev_player.y + ctx.prev_player.height / 2.0)
        - (ctx.prev_target.y + ctx.prev_target.height / 2.0)
    )
    next_dy = jnp.abs(
        (ctx.next_player.y + ctx.next_player.height / 2.0)
        - (ctx.next_target.y + ctx.next_target.height / 2.0)
    )
    y_aligned = jnp.logical_and(
        ctx.destroy_y_align,
        jnp.logical_and(both_active, next_dy < prev_dy),
    )
    in_melee_next = jnp.logical_and(
        both_active, next_dist < jnp.float32(ctx.destroy_standoff_px)
    )
    in_melee_prev = jnp.logical_and(
        both_active, prev_dist < jnp.float32(ctx.destroy_standoff_px)
    )
    backed_off = jnp.logical_and(in_melee_prev, next_dist > prev_dist)
    closed_in = jnp.logical_and(in_melee_next, next_dist < prev_dist)
    shaped = jnp.where(
        jnp.logical_and(ctx.destroy_standoff, backed_off),
        DESTROY_STANDOFF_REWARD,
        jnp.where(
            jnp.logical_and(ctx.destroy_standoff, closed_in),
            DESTROY_STANDOFF_PENALTY,
            jnp.where(y_aligned, DESTROY_Y_ALIGN_REWARD, 0.0),
        ),
    )
    living_tax = jnp.where(
        jnp.logical_and(
            ctx.destroy_living_penalty,
            jnp.logical_and(both_active, jnp.logical_not(success)),
        ),
        DESTROY_LIVING_PENALTY,
        0.0,
    )
    reward = jnp.where(
        success,
        DESTROY_SUCCESS_REWARD,
        jnp.where(contact, DESTROY_FAILURE_REWARD, shaped + living_tax),
    )
    done = jnp.logical_or(success, contact)
    return reward, done


def _collect_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Sparse contact success. Default dense term is approach progress (same
    # idea as NAVIGATE). Optional living tax replaces approach: delay costs,
    # without telling the skill to close the gap. COLLECT_SPARSE_ONLY disables
    # both dense terms (approach and living tax).
    success = jnp.logical_or(
        ctx.overlap_rising_edge, ctx.next_target.was_collected == 1
    )
    prev_dist, next_dist = _approach_distances(ctx)
    both_active = jnp.logical_and(
        ctx.prev_target.active > 0, ctx.next_target.active > 0
    )
    living_tax = jnp.where(
        jnp.logical_and(
            ctx.collect_living_penalty,
            jnp.logical_and(both_active, jnp.logical_not(success)),
        ),
        COLLECT_LIVING_PENALTY,
        0.0,
    )
    progressed = jnp.logical_and(both_active, next_dist < prev_dist)
    shaped = jnp.where(
        ctx.collect_sparse_only,
        0.0,
        jnp.where(
            ctx.collect_living_penalty,
            living_tax,
            jnp.where(progressed, COLLECT_PROGRESS_REWARD, 0.0),
        ),
    )
    reward = jnp.where(success, COLLECT_SUCCESS_REWARD, shaped)
    return reward, success


def _avoid_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Contact is always a failure. was_killed is a failure only in the legacy
    # modes: some games stamp harmful collisions there, but Seaquest uses it
    # for a successful shot. The reference mode ignores it. Life loss is
    # applied later in extract_skill_rewards.
    # Escape shaping: +0.01 when distance increases. Idle earns 0.
    # AVOID_SPARSE_ONLY: +1 and done on the first escape step instead.
    killed_failure = ctx.next_target.was_killed == 1
    if ctx.avoid_killed_event_ignored:
        killed_failure = jnp.zeros_like(killed_failure)
    elif ctx.avoid_killed_event_avoid_label_only:
        # Use the pre-transition label: event stamping can deactivate or replace
        # the post-transition target, while prev_target is the selected object
        # whose trajectory generated this update.
        killed_failure = jnp.logical_and(
            killed_failure, ctx.prev_target.semantic_label == LABEL_AVOID
        )
    failure = jnp.logical_or(ctx.overlap_rising_edge, killed_failure)
    prev_dist, next_dist = _approach_distances(ctx)
    both_active = jnp.logical_and(
        ctx.prev_target.active > 0, ctx.next_target.active > 0
    )
    escaped = jnp.logical_and(
        both_active,
        jnp.logical_and(jnp.logical_not(failure), next_dist > prev_dist),
    )
    escape_r = jnp.where(
        ctx.avoid_sparse_only,
        jnp.float32(AVOID_SUCCESS_REWARD),
        jnp.float32(ctx.avoid_escape_reward),
    )
    reward = jnp.where(
        failure,
        AVOID_FAILURE_REWARD,
        jnp.where(escaped, escape_r, 0.0),
    )
    done = jnp.logical_or(
        failure,
        jnp.logical_and(ctx.avoid_sparse_only, escaped),
    )
    return reward, done


def _navigate_progress_shaped(prev_dist: jnp.ndarray, next_dist: jnp.ndarray) -> jnp.ndarray:
    """Signed dense NAVIGATE shaping: +REWARD per UNIT_PX closer, − per farther."""
    delta_px = prev_dist - next_dist
    return (
        jnp.float32(NAVIGATE_PROGRESS_REWARD)
        * delta_px
        / jnp.float32(NAVIGATE_PROGRESS_UNIT_PX)
    )


def _navigate_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Arrive: same sparse contract as COLLECT (overlap / was_collected).
    # Progress: signed dense shaping proportional to AABB surface-gap change —
    # needed for far static targets (Kangaroo ladders). Retreat is penalized so
    # jump-spam away from the WP cannot be reward-neutral.
    # Life-loss penalty is applied uniformly in extract_skill_rewards.
    success = jnp.logical_or(
        ctx.overlap_rising_edge, ctx.next_target.was_collected == 1
    )
    prev_dist, next_dist = _approach_distances(ctx)
    both_active = jnp.logical_and(
        ctx.prev_target.active > 0, ctx.next_target.active > 0
    )
    shaped = jnp.where(
        both_active,
        _navigate_progress_shaped(prev_dist, next_dist),
        jnp.float32(0.0),
    )
    reward = jnp.where(success, NAVIGATE_SUCCESS_REWARD, shaped)
    return reward, success


def _jump_over_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Success: the relative horizontal offset (player center - target center)
    # changes side between t and t+1 while the player is fully above the target
    # at both frames and never touches it.  Only relative position matters, so
    # this covers both "player leaps over object" and "object rolls underneath".
    # Centers are compared at 2x scale to stay in integer arithmetic.
    prev_dx = (2 * ctx.prev_player.x + ctx.prev_player.width) - (
        2 * ctx.prev_target.x + ctx.prev_target.width
    )
    next_dx = (2 * ctx.next_player.x + ctx.next_player.width) - (
        2 * ctx.next_target.x + ctx.next_target.width
    )
    crossed = (prev_dx > 0) != (next_dx > 0)
    # y grows downward: "above" means player bottom edge at or above target top.
    above_prev = ctx.prev_player.y + ctx.prev_player.height <= ctx.prev_target.y
    above_next = ctx.next_player.y + ctx.next_player.height <= ctx.next_target.y
    no_contact = jnp.logical_not(
        jnp.logical_or(ctx.prev_overlap, ctx.next_overlap)
    )
    # Require the target to still be active at t+1: envs zero positions of
    # deactivated objects, which would fake a crossing.
    success = jnp.logical_and(
        ctx.next_target.active > 0,
        jnp.logical_and(
            crossed,
            jnp.logical_and(jnp.logical_and(above_prev, above_next), no_contact),
        ),
    )
    # Failure: contact — rising-edge AABB overlap or env-stamped was_killed
    # (envs whose collision detection is more accurate than AABB).
    failure = jnp.logical_or(
        ctx.overlap_rising_edge, ctx.next_target.was_killed == 1
    )
    reward = jnp.where(
        success,
        JUMP_SUCCESS_REWARD,
        jnp.where(failure, JUMP_FAILURE_REWARD, 0.0),
    )
    return reward, jnp.logical_or(success, failure)


# Global skill vocabulary. Registry position must equal label - 1
# (skill_idx = label - 1, see label_to_skill_idx in the training modules).
SKILL_REGISTRY: Tuple[SkillSpec, ...] = (
    SkillSpec("destroy", LABEL_DESTROY, _destroy_reward),
    SkillSpec("collect", LABEL_COLLECT, _collect_reward, collect_in_place_fallback=True),
    SkillSpec("avoid", LABEL_AVOID, _avoid_reward),
    SkillSpec("navigate", LABEL_NAVIGATE, _navigate_reward, collect_in_place_fallback=True),
    SkillSpec("jump_over", LABEL_JUMP_OVER, _jump_over_reward),
)

for _idx, _spec in enumerate(SKILL_REGISTRY):
    assert _spec.label == _idx + 1, (
        f"SKILL_REGISTRY[{_idx}] = {_spec.name!r} has label {_spec.label}; "
        "registry position must equal label - 1."
    )

# Skill ordering of the reward/done vectors returned by extract_skill_rewards.
SKILL_REWARD_ORDER = tuple(spec.name for spec in SKILL_REGISTRY)
NUM_SKILL_REWARDS = len(SKILL_REGISTRY)


def extract_skill_rewards(
    prev_raw_obs,
    next_raw_obs,
    prev_target,
    next_target,
    life_lost,
    *,
    avoid_escape_reward: float = AVOID_ESCAPE_REWARD,
    avoid_survival_reward: float | None = None,
    penalize_collect_death: bool = False,
    config=None,
    destroy_y_align: bool | None = None,
    destroy_standoff: bool | None = None,
    destroy_contact_fail: bool | None = None,
    destroy_living_penalty: bool | None = None,
    collect_living_penalty: bool | None = None,
    collect_sparse_only: bool | None = None,
    avoid_sparse_only: bool | None = None,
    collect_death_precedence: str | None = None,
    avoid_killed_event_mode: str | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute rewards and done signals for all skills against a single target.

    This is the only place reward values are computed. Env-specific files call
    this after extracting object positions/dimensions — no reward logic elsewhere.

    Every skill channel evaluates its own mechanics on ``prev_target`` /
    ``next_target``; ``semantic_label`` does **not** zero other skills. The
    heuristic still uses labels to pick an active skill; off-policy learning
    consumes the full reward vector.

    Args:
        prev_raw_obs : raw game observation at t (player + target at t)
        next_raw_obs : raw game observation at t+1 (player at t+1)
        prev_target  : LabeledObjectObservation at t  (stacked or last-frame)
        next_target  : LabeledObjectObservation at t+1 (stacked or last-frame)
        life_lost    : scalar int — number of lives lost this step
        avoid_escape_reward: per-step AVOID escape-shaping magnitude
        avoid_survival_reward: deprecated alias for ``avoid_escape_reward``
        penalize_collect_death: if True, COLLECT also gets SKILL_DEATH_PENALTY
            on life loss (default False; AVOID/DESTROY always do)
        config: optional experiment dict; DESTROY_* flags are read from here
            when the explicit destroy_* kwargs are left None
        destroy_y_align / destroy_standoff / destroy_contact_fail /
        destroy_living_penalty:
            per-run DESTROY extras (see module docstring). ``None`` → config.
        collect_living_penalty:
            per-run COLLECT clock (replaces approach shaping). ``None`` → config.
        collect_sparse_only:
            if True, COLLECT gets no dense term (no approach, no living tax).
            ``None`` → config ``COLLECT_SPARSE_ONLY``.
        avoid_sparse_only:
            if True, AVOID uses +1/done on escape instead of dense escape
            shaping. ``None`` → config ``AVOID_SPARSE_ONLY``.
        collect_death_precedence:
            ``death_first`` (legacy default) or ``success_first``.  In the
            latter mode a same-transition COLLECT success wins over its optional
            life-loss penalty. ``None`` → config ``COLLECT_DEATH_PRECEDENCE``.
        avoid_killed_event_mode:
            ``all_targets`` (legacy default) or ``avoid_label_only``.  Contact
            remains a failure in both modes. ``None`` → config.

    Returns:
        rewards   : float32 [NUM_SKILL_REWARDS] — ordered per SKILL_REWARD_ORDER
        skill_done: bool    [NUM_SKILL_REWARDS] — ordered per SKILL_REWARD_ORDER
    """
    if avoid_survival_reward is not None:
        avoid_escape_reward = float(avoid_survival_reward)
    elif config is not None:
        avoid_escape_reward = avoid_escape_reward_from_config(config)
    flags = destroy_shaping_from_config(config)
    if destroy_y_align is not None:
        flags["destroy_y_align"] = bool(destroy_y_align)
    if destroy_standoff is not None:
        flags["destroy_standoff"] = bool(destroy_standoff)
    if destroy_contact_fail is not None:
        flags["destroy_contact_fail"] = bool(destroy_contact_fail)
    if destroy_living_penalty is not None:
        flags["destroy_living_penalty"] = bool(destroy_living_penalty)
    collect_clock = collect_living_penalty_from_config(config)
    if collect_living_penalty is not None:
        collect_clock = bool(collect_living_penalty)
    collect_sparse = collect_sparse_only_from_config(config)
    if collect_sparse_only is not None:
        collect_sparse = bool(collect_sparse_only)
    avoid_sparse = avoid_sparse_only_from_config(config)
    if avoid_sparse_only is not None:
        avoid_sparse = bool(avoid_sparse_only)
    if avoid_sparse:
        # Dense escape bonus is unused when sparse success is on.
        avoid_escape_reward = 0.0
    collect_death_mode = (
        collect_death_precedence_from_config(config)
        if collect_death_precedence is None
        else _normalise_collect_death_precedence(collect_death_precedence)
    )
    avoid_killed_mode = (
        avoid_killed_event_mode_from_config(config)
        if avoid_killed_event_mode is None
        else _normalise_avoid_killed_event_mode(avoid_killed_event_mode)
    )

    prev_player, next_player, prev_target, next_target, prev_overlap, next_overlap = \
        reward_transition_context(prev_raw_obs, next_raw_obs, prev_target, next_target)

    # Body-gap rewards need the same facing as ego (facing-frame) when enabled.
    facing_deg = 0.0
    try:
        from concept_based_learning.observations.pov_facing_frame import (
            resolve_facing_deg,
            using_pov_facing_frame,
        )

        if using_pov_facing_frame():
            facing_deg = resolve_facing_deg(prev_raw_obs)
        elif hasattr(prev_player, "orientation"):
            facing_deg = prev_player.orientation
    except Exception:
        if hasattr(prev_player, "orientation"):
            facing_deg = prev_player.orientation

    ctx = RewardContext(
        prev_player=prev_player,
        next_player=next_player,
        prev_target=prev_target,
        next_target=next_target,
        prev_overlap=prev_overlap,
        next_overlap=next_overlap,
        # Rising edge: not overlapping at t, overlapping at t+1.
        overlap_rising_edge=jnp.logical_and(
            jnp.logical_not(prev_overlap), next_overlap
        ),
        life_lost=life_lost,
        avoid_escape_reward=avoid_escape_reward,
        avoid_sparse_only=avoid_sparse,
        avoid_killed_event_avoid_label_only=(
            avoid_killed_mode == "avoid_label_only"
        ),
        avoid_killed_event_ignored=(avoid_killed_mode == "none"),
        destroy_y_align=flags["destroy_y_align"],
        destroy_standoff=flags["destroy_standoff"],
        destroy_contact_fail=flags["destroy_contact_fail"],
        destroy_living_penalty=flags["destroy_living_penalty"],
        destroy_standoff_px=flags["destroy_standoff_px"],
        collect_living_penalty=collect_clock,
        collect_sparse_only=collect_sparse,
        y_is_bottom_edge=bool(_uses_bottom_edge_y(prev_raw_obs)),
        facing_deg=facing_deg,
    )

    died = ctx.life_lost > 0
    collect_succeeded = jnp.logical_or(
        ctx.overlap_rising_edge, ctx.next_target.was_collected == 1
    )

    rewards_list = []
    done_list = []
    for spec in SKILL_REGISTRY:
        reward, done = spec.compute(ctx)
        # Death: AVOID/DESTROY always; COLLECT only if opted in.
        # Applies per *skill channel*, independent of the target's label.
        is_combat = jnp.logical_or(
            spec.label == LABEL_AVOID,
            spec.label == LABEL_DESTROY,
        )
        is_collect_opt_in = jnp.logical_and(
            penalize_collect_death,
            spec.label == LABEL_COLLECT,
        )
        if collect_death_mode == "success_first":
            is_collect_opt_in = jnp.logical_and(
                is_collect_opt_in, jnp.logical_not(collect_succeeded)
            )
        death_applies = jnp.logical_and(
            died,
            jnp.logical_or(is_combat, is_collect_opt_in),
        )
        reward = jnp.where(death_applies, SKILL_DEATH_PENALTY, reward)
        done = jnp.logical_or(done, death_applies)
        rewards_list.append(reward)
        done_list.append(done)

    rewards = jnp.stack(rewards_list).astype(jnp.float32)
    skill_done = jnp.stack(done_list).astype(jnp.bool_)

    eligible = object_reward_eligible(prev_target, next_target)
    rewards = jnp.where(eligible, rewards, jnp.zeros_like(rewards))
    skill_done = jnp.logical_and(skill_done, eligible)

    return rewards, skill_done


def destroy_shaping_from_config(config) -> dict:
    """Per-run DESTROY extras; all default off (sparse +1 kill only)."""
    if config is None:
        return {
            "destroy_y_align": False,
            "destroy_standoff": False,
            "destroy_contact_fail": False,
            "destroy_living_penalty": False,
            "destroy_standoff_px": DESTROY_STANDOFF_PX,
        }
    px = config.get("DESTROY_STANDOFF_PX", DESTROY_STANDOFF_PX)
    return {
        "destroy_y_align": bool(config.get("DESTROY_Y_ALIGN", False)),
        "destroy_standoff": bool(config.get("DESTROY_STANDOFF", False)),
        "destroy_contact_fail": bool(config.get("DESTROY_CONTACT_FAIL", False)),
        "destroy_living_penalty": bool(config.get("DESTROY_LIVING_PENALTY", False)),
        "destroy_standoff_px": float(px) if px is not None else DESTROY_STANDOFF_PX,
    }


def avoid_escape_reward_from_config(config) -> float:
    """Per-run AVOID escape shaping; default matches ``AVOID_ESCAPE_REWARD``."""
    if config is None:
        return AVOID_ESCAPE_REWARD
    if "AVOID_ESCAPE_BONUS" in config:
        return float(config["AVOID_ESCAPE_BONUS"])
    # Older configs used AVOID_SURVIVAL_BONUS for the dense AVOID term.
    return float(config.get("AVOID_SURVIVAL_BONUS", AVOID_ESCAPE_REWARD))


def avoid_survival_reward_from_config(config) -> float:
    """Deprecated alias for ``avoid_escape_reward_from_config``."""
    return avoid_escape_reward_from_config(config)

def collect_living_penalty_from_config(config) -> bool:
    """Per-run COLLECT clock; default off (keep approach shaping)."""
    if config is None:
        return False
    return bool(config.get("COLLECT_LIVING_PENALTY", False))


def collect_sparse_only_from_config(config) -> bool:
    """If True, COLLECT has no dense approach / living-tax term (default False)."""
    if config is None:
        return False
    return bool(config.get("COLLECT_SPARSE_ONLY", False))


def avoid_sparse_only_from_config(config) -> bool:
    """If True, AVOID uses +1/done on escape (no dense escape shaping)."""
    if config is None:
        return False
    return bool(config.get("AVOID_SPARSE_ONLY", False))


def penalize_collect_death_from_config(config) -> bool:
    """Whether COLLECT receives the life-loss death penalty (default False)."""
    if config is None:
        return False
    return bool(config.get("SKILL_DEATH_PENALIZE_COLLECT", False))


def _normalise_collect_death_precedence(value) -> str:
    mode = str(value or "death_first").strip().lower()
    aliases = {
        "legacy": "death_first",
        "death": "death_first",
        "success": "success_first",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"death_first", "success_first"}:
        raise ValueError(
            "COLLECT_DEATH_PRECEDENCE must be 'death_first' or 'success_first', "
            f"got {value!r}."
        )
    return mode


def collect_death_precedence_from_config(config) -> str:
    """Same-step COLLECT success/death ordering; legacy death-first by default."""
    if config is None:
        return "death_first"
    return _normalise_collect_death_precedence(
        config.get("COLLECT_DEATH_PRECEDENCE", "death_first")
    )


def _normalise_avoid_killed_event_mode(value) -> str:
    mode = str(value or "all_targets").strip().lower()
    aliases = {
        "legacy": "all_targets",
        "all": "all_targets",
        "label_only": "avoid_label_only",
        "ignore": "none",
        "contact_only": "none",
        "off": "none",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"all_targets", "avoid_label_only", "none"}:
        raise ValueError(
            "AVOID_KILLED_EVENT_MODE must be 'all_targets', "
            f"'avoid_label_only', or 'none', got {value!r}."
        )
    return mode


def avoid_killed_event_mode_from_config(config) -> str:
    """Scope ``was_killed`` as an AVOID failure; legacy all-targets by default."""
    if config is None:
        return "all_targets"
    return _normalise_avoid_killed_event_mode(
        config.get("AVOID_KILLED_EVENT_MODE", "all_targets")
    )
