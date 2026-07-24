"""
Global skill reward contract — env-independent.

Reward magnitudes and trigger conditions:
  AVOID    : +0.01 per frame while tracking an active AVOID target without collision
             -1  on rising edge of AABB overlap (player enters object hitbox)
  COLLECT  : +1  on rising edge of AABB overlap, or next_target.was_collected == 1
  DESTROY  : +1  when next_target.was_killed == 1 (set by env wrapper from raw state)
  NAVIGATE : +1  on rising-edge overlap / was_collected (arrive at waypoint);
             +0.01 when centre-distance to the (still active) waypoint decreases
             between t and t+1 (progress shaping for long-horizon ladders /
             surface runs). COLLECT stays sparse — no per-step progress term.
  JUMP_OVER: +1  when the relative horizontal offset (player center - target
             center) changes side between t and t+1 while the player is fully
             above the target at both frames and never touches it (covers both
             "player leaps over object" and "object passes underneath")
             -1  on rising edge of AABB overlap, or next_target.was_killed == 1

  All skills additionally receive ``SKILL_DEATH_PENALTY`` (-1) and a skill-done
  when ``life_lost > 0``, even if the death was not clearly that skill's fault
  (overrides a same-step success so e.g. suicidal surfacing is not rewarded).

All rewards fire on transition edges so sustained overlap cannot be farmed.
DESTROY relies on was_killed rather than position-based disappearance detection
because JAX-Atari zeros position for both torpedo kills and scroll-offs.

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

# Unified reward magnitudes — single source of truth.
AVOID_SURVIVAL_REWARD  =  0.01
AVOID_FAILURE_REWARD   = -1.0
COLLECT_SUCCESS_REWARD =  1.0
DESTROY_SUCCESS_REWARD =  1.0
DESTROY_FAILURE_REWARD = -1.0  # retained for contact-conditioned destroy fails
NAVIGATE_SUCCESS_REWARD = 1.0
NAVIGATE_PROGRESS_REWARD = 0.01  # distance-to-waypoint decreased this step
JUMP_SUCCESS_REWARD    =  1.0
JUMP_FAILURE_REWARD    = -1.0
# Applied to every skill on life loss (see extract_skill_rewards).
SKILL_DEATH_PENALTY    = -1.0

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
    avoid_survival_reward: float


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

    prev_overlap = _bbox_overlap(prev_player, prev_target, inset_px=OVERLAP_INSET_PX)

    # Standard check: player at t+1 vs target at t+1.
    next_overlap_live = _bbox_overlap(next_player, next_target, inset_px=OVERLAP_INSET_PX)

    # Vanish check: target went active→inactive this step.
    # Some envs (e.g. JAXAtari Seaquest) zero the position at the same step the
    # collision is registered, so next_target.x/y = 0 and the overlap check
    # against next_target always fails.
    #
    # When can_go_offscreen=0 and the label opts into the fallback (COLLECT-like
    # skills, see SkillSpec.collect_in_place_fallback), deactivation means the
    # player collected/triggered the object in place.  Scroll-off objects and
    # other labels are excluded (e.g. DESTROY kills must use was_killed).
    just_deactivated = jnp.logical_and(prev_target.active > 0, next_target.active == 0)
    is_collect_like = jnp.zeros_like(just_deactivated)
    for spec in SKILL_REGISTRY:
        if spec.collect_in_place_fallback:
            is_collect_like = jnp.logical_or(
                is_collect_like, prev_target.semantic_label == spec.label
            )
    collected_in_place = jnp.logical_and(
        just_deactivated,
        jnp.logical_and(prev_target.can_go_offscreen == 0, is_collect_like),
    )
    next_overlap = jnp.where(collected_in_place, jnp.array(True), next_overlap_live)

    return prev_player, next_player, prev_target, next_target, prev_overlap, next_overlap


# ---------------------------------------------------------------------------
# Per-skill reward functions.
#
# Each takes a RewardContext and returns (reward, done) for one transition.
# Label gating (semantic_label == spec.label) is applied by the registry loop
# in extract_skill_rewards, so these encode only the skill's mechanics.
# ---------------------------------------------------------------------------


def _destroy_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # was_killed is stamped on next_target by the env wrapper from raw game state —
    # the only reliable kill signal when the env zeros position for both kills and
    # scroll-offs (e.g. JAX-Atari Seaquest).
    # Life-loss penalty is applied uniformly in extract_skill_rewards.
    success = ctx.next_target.was_killed == 1
    return jnp.where(success, DESTROY_SUCCESS_REWARD, 0.0), success


def _collect_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    success = jnp.logical_or(
        ctx.overlap_rising_edge, ctx.next_target.was_collected == 1
    )
    return jnp.where(success, COLLECT_SUCCESS_REWARD, 0.0), success


def _avoid_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Rising-edge AABB OR env-stamped was_killed (used by envs whose collision
    # detection is more accurate than AABB, e.g. Skiing uses state.collision_type).
    # Life-loss penalty is applied uniformly in extract_skill_rewards.
    failure = jnp.logical_or(
        ctx.overlap_rising_edge, ctx.next_target.was_killed == 1
    )
    survival = jnp.logical_and(
        ctx.prev_target.active > 0, jnp.logical_not(failure)
    )
    reward = jnp.where(
        failure,
        AVOID_FAILURE_REWARD,
        jnp.where(survival, ctx.avoid_survival_reward, 0.0),
    )
    return reward, failure


def _navigate_reward(ctx: RewardContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Arrive: same sparse contract as COLLECT (overlap / was_collected).
    # Progress: dense shaping when centre-distance to the waypoint shrinks —
    # needed for far static targets (Kangaroo ladders). Idle / retreat get 0.
    # Life-loss penalty is applied uniformly in extract_skill_rewards.
    success = jnp.logical_or(
        ctx.overlap_rising_edge, ctx.next_target.was_collected == 1
    )
    prev_px = ctx.prev_player.x + ctx.prev_player.width / 2.0
    prev_py = ctx.prev_player.y + ctx.prev_player.height / 2.0
    next_px = ctx.next_player.x + ctx.next_player.width / 2.0
    next_py = ctx.next_player.y + ctx.next_player.height / 2.0
    prev_tx = ctx.prev_target.x + ctx.prev_target.width / 2.0
    prev_ty = ctx.prev_target.y + ctx.prev_target.height / 2.0
    next_tx = ctx.next_target.x + ctx.next_target.width / 2.0
    next_ty = ctx.next_target.y + ctx.next_target.height / 2.0
    prev_dist = jnp.sqrt((prev_px - prev_tx) ** 2 + (prev_py - prev_ty) ** 2)
    next_dist = jnp.sqrt((next_px - next_tx) ** 2 + (next_py - next_ty) ** 2)
    both_active = jnp.logical_and(
        ctx.prev_target.active > 0, ctx.next_target.active > 0
    )
    progressed = jnp.logical_and(both_active, next_dist < prev_dist)
    reward = jnp.where(
        success,
        NAVIGATE_SUCCESS_REWARD,
        jnp.where(progressed, NAVIGATE_PROGRESS_REWARD, 0.0),
    )
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
    avoid_survival_reward: float = AVOID_SURVIVAL_REWARD,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute rewards and done signals for all skills against a single target.

    This is the only place reward values are computed. Env-specific files call
    this after extracting object positions/dimensions — no reward logic elsewhere.

    Args:
        prev_raw_obs : raw game observation at t (player + target at t)
        next_raw_obs : raw game observation at t+1 (player at t+1)
        prev_target  : LabeledObjectObservation at t  (stacked or last-frame)
        next_target  : LabeledObjectObservation at t+1 (stacked or last-frame)
        life_lost    : scalar int — number of lives lost this step

    Returns:
        rewards   : float32 [NUM_SKILL_REWARDS] — ordered per SKILL_REWARD_ORDER
        skill_done: bool    [NUM_SKILL_REWARDS] — ordered per SKILL_REWARD_ORDER
    """
    prev_player, next_player, prev_target, next_target, prev_overlap, next_overlap = \
        reward_transition_context(prev_raw_obs, next_raw_obs, prev_target, next_target)

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
        avoid_survival_reward=avoid_survival_reward,
    )

    died = ctx.life_lost > 0

    rewards_list = []
    done_list = []
    for spec in SKILL_REGISTRY:
        is_label = prev_target.semantic_label == spec.label
        reward, done = spec.compute(ctx)
        # Death overrides same-step success (e.g. overlapping surface WP while dying).
        reward = jnp.where(died, SKILL_DEATH_PENALTY, reward)
        done = jnp.logical_or(done, died)
        rewards_list.append(jnp.where(is_label, reward, 0.0))
        done_list.append(jnp.logical_and(is_label, done))

    rewards = jnp.stack(rewards_list).astype(jnp.float32)
    skill_done = jnp.stack(done_list).astype(jnp.bool_)

    eligible = object_reward_eligible(prev_target, next_target)
    rewards = jnp.where(eligible, rewards, jnp.zeros_like(rewards))
    skill_done = jnp.logical_and(skill_done, eligible)

    return rewards, skill_done


def avoid_survival_reward_from_config(config) -> float:
    """Per-run avoid shaping; default matches ``AVOID_SURVIVAL_REWARD``."""
    if config is None:
        return AVOID_SURVIVAL_REWARD
    return float(config.get("AVOID_SURVIVAL_BONUS", AVOID_SURVIVAL_REWARD))
