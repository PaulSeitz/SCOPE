import enum
import functools
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.wrappers import JaxatariWrapper, AtariState
from jaxatari.environment import ObjectObservation

# Universal Semantic Labels
# Skill labels must stay contiguous starting at 1: skill_idx = label - 1
# (see label_to_skill_idx).
LABEL_NOOP = 0
LABEL_DESTROY = 1
LABEL_COLLECT = 2
LABEL_AVOID = 3
# NAVIGATE shares the COLLECT reward contract but targets static waypoints,
# so it can be routed to a dedicated skill head.
LABEL_NAVIGATE = 4
LABEL_JUMP_OVER = 5
# Reserved meta-policy slot: selecting this index executes FIRE (not a skill).
LABEL_META_FIRE = 6
# Structural obstacle (wall segment, shield, ...): never a meta target, never
# rewarded — exists only to occlude the skill line-of-sight feature.  Env label
# files emit these as virtual objects (same pattern as virtual waypoints).
LABEL_BLOCKER = 7
NUM_LABELS = 8

# Labels whose objects occlude the player→target line-of-sight ray.
# Solid/hazardous things block (enemies, hazards, structural blockers);
# collect-like objects don't (e.g. Seaquest divers can be shot through).
BLOCKING_LABELS = (LABEL_DESTROY, LABEL_AVOID, LABEL_JUMP_OVER, LABEL_BLOCKER)

# Full skill vocabulary (label - 1).  Configs may use a prefix (NUM_SKILLS < 5);
# label_to_skill_idx remaps the extended labels onto the closest base skill.
NUM_SKILL_LABELS = 5


def label_to_skill_idx(label, num_skills: int):
    """Map semantic label → skill index in ``[0, num_skills)``.

    Base mapping is ``skill_idx = label - 1``.  When ``num_skills`` truncates the
    vocabulary, extended labels fall back so 3-skill runs keep working after
    waypoints emit ``LABEL_NAVIGATE``:
      NAVIGATE  → COLLECT (1) when ``num_skills < 4``
      JUMP_OVER → AVOID (2) when ``num_skills < 5``
    """
    label = jnp.asarray(label)
    skill = label - 1
    skill = jnp.where(
        jnp.logical_and(label == LABEL_NAVIGATE, num_skills < LABEL_NAVIGATE),
        LABEL_COLLECT - 1,
        skill,
    )
    skill = jnp.where(
        jnp.logical_and(label == LABEL_JUMP_OVER, num_skills < LABEL_JUMP_OVER),
        LABEL_AVOID - 1,
        skill,
    )
    return jnp.clip(skill, 0, num_skills - 1)


def project_skill_outcomes(rewards, skill_done, num_skills: int):
    """
    Project full-vocab reward/done vectors onto ``num_skills`` trained heads.

    ``extract_skill_rewards`` always returns width ``NUM_SKILL_LABELS`` (5).
    When configs truncate with ``NUM_SKILLS < 5``, fold remapped channels the
    same way as :func:`label_to_skill_idx`:
      NAVIGATE  → COLLECT when ``num_skills < 4``
      JUMP_OVER → AVOID   when ``num_skills < 5``
    """
    rewards = jnp.asarray(rewards)
    skill_done = jnp.asarray(skill_done)
    num_skills = int(num_skills)
    if num_skills < 1:
        raise ValueError(f"num_skills must be >= 1, got {num_skills}")

    full_w = int(rewards.shape[-1])
    out_w = min(num_skills, full_w)
    out_r = rewards[..., :out_w]
    out_d = skill_done[..., :out_w]

    collect_i = LABEL_COLLECT - 1
    avoid_i = LABEL_AVOID - 1
    navigate_i = LABEL_NAVIGATE - 1
    jump_i = LABEL_JUMP_OVER - 1

    if num_skills < LABEL_NAVIGATE and full_w > navigate_i and out_w > collect_i:
        nav_r = rewards[..., navigate_i]
        nav_d = skill_done[..., navigate_i]
        out_r = out_r.at[..., collect_i].add(nav_r)
        out_d = out_d.at[..., collect_i].set(
            jnp.logical_or(out_d[..., collect_i], nav_d)
        )

    if num_skills < LABEL_JUMP_OVER and full_w > jump_i and out_w > avoid_i:
        jump_r = rewards[..., jump_i]
        jump_d = skill_done[..., jump_i]
        out_r = out_r.at[..., avoid_i].add(jump_r)
        out_d = out_d.at[..., avoid_i].set(
            jnp.logical_or(out_d[..., avoid_i], jump_d)
        )

    if out_w < num_skills:
        pad = num_skills - out_w
        out_r = jnp.concatenate(
            [out_r, jnp.zeros(rewards.shape[:-1] + (pad,), dtype=out_r.dtype)],
            axis=-1,
        )
        out_d = jnp.concatenate(
            [
                out_d,
                jnp.zeros(skill_done.shape[:-1] + (pad,), dtype=out_d.dtype),
            ],
            axis=-1,
        )
    return out_r, out_d


def static_blocker_arrays(rects):
    """
    Build object-array fragments for static occluders (walls, shields, ...).

    ``rects`` is a Python sequence of ``(x, y, width, height)`` screen rects
    (top-left convention).  Returns ``(x, y, w, h, active, labels)`` int32
    arrays of length ``len(rects)`` with ``active=1`` and ``LABEL_BLOCKER``,
    ready to be concatenated into a label file's object groups the same way
    virtual waypoints are.  Blockers are never meta targets and earn no
    rewards; they only occlude the skill line-of-sight feature.
    """
    xs = jnp.array([r[0] for r in rects], dtype=jnp.int32)
    ys = jnp.array([r[1] for r in rects], dtype=jnp.int32)
    ws = jnp.array([r[2] for r in rects], dtype=jnp.int32)
    hs = jnp.array([r[3] for r in rects], dtype=jnp.int32)
    active = jnp.ones(len(rects), dtype=jnp.int32)
    labels = jnp.full(len(rects), LABEL_BLOCKER, dtype=jnp.int32)
    return xs, ys, ws, hs, active, labels


class EgocentricAction(enum.Enum):
    NOOP = 0
    FIRE = 1
    FORWARD = 2
    RIGHT = 3
    LEFT = 4
    BACKWARD = 5
    FORWARD_RIGHT = 6
    FORWARD_LEFT = 7
    BACKWARD_RIGHT = 8
    BACKWARD_LEFT = 9
    FORWARD_FIRE = 10
    RIGHT_FIRE = 11
    LEFT_FIRE = 12
    BACKWARD_FIRE = 13
    FORWARD_RIGHT_FIRE = 14
    FORWARD_LEFT_FIRE = 15
    BACKWARD_RIGHT_FIRE = 16
    BACKWARD_LEFT_FIRE = 17



@struct.dataclass
class LabeledObjectObservation(ObjectObservation):
    """
    Extends Jaxatari's native ObjectObservation to include universal semantic labels.

    `can_go_offscreen`: per-object flag (int32 0/1). When 1, the global reward logic
    uses the object's screen-edge position to distinguish "scrolled off screen" from
    "destroyed on screen" for DESTROY reward gating. Set to 1 for objects that scroll
    (e.g. Seaquest enemies), 0 for objects that are simply removed when gone
    (e.g. fixed goals, virtual waypoints).
    """

    semantic_label: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array(LABEL_NOOP, dtype=jnp.int32)
    )
    can_go_offscreen: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32)
    )
    # Set to 1 by env wrappers when an object was confirmed destroyed this step
    # (not scrolled off screen). Used by DESTROY reward to bypass position-based
    # ambiguity when the env zeros out position for both kills and scroll-offs.
    was_killed: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32)
    )
    # Set to 1 when an env-specific success event fires (e.g. Skiing gate_pass).
    # Used by COLLECT reward when AABB overlap alone is insufficient.
    was_collected: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.int32)
    )


@struct.dataclass
class ConceptObservation:
    """
    The universal observation space for the modular architecture.
    """

    # The original, stacked game observations (used by the Meta-Policy)
    raw_observation: Any

    # shape: [stack_size, max_objects]
    objects: LabeledObjectObservation

    # Optional stacked pixel observations used by pixel-based meta encoders.
    pixel_observation: Any = None


@struct.dataclass
class SemanticState:
    """Wraps the AtariState to track wrapper-specific internal state."""

    atari_state: AtariState
    obs_stack: Any = None


# Backwards-compatible alias used by wrappers
SemanticObservation = ConceptObservation


@chex.dataclass(frozen=True)
class MetaTransition:
    """Transition for the Meta-Policy (System 2) using global rewards."""

    obs: Any
    objects: Any
    pixel_obs: Any
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: Any
    next_objects: Any
    next_pixel_obs: Any
    q_val: chex.Array
    active_mask: chex.Array
    next_active_mask: chex.Array


@chex.dataclass(frozen=True)
class TandemTransition:
    """Unified rollout buffer for tandem (joint System 1 + System 2) training."""

    # Shared
    done: chex.Array

    # Meta-policy (System 2)
    obs: Any
    objects: Any
    pixel_obs: Any
    next_obs: Any
    next_objects: Any
    next_pixel_obs: Any
    target_idx: chex.Array
    meta_q_val: chex.Array
    meta_reward: chex.Array
    active_mask: chex.Array
    next_active_mask: chex.Array
    meta_target_locked: chex.Array

    # Skills (System 1)
    ego_obs: chex.Array
    next_ego_obs: chex.Array
    ego_action: chex.Array
    active_skill: chex.Array
    skill_q_val: chex.Array
    skill_rewards: chex.Array
    skill_done: chex.Array
    ego_obs_all: chex.Array
    next_ego_obs_all: chex.Array
    ego_action_all: chex.Array
    skill_rewards_all: chex.Array
    skill_done_all: chex.Array
    relabel_mask: chex.Array
    # Global object indices aligned with ego_obs_all's object axis (for outline / is_target).
    object_idx_all: chex.Array

    # Raw environment reward (sign-clipped, before blending/penalties) — logged to wandb
    # so episodic env performance is always visible regardless of META_REWARD_LAMBDA
    # or SKILL_REWARD_LAMBDA.
    env_reward: chex.Array

