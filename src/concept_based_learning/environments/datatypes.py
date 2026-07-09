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
LABEL_NOOP = 0
LABEL_DESTROY = 1
LABEL_COLLECT = 2
LABEL_AVOID = 3
# Reserved meta-policy slot: selecting this index executes FIRE (not a skill).
LABEL_META_FIRE = 4
NUM_LABELS = 5


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

    # Raw environment reward (sign-clipped, before blending/penalties) — logged to wandb
    # so episodic env performance is always visible regardless of META_REWARD_LAMBDA
    # or SKILL_REWARD_LAMBDA.
    env_reward: chex.Array

