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
NUM_LABELS = 4


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
    """

    semantic_label: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array(LABEL_NOOP, dtype=jnp.int32)
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
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: Any
    next_objects: Any
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
    next_obs: Any
    next_objects: Any
    target_idx: chex.Array
    meta_q_val: chex.Array
    meta_reward: chex.Array
    active_mask: chex.Array
    next_active_mask: chex.Array

    # Skills (System 1)
    ego_obs: chex.Array
    next_ego_obs: chex.Array
    ego_action: chex.Array
    active_skill: chex.Array
    skill_q_val: chex.Array
    skill_rewards: chex.Array
    skill_done: chex.Array

    # Raw environment reward (sign-clipped, before blending/penalties) — logged to wandb
    # so episodic env performance is always visible regardless of META_REWARD_LAMBDA.
    env_reward: chex.Array

