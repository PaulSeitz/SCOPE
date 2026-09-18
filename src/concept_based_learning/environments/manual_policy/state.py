"""Policy carry state for manual (heuristic) meta-policies.

``PolicyState`` is a flax struct so it can live in JAX training carries and be
vmapped across envs.  A future declarative DSL can emit the same IR nodes and
runtime; this state shape is the stable boundary.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct


NO_TARGET_IDX = -1


@struct.dataclass
class PolicyState:
    """Per-env carry for plan / interrupt bookkeeping.

    Fields
    ------
    plan_idx:
        Current stage index inside a ``Sequence`` plan (0 for a flat DynamicPick).
    mission_target:
        Sticky plan target that survives reactive interrupts (SI-sweep /
        mission_avoid pattern).
    interrupt_latched:
        True while an interrupt is held under hysteresis (within → release).
    aux0:
        Env-specific scratch (e.g. paired ladder-top slot while climbing).
    """

    plan_idx: jnp.ndarray
    mission_target: jnp.ndarray
    interrupt_latched: jnp.ndarray
    aux0: jnp.ndarray

    @staticmethod
    def zero(*, no_target_idx: int = NO_TARGET_IDX) -> "PolicyState":
        return PolicyState(
            plan_idx=jnp.int32(0),
            mission_target=jnp.int32(no_target_idx),
            interrupt_latched=jnp.bool_(False),
            aux0=jnp.int32(no_target_idx),
        )
