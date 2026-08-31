"""Montezuma's Revenge manual policy: room-4 ladder waypoints + skull AVOID.

Plan (first room / room_id 4 — classic key path):
  1. Center ladder BOT (climb down from spawn on the top floor)
  2. Key item (COLLECT until inactive)
  3. Center ladder BOT → TOP (climb back up)
  4. Left door (NAVIGATE until inactive)

Interrupt:
  AVOID rolling skull when within range (hysteresis release).

Slot layout after meta-NOOP (see montezuma_label.py):
  enemies            = 1..3   (AVOID)
  ladder bottoms     = 4..7   (COLLECT waypoints; plan uses center = 4)
  ladder tops        = 8..11  (COLLECT waypoints; plan uses center = 8)
  items              = 12..14 (COLLECT; key = 12)
  doors              = 15..16 (NAVIGATE; left = 15)
  ropes / conveyors / platforms = 17..31
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID
from concept_based_learning.environments.manual_policy import (
    Interrupt,
    ManualPolicySpec,
    NavigateTo,
    Sequence,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX

# Center ladder (index 0) after meta-NOOP.
_L0_BOT, _L0_TOP = 4, 8
_KEY_SLOT = 12
_LEFT_DOOR_SLOT = 15
_ENEMY_SLOT0 = 1
_ENEMY_SLOT1 = 3

LADDER_ARRIVE = 8.0
LADDER_TOP_ARRIVE = 12.0
LADDER_TOP_PLATFORM_DY_MIN = 8.0
LADDER_TOP_PLATFORM_DY_MAX = 22.0
LADDER_TOP_PLATFORM_X_SLACK = 28.0

# Skull AVOID disc + hysteresis (AABB surface gap via Interrupt defaults).
SKULL_AVOID_RANGE = 28.0
SKULL_AVOID_RELEASE = 40.0
# Legacy alias kept for tests.
CLOSE_AVOID_RANGE = SKULL_AVOID_RANGE


def _wp(slot: int, arrive: float = LADDER_ARRIVE) -> NavigateTo:
    return NavigateTo(
        slot=slot,
        arrive_dist=arrive,
        complete_when="arrive",
        skip_if_inactive=False,
    )


def _wp_top(slot: int) -> NavigateTo:
    return NavigateTo(
        slot=slot,
        arrive_dist=LADDER_TOP_ARRIVE,
        complete_when="arrive",
        skip_if_inactive=False,
        platform_reach_dy_min=LADDER_TOP_PLATFORM_DY_MIN,
        platform_reach_dy_max=LADDER_TOP_PLATFORM_DY_MAX,
        platform_reach_x_slack=LADDER_TOP_PLATFORM_X_SLACK,
    )


def _collect_item(slot: int) -> NavigateTo:
    return NavigateTo(
        slot=slot,
        arrive_dist=10.0,
        complete_when="inactive",
        skip_if_inactive=True,
    )


def _open_door(slot: int) -> NavigateTo:
    return NavigateTo(
        slot=slot,
        arrive_dist=12.0,
        complete_when="either",
        skip_if_inactive=True,
    )


def _skull_threat_mask(raw, objects, ctx, plan_target):
    """Restrict AVOID interrupt to enemy slots (rolling skull in room 4)."""
    del raw, objects, plan_target
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    return jnp.logical_and(idxs >= _ENEMY_SLOT0, idxs <= _ENEMY_SLOT1)


MONTEZUMA_L1_STAGES = (
    # Spawn is already on the top floor near the center ladder — go down first.
    _wp(_L0_BOT),
    _collect_item(_KEY_SLOT),
    _wp(_L0_BOT),
    _wp_top(_L0_TOP),
    _open_door(_LEFT_DOOR_SLOT),
)

MONTEZUMA_POLICY = compile_policy(
    ManualPolicySpec(
        plan=Sequence(stages=MONTEZUMA_L1_STAGES),
        interrupts=(
            Interrupt(
                labels=(LABEL_AVOID,),
                within_x=SKULL_AVOID_RANGE,
                within_y=SKULL_AVOID_RANGE,
                release_x=SKULL_AVOID_RELEASE,
                release_y=SKULL_AVOID_RELEASE,
                threat_mask=_skull_threat_mask,
            ),
        ),
        no_target_idx=NO_TARGET_IDX,
    )
)

get_heuristic_target = MONTEZUMA_POLICY.as_legacy_heuristic()
get_heuristic_target_with_state = MONTEZUMA_POLICY.as_stateful_heuristic()
