"""Kangaroo manual policy: explicit L1 climb / collect Sequence.

Plan is a fixed waypoint list (not dynamic nearest-COLLECT).  Reactive
interrupts only for dynamic threats (DESTROY / AVOID); after the interrupt
clears, the Sequence resumes at the same ``plan_idx``.

Level-1 geometry (JAXAtari kangaroo_levels L1), slots after meta-NOOP:
  ladder 0 bottom/top = 1 / 21   (right, ground → mid-low)
  ladder 1 bottom/top = 2 / 22   (left,  mid-low → mid-high)
  ladder 2 bottom/top = 3 / 23   (right, mid-high → top)
  fruit 0 / 1 / 2     = 41 / 42 / 43
  bell / child        = 44 / 45

Zig-zag: climb L0 → fruit0 → climb L1 → fruit1 → climb L2 → fruit2 → bell → child.
Fruits stay labeled COLLECT for skill rewards; the plan still uses NavigateTo
so they are ordered stages, not a dynamic between-climb preference.
"""

from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_DESTROY
from concept_based_learning.environments.manual_policy import (
    Interrupt,
    ManualPolicySpec,
    NavigateTo,
    Sequence,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX

# Ladder bottoms / tops (paired i → 20+i).
_L0_BOT, _L0_TOP = 1, 21
_L1_BOT, _L1_TOP = 2, 22
_L2_BOT, _L2_TOP = 3, 23

# Collectibles.
_FRUIT0, _FRUIT1, _FRUIT2 = 41, 42, 43
_BELL, _CHILD = 44, 45

DESTROY_RANGE = 35.0
# AVOID: taller vertical reach for falling coconuts; tighter sideways.
AVOID_RANGE_X = 20.0
AVOID_RANGE_Y = 45.0
LADDER_ARRIVE = 6.0  # tight: waypoints sit at standing centre on each floor


def _wp(slot: int, arrive: float = LADDER_ARRIVE) -> NavigateTo:
    """Static waypoint: done when the player reaches it (or slot never spawned)."""
    return NavigateTo(slot=slot, arrive_dist=arrive, complete_when="arrive")


def _collect(slot: int) -> NavigateTo:
    """Planned collectible: stay locked until the object is gone."""
    return NavigateTo(slot=slot, arrive_dist=LADDER_ARRIVE, complete_when="inactive")


# Explicit L1 route.  Inactive/already-collected fruits are skipped immediately
# (complete_when=inactive with active=False → done).
KANGAROO_L1_STAGES = (
    _wp(_L0_BOT),
    _wp(_L0_TOP),
    _collect(_FRUIT0),
    _wp(_L1_BOT),
    _wp(_L1_TOP),
    _collect(_FRUIT1),
    _wp(_L2_BOT),
    _wp(_L2_TOP),
    _collect(_FRUIT2),
    _collect(_BELL),
    _collect(_CHILD),
)

KANGAROO_POLICY = compile_policy(
    ManualPolicySpec(
        plan=Sequence(stages=KANGAROO_L1_STAGES),
        interrupts=(
            Interrupt(labels=(LABEL_DESTROY,), within=DESTROY_RANGE),
            Interrupt(
                labels=(LABEL_AVOID,),
                within_x=AVOID_RANGE_X,
                within_y=AVOID_RANGE_Y,
            ),
        ),
        no_target_idx=NO_TARGET_IDX,
    )
)

get_heuristic_target = KANGAROO_POLICY.as_legacy_heuristic()
get_heuristic_target_with_state = KANGAROO_POLICY.as_stateful_heuristic()
