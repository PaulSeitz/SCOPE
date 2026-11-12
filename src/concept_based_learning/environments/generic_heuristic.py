"""
Generic, environment-agnostic heuristic meta-policy.

Built on ``manual_policy`` IR so new envs can reuse the same stickiness /
interrupt pattern without copying JAX.  Targeting rule (priority low → high):

  1. Default: nearest COLLECT (fallback DESTROY, then AVOID / NAVIGATE via
     prefer order) with sticky COLLECT/DESTROY/NAVIGATE/JUMP_OVER missions.
  2. DESTROY within DESTROY_RANGE overrides stickiness.
  3. AVOID within AVOID_RANGE overrides all.
"""

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_JUMP_OVER,
    LABEL_NAVIGATE,
)
from concept_based_learning.environments.manual_policy import (
    DEFAULT_MISSION_LABELS,
    DynamicPick,
    Interrupt,
    ManualPolicySpec,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import (
    NO_TARGET_IDX,
)

# Proximity thresholds (pixels, AABB surface gap — same as skill ego distance).
DESTROY_RANGE = 25.0
AVOID_RANGE = 15.0

GENERIC_POLICY = compile_policy(
    ManualPolicySpec(
        plan=DynamicPick(
            prefer=(
                LABEL_COLLECT,
                LABEL_NAVIGATE,
                LABEL_DESTROY,
                LABEL_JUMP_OVER,
                LABEL_AVOID,
            ),
            sticky=True,
            sticky_labels=DEFAULT_MISSION_LABELS,
        ),
        interrupts=(
            Interrupt(labels=(LABEL_DESTROY,), within=DESTROY_RANGE),
            Interrupt(labels=(LABEL_AVOID,), within=AVOID_RANGE),
        ),
        no_target_idx=NO_TARGET_IDX,
    )
)

get_heuristic_target = GENERIC_POLICY.as_legacy_heuristic()
get_heuristic_target_with_state = GENERIC_POLICY.as_stateful_heuristic()
