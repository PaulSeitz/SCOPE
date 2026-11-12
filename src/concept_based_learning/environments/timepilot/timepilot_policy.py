"""Time Pilot heuristic: nearest DESTROY enemy; AVOID inbound missiles."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_DESTROY
from concept_based_learning.environments.manual_policy import (
    DynamicPick,
    Interrupt,
    ManualPolicySpec,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX
from concept_based_learning.environments.timepilot.timepilot_label import (
    MISSILE_SLOT0,
    MISSILE_SLOT1,
)

# Enemy missiles move 4px/frame — react before contact.
CLOSE_AVOID_RANGE = 40.0
CLOSE_AVOID_RELEASE = 55.0
# Below this per-frame speed^2 we treat the shot as "unknown" (still a threat).
_MIN_MISSILE_SPEED_SQ = 1.0


def _inbound_missile_mask(raw, objects, ctx, plan_target):
    """Missile slots that are approaching the player (or not yet moving)."""
    del raw, plan_target
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    is_missile = jnp.logical_and(idxs >= MISSILE_SLOT0, idxs <= MISSILE_SLOT1)

    vx = objects.x[-1].astype(jnp.float32) - objects.x[-2].astype(jnp.float32)
    vy = objects.y[-1].astype(jnp.float32) - objects.y[-2].astype(jnp.float32)
    dx = (ctx.px - ctx.ox).astype(jnp.float32)
    dy = (ctx.py - ctx.oy).astype(jnp.float32)
    approach = vx * dx + vy * dy
    speed_sq = vx * vx + vy * vy
    # Approaching, or no reliable velocity yet (spawn frame).
    inbound = jnp.logical_or(approach >= 0.0, speed_sq < jnp.float32(_MIN_MISSILE_SPEED_SQ))
    return jnp.logical_and(is_missile, inbound)


TIMEPILOT_POLICY = compile_policy(
    ManualPolicySpec(
        plan=DynamicPick(
            prefer=(LABEL_DESTROY,),
            sticky=False,
        ),
        interrupts=(
            Interrupt(
                labels=(LABEL_AVOID,),
                within=CLOSE_AVOID_RANGE,
                release=CLOSE_AVOID_RELEASE,
                threat_mask=_inbound_missile_mask,
            ),
        ),
        no_target_idx=NO_TARGET_IDX,
    )
)

get_heuristic_target = TIMEPILOT_POLICY.as_legacy_heuristic()
get_heuristic_target_with_state = TIMEPILOT_POLICY.as_stateful_heuristic()
