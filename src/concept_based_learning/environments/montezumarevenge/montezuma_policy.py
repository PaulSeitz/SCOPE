"""Montezuma's Revenge manual policy: room-1 (room_id 4) key path.

Plan (impossible mid-platform jump to the key is deliberately avoided):
  Outbound:
    1. Center ladder BOT (climb down from spawn to mid)
    2. Rope grab COLLECT (high on rope — must jump the right chasm)
    3. Right ladder TOP → BOT (down to bottom)
    4. Left ladder BOT → TOP (across bottom, up to key ledge)
    5. Key item (COLLECT until inactive)
  Return (same route reversed — no left-gap jump):
    6. Left ladder TOP → BOT
    7. Right ladder BOT → TOP
    8. Rope grab (jump chasm back to center mid)
    9. Center ladder BOT → TOP
   10. Left door (NAVIGATE)

Interrupt:
  Rolling skull AVOID only (gap AVOID boxes are labels for skills / jump
  shaping — they must not steal the Sequence).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID
from concept_based_learning.environments.manual_policy import (
    Interrupt,
    ManualPolicySpec,
    NavigateTo,
    Sequence,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX, PolicyState
from concept_based_learning.environments.montezumarevenge.montezuma_label import (
    PLAYER_STATE_DYING,
)

# Ladder indices after meta-NOOP: 0=center, 1=right, 2=left.
_L0_BOT, _L0_TOP = 4, 8
_L1_BOT, _L1_TOP = 5, 9
_L2_BOT, _L2_TOP = 6, 10
_KEY_SLOT = 12
_LEFT_DOOR_SLOT = 15
_ROPE_GRAB_SLOT = 17  # high COLLECT on room-1 rope (see montezuma_label)
_ENEMY_SLOT0 = 1
_ENEMY_SLOT1 = 3

# Tight bot arrive: AABB overlap alone fires mid-climb (tall player box).
# Require gap<=0 AND feet at standing depth (oy+6 == ladder foot).
LADDER_BOT_ARRIVE = 0.0
LADDER_BOT_FEET_OY_MIN = 6.0
LADDER_ARRIVE = 8.0
LADDER_TOP_ARRIVE = 0.0
# Side-ladder tops: no platform_reach (that band matched mid-shaft and skipped
# to KEY). Require overlap + feet near the raised WP (standing feet-oy ≈ 18).
LADDER_SIDE_TOP_FEET_OY_MAX = 20.0
# Center top: raised WP — floor snap when standing on the upper landing.
LADDER_TOP_PLATFORM_DY_MIN = 8.0
LADDER_TOP_PLATFORM_DY_MAX = 22.0
LADDER_TOP_PLATFORM_X_SLACK = 28.0
# Standing on mid under the rope must not complete; jump touch should.
ROPE_GRAB_ARRIVE = 8.0

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


def _wp_bot(slot: int) -> NavigateTo:
    return NavigateTo(
        slot=slot,
        arrive_dist=LADDER_BOT_ARRIVE,
        complete_when="arrive",
        skip_if_inactive=False,
        arrive_feet_oy_min=LADDER_BOT_FEET_OY_MIN,
    )


def _wp_top(slot: int) -> NavigateTo:
    """Center ladder top — platform snap for the raised spawn-floor WP."""
    return NavigateTo(
        slot=slot,
        arrive_dist=12.0,
        complete_when="arrive",
        skip_if_inactive=True,
        platform_reach_dy_min=LADDER_TOP_PLATFORM_DY_MIN,
        platform_reach_dy_max=LADDER_TOP_PLATFORM_DY_MAX,
        platform_reach_x_slack=LADDER_TOP_PLATFORM_X_SLACK,
    )


def _wp_side_top(slot: int) -> NavigateTo:
    """Side ladder top (key ledge) — must not complete mid-shaft."""
    return NavigateTo(
        slot=slot,
        arrive_dist=LADDER_TOP_ARRIVE,
        complete_when="arrive",
        skip_if_inactive=True,
        arrive_feet_oy_max=LADDER_SIDE_TOP_FEET_OY_MAX,
    )


def _rope_grab(slot: int) -> NavigateTo:
    return NavigateTo(
        slot=slot,
        arrive_dist=ROPE_GRAB_ARRIVE,
        complete_when="arrive",
        skip_if_inactive=True,
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
    """Interrupt on skull enemies only — not room-1 gap AVOID boxes."""
    del raw, objects, plan_target
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    return jnp.logical_and(idxs >= _ENEMY_SLOT0, idxs <= _ENEMY_SLOT1)


def _player_last_scalar(arr) -> jnp.ndarray:
    return arr[-1] if jnp.ndim(arr) > 0 else arr


def _player_is_dying(raw, objects=None, ctx=None) -> jnp.ndarray:
    """True during the death animation (stamped on ``player.state``)."""
    del objects, ctx
    player = getattr(raw, "player", None)
    if player is not None and hasattr(player, "state"):
        return _player_last_scalar(player.state) == jnp.int32(PLAYER_STATE_DYING)
    return jnp.bool_(False)


MONTEZUMA_L1_STAGES = (
    # Outbound to key
    _wp_bot(_L0_BOT),
    _rope_grab(_ROPE_GRAB_SLOT),
    _wp(_L1_TOP),
    _wp_bot(_L1_BOT),
    _wp_bot(_L2_BOT),
    _wp_side_top(_L2_TOP),
    _collect_item(_KEY_SLOT),
    # Return via the same route (no left-gap shortcut)
    _wp_bot(_L2_BOT),
    _wp_bot(_L1_BOT),
    _wp(_L1_TOP),
    _rope_grab(_ROPE_GRAB_SLOT),
    _wp_bot(_L0_BOT),
    _wp_top(_L0_TOP),
    _open_door(_LEFT_DOOR_SLOT),
)

_MONTEZUMA_POLICY_CORE = compile_policy(
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


class _DeathFrozenMontezumaPolicy:
    """Hold the Sequence at stage 0 for the whole death animation.

    Outer callers already reset ``PolicyState`` on life-lost, but the corpse can
    still overlap ladder waypoints and advance ``plan_idx`` before respawn.
    Freezing here matches Kangaroo's crash-freeze behaviour.
    """

    def __init__(self, core):
        self._core = core
        self.spec = core.spec
        self.needs_state = True

    def initial_state(self) -> PolicyState:
        return self._core.initial_state()

    def step(self, raw_observation, objects, state, rng):
        dying = _player_is_dying(raw_observation)
        zero = self._core.initial_state()
        state_in = jax.lax.cond(dying, lambda _: zero, lambda s: s, state)
        target, new_state = self._core.step(raw_observation, objects, state_in, rng)
        new_state = jax.lax.cond(dying, lambda _: zero, lambda s: s, new_state)
        target = jnp.where(dying, jnp.int32(NO_TARGET_IDX), target)
        return target.astype(jnp.int32), new_state

    def as_legacy_heuristic(self):
        return self._core.as_legacy_heuristic()

    def as_stateful_heuristic(self):
        def _fn(raw_observation, objects, prev_target_idx, state, rng):
            del prev_target_idx
            return self.step(raw_observation, objects, state, rng)

        return _fn


MONTEZUMA_POLICY = _DeathFrozenMontezumaPolicy(_MONTEZUMA_POLICY_CORE)

get_heuristic_target = MONTEZUMA_POLICY.as_legacy_heuristic()
get_heuristic_target_with_state = MONTEZUMA_POLICY.as_stateful_heuristic()
