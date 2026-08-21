"""Kangaroo manual policy: explicit L1 climb Sequence (waypoints only).

Temporary fruit-free plan for transfer debugging — only ladder bottoms/tops:

  1. Ladder 0 bottom → top (ground → mid-low)
  2. Ladder 1 bottom → top (mid-low → mid-high)
  3. Ladder 2 bottom → top (mid-high → top)

Reactive interrupts (DESTROY / AVOID) still insert for same-floor threats;
falling coco only within ``FALLING_COCO_DODGE_RANGE``.

Level-1 slots after meta-NOOP:
  ladder 0 bottom/top = 1 / 21
  ladder 1 bottom/top = 2 / 22
  ladder 2 bottom/top = 3 / 23
  falling coco        = 46
  monkeys             = 47..50
  thrown cocos        = 51..54

``EnvComponents.reset_policy_on_life_lost=True`` — visualizer / training must
clear ``PolicyState`` on life loss (Kangaroo ``done`` is only true at game over).
"""

import jax.numpy as jnp

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

# Collectibles (unused in the waypoint-only plan; kept for tests / future).
_FRUIT0, _FRUIT1, _FRUIT2 = 41, 42, 43
_BELL, _CHILD = 44, 45

_FALLING_COCO_SLOT = 46
_MONKEY_SLOT0 = 47
_MONKEY_SLOT1 = 50

DESTROY_RANGE = 35.0
DESTROY_PATH_RANGE = 90.0
FALLING_COCO_DODGE_RANGE = 30.0
DESTROY_PATH_Y_SLACK = 28.0
_BELOW_FLOOR_MARGIN_PX = 20.0
AVOID_RANGE_X = 40.0
AVOID_RANGE_Y = 55.0
LADDER_ARRIVE = 6.0
LADDER_TOP_ARRIVE = 10.0


def _wp(slot: int, arrive: float = LADDER_ARRIVE) -> NavigateTo:
    """Static waypoint: done when the player reaches it."""
    return NavigateTo(
        slot=slot,
        arrive_dist=arrive,
        complete_when="arrive",
        skip_if_inactive=False,
    )


def _wp_top(slot: int) -> NavigateTo:
    return _wp(slot, arrive=LADDER_TOP_ARRIVE)


def _player_floor_y(raw) -> jnp.ndarray:
    """Y of the platform the player's feet are standing on (top-edge convention)."""
    player = raw.player
    feet = player.y[-1] + player.height[-1]
    if not hasattr(raw, "platforms"):
        return feet
    plat_y = raw.platforms.y[-1]
    plat_active = raw.platforms.active[-1] > 0.5
    dist = jnp.where(plat_active, jnp.abs(plat_y - feet), jnp.inf)
    return plat_y[jnp.argmin(dist)]


def _not_below_player_floor(raw, ctx) -> jnp.ndarray:
    """Per-slot: object centre is on the player's floor or higher (smaller y)."""
    floor = _player_floor_y(raw)
    return ctx.oy <= floor + _BELOW_FLOOR_MARGIN_PX


def _object_blocks_path(ctx, plan_target: jnp.ndarray, *, y_slack: float) -> jnp.ndarray:
    """Per-slot: object centre lies between player and plan_target on roughly the same floor."""
    valid_tgt = jnp.logical_and(plan_target >= 0, plan_target < ctx.n)
    tx = jnp.where(valid_tgt, ctx.ox[plan_target], ctx.px)
    ty = jnp.where(valid_tgt, ctx.oy[plan_target], ctx.py)
    same_floor = jnp.abs(ctx.oy - ty) <= float(y_slack)
    lo = jnp.minimum(ctx.px, tx)
    hi = jnp.maximum(ctx.px, tx)
    between_x = jnp.logical_and(ctx.ox >= lo, ctx.ox <= hi)
    has_span = (hi - lo) > 4.0
    return jnp.logical_and(
        valid_tgt,
        jnp.logical_and(has_span, jnp.logical_and(same_floor, between_x)),
    )


def _kangaroo_destroy_threat_mask(raw, objects, ctx, plan_target):
    """Monkeys: close-or-blocking on our floor. Falling coco: only within dodge range."""
    del objects
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    is_monkey = jnp.logical_and(idxs >= _MONKEY_SLOT0, idxs <= _MONKEY_SLOT1)
    is_falling = idxs == _FALLING_COCO_SLOT
    close = ctx.distances <= DESTROY_RANGE
    blocks = _object_blocks_path(ctx, plan_target, y_slack=DESTROY_PATH_Y_SLACK)
    monkey_ok = jnp.logical_or(close, blocks)
    falling_ok = ctx.distances <= FALLING_COCO_DODGE_RANGE
    base = jnp.where(
        is_monkey,
        monkey_ok,
        jnp.where(is_falling, falling_ok, True),
    )
    return jnp.logical_and(base, _not_below_player_floor(raw, ctx))


def _kangaroo_avoid_threat_mask(raw, objects, ctx, plan_target):
    """Thrown cocos: only on our floor or above (in-front gating is in the label file)."""
    del objects, plan_target
    return _not_below_player_floor(raw, ctx)


# Waypoint-only L1 climb (no fruits / bell / child).
KANGAROO_L1_STAGES = (
    _wp(_L0_BOT),
    _wp_top(_L0_TOP),
    _wp(_L1_BOT),
    _wp_top(_L1_TOP),
    _wp(_L2_BOT),
    _wp_top(_L2_TOP),
)

KANGAROO_POLICY = compile_policy(
    ManualPolicySpec(
        plan=Sequence(stages=KANGAROO_L1_STAGES),
        interrupts=(
            Interrupt(
                labels=(LABEL_DESTROY,),
                within=DESTROY_PATH_RANGE,
                threat_mask=_kangaroo_destroy_threat_mask,
            ),
            Interrupt(
                labels=(LABEL_AVOID,),
                within_x=AVOID_RANGE_X,
                within_y=AVOID_RANGE_Y,
                threat_mask=_kangaroo_avoid_threat_mask,
            ),
        ),
        no_target_idx=NO_TARGET_IDX,
    )
)

get_heuristic_target = KANGAROO_POLICY.as_legacy_heuristic()
get_heuristic_target_with_state = KANGAROO_POLICY.as_stateful_heuristic()
