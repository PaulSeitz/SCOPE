"""Kangaroo manual policy: L1 ladder climb + fruit COLLECT + reactive threats.

Plan (zig-zag):
  1. Ladder 0 bottom → top → fruit 0
  2. Ladder 1 bottom → top → fruit 1
  3. Ladder 2 bottom → top → fruit 2

Bell / child stay labeled COLLECT for skill rewards but are not in the plan.

Interrupt priority (lowest first, later overrides earlier):
  1. DESTROY — path / melee / emergency; ladder-shaft blockers stay selectable while climbing.
  2. Thrown coconut AVOID — beats nearby monkeys (ground-flush throws share the
     monkey's floor, so DESTROY would otherwise always steal the target).
  3. Falling coconut AVOID — only while actively dropping onto the player
     (``objects.state[falling]=1``); highest reactive priority; climbing-suppressed.

Level-1 slots after meta-NOOP:
  ladder waypoints      = 1..40  (COLLECT; plan uses L0–L2 pairs)
  fruits / bell / child = 41..45 (COLLECT; plan uses fruits 41..43)
  monkeys               = 46..49 (DESTROY)
  falling coconut       = 50     (AVOID)
  thrown cocos          = 51..54 (AVOID)

``EnvComponents.reset_policy_on_life_lost=True`` — visualizer / training must
clear ``PolicyState`` on life loss (Kangaroo ``done`` is only true at game over).
The policy also freezes ``plan_idx`` at 0 while ``player.is_crashing`` so the
falling corpse cannot advance ladder waypoints before respawn.
"""

from concept_based_learning.environments.kangaroo.kangaroo_label import (
    FALLING_COCO_STATE_DROPPING,
    PLAYER_STATE_CLIMBING,
    PLAYER_STATE_CRASHING,
)
import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_DESTROY
from concept_based_learning.environments.manual_policy import (
    Interrupt,
    ManualPolicySpec,
    NavigateTo,
    Sequence,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX, PolicyState

# Ladder bottoms / tops (paired i → 20+i).
_L0_BOT, _L0_TOP = 1, 21
_L1_BOT, _L1_TOP = 2, 22
_L2_BOT, _L2_TOP = 3, 23

# Fruits (COLLECT) between climbs; bell/child stay labeled but off-plan.
_FRUIT0, _FRUIT1, _FRUIT2 = 41, 42, 43
_BELL, _CHILD = 44, 45

# collect(45) | destroy(4) | avoid(5)
_MONKEY_SLOT0 = 46
_MONKEY_SLOT1 = 49
_FALLING_COCO_SLOT = 50
_THROWN_COCO_SLOT0 = 51
_THROWN_COCO_SLOT1 = 54

DESTROY_PATH_RANGE = 90.0
DESTROY_RELEASE = 110.0
DESTROY_PATH_Y_SLACK = 28.0
DESTROY_MELEE_RANGE = 42.0
# Monkeys in the ladder column between player and a vertically offset plan target.
DESTROY_SHAFT_X_SLACK = 16.0
# Last-resort: any monkey this close is selected regardless of path / floor heuristics.
DESTROY_EMERGENCY_RANGE = 26.0
# Falling coco: large vertical ellipse (fast drop) + tight horizontal band.
FALLING_COCO_AVOID_RANGE_X = 32.0
FALLING_COCO_AVOID_RANGE_Y = 95.0
FALLING_COCO_AVOID_RELEASE_X = 42.0
FALLING_COCO_AVOID_RELEASE_Y = 110.0
# Legacy alias (vertical reach — used in tests / dashboards).
FALLING_COCO_DODGE_RANGE = FALLING_COCO_AVOID_RANGE_Y
FALLING_COCO_DODGE_RELEASE = FALLING_COCO_AVOID_RELEASE_Y
# Horizontal: coco centre over player footprint ± slack.
FALLING_COCO_X_MARGIN = 4.0
# Thrown cocos: slower horizontal reach; taller vertical so ground-flush throws
# (centre near feet, ~12 px below player centre) stay inside the ellipse.
THROWN_COCO_AVOID_RANGE_X = 50.0
THROWN_COCO_AVOID_RANGE_Y = 72.0
THROWN_COCO_AVOID_RELEASE_X = 65.0
THROWN_COCO_AVOID_RELEASE_Y = 90.0
# Legacy isotropic aliases (tests / dashboards).
THROWN_COCO_AVOID_RANGE = THROWN_COCO_AVOID_RANGE_X
THROWN_COCO_AVOID_RELEASE = THROWN_COCO_AVOID_RELEASE_X
THROWN_COCO_CORRIDOR_MARGIN_PX = 24.0
# Flattened-circle clip: detection disc cannot extend far below the player.
# Reference = player feet (bbox bottom); margin catches same-floor / overlap threats.
_BELOW_FEET_MARGIN_PX = 32.0
# Thrown coco: vertical band from torso down through feet (+ small under-feet slack
# so ground-flush throws whose centre sits just below the feet line still count).
_THROWN_COCO_BODY_ABOVE_FEET_PX = 40.0
_THROWN_COCO_BELOW_FEET_PX = 8.0
# Nearby thrown coco: skip run-corridor gate (still clipped by the avoid ellipse).
THROWN_COCO_MELEE_RANGE = 28.0
# Player centre must lie in the climb shaft ± slack to count as climbing.
_ON_LADDER_Y_SLACK_PX = 6.0
_ON_LADDER_X_SLACK_PX = 14.0
# Bottom/top COLLECT slots are paired: ladder index i → object slots (i+1, i+21).
_N_LADDER_WAYPOINT_PAIRS = 20
_LADDER_BOT_SLOT_OFFSET = 1
_LADDER_TOP_SLOT_OFFSET = 21
# Bottom: require AABB overlap with the narrow shaft WP (see kangaroo_label).
# arrive_dist=6 on a 16px WP completed the stage ~10px short of the ladder.
LADDER_ARRIVE = 0.0
LADDER_TOP_ARRIVE = 10.0
# Top completion: player centre dy = py - oy in this band (standing on upper floor).
LADDER_TOP_PLATFORM_DY_MIN = 10.0
LADDER_TOP_PLATFORM_DY_MAX = 18.0
LADDER_TOP_PLATFORM_X_SLACK = 28.0


def _wp(slot: int, arrive: float = LADDER_ARRIVE) -> NavigateTo:
    """Static ladder waypoint: done when the player reaches it."""
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


# Fruit collect: skip once feet are clearly on a higher storey than the fruit
# centre (L1 fruit0 hangs between P2/P3 — do not dive back down from P3+).
# Same-floor collect still works: P3 feet≈76 vs fruit1 centre≈90 → not above.
_FRUIT_SKIP_IF_ABOVE_PX = 20.0


def _collect(slot: int) -> NavigateTo:
    """Planned fruit: stay locked until collected, unless we have climbed past it."""
    return NavigateTo(
        slot=slot,
        arrive_dist=LADDER_ARRIVE,
        complete_when="inactive",
        skip_if_player_above_px=_FRUIT_SKIP_IF_ABOVE_PX,
    )


def _player_feet_y(raw) -> jnp.ndarray:
    """Player feet (bbox bottom, screen y grows downward)."""
    player = raw.player
    return player.y[-1] + player.height[-1]


def _not_below_player(raw, ctx) -> jnp.ndarray:
    """Per-slot: object centre is not far below the player's feet."""
    feet = _player_feet_y(raw)
    return ctx.oy <= feet + float(_BELOW_FEET_MARGIN_PX)


def _thrown_coco_on_player_floor(raw, objects, ctx) -> jnp.ndarray:
    """Per-slot: thrown coco overlaps the player's standing body / feet band.

    Ground-flush throws sit with their bbox top on the platform (= feet) and
    centre a couple of pixels below the feet line. A centre-vs-feet clip alone
    is fine with a large margin, but requiring ``oy < py`` (torso-above) would
    drop them. Use the top-left bbox so flush throws clearly overlap the band.
    """
    feet = _player_feet_y(raw)
    o_top = objects.y[-1]
    o_bot = o_top + objects.height[-1]
    band_top = feet - float(_THROWN_COCO_BODY_ABOVE_FEET_PX)
    band_bot = feet + float(_THROWN_COCO_BELOW_FEET_PX)
    return jnp.logical_and(o_top <= band_bot, o_bot >= band_top)


def _coco_centre_x(objects) -> jnp.ndarray:
    return objects.x[-1] + objects.width[-1] * 0.5


def _falling_coco_x_aligned(raw, objects, ctx) -> jnp.ndarray:
    """Per-slot: coco centre is over the player's horizontal footprint."""
    player = raw.player
    px = player.x[-1]
    pw = player.width[-1]
    left = px - float(FALLING_COCO_X_MARGIN)
    right = px + pw + float(FALLING_COCO_X_MARGIN)
    coco_cx = _coco_centre_x(objects)
    return jnp.logical_and(coco_cx >= left, coco_cx <= right)


def _thrown_coco_in_run_corridor(ctx, plan_target, objects) -> jnp.ndarray:
    """Per-slot: thrown coco centre lies in the widened x-band toward the plan target."""
    valid_tgt = jnp.logical_and(plan_target >= 0, plan_target < ctx.n)
    tx = jnp.where(valid_tgt, ctx.ox[plan_target], ctx.px)
    margin = float(THROWN_COCO_CORRIDOR_MARGIN_PX)
    lo = jnp.minimum(ctx.px, tx) - margin
    hi = jnp.maximum(ctx.px, tx) + margin
    coco_cx = _coco_centre_x(objects)
    in_corridor = jnp.logical_and(coco_cx >= lo, coco_cx <= hi)
    has_span = (hi - lo) > 4.0
    return jnp.logical_and(valid_tgt, jnp.logical_and(has_span, in_corridor))


def _ladder_last_frame(arr) -> jnp.ndarray:
    """JAX-atari stacks ladder fields over frames; policy uses the current frame."""
    return arr[-1] if jnp.ndim(arr) > 1 else arr


def _player_on_any_ladder(raw, objects, ctx) -> jnp.ndarray:
    """True when the player is on a ladder column (full shaft, not just the sprite)."""
    del objects
    if not hasattr(raw, "ladders"):
        return jnp.bool_(False)
    ladders = raw.ladders
    lx = _ladder_last_frame(ladders.x)
    ly = _ladder_last_frame(ladders.y)
    lw = _ladder_last_frame(ladders.width)
    lh = _ladder_last_frame(ladders.height)
    active = _ladder_last_frame(ladders.active) > 0.5

    lad_cx = lx + lw * 0.5
    lad_top = ly
    lad_bot = ly + lh
    y_slack = float(_ON_LADDER_Y_SLACK_PX)
    x_slack = float(_ON_LADDER_X_SLACK_PX)

    # Ladder sprites are shorter than the climb: extend the shaft to bot/top waypoints.
    n_lads = lx.shape[0]
    # Waypoints sit at object indices (ladder_i + 1) bottom, (ladder_i + 21) top;
    # index 0 is the meta NOOP slot in labeled observations.
    idx = jnp.arange(n_lads, dtype=jnp.int32)
    bot_i = jnp.clip(idx + _LADDER_BOT_SLOT_OFFSET, 0, max(ctx.n - 1, 0))
    top_i = jnp.clip(idx + _LADDER_TOP_SLOT_OFFSET, 0, max(ctx.n - 1, 0))
    bot_ok = ctx.active[bot_i]
    top_ok = ctx.active[top_i]
    shaft_top = jnp.where(top_ok, jnp.minimum(lad_top, ctx.oy[top_i]), lad_top)
    shaft_bot = jnp.where(bot_ok, jnp.maximum(lad_bot, ctx.oy[bot_i]), lad_bot)

    in_y = jnp.logical_and(
        ctx.py >= shaft_top - y_slack,
        ctx.py <= shaft_bot + y_slack,
    )
    near_x = jnp.abs(ctx.px - lad_cx) <= (lw * 0.5 + x_slack)
    on_lad = jnp.logical_and(active, jnp.logical_and(in_y, near_x))
    return jnp.any(on_lad)


def _player_last_scalar(arr) -> jnp.ndarray:
    return arr[-1] if jnp.ndim(arr) > 0 else arr


def _player_is_climbing(raw, objects, ctx) -> jnp.ndarray:
    """True when JAX-Atari reports active climb (stamped on ``player.state``)."""
    player = getattr(raw, "player", None)
    if player is not None and hasattr(player, "state"):
        return _player_last_scalar(player.state) == jnp.int32(PLAYER_STATE_CLIMBING)
    # Unit tests without stamped state: fall back to shaft geometry.
    return _player_on_any_ladder(raw, objects, ctx)


def _player_is_crashing(raw, objects=None, ctx=None) -> jnp.ndarray:
    """True during the death / fall animation (stamped on ``player.state``)."""
    del objects, ctx
    player = getattr(raw, "player", None)
    if player is not None and hasattr(player, "state"):
        return _player_last_scalar(player.state) == jnp.int32(PLAYER_STATE_CRASHING)
    return jnp.bool_(False)


def _suppress_falling_coco_while_climbing(raw, objects, ctx) -> jnp.ndarray:
    """Disable falling-coco AVOID while the player is in climb state."""
    return _player_is_climbing(raw, objects, ctx)


def _suppress_thrown_coco_while_climbing(raw, objects, ctx) -> jnp.ndarray:
    """Disable thrown-coco AVOID while the player is in climb state."""
    return _player_is_climbing(raw, objects, ctx)


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


def _object_blocks_ladder_shaft(ctx, plan_target: jnp.ndarray) -> jnp.ndarray:
    """Per-slot: object centre is in the ladder column between player and plan target."""
    valid_tgt = jnp.logical_and(plan_target >= 0, plan_target < ctx.n)
    tx = jnp.where(valid_tgt, ctx.ox[plan_target], ctx.px)
    ty = jnp.where(valid_tgt, ctx.oy[plan_target], ctx.py)
    vertical_run = jnp.abs(ty - ctx.py) > float(DESTROY_PATH_Y_SLACK)
    lo_y = jnp.minimum(ctx.py, ty)
    hi_y = jnp.maximum(ctx.py, ty)
    between_y = jnp.logical_and(ctx.oy >= lo_y, ctx.oy <= hi_y)
    x_slack = float(DESTROY_SHAFT_X_SLACK)
    lo_x = jnp.minimum(ctx.px, tx) - x_slack
    hi_x = jnp.maximum(ctx.px, tx) + x_slack
    in_column = jnp.logical_and(ctx.ox >= lo_x, ctx.ox <= hi_x)
    return jnp.logical_and(
        valid_tgt,
        jnp.logical_and(vertical_run, jnp.logical_and(between_y, in_column)),
    )


def _kangaroo_destroy_threat_mask(raw, objects, ctx, plan_target):
    """Monkeys: horizontal path block, ladder-shaft block, same-floor melee, or emergency."""
    del objects, raw
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    is_monkey = jnp.logical_and(idxs >= _MONKEY_SLOT0, idxs <= _MONKEY_SLOT1)
    blocks = jnp.logical_or(
        _object_blocks_path(ctx, plan_target, y_slack=DESTROY_PATH_Y_SLACK),
        _object_blocks_ladder_shaft(ctx, plan_target),
    )
    same_floor = jnp.abs(ctx.oy - ctx.py) <= float(DESTROY_PATH_Y_SLACK)
    in_melee = jnp.logical_and(
        is_monkey,
        jnp.logical_and(same_floor, ctx.distances <= float(DESTROY_MELEE_RANGE)),
    )
    emergency = jnp.logical_and(
        is_monkey,
        ctx.distances <= float(DESTROY_EMERGENCY_RANGE),
    )
    return jnp.logical_or(blocks, jnp.logical_or(in_melee, emergency))


def _kangaroo_falling_coco_threat_mask(raw, objects, ctx, plan_target):
    """Falling coco: only while actively dropping, x-aligned, above the player."""
    del plan_target
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    is_falling = idxs == _FALLING_COCO_SLOT
    # Dropping flag stamped onto objects.state[falling] by KangarooSemanticWrapper.
    dropping = objects.state[-1] == jnp.int32(FALLING_COCO_STATE_DROPPING)
    x_ok = _falling_coco_x_aligned(raw, objects, ctx)
    above = ctx.oy < ctx.py + 8.0
    return jnp.logical_and(
        is_falling,
        jnp.logical_and(
            dropping,
            jnp.logical_and(x_ok, jnp.logical_and(above, _not_below_player(raw, ctx))),
        ),
    )


def _kangaroo_thrown_coco_threat_mask(raw, objects, ctx, plan_target):
    """Thrown cocos in the run corridor (or melee-close) on the player's floor."""
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    is_thrown = jnp.logical_and(
        idxs >= _THROWN_COCO_SLOT0, idxs <= _THROWN_COCO_SLOT1
    )
    in_corridor = _thrown_coco_in_run_corridor(ctx, plan_target, objects)
    melee = ctx.distances <= float(THROWN_COCO_MELEE_RANGE)
    path_ok = jnp.logical_or(in_corridor, melee)
    floor_ok = _thrown_coco_on_player_floor(raw, objects, ctx)
    return jnp.logical_and(is_thrown, jnp.logical_and(path_ok, floor_ok))


KANGAROO_L1_STAGES = (
    _wp(_L0_BOT),
    _wp_top(_L0_TOP),
    _collect(_FRUIT0),
    _wp(_L1_BOT),
    _wp_top(_L1_TOP),
    _collect(_FRUIT1),
    _wp(_L2_BOT),
    _wp_top(_L2_TOP),
    _collect(_FRUIT2),
)

_KANGAROO_POLICY_CORE = compile_policy(
    ManualPolicySpec(
        plan=Sequence(stages=KANGAROO_L1_STAGES),
        interrupts=(
            Interrupt(
                labels=(LABEL_DESTROY,),
                within_x=DESTROY_PATH_RANGE,
                within_y=DESTROY_PATH_RANGE,
                release_x=DESTROY_RELEASE,
                release_y=DESTROY_RELEASE,
                threat_mask=_kangaroo_destroy_threat_mask,
            ),
            # After DESTROY: ground-level throws share the monkey's floor, so
            # monkeys would otherwise always win and the coco is never punched.
            Interrupt(
                labels=(LABEL_AVOID,),
                within_x=THROWN_COCO_AVOID_RANGE_X,
                within_y=THROWN_COCO_AVOID_RANGE_Y,
                release_x=THROWN_COCO_AVOID_RELEASE_X,
                release_y=THROWN_COCO_AVOID_RELEASE_Y,
                threat_mask=_kangaroo_thrown_coco_threat_mask,
                suppress=_suppress_thrown_coco_while_climbing,
            ),
            # Highest reactive priority: actively dropping coco beats monkeys/thrown.
            Interrupt(
                labels=(LABEL_AVOID,),
                within_x=FALLING_COCO_AVOID_RANGE_X,
                within_y=FALLING_COCO_AVOID_RANGE_Y,
                release_x=FALLING_COCO_AVOID_RELEASE_X,
                release_y=FALLING_COCO_AVOID_RELEASE_Y,
                threat_mask=_kangaroo_falling_coco_threat_mask,
                suppress=_suppress_falling_coco_while_climbing,
            ),
        ),
        no_target_idx=NO_TARGET_IDX,
    )
)


class _CrashFrozenKangarooPolicy:
    """Hold the climb Sequence at stage 0 for the whole death animation.

    Outer callers already reset ``PolicyState`` on life-lost, but the falling
    corpse still overlaps ladder bottom waypoints and advances ``plan_idx``.
    After respawn the plan then starts mid-route (killing every life after the
    first). Freezing here makes episode / life boundaries self-healing.
    """

    def __init__(self, core):
        self._core = core
        self.spec = core.spec
        self.needs_state = True

    def initial_state(self) -> PolicyState:
        return self._core.initial_state()

    def step(self, raw_observation, objects, state, rng):
        crashing = _player_is_crashing(raw_observation)
        zero = self._core.initial_state()
        state_in = jax.lax.cond(crashing, lambda _: zero, lambda s: s, state)
        target, new_state = self._core.step(raw_observation, objects, state_in, rng)
        new_state = jax.lax.cond(crashing, lambda _: zero, lambda s: s, new_state)
        target = jnp.where(crashing, jnp.int32(NO_TARGET_IDX), target)
        return target.astype(jnp.int32), new_state

    def as_legacy_heuristic(self):
        return self._core.as_legacy_heuristic()

    def as_stateful_heuristic(self):
        def _fn(raw_observation, objects, prev_target_idx, state, rng):
            del prev_target_idx
            return self.step(raw_observation, objects, state, rng)

        return _fn


KANGAROO_POLICY = _CrashFrozenKangarooPolicy(_KANGAROO_POLICY_CORE)

get_heuristic_target = KANGAROO_POLICY.as_legacy_heuristic()
get_heuristic_target_with_state = KANGAROO_POLICY.as_stateful_heuristic()
