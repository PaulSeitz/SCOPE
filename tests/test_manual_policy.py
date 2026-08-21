"""Unit tests for manual_policy IR runtime and label→skill mapping."""

import jax
import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_JUMP_OVER,
    LABEL_NAVIGATE,
    label_to_skill_idx,
)
from concept_based_learning.environments.generic_heuristic import (
    DESTROY_RANGE,
    GENERIC_POLICY,
    get_heuristic_target,
)
from concept_based_learning.environments.manual_policy import (
    ClimbLadders,
    DynamicPick,
    ForceWhen,
    Interrupt,
    ManualPolicySpec,
    NavigateTo,
    Sequence,
    compile_policy,
    register_predicate,
)
from concept_based_learning.environments.manual_policy.state import (
    NO_TARGET_IDX,
    PolicyState,
)


@struct.dataclass
class _Player:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


@struct.dataclass
class _Raw:
    player: _Player


@struct.dataclass
class _Objects:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray
    semantic_label: jnp.ndarray


def _stack(val, t=4):
    return jnp.stack([val] * t)


def _make_obs(player_xy, objects):
    """objects: list of (x, y, w, h, active, label)."""
    px, py = player_xy
    raw = _Raw(
        player=_Player(
            x=_stack(jnp.int32(px)),
            y=_stack(jnp.int32(py)),
            width=_stack(jnp.int32(4)),
            height=_stack(jnp.int32(4)),
        )
    )
    n = len(objects)
    xs = jnp.array([o[0] for o in objects], dtype=jnp.int32)
    ys = jnp.array([o[1] for o in objects], dtype=jnp.int32)
    ws = jnp.array([o[2] for o in objects], dtype=jnp.int32)
    hs = jnp.array([o[3] for o in objects], dtype=jnp.int32)
    act = jnp.array([o[4] for o in objects], dtype=jnp.int32)
    labs = jnp.array([o[5] for o in objects], dtype=jnp.int32)
    objs = _Objects(
        x=_stack(xs),
        y=_stack(ys),
        width=_stack(ws),
        height=_stack(hs),
        active=_stack(act),
        semantic_label=_stack(labs),
    )
    return raw, objs


def test_label_to_skill_idx_full_vocab():
    assert int(label_to_skill_idx(LABEL_DESTROY, 5)) == 0
    assert int(label_to_skill_idx(LABEL_COLLECT, 5)) == 1
    assert int(label_to_skill_idx(LABEL_AVOID, 5)) == 2
    assert int(label_to_skill_idx(LABEL_NAVIGATE, 5)) == 3
    assert int(label_to_skill_idx(LABEL_JUMP_OVER, 5)) == 4


def test_label_to_skill_idx_truncated_fallback():
    # NUM_SKILLS=3: NAVIGATE → COLLECT, JUMP_OVER → AVOID
    assert int(label_to_skill_idx(LABEL_NAVIGATE, 3)) == 1
    assert int(label_to_skill_idx(LABEL_JUMP_OVER, 3)) == 2
    # NUM_SKILLS=4: NAVIGATE keeps its head; JUMP_OVER → AVOID
    assert int(label_to_skill_idx(LABEL_NAVIGATE, 4)) == 3
    assert int(label_to_skill_idx(LABEL_JUMP_OVER, 4)) == 2


def test_remap_navigate_to_collect_flag():
    from concept_based_learning.environments.datatypes import (
        configure_label_routing_from_config,
        set_remap_navigate_to_collect,
    )

    try:
        set_remap_navigate_to_collect(True)
        assert int(label_to_skill_idx(LABEL_NAVIGATE, 4)) == 1
        assert int(label_to_skill_idx(LABEL_COLLECT, 4)) == 1
        configure_label_routing_from_config({"REMAP_NAVIGATE_TO_COLLECT": False})
        assert int(label_to_skill_idx(LABEL_NAVIGATE, 4)) == 3
    finally:
        set_remap_navigate_to_collect(False)


def test_dynamic_pick_prefers_collect_then_destroy_interrupt():
    # Slot 0 = inactive meta; 1 = far COLLECT; 2 = close DESTROY
    raw, objs = _make_obs(
        (50, 50),
        [
            (0, 0, 1, 1, 0, 0),
            (80, 50, 4, 4, 1, LABEL_COLLECT),
            (55, 50, 4, 4, 1, LABEL_DESTROY),  # within DESTROY_RANGE of player
        ],
    )
    key = jax.random.PRNGKey(0)
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), key)
    assert int(target) == 2  # close DESTROY interrupt


def test_dynamic_pick_sticky_mission():
    raw, objs = _make_obs(
        (50, 50),
        [
            (0, 0, 1, 1, 0, 0),
            (70, 50, 4, 4, 1, LABEL_COLLECT),
            (90, 50, 4, 4, 1, LABEL_COLLECT),
        ],
    )
    key = jax.random.PRNGKey(0)
    # Lock onto slot 2 even though slot 1 is nearer
    target = get_heuristic_target(raw, objs, jnp.int32(2), key)
    assert int(target) == 2


def test_sequence_navigate_advances():
    policy = compile_policy(
        ManualPolicySpec(
            plan=Sequence(
                stages=(
                    NavigateTo(slot=1, arrive_dist=5.0, complete_when="arrive"),
                    NavigateTo(slot=2, arrive_dist=5.0, complete_when="arrive"),
                )
            ),
            no_target_idx=NO_TARGET_IDX,
        )
    )
    # Far from slot 1 → stay on stage 0
    raw, objs = _make_obs(
        (0, 0),
        [
            (0, 0, 1, 1, 0, 0),
            (50, 50, 2, 2, 1, LABEL_NAVIGATE),
            (10, 10, 2, 2, 1, LABEL_NAVIGATE),
        ],
    )
    state = policy.initial_state()
    target, state = policy.step(raw, objs, state, None)
    assert int(target) == 1
    assert int(state.plan_idx) == 0

    # Arrive at slot 1 → advance to stage 1
    raw2, objs2 = _make_obs(
        (50, 50),
        [
            (0, 0, 1, 1, 0, 0),
            (50, 50, 2, 2, 1, LABEL_NAVIGATE),
            (10, 10, 2, 2, 1, LABEL_NAVIGATE),
        ],
    )
    target2, state2 = policy.step(raw2, objs2, state, None)
    assert int(state2.plan_idx) == 1
    assert int(target2) == 2


def test_navigate_collect_waits_until_inactive():
    policy = compile_policy(
        ManualPolicySpec(
            plan=Sequence(
                stages=(
                    NavigateTo(slot=1, complete_when="inactive"),
                    NavigateTo(slot=2, arrive_dist=5.0, complete_when="arrive"),
                )
            ),
            no_target_idx=NO_TARGET_IDX,
        )
    )
    # Standing on fruit but still active → stay on stage 0
    raw, objs = _make_obs(
        (50, 50),
        [
            (0, 0, 1, 1, 0, 0),
            (50, 50, 4, 4, 1, LABEL_COLLECT),
            (10, 10, 2, 2, 1, LABEL_NAVIGATE),
        ],
    )
    state = policy.initial_state()
    target, state = policy.step(raw, objs, state, None)
    assert int(target) == 1
    assert int(state.plan_idx) == 0

    # Fruit gone → advance
    raw2, objs2 = _make_obs(
        (50, 50),
        [
            (0, 0, 1, 1, 0, 0),
            (50, 50, 4, 4, 0, LABEL_COLLECT),
            (10, 10, 2, 2, 1, LABEL_NAVIGATE),
        ],
    )
    target2, state2 = policy.step(raw2, objs2, state, None)
    assert int(state2.plan_idx) == 1
    assert int(target2) == 2


def test_kangaroo_l1_starts_at_first_ladder_not_fruit():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _FRUIT0,
    )

    # Player near fruit0 but far from L0 bottom — plan must still pick ladder.
    slots = [(0, 0, 1, 1, 0, 0)]  # noop
    for i in range(20):
        if i == 0:
            slots.append((136, 168, 2, 2, 1, LABEL_NAVIGATE))
        else:
            slots.append((0, 0, 1, 1, 0, LABEL_NAVIGATE))
    for _ in range(20):
        slots.append((0, 0, 1, 1, 0, LABEL_NAVIGATE))
    slots.extend(
        [
            (119, 108, 4, 4, 1, LABEL_COLLECT),
            (39, 84, 4, 4, 1, LABEL_COLLECT),
            (59, 60, 4, 4, 1, LABEL_COLLECT),
            (93, 36, 4, 4, 1, LABEL_COLLECT),
            (121, 13, 4, 4, 1, LABEL_COLLECT),
        ]
    )
    raw, objs = _make_obs((120, 108), slots)
    target, state = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _L0_BOT
    assert int(target) != _FRUIT0
    assert int(state.plan_idx) == 0


def test_navigate_waits_when_skip_if_inactive_false():
    policy = compile_policy(
        ManualPolicySpec(
            plan=NavigateTo(
                slot=1, arrive_dist=5.0, complete_when="arrive", skip_if_inactive=False
            ),
            no_target_idx=NO_TARGET_IDX,
        )
    )
    raw, objs = _make_obs(
        (50, 50),
        [
            (0, 0, 1, 1, 0, 0),
            (80, 50, 2, 2, 0, LABEL_NAVIGATE),  # gated inactive
        ],
    )
    target, state = policy.step(raw, objs, policy.initial_state(), None)
    assert int(target) == NO_TARGET_IDX
    assert int(state.plan_idx) == 0  # must not skip ahead


def test_destroy_path_blocking_monkey_interrupt():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
    )

    # Plan wants L0 bottom at x=136; monkey sits between player (40) and ladder.
    slots = [(0, 0, 1, 1, 0, 0)]
    for i in range(20):
        if i == 0:
            slots.append((136, 160, 2, 2, 1, LABEL_NAVIGATE))
        else:
            slots.append((0, 0, 1, 1, 0, LABEL_NAVIGATE))
    for _ in range(20):
        slots.append((0, 0, 1, 1, 0, LABEL_NAVIGATE))
    slots.extend(
        [
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_DESTROY),  # falling coco inactive
            (90, 160, 8, 16, 1, LABEL_DESTROY),  # monkey on path
            (0, 0, 1, 1, 0, LABEL_DESTROY),
            (0, 0, 1, 1, 0, LABEL_DESTROY),
            (0, 0, 1, 1, 0, LABEL_DESTROY),
        ]
    )
    raw, objs = _make_obs((40, 160), slots)
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == 47  # first monkey slot
    assert int(target) != _L0_BOT


def test_destroy_ignores_monkey_on_lower_floor():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
    )

    # Player on mid platform (feet ~124); monkey on ground (y=160) must be ignored.
    slots = [(0, 0, 1, 1, 0, 0)]
    for i in range(20):
        if i == 0:
            slots.append((136, 112, 2, 2, 1, LABEL_NAVIGATE))
        else:
            slots.append((0, 0, 1, 1, 0, LABEL_NAVIGATE))
    for _ in range(20):
        slots.append((0, 0, 1, 1, 0, LABEL_NAVIGATE))
    slots.extend(
        [
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_DESTROY),
            (90, 160, 8, 16, 1, LABEL_DESTROY),  # lower-floor monkey
            (0, 0, 1, 1, 0, LABEL_DESTROY),
            (0, 0, 1, 1, 0, LABEL_DESTROY),
            (0, 0, 1, 1, 0, LABEL_DESTROY),
        ]
    )
    # Player centre ~112 with h=4 → feet 114; without platforms, floor=feet.
    # Monkey oy=160+8=168 >> 114+20 → below → ignored; plan keeps L0 bot.
    # Use taller player so feet match a mid-floor reading: y=100,h=24 → feet 124.
    raw, objs = _make_obs((40, 100), slots)
    raw = raw.replace(
        player=raw.player.replace(
            y=_stack(jnp.int32(100)),
            height=_stack(jnp.int32(24)),
        )
    )
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _L0_BOT
    assert int(target) != 47


def test_destroy_ignores_far_falling_coco():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_TOP,
        FALLING_COCO_DODGE_RANGE,
    )

    # On L0 top stage; hanging coco far above must not interrupt.
    slots = [(0, 0, 1, 1, 0, 0)]
    for i in range(20):
        if i == 0:
            slots.append((136, 160, 2, 2, 1, LABEL_NAVIGATE))  # L0 bot
        else:
            slots.append((0, 0, 1, 1, 0, LABEL_NAVIGATE))
    for i in range(20):
        if i == 0:
            slots.append((136, 104, 2, 2, 1, LABEL_NAVIGATE))  # L0 top
        else:
            slots.append((0, 0, 1, 1, 0, LABEL_NAVIGATE))
    slots.extend(
        [
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (0, 0, 1, 1, 0, LABEL_COLLECT),
            (120, 20, 6, 6, 1, LABEL_DESTROY),  # falling coco, far above
            (0, 0, 1, 1, 0, LABEL_DESTROY),
            (0, 0, 1, 1, 0, LABEL_DESTROY),
            (0, 0, 1, 1, 0, LABEL_DESTROY),
            (0, 0, 1, 1, 0, LABEL_DESTROY),
        ]
    )
    raw, objs = _make_obs((136, 130), slots)
    raw = raw.replace(
        player=raw.player.replace(
            y=_stack(jnp.int32(130)),
            height=_stack(jnp.int32(24)),
        )
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(1))  # L0 top
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _L0_TOP
    assert int(target) != 46
    assert FALLING_COCO_DODGE_RANGE == 30.0


def test_kangaroo_plan_is_waypoints_only():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_L1_STAGES,
        _L0_BOT,
        _L0_TOP,
        _L2_TOP,
    )

    slots = [s.slot for s in KANGAROO_L1_STAGES]
    assert slots[0] == _L0_BOT
    assert slots[1] == _L0_TOP
    assert slots[-1] == _L2_TOP
    assert 41 not in slots  # no fruits
    assert 44 not in slots  # no bell
    assert 45 not in slots  # no child


def test_falling_coco_covers_ladder_helper():
    from concept_based_learning.environments.kangaroo.kangaroo_label import (
        _falling_coco_covers_ladders,
    )

    @struct.dataclass
    class _Ent:
        x: jnp.ndarray
        y: jnp.ndarray
        width: jnp.ndarray
        height: jnp.ndarray
        active: jnp.ndarray

    ladders = _Ent(
        x=jnp.array([132, 20], dtype=jnp.float32),
        y=jnp.array([132, 84], dtype=jnp.float32),
        width=jnp.array([8, 8], dtype=jnp.float32),
        height=jnp.array([36, 36], dtype=jnp.float32),
        active=jnp.array([1, 1], dtype=jnp.int32),
    )
    # Coco over ladder 0 column, above its foot.
    coco = _Ent(
        x=jnp.float32(134),
        y=jnp.float32(100),
        width=jnp.float32(6),
        height=jnp.float32(6),
        active=jnp.int32(1),
    )
    cover = _falling_coco_covers_ladders(coco, ladders)
    assert bool(cover[0])
    assert not bool(cover[1])


def test_force_when_beats_interrupt():
    def always(_raw, _objs, _ctx):
        return jnp.bool_(True)

    # Register under a unique name for this test process.
    name = "test.always_force"
    if name not in __import__(
        "concept_based_learning.environments.manual_policy.ir", fromlist=["PREDICATE_REGISTRY"]
    ).PREDICATE_REGISTRY:
        register_predicate(name, always)

    policy = compile_policy(
        ManualPolicySpec(
            plan=DynamicPick(prefer=(LABEL_COLLECT,), sticky=False),
            force=(ForceWhen(predicate=name, target_slot=3),),
            force_beats_interrupt=True,
            interrupts=(Interrupt(labels=(LABEL_AVOID,), within=100.0),),
            no_target_idx=NO_TARGET_IDX,
        )
    )
    raw, objs = _make_obs(
        (50, 50),
        [
            (0, 0, 1, 1, 0, 0),
            (55, 50, 4, 4, 1, LABEL_AVOID),
            (80, 50, 4, 4, 1, LABEL_COLLECT),
            (90, 90, 2, 2, 1, LABEL_NAVIGATE),
        ],
    )
    target, _ = policy.step(raw, objs, policy.initial_state(), None)
    assert int(target) == 3


def test_climb_ladders_pairs_bottom_to_top():
    policy = compile_policy(
        ManualPolicySpec(
            plan=ClimbLadders(
                bottom_start=1,
                top_start=3,
                n_ladders=2,
                arrive_dist=5.0,
                between_prefer=(),
            ),
            no_target_idx=NO_TARGET_IDX,
        )
    )
    # Bottoms at 1,2 ; tops at 3,4
    raw, objs = _make_obs(
        (10, 40),
        [
            (0, 0, 1, 1, 0, 0),
            (10, 40, 2, 2, 1, LABEL_NAVIGATE),  # bottom 0 — at player
            (80, 40, 2, 2, 1, LABEL_NAVIGATE),  # bottom 1
            (10, 10, 2, 2, 1, LABEL_NAVIGATE),  # top 0
            (80, 10, 2, 2, 1, LABEL_NAVIGATE),  # top 1
        ],
    )
    state = policy.initial_state()
    # First step at bottom → should lock paired top (slot 3)
    target, state = policy.step(raw, objs, state, None)
    assert int(target) == 3
    assert int(state.aux0) == 3


def test_avoid_interrupt_taller_vertically():
    """Axis-weighted AVOID: far overhead threat triggers; far side does not."""
    policy = compile_policy(
        ManualPolicySpec(
            plan=DynamicPick(prefer=(LABEL_COLLECT,), sticky=False),
            interrupts=(
                Interrupt(labels=(LABEL_AVOID,), within_x=20.0, within_y=45.0),
            ),
            no_target_idx=NO_TARGET_IDX,
        )
    )
    # Overhead coconut: dx=0, dy=40 → inside ellipse; side coconut dx=40, dy=0 → outside.
    raw, objs = _make_obs(
        (50, 100),
        [
            (0, 0, 1, 1, 0, 0),
            (80, 100, 4, 4, 1, LABEL_COLLECT),
            (50, 60, 4, 4, 1, LABEL_AVOID),  # above, dy≈40
        ],
    )
    target, _ = policy.step(raw, objs, policy.initial_state(), None)
    assert int(target) == 2

    raw2, objs2 = _make_obs(
        (50, 100),
        [
            (0, 0, 1, 1, 0, 0),
            (80, 100, 4, 4, 1, LABEL_COLLECT),
            (90, 100, 4, 4, 1, LABEL_AVOID),  # side, dx≈40
        ],
    )
    target2, _ = policy.step(raw2, objs2, policy.initial_state(), None)
    assert int(target2) == 1  # stay on collect; side avoid too far


def test_generic_policy_exports_legacy_fn():
    assert callable(get_heuristic_target)
    assert GENERIC_POLICY.needs_state

