"""Unit tests for manual_policy IR runtime and label→skill mapping."""

import jax
import jax.numpy as jnp
from typing import Any
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
    state: jnp.ndarray


@struct.dataclass
class _Raw:
    player: _Player


@struct.dataclass
class _RawKangaroo:
    player: _Player
    ladders: Any


@struct.dataclass
class _Objects:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray
    semantic_label: jnp.ndarray
    state: jnp.ndarray


def _stack(val, t=4):
    return jnp.stack([val] * t)


def _make_obs(player_xy, objects, *, object_states=None):
    """objects: list of (x, y, w, h, active, label).

    ``object_states`` optional per-slot int state (e.g. falling-coco dropping).
    """
    px, py = player_xy
    raw = _Raw(
        player=_Player(
            x=_stack(jnp.int32(px)),
            y=_stack(jnp.int32(py)),
            width=_stack(jnp.int32(4)),
            height=_stack(jnp.int32(4)),
            state=_stack(jnp.int32(0)),
        )
    )
    n = len(objects)
    xs = jnp.array([o[0] for o in objects], dtype=jnp.int32)
    ys = jnp.array([o[1] for o in objects], dtype=jnp.int32)
    ws = jnp.array([o[2] for o in objects], dtype=jnp.int32)
    hs = jnp.array([o[3] for o in objects], dtype=jnp.int32)
    act = jnp.array([o[4] for o in objects], dtype=jnp.int32)
    labs = jnp.array([o[5] for o in objects], dtype=jnp.int32)
    if object_states is None:
        st = jnp.zeros((n,), dtype=jnp.int32)
    else:
        st = jnp.asarray(object_states, dtype=jnp.int32)
    objs = _Objects(
        x=_stack(xs),
        y=_stack(ys),
        width=_stack(ws),
        height=_stack(hs),
        active=_stack(act),
        semantic_label=_stack(labs),
        state=_stack(st),
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
    )

    _FRUIT0 = 40  # first collectible slot after 40 ladder waypoints
    tail = [
        (119, 108, 4, 4, 1, LABEL_COLLECT),
        (39, 84, 4, 4, 1, LABEL_COLLECT),
        (59, 60, 4, 4, 1, LABEL_COLLECT),
        (93, 36, 4, 4, 1, LABEL_COLLECT),
        (121, 13, 4, 4, 1, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(l0_bot=(136, 168, 2, 2), collect_tail=tail)
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


def _kangaroo_policy_slots(
    *,
    l0_bot=(136, 160, 2, 2),
    l0_top=None,
    l1_bot=None,
    l1_top=None,
    l2_bot=None,
    l2_top=None,
    collect_tail=None,
):
    """Minimal 55-slot Kangaroo layout: 40 WPs + 5 collect + 4 destroy + 5 avoid + meta-NOOP.

    Meta-NOOP is trailing (index 54), matching ``append_meta_reserved_slots``.
    """
    bottom_overrides = {0: l0_bot, 1: l1_bot, 2: l2_bot}
    top_overrides = {0: l0_top, 1: l1_top, 2: l2_top}
    slots = []
    for i in range(20):
        if i in bottom_overrides and bottom_overrides[i] is not None:
            slots.append((*bottom_overrides[i], 1, LABEL_COLLECT))
        elif i == 0:
            slots.append((*l0_bot, 1, LABEL_COLLECT))
        else:
            slots.append((0, 0, 1, 1, 0, LABEL_COLLECT))
    for i in range(20):
        if i in top_overrides and top_overrides[i] is not None:
            slots.append((*top_overrides[i], 1, LABEL_COLLECT))
        elif l0_top is not None and i == 0:
            slots.append((*l0_top, 1, LABEL_COLLECT))
        else:
            slots.append((0, 0, 1, 1, 0, LABEL_COLLECT))
    tail = collect_tail or [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots.extend(tail)
    slots.append((0, 0, 1, 1, 0, 0))  # trailing meta-NOOP
    return slots


def test_destroy_path_blocking_monkey_interrupt():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _MONKEY_SLOT0,
    )

    # Plan wants L0 bottom at x=136; monkey sits between player (40) and ladder.
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (90, 160, 8, 16, 1, LABEL_DESTROY),  # monkey on path (slot 45)
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    raw, objs = _make_obs((40, 160), slots)
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _MONKEY_SLOT0
    assert int(target) != _L0_BOT


def test_destroy_targets_nearby_monkey_on_same_floor():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L1_TOP,
        _MONKEY_SLOT0,
    )

    # On L1 top platform; monkey adjacent (not between player and waypoint).
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (35, 56, 8, 16, 1, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        l1_top=(20, 56, 2, 2),
        collect_tail=tail,
    )
    raw, objs = _make_obs((30, 56), slots)
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _MONKEY_SLOT0
    assert int(target) != _L1_TOP


def test_destroy_emergency_close_monkey_ignores_floor_heuristic():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L1_TOP,
        _MONKEY_SLOT0,
    )

    # Monkey slightly off "same floor" slack but within emergency range.
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (34, 48, 8, 16, 1, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        l1_top=(20, 44, 2, 2),
        collect_tail=tail,
    )
    raw, objs = _make_obs((30, 56), slots)
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _MONKEY_SLOT0
    assert int(target) != _L1_TOP


def test_destroy_suppressed_while_on_ladder():
    """Off-shaft monkeys stay ignored while climbing; only shaft blockers interrupt."""
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L1_TOP,
        _MONKEY_SLOT0,
    )

    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (34, 48, 8, 16, 1, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        l1_top=(20, 44, 2, 2),
        collect_tail=tail,
    )
    raw, objs = _make_obs((18, 84), slots)
    raw = raw.replace(
        player=raw.player.replace(
            x=_stack(jnp.int32(18)),
            y=_stack(jnp.int32(84)),
            width=_stack(jnp.int32(4)),
            height=_stack(jnp.int32(24)),
        )
    )

    @struct.dataclass
    class _Ladders:
        x: jnp.ndarray
        y: jnp.ndarray
        width: jnp.ndarray
        height: jnp.ndarray
        active: jnp.ndarray

    raw = _RawKangaroo(
        player=raw.player,
        ladders=_Ladders(
            x=jnp.array([132.0, 20.0] + [0.0] * 18, dtype=jnp.float32),
            y=jnp.array([132.0, 84.0] + [0.0] * 18, dtype=jnp.float32),
            width=jnp.full(20, 8.0, dtype=jnp.float32),
            height=jnp.full(20, 36.0, dtype=jnp.float32),
            active=jnp.array([1, 1] + [0] * 18, dtype=jnp.int32),
        ),
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _L1_TOP
    assert int(target) != _MONKEY_SLOT0


def test_destroy_shaft_blocking_monkey_while_climbing():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L1_TOP,
        _MONKEY_SLOT0,
    )

    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (22, 80, 8, 16, 1, LABEL_DESTROY),  # between player and L1 top waypoint
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        l1_top=(20, 56, 2, 2),
        collect_tail=tail,
    )
    raw, objs = _make_obs((18, 100), slots)
    raw = raw.replace(
        player=raw.player.replace(
            x=_stack(jnp.int32(18)),
            y=_stack(jnp.int32(100)),
            width=_stack(jnp.int32(4)),
            height=_stack(jnp.int32(24)),
        )
    )

    @struct.dataclass
    class _Ladders:
        x: jnp.ndarray
        y: jnp.ndarray
        width: jnp.ndarray
        height: jnp.ndarray
        active: jnp.ndarray

    raw = _RawKangaroo(
        player=raw.player,
        ladders=_Ladders(
            x=jnp.array([0.0, 20.0] + [0.0] * 18, dtype=jnp.float32),
            y=jnp.array([0.0, 84.0] + [0.0] * 18, dtype=jnp.float32),
            width=jnp.full(20, 8.0, dtype=jnp.float32),
            height=jnp.full(20, 36.0, dtype=jnp.float32),
            active=jnp.array([0, 1] + [0] * 18, dtype=jnp.int32),
        ),
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _MONKEY_SLOT0
    assert int(target) != _L1_TOP


def test_l1_top_does_not_complete_mid_climb():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L1_TOP,
        _L2_BOT,
    )

    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        l1_top=(20, 44, 2, 2),
        l2_bot=(136, 56, 2, 2),
    )
    raw, objs = _make_obs((18, 84), slots)
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, state2 = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(state2.plan_idx) == 6
    assert int(target) == _L1_TOP
    assert int(target) != _L2_BOT


def test_l1_top_stage_completes_on_platform():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _FRUIT1,
        _L1_TOP,
    )

    # Active fruit1 so the stage after L1_TOP has a real COLLECT target.
    # Real L1 fruit1 is at (39, 84); use full fruit height so the above-skip
    # margin matches in-game geometry (same-floor collect from P3).
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (39, 84, 8, 12, 1, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        l1_top=(20, 44, 2, 2),
        l2_bot=(136, 56, 2, 2),
        collect_tail=tail,
    )
    # Standing on mid-high (P3 y=76): top-left y = 52, feet = 76.
    raw, objs = _make_obs((18, 52), slots)
    raw = raw.replace(
        player=raw.player.replace(
            y=_stack(jnp.int32(52)),
            height=_stack(jnp.int32(24)),
            width=_stack(jnp.int32(8)),
        )
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, state2 = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(state2.plan_idx) == 7
    assert int(target) == _FRUIT1
    assert int(target) != _L1_TOP


def test_destroy_ignores_monkey_on_lower_floor():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _MONKEY_SLOT0,
    )

    # Player on mid platform (feet ~124); monkey on ground (y=160) must be ignored.
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (90, 160, 8, 16, 1, LABEL_DESTROY),  # lower-floor monkey
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l0_bot=(136, 112, 2, 2),
        collect_tail=tail,
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
    assert int(target) != _MONKEY_SLOT0


def test_destroy_ignores_far_falling_coco():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_TOP,
        _FALLING_COCO_SLOT,
        FALLING_COCO_DODGE_RANGE,
    )

    # On L0 top stage; hanging coco far above must not interrupt.
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (120, 20, 6, 6, 1, LABEL_AVOID),  # falling coco, far above
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l0_top=(136, 104, 2, 2),
        collect_tail=tail,
    )
    raw, objs = _make_obs((136, 130), slots)
    raw = raw.replace(
        player=raw.player.replace(
            y=_stack(jnp.int32(130)),
            height=_stack(jnp.int32(24)),
        )
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(2))  # L0 top
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _L0_TOP
    assert int(target) != _FALLING_COCO_SLOT
    assert FALLING_COCO_DODGE_RANGE == 95.0
    assert FALLING_COCO_DODGE_RANGE > 50.0  # earlier than thrown symmetric range


def test_falling_coco_ignored_when_not_x_aligned():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _FALLING_COCO_SLOT,
    )

    # Overhead coco far to the side — outside the player's run-into x band.
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (20, 80, 6, 6, 1, LABEL_AVOID),  # falling coco above but x-far from player@40
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    raw, objs = _make_obs((40, 100), slots)
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _L0_BOT
    assert int(target) != _FALLING_COCO_SLOT


def test_falling_coco_ignored_when_hovering_not_dropping():
    """Bobbing / x-tracking coco must not interrupt until it actually drops."""
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _FALLING_COCO_SLOT,
    )

    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (38, 72, 6, 6, 1, LABEL_AVOID),  # x-aligned and close, but not dropping
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    raw, objs = _make_obs((40, 100), slots)  # state defaults to 0 = not dropping
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _L0_BOT
    assert int(target) != _FALLING_COCO_SLOT


def test_falling_coco_selected_when_x_aligned_and_close():
    from concept_based_learning.environments.kangaroo.kangaroo_label import (
        FALLING_COCO_STATE_DROPPING,
    )
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _FALLING_COCO_SLOT,
    )

    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (38, 72, 6, 6, 1, LABEL_AVOID),  # directly above player@40, within 30 px
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    states = [0] * len(slots)
    states[_FALLING_COCO_SLOT] = FALLING_COCO_STATE_DROPPING
    raw, objs = _make_obs((40, 100), slots, object_states=states)
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _FALLING_COCO_SLOT
    assert int(target) != _L0_BOT


def test_thrown_coco_ignored_when_off_run_path():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _THROWN_COCO_SLOT0,
    )

    # Plan runs right toward L0 bot; thrown coco left of the player is not in corridor.
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (5, 158, 6, 6, 1, LABEL_AVOID),  # thrown coco left of widened corridor
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    raw, objs = _make_obs((40, 160), slots)
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _L0_BOT
    assert int(target) != _THROWN_COCO_SLOT0


def test_thrown_coco_selected_when_in_corridor_and_close():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _THROWN_COCO_SLOT0,
    )

    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (55, 158, 6, 6, 1, LABEL_AVOID),  # between player@40 and ladder@136
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    raw, objs = _make_obs((40, 160), slots)
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _THROWN_COCO_SLOT0
    assert int(target) != _L0_BOT


def test_thrown_coco_selected_when_ground_flush():
    """Low throws sit on the platform (top ≈ feet); must still be AVOID targets."""
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L0_BOT,
        _THROWN_COCO_SLOT0,
    )

    # Player standing on platform y=172 (top-left y=148, h=24). Real thrown coco
    # is 2×3; low spawn puts top at monkey_feet - 3 = 169 (flush with the floor).
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (55, 169, 2, 3, 1, LABEL_AVOID),  # ground-flush throw in corridor
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    raw, objs = _make_obs((40, 148), slots)
    raw = raw.replace(
        player=raw.player.replace(
            y=_stack(jnp.int32(148)),
            height=_stack(jnp.int32(24)),
            width=_stack(jnp.int32(8)),
        )
    )
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _THROWN_COCO_SLOT0
    assert int(target) != _L0_BOT


def test_ground_thrown_coco_beats_nearby_monkey():
    """Low throws share the monkey's floor — AVOID must win over DESTROY."""
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _MONKEY_SLOT0,
        _THROWN_COCO_SLOT0,
    )

    # Monkey at (80,157) on ground floor; low coco at (55,169) in the run path.
    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (80, 157, 8, 15, 1, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (55, 169, 2, 3, 1, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    raw, objs = _make_obs((40, 148), slots)
    raw = raw.replace(
        player=raw.player.replace(
            y=_stack(jnp.int32(148)),
            height=_stack(jnp.int32(24)),
            width=_stack(jnp.int32(8)),
        )
    )
    target, _ = KANGAROO_POLICY.step(
        raw, objs, KANGAROO_POLICY.initial_state(), None
    )
    assert int(target) == _THROWN_COCO_SLOT0
    assert int(target) != _MONKEY_SLOT0


def test_thrown_coco_avoid_suppressed_while_on_ladder():
    """Thrown-coco AVOID must not fire while climbing (stamped climb state)."""
    from concept_based_learning.environments.kangaroo.kangaroo_label import (
        PLAYER_STATE_CLIMBING,
    )
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _L1_TOP,
        _THROWN_COCO_SLOT0,
    )

    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (18, 84, 6, 6, 1, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        l1_top=(20, 44, 2, 2),
        collect_tail=tail,
    )
    raw, objs = _make_obs((18, 84), slots)
    raw = raw.replace(
        player=raw.player.replace(
            x=_stack(jnp.int32(18)),
            y=_stack(jnp.int32(84)),
            width=_stack(jnp.int32(4)),
            height=_stack(jnp.int32(24)),
            state=_stack(jnp.int32(PLAYER_STATE_CLIMBING)),
        )
    )

    @struct.dataclass
    class _Ladders:
        x: jnp.ndarray
        y: jnp.ndarray
        width: jnp.ndarray
        height: jnp.ndarray
        active: jnp.ndarray

    raw = _RawKangaroo(
        player=raw.player,
        ladders=_Ladders(
            x=jnp.array([132.0, 20.0] + [0.0] * 18, dtype=jnp.float32),
            y=jnp.array([132.0, 84.0] + [0.0] * 18, dtype=jnp.float32),
            width=jnp.full(20, 8.0, dtype=jnp.float32),
            height=jnp.full(20, 36.0, dtype=jnp.float32),
            active=jnp.array([1, 1] + [0] * 18, dtype=jnp.int32),
        ),
    )

    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _L1_TOP
    assert int(target) != _THROWN_COCO_SLOT0


def test_falling_coco_suppressed_in_upper_ladder_shaft():
    """Shaft extends to top waypoint — sprite top alone misses upper climb."""
    from concept_based_learning.environments.kangaroo.kangaroo_label import (
        PLAYER_STATE_CLIMBING,
    )
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _FALLING_COCO_SLOT,
        _L1_TOP,
    )

    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (140, 110, 6, 6, 1, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(140, 160, 2, 2),
        l1_top=(140, 95, 2, 2),
        collect_tail=tail,
    )
    raw, objs = _make_obs((136, 110), slots)
    raw = raw.replace(
        player=raw.player.replace(
            x=_stack(jnp.int32(136)),
            y=_stack(jnp.int32(110)),
            width=_stack(jnp.int32(4)),
            height=_stack(jnp.int32(24)),
            state=_stack(jnp.int32(PLAYER_STATE_CLIMBING)),
        )
    )

    @struct.dataclass
    class _Ladders:
        x: jnp.ndarray
        y: jnp.ndarray
        width: jnp.ndarray
        height: jnp.ndarray
        active: jnp.ndarray

    # Right-side ladder (plan L1 / slot 2): sprite ends at y=132, top wp at y=95.
    raw = _RawKangaroo(
        player=raw.player,
        ladders=_Ladders(
            x=jnp.array([0.0, 132.0] + [0.0] * 18, dtype=jnp.float32),
            y=jnp.array([0.0, 132.0] + [0.0] * 18, dtype=jnp.float32),
            width=jnp.full(20, 8.0, dtype=jnp.float32),
            height=jnp.full(20, 36.0, dtype=jnp.float32),
            active=jnp.array([0, 1] + [0] * 18, dtype=jnp.int32),
        ),
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _L1_TOP
    assert int(target) != _FALLING_COCO_SLOT


def test_falling_coco_avoid_suppressed_while_on_ladder():
    from concept_based_learning.environments.kangaroo.kangaroo_label import (
        PLAYER_STATE_CLIMBING,
    )
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _FALLING_COCO_SLOT,
        _L1_TOP,
    )

    tail = [
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (18, 84, 6, 6, 1, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        l1_top=(20, 56, 2, 2),
        collect_tail=tail,
    )
    raw, objs = _make_obs((18, 84), slots)
    raw = raw.replace(
        player=raw.player.replace(
            x=_stack(jnp.int32(18)),
            y=_stack(jnp.int32(84)),
            width=_stack(jnp.int32(4)),
            height=_stack(jnp.int32(24)),
            state=_stack(jnp.int32(PLAYER_STATE_CLIMBING)),
        )
    )

    @struct.dataclass
    class _Ladders:
        x: jnp.ndarray
        y: jnp.ndarray
        width: jnp.ndarray
        height: jnp.ndarray
        active: jnp.ndarray

    # Ladder 1 geometry aligned with player mid-climb.
    raw = _RawKangaroo(
        player=raw.player,
        ladders=_Ladders(
            x=jnp.array([132.0, 20.0] + [0.0] * 18, dtype=jnp.float32),
            y=jnp.array([132.0, 84.0] + [0.0] * 18, dtype=jnp.float32),
            width=jnp.full(20, 8.0, dtype=jnp.float32),
            height=jnp.full(20, 36.0, dtype=jnp.float32),
            active=jnp.array([1, 1] + [0] * 18, dtype=jnp.int32),
        ),
    )

    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(6))
    target, _ = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _L1_TOP
    assert int(target) != _FALLING_COCO_SLOT


def test_kangaroo_plan_includes_fruits_between_climbs():
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_L1_STAGES,
        _FRUIT0,
        _FRUIT1,
        _FRUIT2,
        _L0_BOT,
        _L0_MID,
        _L0_TOP,
        _L2_TOP,
    )

    slots = [s.slot for s in KANGAROO_L1_STAGES]
    assert slots[0] == _L0_BOT
    assert slots[1] == _L0_MID
    assert slots[2] == _L0_TOP
    assert slots[3] == _FRUIT0
    assert slots[7] == _FRUIT1
    assert slots[-2] == _L2_TOP
    assert slots[-1] == _FRUIT2
    assert 43 not in slots  # no bell
    assert 44 not in slots  # no child


def test_fruit0_skipped_when_player_on_floor_above():
    """L1 fruit0 hangs under mid-high; once on P3 do not path back down to it."""
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _FRUIT0,
        _L1_BOT,
    )

    # Fruit0 at L1 pos (119, 108), h=12 → centre ≈ 114. P3 feet at 76 is above.
    tail = [
        (119, 108, 8, 12, 1, LABEL_COLLECT),  # fruit0
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(
        l1_bot=(20, 112, 2, 2),
        collect_tail=tail,
    )
    # Standing on mid-high platform (P3 y=76): top-left y = 76 - 24 = 52.
    raw, objs = _make_obs((40, 52), slots)
    raw = raw.replace(
        player=raw.player.replace(
            y=_stack(jnp.int32(52)),
            height=_stack(jnp.int32(24)),
            width=_stack(jnp.int32(8)),
        )
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(3))  # fruit0
    target, state2 = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) != _FRUIT0
    assert int(state2.plan_idx) >= 4  # advanced past fruit0 (L1 bot stage)
    assert int(target) == _L1_BOT


def test_fruit0_selected_when_player_on_floor_below():
    """From mid-low (P2) the hanging fruit0 remains a valid COLLECT target."""
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
        _FRUIT0,
    )

    tail = [
        (119, 108, 8, 12, 1, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_COLLECT),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_DESTROY),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
        (0, 0, 1, 1, 0, LABEL_AVOID),
    ]
    slots = _kangaroo_policy_slots(collect_tail=tail)
    # P2 y=124 → standing top-left y = 100.
    raw, objs = _make_obs((100, 100), slots)
    raw = raw.replace(
        player=raw.player.replace(
            y=_stack(jnp.int32(100)),
            height=_stack(jnp.int32(24)),
            width=_stack(jnp.int32(8)),
        )
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(3))  # fruit0
    target, state2 = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(target) == _FRUIT0
    assert int(state2.plan_idx) == 3


def test_kangaroo_crash_freezes_plan_at_stage_zero():
    """Falling corpse must not advance ladder stages before respawn."""
    from concept_based_learning.environments.kangaroo.kangaroo_label import (
        PLAYER_STATE_CRASHING,
    )
    from concept_based_learning.environments.kangaroo.kangaroo_policy import (
        KANGAROO_POLICY,
    )

    # Player overlaps L0 bottom while "crashing" — without the freeze, arrive
    # would advance plan_idx past stage 0.
    slots = _kangaroo_policy_slots(l0_bot=(136, 160, 2, 2))
    raw, objs = _make_obs((136, 160), slots)
    raw = raw.replace(
        player=raw.player.replace(
            state=_stack(jnp.int32(PLAYER_STATE_CRASHING)),
        )
    )
    state = KANGAROO_POLICY.initial_state().replace(plan_idx=jnp.int32(2))
    target, new_state = KANGAROO_POLICY.step(raw, objs, state, None)
    assert int(new_state.plan_idx) == 0
    assert int(target) == int(NO_TARGET_IDX)


def test_falling_coco_covers_ladder_helper():
    from concept_based_learning.environments.kangaroo.kangaroo_label import (
        _falling_coco_covers_ladders,
        _player_on_ladders,
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
    # Coco in ladder 0 column, above the foot — proactive gate (even if still high).
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

    # Coco has fallen past the ladder foot — column clear.
    passed = _Ent(
        x=jnp.float32(134),
        y=jnp.float32(180),
        width=jnp.float32(6),
        height=jnp.float32(6),
        active=jnp.int32(1),
    )
    assert not bool(_falling_coco_covers_ladders(passed, ladders)[0])

    # Player mid-shaft on ladder 0 — coco still "covers", but climb bypass applies.
    player = _Ent(
        x=jnp.float32(130),
        y=jnp.float32(140),
        width=jnp.float32(8),
        height=jnp.float32(24),
        active=jnp.int32(1),
    )
    on = _player_on_ladders(player, ladders)
    assert bool(on[0])
    assert not bool(on[1])
    blocks = jnp.logical_and(cover, jnp.logical_not(on))
    assert not bool(blocks[0])


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

