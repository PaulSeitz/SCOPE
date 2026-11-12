"""Unit tests for Montezuma's Revenge ego patch + L1 waypoint heuristic."""

from __future__ import annotations

import pytest

import jax
import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
)
from concept_based_learning.environments.manual_policy.state import PolicyState
from concept_based_learning.environments.montezumarevenge.montezuma_label import (
    MONTEZUMA_RENDER_Y_OFFSET,
)
from concept_based_learning.environments.montezumarevenge.montezuma_policy import (
    CLOSE_AVOID_RANGE,
    NO_TARGET_IDX,
    WP0,
    WP1,
    WP2,
    WP3,
    WP4,
    WP6,
    _EDGE_COLLECT_SLOT,
    _KEY_SLOT,
    _L0_BOT,
    _L0_TOP,
    _L1_BOT,
    _L1_TOP,
    _L2_BOT,
    _L2_TOP,
    _LEFT_DOOR_SLOT,
    _ROPE_GRAB_SLOT,
    _SHAFT_BEAD_SLOT,
    room4_path_stamps,
    get_heuristic_target,
    get_heuristic_target_with_state,
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


def _make_obs(player_xy, objects, n_slots=32):
    """Build stacked raw/objects. ``objects`` entries: (x,y,w,h,active,label)."""
    px, py = player_xy
    raw = _Raw(
        player=_Player(
            x=_stack(jnp.int32(px)),
            y=_stack(jnp.int32(py)),
            width=_stack(jnp.int32(7)),
            height=_stack(jnp.int32(20)),
        )
    )
    xs = jnp.zeros(n_slots, dtype=jnp.int32)
    ys = jnp.zeros(n_slots, dtype=jnp.int32)
    ws = jnp.ones(n_slots, dtype=jnp.int32)
    hs = jnp.ones(n_slots, dtype=jnp.int32)
    act = jnp.zeros(n_slots, dtype=jnp.int32)
    labs = jnp.zeros(n_slots, dtype=jnp.int32)
    for i, o in enumerate(objects):
        if i >= n_slots:
            break
        xs = xs.at[i].set(jnp.int32(o[0]))
        ys = ys.at[i].set(jnp.int32(o[1]))
        ws = ws.at[i].set(jnp.int32(o[2]))
        hs = hs.at[i].set(jnp.int32(o[3]))
        act = act.at[i].set(jnp.int32(o[4]))
        labs = labs.at[i].set(jnp.int32(o[5]))
    objs = _Objects(
        x=_stack(xs),
        y=_stack(ys),
        width=_stack(ws),
        height=_stack(hs),
        active=_stack(act),
        semantic_label=_stack(labs),
    )
    return raw, objs


def _room4_objects(*, skull_xy=None, skull_active=0):
    """Room-4 slots stamped from policy ``WP*`` centres (obs space)."""
    objs = [(0, 0, 1, 1, 0, 0)] * 32
    objs = list(objs)
    for slot, (x, y, w, h, lab) in room4_path_stamps().items():
        objs[slot] = (x, y, w, h, 1, lab)
    objs[_ROPE_GRAB_SLOT] = (107, 98, 8, 8, 1, LABEL_COLLECT)
    objs[_KEY_SLOT] = (13, 99, 6, 8, 1, LABEL_COLLECT)
    objs[_LEFT_DOOR_SLOT] = (16, 54, 4, 38, 1, LABEL_COLLECT)
    if skull_xy is not None:
        objs[0] = (skull_xy[0], skull_xy[1], 6, 14, skull_active, LABEL_AVOID)
    return objs


def _near(xy, dx=0.0, dy=0.0):
    """Player top-left so its centre sits on ``xy`` (+optional nudge)."""
    return (xy[0] - 3.5 + dx, xy[1] - 10.0 + dy)


def test_montezuma_avoids_close_enemy():
    raw, objs = _make_obs((80, 160), _room4_objects(skull_xy=(90, 167), skull_active=1))
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 0
    assert CLOSE_AVOID_RANGE > 0


def test_montezuma_plan_starts_at_shaft_bead():
    raw, objs = _make_obs((77, 73), _room4_objects())
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == _SHAFT_BEAD_SLOT
    assert int(target) != _KEY_SLOT
    assert int(target) != _L0_BOT


def test_montezuma_stateful_shaft_then_edge():
    raw, objs = _make_obs((77, 73), _room4_objects())
    state = PolicyState.zero(no_target_idx=NO_TARGET_IDX)
    target, state = get_heuristic_target_with_state(
        raw, objs, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(0)
    )
    assert int(target) == _SHAFT_BEAD_SLOT
    assert int(state.plan_idx) == 0

    # Overlap WP0 → advance to edge.
    raw_shaft, objs_shaft = _make_obs(_near(WP0), _room4_objects())
    target_shaft, state_shaft = get_heuristic_target_with_state(
        raw_shaft, objs_shaft, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(2)
    )
    assert int(state_shaft.plan_idx) >= 1
    assert int(target_shaft) == _EDGE_COLLECT_SLOT

    # Still in the ladder shaft (well left of WP1 box) — must NOT complete EDGE.
    raw_shaft_foot, objs_shaft_foot = _make_obs((70.0, WP1[1] - 10.0), _room4_objects())
    target_mid, state_mid = get_heuristic_target_with_state(
        raw_shaft_foot,
        objs_shaft_foot,
        jnp.int32(NO_TARGET_IDX),
        state_shaft,
        jax.random.PRNGKey(3),
    )
    assert int(state_mid.plan_idx) == int(state_shaft.plan_idx)
    assert int(target_mid) == _EDGE_COLLECT_SLOT

    # Overlap WP1 → rope COLLECT.
    raw_edge, objs_edge = _make_obs(_near(WP1), _room4_objects())
    target_edge, state_edge = get_heuristic_target_with_state(
        raw_edge, objs_edge, jnp.int32(NO_TARGET_IDX), state_shaft, jax.random.PRNGKey(1)
    )
    assert int(state_edge.plan_idx) >= 2
    assert int(target_edge) == _ROPE_GRAB_SLOT


def test_montezuma_l1_top_advances_on_overlap():
    """Overlapping L1_TOP (COLLECT contact) must advance — not centre-radius only."""
    # Player box overlaps WP3 stamp but centre can be > old ARRIVE_R=6 away in y.
    raw, objs = _make_obs((WP3[0] - 3.5, WP3[1] - 18.0), _room4_objects())
    state = PolicyState.zero(no_target_idx=NO_TARGET_IDX).replace(plan_idx=jnp.int32(3))
    target, state2 = get_heuristic_target_with_state(
        raw, objs, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(0)
    )
    assert int(state2.plan_idx) >= 4
    assert int(target) == _L1_BOT


def test_montezuma_l1_top_does_not_complete_far_away():
    raw, objs = _make_obs((WP3[0] - 3.5, WP3[1] - 50.0), _room4_objects())
    state = PolicyState.zero(no_target_idx=NO_TARGET_IDX).replace(plan_idx=jnp.int32(3))
    target, state2 = get_heuristic_target_with_state(
        raw, objs, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(0)
    )
    assert int(target) == _L1_TOP
    assert int(state2.plan_idx) == 3


def test_montezuma_rope_collect_advances_to_l1_top():
    raw, objs = _make_obs(_near(WP1), _room4_objects())
    state = PolicyState.zero(no_target_idx=NO_TARGET_IDX).replace(plan_idx=jnp.int32(2))
    target, state2 = get_heuristic_target_with_state(
        raw, objs, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(0)
    )
    assert int(target) == _ROPE_GRAB_SLOT
    assert int(state2.plan_idx) == 2

    # Overlap rope WP2 → L1_TOP.
    raw_rope, objs_rope = _make_obs(_near(WP2), _room4_objects())
    target_rope, state_rope = get_heuristic_target_with_state(
        raw_rope, objs_rope, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(1)
    )
    assert int(state_rope.plan_idx) >= 3
    assert int(target_rope) == _L1_TOP


def test_montezuma_side_top_does_not_skip_to_key_mid_climb():
    from concept_based_learning.environments.montezumarevenge.montezuma_policy import (
        MONTEZUMA_L1_STAGES,
    )

    l2_top_idx = next(
        i for i, st in enumerate(MONTEZUMA_L1_STAGES) if st.slot == _L2_TOP
    )
    # Low in the left shaft — far from WP6 box.
    raw, objs = _make_obs((WP6[0] - 3.5, WP4[1] - 10.0), _room4_objects())
    state = PolicyState.zero(no_target_idx=NO_TARGET_IDX).replace(
        plan_idx=jnp.int32(l2_top_idx)
    )
    target, state2 = get_heuristic_target_with_state(
        raw, objs, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(0)
    )
    assert int(target) == _L2_TOP
    assert int(state2.plan_idx) == l2_top_idx
    assert int(target) != _KEY_SLOT

    raw_top, objs_top = _make_obs(_near(WP6), _room4_objects())
    target_top, state_top = get_heuristic_target_with_state(
        raw_top, objs_top, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(1)
    )
    assert int(state_top.plan_idx) >= l2_top_idx + 1
    assert int(target_top) == _KEY_SLOT


def test_montezuma_inactive_ladder_returns_no_target_or_shaft():
    raw, objs = _make_obs((80, 80), [(0, 0, 1, 1, 0, 0)])
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) in (NO_TARGET_IDX, _SHAFT_BEAD_SLOT)


@pytest.mark.xfail(reason="Montezuma is not part of the thesis evaluation; written against the earlier JAXtari fork", strict=False)
def test_montezuma_player_shape_and_orientation():
    from concept_based_learning.environments.env_manager import EnvManager

    mgr = EnvManager()
    env, _ = mgr.make_env(
        "montezumarevenge",
        {
            "FRAME_STACK_SIZE": 4,
            "FRAME_SKIP": 1,
            "FIRST_FIRE": False,
            "STICKY_ACTIONS": False,
        },
        train=False,
    )
    obs, state = env.reset(jax.random.PRNGKey(0))
    player = obs.raw_observation.player
    assert player.x.ndim == 1
    assert tuple(player.x.shape) == (4,)
    assert float(player.orientation[-1]) in (90.0, 270.0)

    left = jnp.int32(4)
    obs2, state2, *_ = env.step(state, left)
    assert float(obs2.raw_observation.player.orientation[-1]) == 270.0
    right = jnp.int32(3)
    obs3, _, *_ = env.step(state2, right)
    assert float(obs3.raw_observation.player.orientation[-1]) == 90.0

    act = obs.objects.active[-1]
    assert int(act[_L0_BOT]) == 1
    assert int(act[_L0_TOP]) == 0
    assert int(obs.objects.semantic_label[-1][_L0_TOP]) == LABEL_COLLECT
    assert int(act[_L1_BOT]) == 1 and int(act[_L2_BOT]) == 1
    assert int(act[_L1_TOP]) == 1 and int(act[_L2_TOP]) == 1
    assert int(act[_SHAFT_BEAD_SLOT]) == 1
    assert int(act[_EDGE_COLLECT_SLOT]) == 1
    assert int(act[_ROPE_GRAB_SLOT]) == 1
    assert int(obs.objects.semantic_label[-1][_ROPE_GRAB_SLOT]) == LABEL_COLLECT
    assert int(obs.objects.semantic_label[-1][_LEFT_DOOR_SLOT]) == LABEL_COLLECT
    # Stamped centres match policy WP* constants.
    shaft_cx = float(obs.objects.x[-1][_SHAFT_BEAD_SLOT]) + float(
        obs.objects.width[-1][_SHAFT_BEAD_SLOT]
    ) * 0.5
    shaft_cy = float(obs.objects.y[-1][_SHAFT_BEAD_SLOT]) + float(
        obs.objects.height[-1][_SHAFT_BEAD_SLOT]
    ) * 0.5
    assert abs(shaft_cx - WP0[0]) < 0.6
    assert abs(shaft_cy - WP0[1]) < 0.6
    rope_cx = float(obs.objects.x[-1][_ROPE_GRAB_SLOT]) + float(
        obs.objects.width[-1][_ROPE_GRAB_SLOT]
    ) * 0.5
    rope_cy = float(obs.objects.y[-1][_ROPE_GRAB_SLOT]) + float(
        obs.objects.height[-1][_ROPE_GRAB_SLOT]
    ) * 0.5
    assert abs(rope_cx - WP2[0]) < 0.6
    assert abs(rope_cy - WP2[1]) < 0.6


@pytest.mark.xfail(reason="Montezuma is not part of the thesis evaluation; written against the earlier JAXtari fork", strict=False)
def test_montezuma_obs_y_matches_render_offset():
    from concept_based_learning.environments.env_manager import EnvManager

    mgr = EnvManager()
    env, _ = mgr.make_env(
        "montezumarevenge",
        {
            "FRAME_STACK_SIZE": 4,
            "FRAME_SKIP": 1,
            "FIRST_FIRE": False,
            "STICKY_ACTIONS": False,
        },
        train=False,
    )
    obs, state = env.reset(jax.random.PRNGKey(0))
    st = state
    for _ in range(8):
        if hasattr(st, "env_state"):
            gs = st.env_state
            break
        if hasattr(st, "atari_state"):
            st = st.atari_state
            continue
        gs = st
        break
    else:
        gs = state
    py_obs = int(obs.raw_observation.player.y[-1])
    py_state = int(gs.player_y)
    assert py_obs == py_state + MONTEZUMA_RENDER_Y_OFFSET
    key_y_obs = int(obs.objects.y[-1][_KEY_SLOT])
    key_y_state = int(gs.items_y[0])
    assert key_y_obs == key_y_state + MONTEZUMA_RENDER_Y_OFFSET


def test_montezuma_rope_grab_is_high_collect():
    from concept_based_learning.environments.env_manager import EnvManager
    from concept_based_learning.observations.egocentric import player_object_aabb_gaps

    mgr = EnvManager()
    env, _ = mgr.make_env(
        "montezumarevenge",
        {
            "FRAME_STACK_SIZE": 4,
            "FRAME_SKIP": 1,
            "FIRST_FIRE": False,
            "STICKY_ACTIONS": False,
        },
        train=False,
    )
    obs, _ = env.reset(jax.random.PRNGKey(0))
    assert int(obs.objects.active[-1][_ROPE_GRAB_SLOT]) == 1
    assert int(obs.objects.semantic_label[-1][_ROPE_GRAB_SLOT]) == LABEL_COLLECT
    rope_y = int(obs.objects.y[-1][_ROPE_GRAB_SLOT])
    assert rope_y < 110
    # Room-4 stamps rope from WP2 with the standard (2·ARRIVE_R)² arrive box.
    assert int(obs.objects.height[-1][_ROPE_GRAB_SLOT]) == 12

    mid_y = 68 + MONTEZUMA_RENDER_Y_OFFSET
    raw, objs = _make_obs((111, mid_y), _room4_objects())
    objs = objs.replace(
        x=objs.x.at[:, _ROPE_GRAB_SLOT].set(
            jnp.int32(obs.objects.x[-1][_ROPE_GRAB_SLOT])
        ),
        y=objs.y.at[:, _ROPE_GRAB_SLOT].set(
            jnp.int32(obs.objects.y[-1][_ROPE_GRAB_SLOT])
        ),
    )
    gaps, *_ = player_object_aabb_gaps(raw, objs)
    assert float(gaps[_ROPE_GRAB_SLOT]) > 8.0


def test_montezuma_get_ego_obs_pov_facing_xy():
    from concept_based_learning.environments.env_manager import EnvManager
    from concept_based_learning.observations.egocentric import get_ego_obs
    from concept_based_learning.observations.pov_facing_frame import (
        configure_pov_facing_frame_from_config,
    )
    from concept_based_learning.observations.pov_relative import (
        configure_pov_relative_from_config,
    )

    configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": True})
    configure_pov_facing_frame_from_config(
        {
            "USE_POV_RELATIVE_MAPPER": True,
            "USE_POV_FACING_FRAME": True,
            "EGO_POSE_XY": True,
            "EGO_WEAPON_FEATURES": True,
            "EGO_BLOCKING_FEATURES": True,
        }
    )
    try:
        mgr = EnvManager()
        env, _ = mgr.make_env(
            "montezumarevenge",
            {
                "FRAME_STACK_SIZE": 4,
                "FRAME_SKIP": 1,
                "FIRST_FIRE": False,
                "STICKY_ACTIONS": False,
            },
            train=False,
        )
        obs, _ = env.reset(jax.random.PRNGKey(1))
        ego = get_ego_obs(
            obs.raw_observation,
            obs.objects,
            jnp.int32(_SHAFT_BEAD_SLOT),
        )
        assert ego.ndim == 1
        assert ego.shape[0] > 0
    finally:
        configure_pov_facing_frame_from_config({"USE_POV_FACING_FRAME": False})
        configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})


@pytest.mark.xfail(reason="Montezuma is not part of the thesis evaluation; written against the earlier JAXtari fork", strict=False)
def test_montezuma_heuristic_smoke_on_reset():
    from concept_based_learning.environments.env_manager import EnvManager

    mgr = EnvManager()
    env, _ = mgr.make_env(
        "montezumarevenge",
        {
            "FRAME_STACK_SIZE": 4,
            "FRAME_SKIP": 1,
            "FIRST_FIRE": False,
            "STICKY_ACTIONS": False,
        },
        train=False,
    )
    obs, _ = env.reset(jax.random.PRNGKey(2))
    target = get_heuristic_target(
        obs.raw_observation,
        obs.objects,
        jnp.int32(NO_TARGET_IDX),
        jax.random.PRNGKey(0),
    )
    assert int(target) == _SHAFT_BEAD_SLOT
    assert int(target) != _KEY_SLOT


def test_arrive_uses_overlap_not_centre_only():
    """Touching the WP box advances; missing it does not."""
    state0 = PolicyState.zero(no_target_idx=NO_TARGET_IDX)

    raw_far, objs_far = _make_obs((WP0[0] - 3.5, WP0[1] - 40.0), _room4_objects())
    _, state_far = get_heuristic_target_with_state(
        raw_far, objs_far, jnp.int32(NO_TARGET_IDX), state0, jax.random.PRNGKey(0)
    )
    assert int(state_far.plan_idx) == 0

    raw_near, objs_near = _make_obs(_near(WP0), _room4_objects())
    _, state_near = get_heuristic_target_with_state(
        raw_near, objs_near, jnp.int32(NO_TARGET_IDX), state0, jax.random.PRNGKey(1)
    )
    assert int(state_near.plan_idx) >= 1
