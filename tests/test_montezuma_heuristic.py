"""Unit tests for Montezuma's Revenge ego patch + L1 waypoint heuristic."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_NAVIGATE,
)
from concept_based_learning.environments.manual_policy.state import PolicyState
from concept_based_learning.environments.montezumarevenge.montezuma_label import (
    MONTEZUMA_RENDER_Y_OFFSET,
)
from concept_based_learning.environments.montezumarevenge.montezuma_policy import (
    CLOSE_AVOID_RANGE,
    NO_TARGET_IDX,
    ROPE_GRAB_ARRIVE,
    _KEY_SLOT,
    _L0_BOT,
    _L0_TOP,
    _L1_TOP,
    _ROPE_GRAB_SLOT,
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
    """Minimal room-4-like slots: meta NOOP + enemies + ladder wps + key + door."""
    objs = [(0, 0, 1, 1, 0, 0)] * 32
    objs = list(objs)
    objs[_L0_BOT] = (80, 78, 8, 8, 1, LABEL_COLLECT)
    objs[_L0_TOP] = (80, 27, 8, 8, 1, LABEL_COLLECT)
    objs[_L1_TOP] = (136, 70, 8, 8, 1, LABEL_COLLECT)
    objs[5] = (136, 120, 8, 8, 1, LABEL_COLLECT)  # L1_BOT
    objs[6] = (24, 120, 8, 8, 1, LABEL_COLLECT)  # L2_BOT
    objs[10] = (24, 70, 8, 8, 1, LABEL_COLLECT)  # L2_TOP
    # High rope grab (state-y ~51): above mid-floor stand (~68).
    objs[_ROPE_GRAB_SLOT] = (107, 51, 8, 8, 1, LABEL_COLLECT)
    objs[_KEY_SLOT] = (13, 52, 6, 8, 1, LABEL_COLLECT)
    objs[15] = (16, 7, 4, 38, 1, LABEL_NAVIGATE)
    if skull_xy is not None:
        objs[1] = (skull_xy[0], skull_xy[1], 6, 14, skull_active, LABEL_AVOID)
    return objs


def test_montezuma_avoids_close_enemy():
    raw, objs = _make_obs((80, 110), _room4_objects(skull_xy=(90, 119), skull_active=1))
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == 1
    assert CLOSE_AVOID_RANGE > 0


def test_montezuma_plan_starts_at_center_ladder_bot():
    # Spawn is on the top floor — first stage is climb down to center ladder BOT.
    raw, objs = _make_obs((77, 26), _room4_objects())
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) == _L0_BOT
    assert int(target) != _KEY_SLOT


def test_montezuma_stateful_holds_bot_until_arrive():
    raw, objs = _make_obs((77, 26), _room4_objects())
    state = PolicyState.zero(no_target_idx=NO_TARGET_IDX)
    target, state = get_heuristic_target_with_state(
        raw, objs, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(0)
    )
    assert int(target) == _L0_BOT
    assert int(state.plan_idx) == 0

    # Mid-climb: player box already overlaps the bot WP, but feet are still above
    # the WP centre — must not advance.
    raw_mid, objs_mid = _make_obs((80, 58), _room4_objects())
    target_mid, state_mid = get_heuristic_target_with_state(
        raw_mid, objs_mid, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(2)
    )
    assert int(state_mid.plan_idx) == 0
    assert int(target_mid) == _L0_BOT

    # Standing on the mid landing (feet at ladder bottom) → rope grab next.
    # feet_oy_min=6 requires full stand (was 4 → 2px early on ground ladders).
    raw2, objs2 = _make_obs((80, 68), _room4_objects())
    target2, state2 = get_heuristic_target_with_state(
        raw2, objs2, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(1)
    )
    assert int(state2.plan_idx) >= 1
    assert int(target2) == _ROPE_GRAB_SLOT
    assert int(target2) != _KEY_SLOT
    assert int(target2) != _L1_TOP


def test_montezuma_side_top_does_not_skip_to_key_mid_climb():
    """L2_TOP must not complete mid-shaft (old platform_reach → KEY early)."""
    from concept_based_learning.environments.montezumarevenge.montezuma_policy import (
        MONTEZUMA_POLICY,
        _L2_TOP,
    )

    # Advance plan to L2_TOP stage (index of _wp_side_top in MONTEZUMA_L1_STAGES).
    raw, objs = _make_obs((20, 140), _room4_objects())
    state = PolicyState.zero(no_target_idx=NO_TARGET_IDX)
    # Manually set plan_idx to the L2_TOP stage (outbound index 5).
    state = state.replace(plan_idx=jnp.int32(5))
    target, state2 = get_heuristic_target_with_state(
        raw, objs, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(0)
    )
    assert int(target) == _L2_TOP
    assert int(state2.plan_idx) == 5
    assert int(target) != _KEY_SLOT

    # Standing on key ledge → may advance to KEY.
    raw_top, objs_top = _make_obs((20, 70), _room4_objects())
    target_top, state_top = get_heuristic_target_with_state(
        raw_top, objs_top, jnp.int32(NO_TARGET_IDX), state, jax.random.PRNGKey(1)
    )
    assert int(state_top.plan_idx) >= 6
    assert int(target_top) == _KEY_SLOT


def test_montezuma_inactive_ladder_returns_no_target_or_bot():
    # Inactive bot slot: NavigateTo returns NO_TARGET while skip_if_inactive=False
    # keeps the plan from advancing.
    raw, objs = _make_obs((80, 80), [(0, 0, 1, 1, 0, 0)])
    target = get_heuristic_target(
        raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0)
    )
    assert int(target) in (NO_TARGET_IDX, _L0_BOT)


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
    # Spawn is already on the top landing — center TOP WP starts inactive.
    assert int(act[_L0_TOP]) == 0
    assert int(obs.objects.semantic_label[-1][_L0_TOP]) == LABEL_COLLECT
    # Side ladders are active for the right→bottom→left key path.
    assert int(act[5]) == 1 and int(act[6]) == 1
    assert int(act[9]) == 1 and int(act[10]) == 1


def test_montezuma_obs_y_matches_render_offset():
    """Player/object Y in obs must match renderer (state_y + 47)."""
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
    # Unwrap to env_state
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
    # Key item slot (meta + ITEM0 = 12)
    key_y_obs = int(obs.objects.y[-1][_KEY_SLOT])
    key_y_state = int(gs.items_y[0])
    assert key_y_obs == key_y_state + MONTEZUMA_RENDER_Y_OFFSET


def test_montezuma_rope_grab_is_high_collect():
    """Rope slot is a high COLLECT; mid-floor stand does not arrive."""
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
    # Grab sits near rope top (state 49+2 + render 47 ≈ 98), not mid floor.
    rope_y = int(obs.objects.y[-1][_ROPE_GRAB_SLOT])
    assert rope_y < 110
    assert int(obs.objects.height[-1][_ROPE_GRAB_SLOT]) == 8

    # Place player on mid floor under/near rope (state y~68 → obs y~115).
    mid_y = 68 + MONTEZUMA_RENDER_Y_OFFSET
    raw, objs = _make_obs(
        (111, mid_y),
        _room4_objects(),
    )
    # Use env rope coords (with offset) for a realistic gap check.
    objs = objs.replace(
        x=objs.x.at[:, _ROPE_GRAB_SLOT].set(
            jnp.int32(obs.objects.x[-1][_ROPE_GRAB_SLOT])
        ),
        y=objs.y.at[:, _ROPE_GRAB_SLOT].set(
            jnp.int32(obs.objects.y[-1][_ROPE_GRAB_SLOT])
        ),
    )
    gaps, *_ = player_object_aabb_gaps(raw, objs)
    assert float(gaps[_ROPE_GRAB_SLOT]) > ROPE_GRAB_ARRIVE


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
            jnp.int32(_L0_BOT),
        )
        assert ego.ndim == 1
        assert ego.shape[0] > 0
    finally:
        configure_pov_facing_frame_from_config({"USE_POV_FACING_FRAME": False})
        configure_pov_relative_from_config({"USE_POV_RELATIVE_MAPPER": False})


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
    assert int(target) == _L0_BOT
    assert int(target) != _KEY_SLOT
