"""Unit tests for Asterix heuristic targeting rules."""

import jax
import jax.numpy as jnp
from flax import struct

from concept_based_learning.environments.asterix.asterix_policy import (
    CLOSE_AVOID_RANGE,
    LANE_Y_THRESH,
    NO_TARGET_IDX,
    PREFERRED_COLLECT_CY,
    get_heuristic_target,
)
from concept_based_learning.environments.datatypes import LABEL_AVOID, LABEL_COLLECT


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
    px, py = player_xy
    raw = _Raw(
        player=_Player(
            x=_stack(jnp.int32(px)),
            y=_stack(jnp.int32(py)),
            width=_stack(jnp.int32(8)),
            height=_stack(jnp.int32(8)),
        )
    )
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


def test_asterix_avoids_same_lane_enemy():
    raw, objs = _make_obs(
        (80, 75),
        [
            (0, 0, 1, 1, 0, 0),
            (120, 75, 8, 8, 1, LABEL_AVOID),
            (40, 75, 8, 8, 1, LABEL_COLLECT),
        ],
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0))
    assert int(target) == 1


def test_asterix_collects_when_lane_clear():
    raw, objs = _make_obs(
        (80, 75),
        [
            (0, 0, 1, 1, 0, 0),
            (40, 75, 8, 8, 1, LABEL_COLLECT),
            (120, 120, 8, 8, 1, LABEL_AVOID),  # other lane
        ],
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0))
    assert int(target) == 1


def test_asterix_skips_collect_blocked_by_enemy():
    # Enemy between player and collectible on same lane → prefer AVOID.
    raw, objs = _make_obs(
        (80, 75),
        [
            (0, 0, 1, 1, 0, 0),
            (40, 75, 8, 8, 1, LABEL_COLLECT),
            (60, 75, 8, 8, 1, LABEL_AVOID),
        ],
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0))
    assert int(target) == 2


def test_asterix_avoids_close_off_lane():
    # Nearly overlapping AVOID on a neighbouring lane → within CLOSE_AVOID_RANGE.
    raw, objs = _make_obs(
        (80, 75),
        [
            (0, 0, 1, 1, 0, 0),
            (82, 75 + 6, 8, 8, 1, LABEL_AVOID),
            (20, 40, 8, 8, 1, LABEL_COLLECT),
        ],
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0))
    assert int(target) == 1
    assert CLOSE_AVOID_RANGE > 0


def test_asterix_prefers_middle_lane_collect_over_closer_top():
    # Top-lane item is nearer in gap distance but middle-lane rank should win.
    raw, objs = _make_obs(
        (80, 75),
        [
            (0, 0, 1, 1, 0, 0),
            (70, 27, 8, 8, 1, LABEL_COLLECT),  # top lane, ~15 px gap
            (120, 75, 8, 8, 1, LABEL_COLLECT),  # middle lane, ~40 px gap
        ],
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0))
    assert int(target) == 2
    assert PREFERRED_COLLECT_CY > 0


def test_asterix_ignores_off_lane_enemy_below_for_avoid():
    # Off-lane enemy below the player should not override a clear COLLECT.
    raw, objs = _make_obs(
        (80, 75),
        [
            (0, 0, 1, 1, 0, 0),
            (40, 75, 8, 8, 1, LABEL_COLLECT),
            (82, 95, 8, 8, 1, LABEL_AVOID),  # below, neighbouring lane, close on x
        ],
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), jax.random.PRNGKey(0))
    assert int(target) == 1


def test_asterix_registered_in_env_manager():
    from concept_based_learning.environments.env_manager import EnvManager

    mgr = EnvManager()
    assert "asterix" in mgr.available_envs()
    comps = mgr.get_components("asterix")
    assert comps.jax_game_id == "asterix"
    assert comps.get_heuristic_target is get_heuristic_target
