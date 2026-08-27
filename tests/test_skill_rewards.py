"""Unit tests for global skill reward contract (NAVIGATE progress, death)."""

import jax.numpy as jnp
import pytest
from flax import struct

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NAVIGATE,
)
from concept_based_learning.environments.skill_rewards import (
    AVOID_ESCAPE_REWARD,
    COLLECT_LIVING_PENALTY,
    COLLECT_PROGRESS_REWARD,
    COLLECT_SUCCESS_REWARD,
    DESTROY_FAILURE_REWARD,
    DESTROY_LIVING_PENALTY,
    DESTROY_STANDOFF_PENALTY,
    DESTROY_STANDOFF_REWARD,
    DESTROY_SUCCESS_REWARD,
    DESTROY_Y_ALIGN_REWARD,
    NAVIGATE_PROGRESS_REWARD,
    NAVIGATE_PROGRESS_UNIT_PX,
    NAVIGATE_SUCCESS_REWARD,
    SKILL_DEATH_PENALTY,
    extract_skill_rewards,
)


@struct.dataclass
class _Ent:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


@struct.dataclass
class _Raw:
    player: _Ent


@struct.dataclass
class _Tgt:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray
    semantic_label: jnp.ndarray
    was_killed: jnp.ndarray
    was_collected: jnp.ndarray
    can_go_offscreen: jnp.ndarray


def _S(v, dtype=jnp.int32):
    a = jnp.asarray(v, dtype=dtype)
    return jnp.stack([a, a, a, a])


def _centre_dist(px, py, tx, ty, *, pw=8, ph=11, tw=1, th=1) -> float:
    pcx, pcy = px + pw / 2.0, py + ph / 2.0
    tcx, tcy = tx + tw / 2.0, ty + th / 2.0
    return float(((pcx - tcx) ** 2 + (pcy - tcy) ** 2) ** 0.5)


def _nav_shaped(prev_dist: float, next_dist: float) -> float:
    return NAVIGATE_PROGRESS_REWARD * (prev_dist - next_dist) / NAVIGATE_PROGRESS_UNIT_PX


def _mk(px, py, tx, ty, label, *, pw=8, ph=11, tw=1, th=1, active=1, collected=0):
    raw = _Raw(_Ent(_S(px), _S(py), _S(pw), _S(ph)))
    tgt = _Tgt(
        _S(tx),
        _S(ty),
        _S(tw),
        _S(th),
        _S(active),
        _S(label),
        _S(0),
        _S(collected),
        _S(0),
    )
    return raw, tgt


def test_navigate_progress_when_distance_decreases():
    # Waypoint above/left; player moves closer → scaled +progress, not done.
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_NAVIGATE)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    expected = _nav_shaped(
        _centre_dist(50, 100, 50, 40),
        _centre_dist(50, 80, 50, 40),
    )
    assert float(r[3]) == pytest.approx(expected)
    assert expected > 0.0
    assert not bool(d[3])


def test_navigate_no_progress_when_idle_or_farther():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    idle_r, idle_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    r0, _ = extract_skill_rewards(prev_r, idle_r, prev_t, idle_t, jnp.int32(0))
    assert float(r0[3]) == 0.0

    away_r, away_t = _mk(50, 120, 50, 40, LABEL_NAVIGATE)
    r1, _ = extract_skill_rewards(prev_r, away_r, prev_t, away_t, jnp.int32(0))
    expected = _nav_shaped(
        _centre_dist(50, 100, 50, 40),
        _centre_dist(50, 120, 50, 40),
    )
    assert float(r1[3]) == pytest.approx(expected)
    assert expected < 0.0


def test_navigate_arrive_overrides_progress():
    # Move onto the 1px waypoint → sparse success, not progress.
    prev_r, prev_t = _mk(50, 50, 50, 44, LABEL_NAVIGATE, ph=8)
    next_r, next_t = _mk(50, 40, 50, 44, LABEL_NAVIGATE, ph=8)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[3]) == NAVIGATE_SUCCESS_REWARD
    assert bool(d[3])


def test_navigate_ten_px_close_gives_point_oh_five():
    # Exactly 10px closer → +0.05 by construction of the scale.
    prev_r, prev_t = _mk(0, 0, 100, 0, LABEL_NAVIGATE, pw=0, ph=0, tw=0, th=0)
    next_r, next_t = _mk(10, 0, 100, 0, LABEL_NAVIGATE, pw=0, ph=0, tw=0, th=0)
    r, _ = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[3]) == pytest.approx(0.05)


def test_collect_approach_progress_when_distance_decreases():
    prev_r, prev_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_r, next_t = _mk(20, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[1]) == pytest.approx(COLLECT_PROGRESS_REWARD)
    assert not bool(d[1])


def test_collect_idle_gets_no_progress():
    prev_r, prev_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_r, next_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[1]) == 0.0
    assert not bool(d[1])


def test_avoid_escape_progress_when_distance_increases():
    # Move away from threat → escape shaping, not survival farm / not done.
    prev_r, prev_t = _mk(50, 100, 80, 100, LABEL_AVOID, tw=8, th=1)
    next_r, next_t = _mk(30, 100, 80, 100, LABEL_AVOID, tw=8, th=1)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[2]) == pytest.approx(AVOID_ESCAPE_REWARD)
    assert not bool(d[2])


def test_avoid_idle_near_threat_gets_zero():
    # Old survival farm would pay +0.01 here; escape shaping must not.
    prev_r, prev_t = _mk(50, 100, 80, 100, LABEL_AVOID, tw=8, th=1)
    next_r, next_t = _mk(50, 100, 80, 100, LABEL_AVOID, tw=8, th=1)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[2]) == 0.0
    assert not bool(d[2])


def test_avoid_closing_in_gets_zero():
    prev_r, prev_t = _mk(30, 100, 80, 100, LABEL_AVOID, tw=8, th=1)
    next_r, next_t = _mk(50, 100, 80, 100, LABEL_AVOID, tw=8, th=1)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[2]) == 0.0
    assert not bool(d[2])


def test_death_does_not_penalize_navigate_or_collect():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_NAVIGATE)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(1))
    # Progress still applies; death does not override NAVIGATE.
    expected = _nav_shaped(
        _centre_dist(50, 100, 50, 40),
        _centre_dist(50, 80, 50, 40),
    )
    assert float(r[3]) == pytest.approx(expected)
    assert not bool(d[3])

    prev_c, prev_tc = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_c, next_tc = _mk(20, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    rc, dc = extract_skill_rewards(prev_c, next_c, prev_tc, next_tc, jnp.int32(1))
    # Approach progress still applies when COLLECT death is not opted in.
    assert float(rc[1]) == pytest.approx(COLLECT_PROGRESS_REWARD)
    assert not bool(dc[1])


def test_death_penalizes_collect_when_opted_in():
    prev_c, prev_tc = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_c, next_tc = _mk(20, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    rc, dc = extract_skill_rewards(
        prev_c,
        next_c,
        prev_tc,
        next_tc,
        jnp.int32(1),
        penalize_collect_death=True,
    )
    assert float(rc[1]) == SKILL_DEATH_PENALTY
    assert bool(dc[1])

    # NAVIGATE stays exempt even when COLLECT death is on.
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_NAVIGATE)
    r, d = extract_skill_rewards(
        prev_r,
        next_r,
        prev_t,
        next_t,
        jnp.int32(1),
        penalize_collect_death=True,
    )
    assert float(r[3]) == pytest.approx(
        _nav_shaped(_centre_dist(50, 100, 50, 40), _centre_dist(50, 80, 50, 40))
    )
    assert not bool(d[3])


def test_penalize_collect_death_from_config():
    from concept_based_learning.environments.skill_rewards import (
        penalize_collect_death_from_config,
    )

    assert penalize_collect_death_from_config(None) is False
    assert penalize_collect_death_from_config({}) is False
    assert penalize_collect_death_from_config({"SKILL_DEATH_PENALIZE_COLLECT": True}) is True
    assert penalize_collect_death_from_config({"SKILL_DEATH_PENALIZE_COLLECT": False}) is False


def test_death_penalizes_avoid_and_destroy():
    prev_r, prev_t = _mk(50, 100, 80, 100, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 100, 80, 100, LABEL_DESTROY, tw=8, th=7)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(1))
    assert float(r[0]) == SKILL_DEATH_PENALTY
    assert bool(d[0])

    prev_a, prev_ta = _mk(50, 100, 80, 100, LABEL_AVOID, tw=8, th=1)
    next_a, next_ta = _mk(50, 100, 80, 100, LABEL_AVOID, tw=8, th=1)
    ra, da = extract_skill_rewards(prev_a, next_a, prev_ta, next_ta, jnp.int32(1))
    assert float(ra[2]) == SKILL_DEATH_PENALTY
    assert bool(da[2])


def test_destroy_stays_sparse_without_flags():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_DESTROY, tw=8, th=7)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[0]) == 0.0
    assert not bool(d[0])


def test_destroy_kill_still_plus_one():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 100, 50, 40, LABEL_DESTROY, tw=8, th=7)
    next_t = next_t.replace(was_killed=_S(1))
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[0]) == pytest.approx(DESTROY_SUCCESS_REWARD)
    assert bool(d[0])


def test_destroy_y_align_when_abs_dy_shrinks():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_DESTROY, tw=8, th=7)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), destroy_y_align=True
    )
    assert float(r[0]) == pytest.approx(DESTROY_Y_ALIGN_REWARD)
    assert not bool(d[0])


def test_destroy_standoff_backs_off_in_melee():
    # Same lane, overlapping x; step away horizontally.
    prev_r, prev_t = _mk(48, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(20, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), destroy_standoff=True
    )
    assert float(r[0]) == pytest.approx(DESTROY_STANDOFF_REWARD)
    assert not bool(d[0])


def test_destroy_standoff_penalizes_closing_in_melee():
    prev_r, prev_t = _mk(30, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(48, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), destroy_standoff=True
    )
    assert float(r[0]) == pytest.approx(DESTROY_STANDOFF_PENALTY)
    assert not bool(d[0])


def test_destroy_contact_fail_on_overlap_rising_edge():
    prev_r, prev_t = _mk(10, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), destroy_contact_fail=True
    )
    assert float(r[0]) == pytest.approx(DESTROY_FAILURE_REWARD)
    assert bool(d[0])


def test_destroy_kill_overrides_contact_fail():
    prev_r, prev_t = _mk(10, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    next_t = next_t.replace(was_killed=_S(1))
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), destroy_contact_fail=True
    )
    assert float(r[0]) == pytest.approx(DESTROY_SUCCESS_REWARD)
    assert bool(d[0])


def test_destroy_living_penalty_while_enemy_alive():
    prev_r, prev_t = _mk(50, 100, 80, 100, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 100, 80, 100, LABEL_DESTROY, tw=8, th=7)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), destroy_living_penalty=True
    )
    assert float(r[0]) == pytest.approx(DESTROY_LIVING_PENALTY)
    assert not bool(d[0])


def test_destroy_living_penalty_skipped_on_kill():
    prev_r, prev_t = _mk(50, 100, 80, 100, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 100, 80, 100, LABEL_DESTROY, tw=8, th=7)
    next_t = next_t.replace(was_killed=_S(1))
    r, d = extract_skill_rewards(
        prev_r,
        next_r,
        prev_t,
        next_t,
        jnp.int32(0),
        destroy_living_penalty=True,
        destroy_contact_fail=True,
    )
    assert float(r[0]) == pytest.approx(DESTROY_SUCCESS_REWARD)
    assert bool(d[0])


def test_destroy_contact_fail_beats_living_penalty():
    prev_r, prev_t = _mk(10, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    next_r, next_t = _mk(50, 50, 50, 50, LABEL_DESTROY, tw=8, th=7)
    r, d = extract_skill_rewards(
        prev_r,
        next_r,
        prev_t,
        next_t,
        jnp.int32(0),
        destroy_living_penalty=True,
        destroy_contact_fail=True,
    )
    assert float(r[0]) == pytest.approx(DESTROY_FAILURE_REWARD)
    assert bool(d[0])


def test_collect_living_penalty_replaces_approach():
    prev_r, prev_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_r, next_t = _mk(20, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), collect_living_penalty=True
    )
    assert float(r[1]) == pytest.approx(COLLECT_LIVING_PENALTY)
    assert not bool(d[1])


def test_collect_living_penalty_idle_still_taxes():
    prev_r, prev_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_r, next_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), collect_living_penalty=True
    )
    assert float(r[1]) == pytest.approx(COLLECT_LIVING_PENALTY)
    assert not bool(d[1])


def test_collect_success_overrides_living_penalty():
    prev_r, prev_t = _mk(10, 50, 20, 50, LABEL_COLLECT, tw=8, th=8)
    next_r, next_t = _mk(20, 50, 20, 50, LABEL_COLLECT, tw=8, th=8)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), collect_living_penalty=True
    )
    assert float(r[1]) == pytest.approx(COLLECT_SUCCESS_REWARD)
    assert bool(d[1])


def test_navigate_keeps_approach_when_collect_living_on():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_NAVIGATE)
    r, d = extract_skill_rewards(
        prev_r, next_r, prev_t, next_t, jnp.int32(0), collect_living_penalty=True
    )
    assert float(r[3]) == pytest.approx(
        _nav_shaped(_centre_dist(50, 100, 50, 40), _centre_dist(50, 80, 50, 40))
    )
    assert not bool(d[3])


def test_collect_living_penalty_from_config():
    from concept_based_learning.environments.skill_rewards import (
        collect_living_penalty_from_config,
    )

    assert collect_living_penalty_from_config(None) is False
    assert collect_living_penalty_from_config({}) is False
    assert collect_living_penalty_from_config({"COLLECT_LIVING_PENALTY": True}) is True
    prev_r, prev_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_r, next_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    r, d = extract_skill_rewards(
        prev_r,
        next_r,
        prev_t,
        next_t,
        jnp.int32(0),
        config={"COLLECT_LIVING_PENALTY": True},
    )
    assert float(r[1]) == pytest.approx(COLLECT_LIVING_PENALTY)
    assert not bool(d[1])
