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
    NAVIGATE_PROGRESS_REWARD,
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
    # Waypoint above/left; player moves closer → +progress, not done.
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_NAVIGATE)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[3]) == pytest.approx(NAVIGATE_PROGRESS_REWARD)
    assert not bool(d[3])


def test_navigate_no_progress_when_idle_or_farther():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    idle_r, idle_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    r0, _ = extract_skill_rewards(prev_r, idle_r, prev_t, idle_t, jnp.int32(0))
    assert float(r0[3]) == 0.0

    away_r, away_t = _mk(50, 120, 50, 40, LABEL_NAVIGATE)
    r1, _ = extract_skill_rewards(prev_r, away_r, prev_t, away_t, jnp.int32(0))
    assert float(r1[3]) == 0.0


def test_navigate_arrive_overrides_progress():
    # Move onto the 1px waypoint → sparse success, not progress.
    prev_r, prev_t = _mk(50, 50, 50, 44, LABEL_NAVIGATE, ph=8)
    next_r, next_t = _mk(50, 40, 50, 44, LABEL_NAVIGATE, ph=8)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[3]) == NAVIGATE_SUCCESS_REWARD
    assert bool(d[3])


def test_collect_stays_sparse_no_progress_term():
    prev_r, prev_t = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_r, next_t = _mk(20, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(0))
    assert float(r[1]) == 0.0
    assert not bool(d[1])


def test_death_does_not_penalize_navigate_or_collect():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_NAVIGATE)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(1))
    # Progress still applies; death does not override NAVIGATE.
    assert float(r[3]) == pytest.approx(NAVIGATE_PROGRESS_REWARD)
    assert not bool(d[3])

    prev_c, prev_tc = _mk(10, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    next_c, next_tc = _mk(20, 50, 40, 50, LABEL_COLLECT, tw=4, th=4)
    rc, dc = extract_skill_rewards(prev_c, next_c, prev_tc, next_tc, jnp.int32(1))
    assert float(rc[1]) == 0.0
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
    assert float(r[3]) == pytest.approx(NAVIGATE_PROGRESS_REWARD)
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
