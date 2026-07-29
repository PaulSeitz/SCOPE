"""Unit tests for global skill reward contract (NAVIGATE progress, death)."""

import jax.numpy as jnp
import pytest
from flax import struct

from concept_based_learning.environments.datatypes import (
    LABEL_COLLECT,
    LABEL_NAVIGATE,
)
from concept_based_learning.environments.skill_rewards import (
    COLLECT_SUCCESS_REWARD,
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


def test_death_overrides_navigate_progress():
    prev_r, prev_t = _mk(50, 100, 50, 40, LABEL_NAVIGATE)
    next_r, next_t = _mk(50, 80, 50, 40, LABEL_NAVIGATE)
    r, d = extract_skill_rewards(prev_r, next_r, prev_t, next_t, jnp.int32(1))
    assert float(r[3]) == SKILL_DEATH_PENALTY
    assert bool(d[3])
