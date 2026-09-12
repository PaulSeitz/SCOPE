"""Unit tests for same-scaffolding fixed ego skill rules."""

import jax.numpy as jnp
import pytest

from concept_based_learning.baselines.fixed_ego_skills import (
    EGO_BACKWARD,
    EGO_FIRE,
    EGO_FORWARD,
    EGO_LEFT,
    EGO_RIGHT,
    constant_ego_action,
    ego_move_for_bearing,
    fixed_ego_action_from_phi,
    skill_policy_mode_from_config,
    uses_fixed_skill_policy,
    wrap_pi,
)


SKILL_NAMES = ("DESTROY", "COLLECT", "AVOID", "NAVIGATE")


def test_skill_policy_mode_helpers():
    assert skill_policy_mode_from_config({}) == "learned"
    assert uses_fixed_skill_policy({"SKILL_POLICY_MODE": "fixed_ego_rules"})
    assert uses_fixed_skill_policy({"SKILL_POLICY_MODE": "constant_ego"})
    assert not uses_fixed_skill_policy({"SKILL_POLICY_MODE": "learned"})
    with pytest.raises(ValueError):
        skill_policy_mode_from_config({"SKILL_POLICY_MODE": "nope"})


def test_ego_move_for_bearing_cardinals():
    assert int(ego_move_for_bearing(jnp.float32(0.0))) == EGO_FORWARD
    assert int(ego_move_for_bearing(jnp.float32(jnp.pi / 2))) == EGO_RIGHT
    assert int(ego_move_for_bearing(jnp.float32(-jnp.pi / 2))) == EGO_LEFT
    assert int(ego_move_for_bearing(jnp.float32(jnp.pi))) == EGO_BACKWARD


def test_collect_approaches_when_aligned():
    a = fixed_ego_action_from_phi(
        1, jnp.float32(0.0), jnp.float32(20.0), skill_names=SKILL_NAMES
    )
    assert int(a) == EGO_FORWARD


def test_collect_turns_toward_target():
    a = fixed_ego_action_from_phi(
        1, jnp.float32(jnp.pi / 2), jnp.float32(20.0), skill_names=SKILL_NAMES
    )
    assert int(a) == EGO_RIGHT


def test_avoid_flees_opposite_bearing():
    # Threat ahead → flee backward.
    a = fixed_ego_action_from_phi(
        2, jnp.float32(0.0), jnp.float32(8.0), skill_names=SKILL_NAMES
    )
    assert int(a) == EGO_BACKWARD
    # Threat to the right → flee left.
    a = fixed_ego_action_from_phi(
        2, jnp.float32(jnp.pi / 2), jnp.float32(8.0), skill_names=SKILL_NAMES
    )
    assert int(a) == EGO_LEFT


def test_destroy_fires_when_aligned():
    a = fixed_ego_action_from_phi(
        0, jnp.float32(0.05), jnp.float32(30.0), skill_names=SKILL_NAMES
    )
    assert int(a) == EGO_FIRE


def test_destroy_turns_when_misaligned():
    a = fixed_ego_action_from_phi(
        0, jnp.float32(-jnp.pi / 2), jnp.float32(30.0), skill_names=SKILL_NAMES
    )
    assert int(a) == EGO_LEFT


def test_navigate_shares_collect_rule():
    a = fixed_ego_action_from_phi(
        3, jnp.float32(0.0), jnp.float32(12.0), skill_names=SKILL_NAMES
    )
    assert int(a) == EGO_FORWARD


def test_constant_ego_actions():
    assert int(constant_ego_action(0, skill_names=SKILL_NAMES)) == EGO_FIRE
    assert int(constant_ego_action(1, skill_names=SKILL_NAMES)) == EGO_FORWARD
    assert int(constant_ego_action(2, skill_names=SKILL_NAMES)) == EGO_BACKWARD
    assert int(constant_ego_action(3, skill_names=SKILL_NAMES)) == EGO_FORWARD


def test_wrap_pi():
    assert float(wrap_pi(jnp.float32(3 * jnp.pi / 2))) == pytest.approx(
        float(-jnp.pi / 2), abs=1e-5
    )


def test_select_fixed_ego_action_jittable():
    import jax

    from concept_based_learning.baselines.fixed_ego_skills import select_fixed_ego_action

    # Minimal fake obs/objects with player + one target AABB.
    class _Box:
        def __init__(self):
            self.x = jnp.array([80.0, 80.0], dtype=jnp.float32)
            self.y = jnp.array([100.0, 100.0], dtype=jnp.float32)
            self.width = jnp.array([8.0, 8.0], dtype=jnp.float32)
            self.height = jnp.array([8.0, 8.0], dtype=jnp.float32)
            self.orientation = jnp.array([0, 0], dtype=jnp.int32)

    class _Objs:
        def __init__(self):
            self.x = jnp.zeros((2, 2), dtype=jnp.float32).at[:, 0].set(90.0)
            self.y = jnp.zeros((2, 2), dtype=jnp.float32).at[:, 0].set(100.0)
            self.width = jnp.ones((2, 2), dtype=jnp.float32) * 6.0
            self.height = jnp.ones((2, 2), dtype=jnp.float32) * 6.0
            self.semantic_label = jnp.ones((2, 2), dtype=jnp.int32)

    class _Obs:
        def __init__(self):
            self.player = _Box()

    cfg = {
        "SKILL_POLICY_MODE": "fixed_ego_rules",
        "SKILL_NAMES": SKILL_NAMES,
        "NUM_SKILLS": 4,
        "FIXED_EGO_ALIGN_DEG": 22.5,
    }
    raw = _Obs()
    objs = _Objs()

    @jax.jit
    def f(sk, tid):
        return select_fixed_ego_action(cfg, sk, raw, objs, tid)

    a = f(jnp.int32(1), jnp.int32(0))
    assert int(a) in {EGO_FORWARD, EGO_RIGHT, EGO_LEFT, EGO_BACKWARD, EGO_FIRE}
