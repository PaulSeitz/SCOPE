import jax.numpy as jnp
import pytest

from concept_based_learning.training.trajectory_cuts import (
    SKILL_TARGET_CUT_GLOBAL,
    SKILL_TARGET_CUT_INVOLVED,
    SKILL_TARGET_CUT_LIFETIME,
    normalize_skill_target_cut,
    skill_target_switch_cut,
    skill_target_switch_cut_env,
)


def test_normalize_aliases():
    assert normalize_skill_target_cut("HER") == SKILL_TARGET_CUT_LIFETIME
    assert normalize_skill_target_cut("involved_only") == SKILL_TARGET_CUT_INVOLVED
    assert normalize_skill_target_cut("all") == SKILL_TARGET_CUT_GLOBAL


def test_lifetime_no_switch_cuts():
    target_idx = jnp.array([[0], [1]])  # [T=2, E=1], switch 0->1
    cut = skill_target_switch_cut(target_idx, num_objects=3, mode=SKILL_TARGET_CUT_LIFETIME)
    assert cut.shape == (1, 1, 3)
    assert jnp.all(cut == 0.0)


def test_global_cuts_all_slots():
    target_idx = jnp.array([[0], [1]])
    cut = skill_target_switch_cut(target_idx, num_objects=3, mode=SKILL_TARGET_CUT_GLOBAL)
    assert jnp.all(cut[0, 0] == 1.0)


def test_involved_cuts_only_endpoints():
    # A=0, B=1 switch; slot C=2 untouched
    target_idx = jnp.array([[0], [1]])
    cut = skill_target_switch_cut(target_idx, num_objects=3, mode=SKILL_TARGET_CUT_INVOLVED)
    assert float(cut[0, 0, 0]) == 1.0  # old target A
    assert float(cut[0, 0, 1]) == 1.0  # new target B
    assert float(cut[0, 0, 2]) == 0.0  # passive C


def test_env_cut_lifetime_vs_global():
    target_idx = jnp.array([[0], [1]])
    assert jnp.all(
        skill_target_switch_cut_env(target_idx, SKILL_TARGET_CUT_LIFETIME) == 0.0
    )
    assert skill_target_switch_cut_env(target_idx, SKILL_TARGET_CUT_GLOBAL)[0, 0] == 1.0


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown SKILL_TARGET_CUT"):
        normalize_skill_target_cut("bogus")
