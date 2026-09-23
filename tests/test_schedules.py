import pytest

from concept_based_learning.training.schedules import (
    eps_decay_transition_steps,
    meta_learning_rate_schedule,
    num_updates_decay,
    optax_optimizer_transition_steps,
    skill_learning_rate_schedule,
)


def test_eps_decay_small_values_are_transition_steps_not_fraction():
    # Legacy semantics: passed directly to optax (0.8 -> 1 update, not 80% of training).
    assert eps_decay_transition_steps(0.8) == 1
    assert eps_decay_transition_steps(0.3) == 1


def test_eps_decay_large_values_are_explicit_steps():
    assert eps_decay_transition_steps(100) == 100
    assert eps_decay_transition_steps(781) == 781


def test_skill_lr_constant_without_finish():
    lr = skill_learning_rate_schedule({"SKILL_LR": 5e-4, "LR": 1e-4}, num_updates=1000)
    assert lr == 5e-4


def test_skill_lr_linear_decay_when_finish_set():
    sched = skill_learning_rate_schedule(
        {
            "SKILL_LR": 5e-4,
            "SKILL_LR_FINISH": 1e-4,
            "SKILL_LR_DECAY": 0.5,
            "LR": 1e-4,
        },
        num_updates=1000,
    )
    assert float(sched(0)) == pytest.approx(5e-4)
    assert float(sched(500)) == pytest.approx(1e-4)
    assert float(sched(999)) == pytest.approx(1e-4)


def test_lr_linear_decay_matches_flat_pqn_optax_steps():
    cfg = {
        "TOTAL_TIMESTEPS": 50_000_000,
        "TOTAL_TIMESTEPS_DECAY": 50_000_000,
        "NUM_STEPS": 32,
        "NUM_ENVS": 1024,
        "NUM_EPOCHS": 2,
        "NUM_MINIBATCHES": 256,
        "SKILL_LR": 2.5e-4,
        "META_LR": 2.5e-4,
        "LR": 2.5e-4,
        "LR_LINEAR_DECAY": True,
    }
    n_upd = num_updates_decay(cfg)
    assert n_upd == 1525
    n_opt = optax_optimizer_transition_steps(cfg, n_upd)
    assert n_opt == 1525 * 256 * 2

    skill = skill_learning_rate_schedule(cfg)
    meta = meta_learning_rate_schedule(cfg)
    assert float(skill(0)) == pytest.approx(2.5e-4)
    assert float(meta(0)) == pytest.approx(2.5e-4)
    assert float(skill(n_opt)) == pytest.approx(1e-20)
    assert float(meta(n_opt)) == pytest.approx(1e-20)
