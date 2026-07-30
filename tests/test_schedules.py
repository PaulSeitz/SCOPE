import pytest

from concept_based_learning.training.schedules import (
    eps_decay_transition_steps,
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
