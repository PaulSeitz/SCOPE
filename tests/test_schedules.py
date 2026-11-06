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


def test_meta_lr_counts_its_own_epoch_override():
    cfg = {"TOTAL_TIMESTEPS": 160, "NUM_STEPS": 4, "NUM_ENVS": 2,
           "NUM_MINIBATCHES": 2, "NUM_EPOCHS": 1, "META_NUM_EPOCHS": 3,
           "LR": .001, "SKILL_LR": .001, "META_LR": .001, "LR_LINEAR_DECAY": True}
    skill, meta = skill_learning_rate_schedule(cfg), meta_learning_rate_schedule(cfg)
    # 20 rollout updates: 40 skill optimizer steps, 120 meta optimizer steps.
    assert float(skill(40)) == pytest.approx(1e-20)
    assert float(meta(40)) == pytest.approx(.001 * 2 / 3)
    assert float(meta(120)) == pytest.approx(1e-20)
