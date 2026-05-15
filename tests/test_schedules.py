from concept_based_learning.training.schedules import eps_decay_transition_steps


def test_eps_decay_small_values_are_transition_steps_not_fraction():
    # Legacy semantics: passed directly to optax (0.8 -> 1 update, not 80% of training).
    assert eps_decay_transition_steps(0.8) == 1
    assert eps_decay_transition_steps(0.3) == 1


def test_eps_decay_large_values_are_explicit_steps():
    assert eps_decay_transition_steps(100) == 100
    assert eps_decay_transition_steps(781) == 781
