"""Independent arithmetic oracles for PQN targets, including random rollouts.

The forward oracle enumerates n-step returns and their mixture weights; it
does not implement the reverse recursion used by the trainer.
"""

from fractions import Fraction
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from concept_based_learning.training.returns import lambda_return_targets, object_lambda_return_targets


def forward_view(rewards, dones, next_values, gamma, lam):
    """Exact rational forward view for one time-major scalar trajectory."""
    f = lambda v: Fraction(str(float(v)))
    r, v = list(map(f, rewards)), list(map(f, next_values))
    gamma, lam = f(gamma), f(lam)
    result = []
    for start in range(len(r)):
        horizons = []
        accumulated = Fraction(0)
        for end in range(start, len(r)):
            n = end - start + 1
            accumulated += gamma ** (n - 1) * r[end]
            terminal = bool(dones[end])
            horizons.append(accumulated if terminal else accumulated + gamma ** n * v[end])
            if terminal:
                break
        total = Fraction(0)
        for n, value in enumerate(horizons, 1):
            weight = lam ** (n - 1)
            if n < len(horizons):
                weight *= 1 - lam
            total += weight * value
        result.append(float(total))
    return np.array(result)


@pytest.mark.parametrize("lam,expected", [
    (0, [19, 29, 39]),
    (0.5, [24.8725, 33.05, 39]),
    (1, [34.39, 37.1, 39]),
])
def test_hand_calculated_three_steps(lam, expected):
    actual = lambda_return_targets(
        jnp.array([1., 2., 3.]), jnp.zeros(3, dtype=bool),
        jnp.array([20., 30., 40.]), 0.9, lam,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_terminal_cuts_both_bootstrap_and_future_rewards():
    actual = lambda_return_targets(
        jnp.array([1., 2., 3.]), jnp.array([False, True, False]),
        jnp.array([20., 30., 40.]), 0.9, 0.5,
    )
    np.testing.assert_allclose(actual, [10.9, 2., 39.], rtol=1e-6)


def test_seeded_hand_audit_examples():
    rng = np.random.default_rng(20261024)
    expected_cases = [
        [4.3236046875, .6471875, -2.3125, 4.5],
        [1.01, 1., -1.4435, -3.1],
        [6.1032215, 2.1679, 1.74, 0.],
    ]
    for case, expected in enumerate(expected_cases):
        r = rng.integers(-4, 5, 4).astype(np.float32)
        v = rng.integers(-5, 11, 4).astype(np.float32)
        done = np.zeros(4, bool)
        done[1 if case == 1 else 3] = case > 0
        np.testing.assert_allclose(forward_view(r, done, v, .9, .65), expected, atol=1e-12)
        np.testing.assert_allclose(lambda_return_targets(r, done, v, .9, .65), expected, rtol=1e-6)


@pytest.mark.parametrize("lam", [0., 0.5, 0.65, 1.])
@pytest.mark.parametrize("time", [1, 2, 5, 32])
def test_random_batches_against_exact_forward_view(lam, time):
    rng = np.random.default_rng(20261024 + time)
    # Independent environment, relabel-object and skill dimensions.
    shape = (time, 3, 4, 3)
    rewards = rng.integers(-4, 5, shape).astype(np.float32) / 4
    dones = rng.random(shape) < 0.2
    values = rng.integers(-20, 21, shape).astype(np.float32) / 4
    actual = jax.jit(lambda_return_targets)(
        jnp.asarray(rewards), jnp.asarray(dones), jnp.asarray(values), 0.99, lam,
    )
    for env, obj, skill in np.ndindex(shape[1:]):
        idx = (slice(None), env, obj, skill)
        expected = forward_view(rewards[idx], dones[idx], values[idx], 0.99, lam)
        np.testing.assert_allclose(actual[idx], expected, rtol=2e-6, atol=2e-6)


def test_terminal_blocks_nonfinite_reset_value():
    targets = lambda_return_targets(
        jnp.array([1.]), jnp.array([True]), jnp.array([jnp.nan]), .99, .65,
    )
    np.testing.assert_array_equal(targets, [1.])


def test_targets_are_detached_from_online_network():
    grad = jax.grad(lambda values: lambda_return_targets(
        jnp.array([1., 2.]), jnp.array([False, False]), values, .99, .65,
    ).sum())(jnp.array([3., 4.]))
    np.testing.assert_array_equal(grad, [0., 0.])


def test_shape_mismatch_fails_instead_of_broadcasting_across_envs():
    with pytest.raises(ValueError, match="identical"):
        lambda_return_targets(jnp.zeros((3, 2)), jnp.zeros((3, 1)), jnp.zeros((3, 2)), .99, .65)


def test_empty_rollout_fails():
    with pytest.raises(ValueError, match="nonempty"):
        lambda_return_targets(jnp.zeros(0), jnp.zeros(0), jnp.zeros(0), .99, .65)


def test_relabel_permutations_preserve_target_returns_by_hand():
    # t0: [A,B], t1: [B,A], t2: [A,B]. B's reward must never enter A's return.
    ids = jnp.array([[[7, 12]], [[12, 7]], [[7, 12]]])
    rewards = jnp.array([[[1., 10.]], [[20., 2.]], [[3., 30.]]])
    values = jnp.array([[[20., 200.]], [[300., 30.]], [[40., 400.]]])
    targets = object_lambda_return_targets(rewards, jnp.zeros_like(rewards, bool), values, ids, jnp.ones_like(rewards, bool), .9, .5)
    np.testing.assert_allclose(targets[:, 0], [[24.8725, 248.725], [330.5, 33.05], [39., 390.]], rtol=1e-6)


def test_missing_relabel_target_ends_trace_with_bootstrap():
    # A was sampled at t0 but not at t1: do not take the future reward 999.
    ids = jnp.array([[[7]], [[12]]])
    targets = object_lambda_return_targets(
        jnp.array([[[1.]], [[999.]]]), jnp.zeros((2, 1, 1), bool),
        jnp.array([[[20.]], [[40.]]]), ids, jnp.ones((2, 1, 1), bool), .9, .65,
    )
    np.testing.assert_allclose(targets[:, 0, 0], [19., 1035.], rtol=1e-6)


def test_invalid_future_row_is_not_used_as_trace_continuation():
    targets = object_lambda_return_targets(
        jnp.array([[[1.]], [[999.]]]), jnp.zeros((2, 1, 1), bool),
        jnp.array([[[20.]], [[40.]]]), jnp.full((2, 1, 1), 7),
        jnp.array([[[True]], [[False]]]), .9, .65,
    )
    assert float(targets[0, 0, 0]) == pytest.approx(19.)


@pytest.mark.parametrize("lam", [0., .5, .65, 1.])
def test_random_resampled_targets_against_forward_enumeration(lam):
    rng = np.random.default_rng(1525)
    time, envs, slots, k = 8, 3, 7, 4
    ids = np.stack([[rng.choice(slots, k, replace=False) for _ in range(envs)] for _ in range(time)])
    rewards = rng.integers(-4, 5, (time, envs, k)).astype(np.float32)
    values = rng.integers(-5, 8, (time, envs, k)).astype(np.float32)
    dones = rng.random(rewards.shape) < .2
    valid = rng.random(rewards.shape) > .15
    actual = jax.jit(object_lambda_return_targets)(rewards, dones, values, ids, valid, .9, lam)
    for t, env, row in np.ndindex(rewards.shape):
        identity = ids[t, env, row]
        r, v, d = [], [], []
        for future in range(t, time):
            matches = np.flatnonzero((ids[future, env] == identity) & valid[future, env])
            if future == t:
                pos = row
            elif matches.size:
                pos = int(matches[0])
            else:
                break
            r.append(rewards[future, env, pos])
            v.append(values[future, env, pos])
            d.append(dones[future, env, pos])
            if d[-1]:
                break
        expected = forward_view(r, d, v, .9, lam)[0]
        np.testing.assert_allclose(actual[t, env, row], expected, rtol=2e-6, atol=2e-6)


def test_matches_corrected_nexus_reference():
    path = Path(__file__).resolve().parents[1] / "third_party/symbolic_options/src/symbolic_options/hierarchical_pqn_jaxtari.py"
    if not path.exists():
        pytest.skip("NEXUS submodule is not checked out")
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "lambda_return_targets")
    namespace = {"jax": jax, "jnp": jnp}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    reference = namespace["lambda_return_targets"]
    rng = np.random.default_rng(491)
    r = rng.integers(-3, 4, (7, 3)).astype(np.float32)
    done = rng.random(r.shape) < .2
    states = rng.integers(-8, 9, (8, 3)).astype(np.float32)
    for lam in [0., .5, .65, 1.]:
        expected = reference(jnp.array(r), jnp.array(done), jnp.array(states[:-1]), jnp.array(states[-1]), .99, lam)
        actual = lambda_return_targets(r, done, states[1:], .99, lam)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
