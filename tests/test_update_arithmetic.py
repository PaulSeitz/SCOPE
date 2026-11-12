"""Exercise actual trainer loss bodies against hand-derived tabular updates.

Load the production nested functions through AST extraction so these checks
test action indexing, loss scaling, and masks without duplicating that code
or compiling an entire game. Tabular outputs give analytic gradients.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "concept_based_learning"


def production_function(relative_path, name, namespace, occurrence=0):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    functions.sort(key=lambda n: n.lineno)
    node = functions[occurrence]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class TableNetwork:
    def apply(self, variables, obs, *args, mutable=None, **kwargs):
        del args, kwargs
        params = variables["params"]
        if isinstance(params, dict):
            params = params["q"]
        if mutable is None:
            return params[obs]
        return params, {"batch_stats": {}}


@pytest.mark.parametrize("trainer", ["train_skills.py", "train_tandem.py"])
@pytest.mark.parametrize("off_policy", [True, False])
@pytest.mark.parametrize("n_objects", [1, 3])
def test_skill_loss_and_gradient_match_executed_actions(trainer, off_policy, n_objects):
    rng = np.random.default_rng(9143 + n_objects)
    batch, actions = 7, 18
    q = rng.integers(-8, 9, (batch, n_objects, actions)).astype(np.float32) / 4
    chosen = rng.integers(0, actions, (batch, n_objects))
    targets = rng.integers(-8, 9, (batch, n_objects)).astype(np.float32) / 4
    valid = rng.random((batch, n_objects)) > .2
    active = np.array([0, 1, 2, 1, 0, 2, 1])
    object_ids = np.broadcast_to(np.arange(n_objects) + 10, (batch, n_objects))
    selected = object_ids[:, 0]
    state = SimpleNamespace(batch_stats={}, n_updates=0)
    minibatch = {
        "ego_obs_all": jnp.zeros((batch, n_objects, 2)),
        "ego_action_all": jnp.asarray(chosen),
        "object_idx_all": jnp.asarray(object_ids),
        "target_idx": jnp.asarray(selected),
        "relabel_mask": jnp.asarray(valid),
        "active_skill": jnp.asarray(active),
    }
    namespace = dict(
        jnp=jnp, jax=jax, config={"OFF_POLICY": off_policy},
        minibatch=minibatch, target=jnp.asarray(targets), skill_idx=1,
        skill_network=TableNetwork(), debug_inputs=False, use_pixel_skill=False,
        use_object_skill=False, use_oc_psi=False, _ego=lambda x: x,
        train_state=state, s_state_phase=state,
    )
    loss_fn = production_function("training/" + trainer, "_loss_fn", namespace, occurrence=0 if trainer == "train_skills.py" else 1)
    (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(jnp.asarray(q))
    eligible = valid.copy()
    if not off_policy:
        eligible &= active[:, None] == 1
        if n_objects > 1:
            eligible &= object_ids == selected[:, None]
    denominator = eligible.sum() + 1e-8
    expected_gradient = np.zeros_like(q)
    expected_loss = 0.
    for b, o in np.ndindex(batch, n_objects):
        if eligible[b, o]:
            error = q[b, o, chosen[b, o]] - targets[b, o]
            expected_loss += 0.5 * float(error * error) / denominator
            expected_gradient[b, o, chosen[b, o]] = error / denominator
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(grad, expected_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("trainer,scale", [("flat", .5), ("meta", .5)])
def test_flat_and_meta_losses_select_correct_output(trainer, scale):
    q = np.arange(30, dtype=np.float32).reshape(5, 6) / 10
    chosen = np.array([4, 0, 5, 1, 3])
    targets = np.array([1., -1., 2., 0., 1.])
    valid = np.array([True, True, False, True, True]) if trainer == "meta" else np.ones(5, bool)
    minibatch = SimpleNamespace(obs=jnp.arange(5), action=jnp.asarray(chosen))
    state = SimpleNamespace(batch_stats={})
    namespace = dict(jax=jax, jnp=jnp, minibatch=minibatch, target=jnp.asarray(targets), train_state=state, network=TableNetwork())
    if trainer == "meta":
        namespace.update(
            minibatch={"obs": jnp.arange(5), "target_idx": jnp.asarray(chosen),
                       "objects": jnp.zeros(5), "pixel_obs": jnp.zeros(5),
                       "meta_target_locked": jnp.zeros(5),
                       "active_mask": jnp.asarray(valid[:, None])},
            meta_network=TableNetwork(),
        )
    path = "baselines/flat_pqn_train.py" if trainer == "flat" else "training/train_tandem.py"
    fn = production_function(path, "_loss_fn", namespace)
    (loss, _), grad = jax.value_and_grad(fn, has_aux=True)(jnp.asarray(q))
    errors = q[np.arange(5), chosen] - targets
    denominator = valid.sum() + (1e-8 if trainer == "meta" else 0)
    expected = np.zeros_like(q)
    expected[np.arange(5), chosen] = 2 * scale * errors * valid / denominator
    np.testing.assert_allclose(loss, scale * np.sum(errors ** 2 * valid) / denominator, rtol=1e-6)
    np.testing.assert_allclose(grad, expected, rtol=1e-6, atol=1e-7)


def test_two_radam_updates_and_global_clipping_by_hand():
    # Early RAdam steps use the bias-corrected first moment (rho < 5).
    tx = optax.chain(optax.clip_by_global_norm(2.), optax.radam(.1))
    p = jnp.array([1., -2.])
    state = tx.init(p)
    update, state = tx.update(jnp.array([3., 4.]), state, p)
    p = optax.apply_updates(p, update)
    # norm=5 -> clipped gradient [1.2, 1.6]. First moment / (1-.9).
    np.testing.assert_allclose(p, [.88, -2.16], rtol=2e-6)
    update, state = tx.update(jnp.array([-3., 0.]), state, p)
    p = optax.apply_updates(p, update)
    # Next clipped gradient [-2,0]; m=.9*[.12,.16]+.1*[-2,0].
    m = np.array([-.092, .144])
    np.testing.assert_allclose(p, np.array([.88, -2.16]) - .1 * m / .19, rtol=2e-6)


@pytest.mark.parametrize("trainer", ["train_skills.py", "train_tandem.py"])
def test_minibatch_shuffle_keeps_fields_and_targets_aligned(trainer):
    namespace = dict(jax=jax, jnp=jnp, config={"NUM_MINIBATCHES": 3})
    fn = production_function("training/" + trainer, "preprocess_transition", namespace)
    ids = jnp.arange(12).reshape(4, 3)
    key = jax.random.PRNGKey(381)
    shuffled = fn(ids, key)
    target = fn(ids * 7 + 3, key)
    np.testing.assert_array_equal(target, shuffled * 7 + 3)
    np.testing.assert_array_equal(np.sort(np.asarray(shuffled).ravel()), np.arange(12))


@pytest.mark.parametrize("level", ["skill", "meta"])
@pytest.mark.parametrize("frozen", [True, False])
def test_frozen_tandem_preserves_nonzero_optimizer_momentum(level, frozen):
    from concept_based_learning.networks.skill_network import CustomTrainState

    params = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    if level == "skill":
        params = params[:, None, :]
    state = CustomTrainState.create(apply_fn=TableNetwork().apply, params={"q": params},
                                    batch_stats={}, tx=optax.radam(.1))
    # Prime RAdam momentum so a zero-gradient freeze would demonstrably move.
    state = state.apply_gradients(grads={"q": jnp.ones_like(params)})
    mb = {"ego_obs_all": jnp.zeros((3, 1, 2)), "object_idx_all": jnp.zeros((3, 1), int),
          "ego_action_all": jnp.array([[2], [1], [0]]), "relabel_mask": jnp.ones((3, 1), bool),
          "active_skill": jnp.zeros(3, int), "target_idx": jnp.array([2, 1, 0]),
          "obs": jnp.arange(3), "objects": jnp.zeros(3), "pixel_obs": jnp.zeros(3),
          "meta_target_locked": jnp.zeros(3), "active_mask": jnp.ones((3, 4), bool)}
    ns = dict(jax=jax, jnp=jnp, config={"OFF_POLICY": True}, skill_idx=0,
              skills_frozen=frozen, skip_meta_update=frozen, debug_inputs=False,
              skill_network=TableNetwork(), meta_network=TableNetwork(),
              use_pixel_skill=False, use_object_skill=False)
    fn = production_function("training/train_tandem.py", "_learn_phase" if level == "skill" else "_meta_learn_phase", ns)
    target = jnp.zeros((3, 1)) if level == "skill" else jnp.zeros(3)
    (updated, _), _ = fn((state, jax.random.PRNGKey(0)), (mb, target))
    if frozen:
        for old, new in zip(jax.tree.leaves(state), jax.tree.leaves(updated)):
            np.testing.assert_array_equal(old, new)
    else:
        assert not np.array_equal(state.params["q"], updated.params["q"])
        assert updated.step == state.step + 1


@pytest.mark.parametrize("trainer", ["train_skills.py", "train_tandem.py"])
def test_shared_linear_weights_sum_gradients_from_multiple_transitions(trainer):
    class LinearNetwork:
        def apply(self, variables, obs, **kwargs):
            return obs @ variables["params"], {"batch_stats": {}}

    # All samples contribute to the same weight matrix, including repeated
    # action indices. Compute the weight-gradient sum by scalar arithmetic.
    x = np.array([[[1., 2.]], [[-1., 3.]], [[2., -2.]]], np.float32)
    weights = np.array([[.5, 1., -.5], [.25, -.5, 2.]], np.float32)
    actions = np.array([[1], [1], [2]])
    targets = np.array([[2.], [-1.], [3.]], np.float32)
    mb = {"ego_obs_all": jnp.array(x), "object_idx_all": jnp.zeros((3, 1), int),
          "ego_action_all": jnp.array(actions), "active_skill": jnp.array([0, 1, 2]),
          "target_idx": jnp.zeros(3, int), "relabel_mask": jnp.ones((3, 1), bool)}
    state = SimpleNamespace(batch_stats={}, n_updates=0)
    ns = dict(jax=jax, jnp=jnp, config={"OFF_POLICY": True}, skill_idx=1,
              debug_inputs=False, use_pixel_skill=False, use_object_skill=False,
              use_oc_psi=False, skill_network=LinearNetwork(), minibatch=mb,
              target=jnp.array(targets), train_state=state, s_state_phase=state,
              _ego=lambda a: a)
    fn = production_function("training/" + trainer, "_loss_fn", ns, occurrence=0 if trainer == "train_skills.py" else 1)
    (_, _), grad = jax.value_and_grad(fn, has_aux=True)(jnp.array(weights))
    expected = np.zeros_like(weights)
    for b in range(3):
        action = actions[b, 0]
        predicted = sum(float(x[b, 0, f] * weights[f, action]) for f in range(2))
        for feature in range(2):
            expected[feature, action] += x[b, 0, feature] * (predicted - targets[b, 0]) / 3
    np.testing.assert_allclose(grad, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("typed", [False, True])
def test_tandem_regions_have_distinct_reproducible_rng_streams(typed):
    ns = dict(jax=jax)
    fn = production_function("training/train_tandem.py", "split_tandem_update_rng", ns)
    key = jax.random.key(44) if typed else jax.random.PRNGKey(44)
    seed_keys = jax.random.split(key, 3)
    first = fn(seed_keys)
    again = fn(seed_keys)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(jax.random.key_data(a), jax.random.key_data(b))
    # Three seeds times four regions; no shared root or reused update key.
    raw = np.stack([np.asarray(jax.random.key_data(k)) for k in first]).reshape(-1, 2)
    assert len(np.unique(raw, axis=0)) == 12
    second = np.stack([np.asarray(jax.random.key_data(k)) for k in fn(first[0])]).reshape(-1, 2)
    assert len(np.unique(np.concatenate([raw, second]), axis=0)) == 24
