"""Numerical tests of production target-construction blocks with known Qs."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from concept_based_learning.training.returns import lambda_return_targets, object_lambda_return_targets
from concept_based_learning.training.skill_bootstrap import skill_network_q_max
from concept_based_learning.training.trajectory_cuts import (
    skill_target_switch_cut, skill_target_switch_cut_env_padded,
)
from tests.test_lambda_returns import forward_view

ROOT = Path(__file__).resolve().parents[1] / "src" / "concept_based_learning"


@dataclass
class State:
    params: object = None
    batch_stats: object = None
    timesteps: int = 0
    n_updates: int = 0

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class KnownQ:
    def apply(self, variables, obs, *args, **kwargs):
        del args, kwargs
        # The feature itself encodes known V(s). A distractor action is smaller.
        v = obs[..., 0]
        if isinstance(variables["params"], dict):
            v = v + variables["params"].get("offset", 0.)
        return jnp.stack([v - 3, v, v - 1], -1)


def target_prefix(path, function, target, namespace):
    """Compile unmodified production statements up to the optimizer phase."""
    source = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == function)
    prefix = []
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name in ("_learn_epoch", "_meta_learn_epoch"):
            break
        prefix.append(stmt)
    else:
        raise AssertionError("optimizer boundary missing")
    node.body = prefix + [ast.Return(value=ast.Name(id=target, ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(ROOT / path), "exec"), namespace)
    return namespace[function]


@pytest.mark.parametrize("trainer", ["train_skills.py", "train_tandem.py"])
@pytest.mark.parametrize("all_objects", [False, True])
@pytest.mark.parametrize("one_step", [False, True])
@pytest.mark.parametrize("time", [1, 5])
def test_skill_target_wiring_matches_independent_forward_view(trainer, all_objects, one_step, time):
    rng = np.random.default_rng(20261024 + time)
    envs, objects, skills = 2, 3 if all_objects else 1, 3
    rewards = rng.integers(-2, 3, (time, envs, objects, skills)).astype(np.float32)
    done = rng.random((time, envs)) < .2
    outcomes = rng.random(rewards.shape) < .15
    values = rng.integers(-4, 7, (time, envs, objects)).astype(np.float32)
    targets = np.array([[0, 1], [0, 1], [1, 1], [1, 2], [1, 2]])[:time]
    # Final row is deliberately nonterminal: must bootstrap at the boundary.
    done[-1] = False
    outcomes[-1] = False
    transitions = SimpleNamespace(
        target_idx=jnp.array(targets), object_idx_all=jnp.broadcast_to(jnp.arange(objects), (time, envs, objects)),
        done=jnp.array(done), skill_done_all=jnp.array(outcomes), skill_done=jnp.array(outcomes[:, :, 0]),
        rewards_all=jnp.array(rewards), rewards=jnp.array(rewards[:, :, 0]),
        skill_rewards_all=jnp.array(rewards), skill_rewards=jnp.array(rewards[:, :, 0]),
        next_ego_obs_all=jnp.array(values[..., None]), next_ego_obs=jnp.array(values[:, :, 0, None]),
        relabel_mask=jnp.ones((time, envs, objects), dtype=bool),
    )
    ns = dict(
        jax=jax, jnp=jnp, config={"NUM_STEPS": time, "NUM_ENVS": envs, "LAMBDA": .1},
        transitions=transitions, relabel_all_objects=all_objects,
        skill_bootstrap="one_step" if one_step else "q_lambda",
        skill_gamma=.9, skill_lambda=.65, skill_target_cut="global",
        skill_bootstrap_chunk_steps=2, skill_network=KnownQ(), skill_network_q_max=skill_network_q_max,
        use_pixel_skill=False, use_object_skill=False, use_oc_psi=False, pose_only=False,
        lambda_return_targets=lambda_return_targets,
        object_lambda_return_targets=object_lambda_return_targets,
        skill_target_switch_cut=skill_target_switch_cut,
        skill_target_switch_cut_env_padded=skill_target_switch_cut_env_padded,
    )
    is_tandem = trainer == "train_tandem.py"
    fn = target_prefix("training/" + trainer, "_update_single_skill" if is_tandem else "_update_skill", "skill_targets" if is_tandem else "train_targets", ns)
    for skill in range(skills):
        actual = np.asarray(fn(State(params={"offset": 10. * skill}), skill, jax.random.PRNGKey(0)))
        if not all_objects:
            actual = actual[..., 0]
        expected = np.empty((time, envs, objects))
        for env in range(envs):
            for obj in range(objects):
                terminals = outcomes[:, env, obj, skill] | done[:, env]
                terminals = terminals.copy()
                terminals[:-1] |= targets[:-1, env] != targets[1:, env]
                lam = 0 if one_step else .65
                expected[:, env, obj] = forward_view(
                    rewards[:, env, obj, skill], terminals, values[:, env, obj] + 10. * skill, .9, lam,
                )
        if not all_objects:
            expected = expected[..., 0]
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_meta_target_wiring_uses_next_time_and_valid_candidates():
    # s0..s3, two envs, three candidate actions; invalid Q=999 must not win.
    current_q = jnp.array([
        [[1., 999., 3.], [2., 4., 999.]],
        [[5., 999., 6.], [7., 8., 999.]],
        [[9., 999., 10.], [11., 12., 999.]],
    ])
    mask = jnp.array([[[True, False, True], [True, True, False]]] * 3)
    tr = SimpleNamespace(
        next_obs=jnp.array([[[0.], [0.]], [[0.], [0.]], [[13.], [14.]]]),
        next_objects=jnp.zeros((3, 2)), next_pixel_obs=jnp.zeros((3, 2)),
        active_mask=mask, next_active_mask=jnp.ones((3, 2, 3), dtype=bool),
        meta_q_val=current_q, meta_reward=jnp.array([[1., 2.], [3., 4.], [5., 6.]]),
        done=jnp.array([[False, False], [True, False], [False, False]]),
    )
    ns = dict(
        jax=jax, jnp=jnp, config={"NUM_STEPS": 3, "NUM_ENVS": 2, "GAMMA": .9, "LAMBDA": .5},
        skill_warmup_updates=0, meta_learn_during_warmup=False,
        compute_meta_target_locked=lambda *a, **kw: jnp.zeros(2, dtype=bool),
        no_target_idx=2, max_objects=3, meta_network=KnownQ(), lambda_return_targets=lambda_return_targets,
    )
    fn = target_prefix("training/train_tandem.py", "meta_bootstrap_and_train", "meta_lambda_targets", ns)
    actual = fn(State(), tr, jax.random.PRNGKey(0), (jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)))
    for env in range(2):
        next_v = [6., 10., 13.] if env == 0 else [8., 12., 14.]
        expected = forward_view(np.asarray(tr.meta_reward)[:, env], np.asarray(tr.done)[:, env], next_v, .9, .5)
        np.testing.assert_allclose(actual[:, env], expected, rtol=1e-6)


def test_tandem_terminal_transition_retains_executed_target():
    # Execute the actual reset assignment and Transition keyword expressions.
    tree = ast.parse((ROOT / "training/train_tandem.py").read_text())
    reset = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "carried_target_obj_idxs" for t in n.targets))
    namespace = dict(jnp=jnp, new_done=jnp.array([True, False]), target_obj_idxs=jnp.array([7, 9]), no_target_idx=35)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reset], type_ignores=[])), "target_reset", "exec"), namespace)
    transition = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "TandemTransition")
    expression = next(k.value for k in transition.keywords if k.arg == "target_idx")
    stored = eval(compile(ast.Expression(expression), "target_store", "eval"), namespace)
    np.testing.assert_array_equal(stored, [7, 9])
    np.testing.assert_array_equal(namespace["carried_target_obj_idxs"], [35, 9])


def test_flat_target_wiring_matches_next_state_values():
    path = ROOT / "baselines/flat_pqn_train.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_update_step")
    start = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "last_q" for t in n.targets))
    end = next(i for i, n in enumerate(node.body) if isinstance(n, ast.FunctionDef) and n.name == "_learn_epoch")
    v = jnp.array([[10., 11.], [20., 21.], [30., 31.]])
    q = jnp.stack([v - 10, v], -1)
    tr = SimpleNamespace(q_val=q, next_obs=jnp.array([[[20.], [21.]], [[30.], [31.]], [[40.], [41.]]]),
                         reward=jnp.array([[1., 2.], [3., 4.], [5., 6.]]),
                         done=jnp.array([[False, False], [False, True], [False, False]]))
    ns = dict(jnp=jnp, cfg={"GAMMA": .9, "LAMBDA": .65}, network=KnownQ(),
              train_state=State(), transitions=tr, lambda_return_targets=lambda_return_targets)
    exec(compile(ast.fix_missing_locations(ast.Module(body=node.body[start:end], type_ignores=[])), str(path), "exec"), ns)
    for env in range(2):
        expected = forward_view(np.asarray(tr.reward)[:, env], np.asarray(tr.done)[:, env], [20 + env, 30 + env, 40 + env], .9, .65)
        np.testing.assert_allclose(ns["lambda_targets"][:, env], expected, rtol=1e-6)
