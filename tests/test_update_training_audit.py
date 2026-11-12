"""Small real training runs with independent checks of runtime targets."""

import importlib
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from tests.test_lambda_returns import forward_view

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.slow
@pytest.mark.parametrize("arm", [
    "skills_ego", "skills_pixel_outline", "skills_oc_flat_rel", "skills_relabel_k",
    "tandem_oc", "tandem_pixel", "tandem_relabel_k", "pqn_oc", "pqn_pixel",
])
def test_final_recipe_training_targets_at_runtime(arm, monkeypatch, tmp_path):
    from main import outer_make_train, outer_make_tandem_train
    from concept_based_learning.baselines.flat_pqn_train import make_train
    from concept_based_learning.training.returns import lambda_return_targets, object_lambda_return_targets

    file_arm = {"skills_relabel_k": "skills_ego", "tandem_relabel_k": "tandem_oc"}.get(arm, arm)
    raw = yaml.safe_load((ROOT / "configs/experiment/paper_eval/thesis_final" / f"tf_seaquest_{file_arm}_seed44.yaml").read_text())
    alg = raw.pop("alg", {})
    cfg = {**raw, **alg, "NUM_SEEDS": 1, "NUM_ENVS": 2, "NUM_STEPS": 4,
           "NUM_MINIBATCHES": 2, "NUM_EPOCHS": 2, "META_NUM_EPOCHS": 2,
           "TOTAL_TIMESTEPS": 16, "TOTAL_TIMESTEPS_DECAY": 16,
           "WANDB_MODE": "disabled", "RTPT_ENABLED": False,
           "RECORD_VIDEO": False, "VIDEO_AT_END": False,
           "SAVE_TRAINING_METRICS_CSV": False, "TEST_DURING_TRAINING": False}
    if arm.endswith("relabel_k"):
        cfg.update(RELABEL_ALL_OBJECTS=True, RELABEL_K_OBJECTS=3)
    checked = []
    records = []
    sampling_rng = np.random.default_rng(20261024)

    def inspect(rewards, dones, values, targets, gamma, lam):
        rewards, dones, values, targets = map(np.asarray, (rewards, dones, values, targets))
        for coord in np.ndindex(rewards.shape[1:]):
            idx = (slice(None), *coord)
            expected = forward_view(rewards[idx], dones[idx], values[idx], gamma, lam)
            np.testing.assert_allclose(targets[idx], expected, rtol=3e-6, atol=3e-6)
        checked.append(rewards.shape)
        coord = tuple(int(sampling_rng.integers(n)) for n in rewards.shape[1:])
        idx = (slice(None), *coord)
        records.append({"kind": "standard", "coordinate": list(coord),
                        "rewards": rewards[idx].tolist(), "dones": dones[idx].tolist(),
                        "next_values": values[idx].tolist(), "targets": targets[idx].tolist(),
                        "gamma": float(gamma), "lambda": float(lam),
                        "independent_forward_view": forward_view(rewards[idx], dones[idx], values[idx], gamma, lam).tolist()})

    def verified(rewards, dones, values, gamma, lam):
        targets = lambda_return_targets(rewards, dones, values, gamma, lam)
        jax.debug.callback(inspect, rewards, dones, values, targets, gamma, lam)
        return targets

    def inspect_objects(rewards, dones, values, ids, valid, targets, gamma, lam):
        rewards, dones, values, ids, valid, targets = map(np.asarray, (rewards, dones, values, ids, valid, targets))
        for t, env, row in np.ndindex(rewards.shape):
            identity = ids[t, env, row]
            r, v, d = [], [], []
            for future in range(t, rewards.shape[0]):
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
            expected = forward_view(r, d, v, gamma, lam)[0]
            np.testing.assert_allclose(targets[t, env, row], expected, rtol=3e-6, atol=3e-6)
        checked.append(rewards.shape)
        env = int(sampling_rng.integers(rewards.shape[1]))
        row = int(sampling_rng.integers(rewards.shape[2]))
        records.append({"kind": "sampled_objects", "environment": env, "selected_row": row,
                        "object_ids": ids[:, env].tolist(), "valid": valid[:, env].tolist(),
                        "rewards": rewards[:, env].tolist(), "dones": dones[:, env].tolist(),
                        "next_values": values[:, env].tolist(), "targets": targets[:, env].tolist(),
                        "gamma": float(gamma), "lambda": float(lam)})

    def verified_objects(rewards, dones, values, ids, valid, gamma, lam):
        targets = object_lambda_return_targets(rewards, dones, values, ids, valid, gamma, lam)
        jax.debug.callback(inspect_objects, rewards, dones, values, ids, valid, targets, gamma, lam)
        return targets

    module_name = ("training.train_tandem" if arm.startswith("tandem") else
                   "baselines.flat_pqn_train" if arm.startswith("pqn") else "training.train_skills")
    module = importlib.import_module("concept_based_learning." + module_name)
    monkeypatch.setattr(module, "lambda_return_targets", verified)
    if hasattr(module, "object_lambda_return_targets"):
        monkeypatch.setattr(module, "object_lambda_return_targets", verified_objects)
    key = jax.random.PRNGKey(44)
    if arm.startswith("tandem"):
        bundle, _ = outer_make_tandem_train(cfg)
        keys = jax.random.split(key, 1)
        initial = jax.block_until_ready(bundle["init_fn"](keys, None, None))
        initial_meta_params = jax.tree.map(np.asarray, initial[0].params)
        out = jax.block_until_ready(bundle["run_training"](keys, None, None))
        assert any(np.any(a != np.asarray(b)) for a, b in zip(jax.tree.leaves(initial_meta_params), jax.tree.leaves(out["runner_state"][0].params)))
    elif arm.startswith("pqn"):
        fn = make_train(cfg)
        out = jax.block_until_ready(jax.jit(fn)(key))
    else:
        fn, _, _ = outer_make_train(cfg)
        out = jax.block_until_ready(jax.jit(fn)(key, None, None))
    jax.effects_barrier()
    assert checked, "the production return helper was never executed"
    for name, value in out["metrics"].items():
        if "td_loss" in name:
            assert np.isfinite(np.asarray(value)).all(), name
    report_dir = Path(os.environ.get("CBL_UPDATE_AUDIT_REPORT_DIR", str(tmp_path)))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{arm}.json").write_text(json.dumps(
        {"arm": arm, "training_seed": 44, "sampling_seed": 20261024,
         "checked_return_batches": len(checked), "samples": records}, indent=2) + "\n")
