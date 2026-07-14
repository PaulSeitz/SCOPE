"""Parity and compile smoketests for the tandem JIT split plan.

See ``docs/tandem_jit_split_plan.md`` for region boundaries and the target API.

When ``make_tandem_train`` returns the Phase 6 region bundle
``{collect_jit, meta_jit, skills_jit, init_fn, run_training, compile_regions}``,
tests exercise split deterministic replay, per-region compile smoke, and gates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import outer_make_tandem_train  # noqa: E402

SPLIT_REGION_KEYS = ("collect_jit", "meta_jit", "skills_jit", "init_fn")

KEY_LOSS_METRICS = (
    "meta_policy/td_loss",
    "skill/td_loss_mean",
)

PARAM_RTOL = 1e-4
PARAM_ATOL = 1e-5
METRIC_RTOL = 1e-4
METRIC_ATOL = 1e-5

def _load_experiment_config(name: str, *, subdir: str = "") -> dict:
    base = ROOT / "configs" / "experiment"
    path = (base / subdir / f"{name}.yaml") if subdir else (base / f"{name}.yaml")
    with path.open() as f:
        raw = yaml.safe_load(f)
    alg = raw.pop("alg", {})
    return {**raw, **alg}


def _parity_overrides(**extra: Any) -> dict[str, Any]:
    return {
        "WANDB_MODE": "disabled",
        "RTPT_ENABLED": False,
        "RECORD_VIDEO": False,
        "NUM_SEEDS": 1,
        "SEED": 42,
        "NUM_ENVS": 4,
        "NUM_STEPS": 4,
        "NUM_MINIBATCHES": 4,
        "NUM_EPOCHS": 1,
        "META_NUM_EPOCHS": 1,
        "TOTAL_TIMESTEPS": 64,  # NUM_ENVS * NUM_STEPS
        "CURRICULUM_ENABLED": False,
        **extra,
    }


def _is_split_bundle(obj: Any) -> bool:
    return isinstance(obj, dict) and all(key in obj for key in SPLIT_REGION_KEYS)


def _resolve_tandem_api(config: dict) -> Any:
    train_api, _ = outer_make_tandem_train(config)
    return train_api


def _has_accelerator() -> bool:
    try:
        return any(
            d.platform in ("gpu", "cuda", "rocm", "tpu")
            for d in jax.devices()
        )
    except Exception:
        return False


def _require_accelerator_or_skip(reason: str) -> None:
    if not _has_accelerator():
        pytest.skip(f"No JAX accelerator available: {reason}")


def _final_tree_leaf(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x[-1] if getattr(x, "ndim", 0) > 0 else x, tree)


def _extract_snapshot(out: dict[str, Any]) -> dict[str, Any]:
    """Snapshot comparable artifacts after a full tandem run."""
    meta_state, skill_states, *_rest = out["runner_state"]
    final_meta = _final_tree_leaf(meta_state)
    final_skills = _final_tree_leaf(skill_states)
    metrics = out["metrics"]

    loss_metrics = {}
    for key in KEY_LOSS_METRICS:
        if key not in metrics:
            continue
        val = metrics[key]
        arr = np.asarray(val[-1] if getattr(val, "ndim", 0) > 0 else val)
        loss_metrics[key] = float(arr.reshape(-1)[0])

    meta_n = np.asarray(jax.device_get(final_meta.n_updates)).reshape(-1)[0]
    meta_ts = np.asarray(jax.device_get(final_meta.timesteps)).reshape(-1)[0]
    return {
        "meta_params": final_meta.params,
        "skill_params": final_skills.params,
        "meta_n_updates": int(meta_n),
        "meta_timesteps": int(meta_ts),
        "loss_metrics": loss_metrics,
    }


def _assert_params_allclose(a: Any, b: Any, *, label: str) -> None:
    def _check(path, x, y):
        if not jnp.allclose(x, y, rtol=PARAM_RTOL, atol=PARAM_ATOL, equal_nan=True):
            diff = float(jnp.max(jnp.abs(x - y)))
            raise AssertionError(f"{label} param mismatch at {path}: max abs diff={diff}")

    jax.tree_util.tree_map_with_path(_check, a, b)


def _assert_snapshots_allclose(left: dict[str, Any], right: dict[str, Any], *, label: str) -> None:
    _assert_params_allclose(left["meta_params"], right["meta_params"], label=f"{label} meta")
    _assert_params_allclose(left["skill_params"], right["skill_params"], label=f"{label} skill")
    assert left["meta_n_updates"] == right["meta_n_updates"], (
        f"{label}: meta n_updates {left['meta_n_updates']} != {right['meta_n_updates']}"
    )
    assert left["meta_timesteps"] == right["meta_timesteps"], (
        f"{label}: meta timesteps {left['meta_timesteps']} != {right['meta_timesteps']}"
    )
    for key in KEY_LOSS_METRICS:
        if key not in left["loss_metrics"] or key not in right["loss_metrics"]:
            continue
        lv, rv = left["loss_metrics"][key], right["loss_metrics"][key]
        assert np.allclose(lv, rv, rtol=METRIC_RTOL, atol=METRIC_ATOL, equal_nan=True), (
            f"{label}: metric {key} {lv} != {rv}"
        )


def _run_split(config: dict, *, seed: int = 42) -> dict[str, Any]:
    bundle = _resolve_tandem_api(config)
    if not _is_split_bundle(bundle):
        pytest.skip("Split JIT API not implemented yet (Phase 6)")
    run_training = bundle.get("run_training")
    if run_training is None:
        pytest.skip("Split bundle missing run_training")
    rng = jax.random.split(jax.random.PRNGKey(seed), config["NUM_SEEDS"])
    return jax.block_until_ready(run_training(rng, None, None))


def _compile_split_regions(bundle: dict[str, Any], config: dict, *, seed: int = 0) -> None:
    """Lower/compile all split regions once via bundle.compile_regions."""
    compile_regions = bundle.get("compile_regions")
    if compile_regions is None:
        pytest.skip("Split bundle missing compile_regions")
    rng = jax.random.split(jax.random.PRNGKey(seed), config["NUM_SEEDS"])
    compile_regions(rng, None, None)


@pytest.mark.slow
def test_tandem_split_deterministic_replay():
    """Fixed seed → identical params and key losses across two split runs."""
    config = {**_load_experiment_config("seaquest_tandem"), **_parity_overrides()}
    train_api = _resolve_tandem_api(config)
    if not _is_split_bundle(train_api):
        pytest.skip("Split JIT API required")

    snap_a = _extract_snapshot(_run_split(config, seed=42))
    snap_b = _extract_snapshot(_run_split(config, seed=42))
    _assert_snapshots_allclose(snap_a, snap_b, label="split replay")


@pytest.mark.slow
def test_tandem_split_monolith_parity():
    """Monolith vs split paths match when monolith_train is retained in the bundle."""
    config = {**_load_experiment_config("seaquest_tandem"), **_parity_overrides()}
    train_api = _resolve_tandem_api(config)
    if not _is_split_bundle(train_api):
        pytest.skip("Split JIT API required")

    split_snap = _extract_snapshot(_run_split(config, seed=42))
    monolith_fn = train_api.get("monolith_train")
    if monolith_fn is None or not callable(monolith_fn):
        pytest.skip("monolith_train not retained in bundle for parity check")

    rng = jax.random.split(jax.random.PRNGKey(42), config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(monolith_fn, in_axes=(0, None, None)))
    mono_out = jax.block_until_ready(train_vjit(rng, None, None))
    mono_snap = _extract_snapshot(mono_out)
    _assert_snapshots_allclose(mono_snap, split_snap, label="monolith vs split")


@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "TODO(Phase 1): expose standalone collect_rollout / R1 jit for isolation. "
        "Compare transitions env_reward, done, meta_reward, skill_rewards_all."
    )
)
def test_r1_collect_rollout_region_isolation():
    """Same rng + pre-update state → identical TandemTransition buffer from R1 only."""
    config = {**_load_experiment_config("seaquest_tandem"), **_parity_overrides()}
    train_api = _resolve_tandem_api(config)
    if not _is_split_bundle(train_api):
        pytest.skip("Split JIT API required for R1 isolation")

    collect_jit = train_api["collect_jit"]
    init_fn = train_api["init_fn"]
    rng = jax.random.PRNGKey(42)
    state_a = init_fn(rng, None, None)
    state_b = init_fn(rng, None, None)

    _, transitions_a = collect_jit(state_a)
    _, transitions_b = collect_jit(state_b)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), transitions_a, transitions_b)
    )


@pytest.mark.slow
def test_tandem_split_region_compile_smoke():
    """All split regions lower/compile once on tiny tandem config."""
    config = {**_load_experiment_config("seaquest_tandem"), **_parity_overrides()}
    train_api = _resolve_tandem_api(config)
    if not _is_split_bundle(train_api):
        pytest.skip("Split JIT API not implemented yet (Phase 6)")

    _compile_split_regions(train_api, config)


@pytest.mark.slow
def test_tandem_dual_pixel_compile_smoke():
    """Compile smoke on dual-pixel ablation config (heavy; prefers accelerator)."""
    _require_accelerator_or_skip("dual-pixel compile smoke")
    config = {
        **_load_experiment_config(
            "seaquest_mi_pixel_embed_si_pixel_embed",
            subdir="input_ablation",
        ),
        **_parity_overrides(),
    }
    train_api = _resolve_tandem_api(config)
    if not _is_split_bundle(train_api):
        pytest.skip("Split JIT API required for dual-pixel compile smoke")
    _compile_split_regions(train_api, config)


@pytest.mark.slow
@pytest.mark.parametrize(
    "gate_overrides",
    [
        pytest.param({"SKILL_WARMUP_UPDATES": 1}, id="skill_warmup"),
        pytest.param({"SKILL_FREEZE_AFTER_UPDATES": 1}, id="skill_freeze"),
        pytest.param({"RELABEL_K_OBJECTS": 5}, id="relabel_k"),
    ],
)
def test_tandem_training_gates_smoke(gate_overrides: dict[str, Any]):
    """Warmup, freeze, and relabel-k gates compile and run one tiny tandem rollout."""
    config = {
        **_load_experiment_config("seaquest_tandem"),
        **_parity_overrides(**gate_overrides),
    }
    if gate_overrides.get("SKILL_WARMUP_UPDATES"):
        config["META_LEARN_DURING_WARMUP"] = False
    if gate_overrides.get("RELABEL_K_OBJECTS") is not None:
        config["RELABEL_ALL_OBJECTS"] = True

    train_api = _resolve_tandem_api(config)
    if not _is_split_bundle(train_api):
        pytest.skip("Split JIT API required")
    out = _run_split(config, seed=42)

    snap = _extract_snapshot(out)
    assert snap["meta_n_updates"] == config["NUM_UPDATES"]
    assert snap["meta_timesteps"] == config["TOTAL_TIMESTEPS"]

    if "SKILL_WARMUP_UPDATES" in gate_overrides:
        assert snap["loss_metrics"].get("meta_policy/td_loss") is not None
    if "SKILL_FREEZE_AFTER_UPDATES" in gate_overrides:
        assert snap["loss_metrics"].get("skill/td_loss_mean") is not None
    if "RELABEL_K_OBJECTS" in gate_overrides:
        assert snap["loss_metrics"].get("skill/td_loss_mean") is not None
