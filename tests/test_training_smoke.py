"""Compile / one-step smoketests for training loops (wandb off, tiny rollouts)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import outer_make_train, outer_make_tandem_train  # noqa: E402


def _load_experiment_config(name: str) -> dict:
    path = ROOT / "configs" / "experiment" / f"{name}.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)
    alg = raw.pop("alg", {})
    return {**raw, **alg}


@pytest.mark.slow
def test_pov_turn_primitives_compile_and_one_step():
    """Persistent heading must survive the rollout carry, mapper and next-obs."""
    config = {
        **_load_experiment_config(
            "paper_eval/rq1/seaquest_skills_ego_pov_clip50_vel_psibins8_turn"
        ),
        **_smoke_overrides(),
    }
    assert config["USE_POV_TURN_PRIMITIVES"] is True
    assert config["EGO_ACTION_DIM"] == 6
    train_fn, _, _ = outer_make_train(config)
    train_vjit = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None)))
    rng = jax.random.split(jax.random.PRNGKey(0), config["NUM_SEEDS"])
    print(
        "Compiling turn-primitive train (first GPU compile for Seaquest POV can "
        "take several minutes; the step itself is fast once compiled)...",
        flush=True,
    )
    compiled = train_vjit.lower(rng, None, None).compile()
    out = jax.block_until_ready(compiled(rng, None, None))
    assert "runner_state" in out


@pytest.mark.slow
def test_skills_ego_embed_compile_and_one_step():
    """SKILL_INPUT_MODE=ego_embed: Dense projection of ego features before Q MLP."""
    config = {
        **_load_experiment_config("feature_eval/seaquest_skills_heuristic_ego_embed"),
        **_smoke_overrides(),
    }
    assert str(config.get("SKILL_INPUT_MODE", "")).lower().strip() == "ego_embed"
    train_fn, _, _ = outer_make_train(config)
    train_vjit = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None)))
    rng = jax.random.split(jax.random.PRNGKey(0), config["NUM_SEEDS"])
    compiled = train_vjit.lower(rng, None, None).compile()
    out = jax.block_until_ready(compiled(rng, None, None))
    assert "runner_state" in out


def _smoke_overrides() -> dict:
    return {
        "WANDB_MODE": "disabled",
        "RTPT_ENABLED": False,
        "RECORD_VIDEO": False,
        "NUM_SEEDS": 1,
        "SEED": 0,
        "NUM_ENVS": 4,
        "NUM_STEPS": 4,
        "NUM_MINIBATCHES": 4,
        "NUM_EPOCHS": 1,
        "TOTAL_TIMESTEPS": 64,  # NUM_ENVS * NUM_STEPS
        "CURRICULUM_ENABLED": False,
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "experiment",
    [
        "phoenix_skills",
        "spaceinvaders_skills",
    ],
)
def test_skills_training_compile_and_one_step(experiment: str):
    config = {**_load_experiment_config(experiment), **_smoke_overrides()}
    train_fn, _, _ = outer_make_train(config)
    train_vjit = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None)))
    rng = jax.random.split(jax.random.PRNGKey(0), config["NUM_SEEDS"])
    compiled = train_vjit.lower(rng, None, None).compile()
    out = jax.block_until_ready(compiled(rng, None, None))
    assert "runner_state" in out


@pytest.mark.slow
def test_tandem_training_compile_and_one_step():
    config = {**_load_experiment_config("seaquest_tandem"), **_smoke_overrides()}
    tandem_regions, _ = outer_make_tandem_train(config)
    rng = jax.random.split(jax.random.PRNGKey(0), config["NUM_SEEDS"])
    tandem_regions["compile_regions"](rng, None, None)
    out = jax.block_until_ready(tandem_regions["run_training"](rng, None, None))
    assert "runner_state" in out
    assert "metrics" in out
