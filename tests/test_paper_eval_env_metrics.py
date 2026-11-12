"""Unit tests for paper_eval env-aligned metric helpers."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.paper_eval.csv_export import results_to_episode_rows, results_to_summary_rows
from concept_based_learning.paper_eval.env_metrics import (
    EPISODE_METRIC_KEYS,
    aggregate_episode_metrics,
    init_episode_metrics,
    make_metric_fns,
    metrics_to_python,
)


def test_init_episode_metrics_keys():
    m = init_episode_metrics()
    assert set(m) == set(EPISODE_METRIC_KEYS)
    assert float(m["env_chicken_y_min"]) > 1e5


def test_metrics_to_python_sentinel():
    m = init_episode_metrics()
    out = metrics_to_python(m)
    assert out["env_kills"] == 0.0
    assert out["env_chicken_y_min"] != out["env_chicken_y_min"]  # nan


def test_seaquest_update_from_fake_state():
    class GS:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Wrap:
        def __init__(self, gs):
            self.env_state = gs

    class Objects:
        def __init__(self):
            # [stack=1, objects]: noop + shark killed + diver collected
            self.was_killed = jnp.array([[0, 1, 0]], dtype=jnp.int32)
            self.was_collected = jnp.array([[0, 0, 1]], dtype=jnp.int32)

    class Obs:
        def __init__(self):
            self.objects = Objects()

    update, finalize = make_metric_fns("Seaquest", has_semantic_obs=True)
    metrics = init_episode_metrics()
    prev = Wrap(GS(lives=3, score=0, divers_collected=0, successful_rescues=0, just_surfaced=0))
    nxt = Wrap(
        GS(lives=2, score=40, divers_collected=2, successful_rescues=1, just_surfaced=1)
    )
    metrics = update(metrics, prev, nxt, Obs(), jnp.float32(1.0))
    metrics = finalize(metrics, nxt)
    py = metrics_to_python(metrics)
    assert py["env_kills"] == 1.0
    assert py["env_collects"] == 1.0
    assert py["env_rescues"] == 1.0
    assert py["env_surface_events"] == 1.0
    assert py["env_divers_max"] == 2.0
    assert py["env_life_losses"] == 1.0
    assert py["env_score_final"] == 40.0
    assert py["env_positive_reward_steps"] == 1.0


def test_csv_includes_episode_metrics():
    payload = {
        "agent_kind": "skills",
        "env": "Seaquest",
        "results": {
            "base": {
                "mean": 10.0,
                "std": 0.0,
                "returns": [10.0],
                "episode_metrics": [
                    {"env_kills": 3.0, "env_rescues": 1.0, "env_steps": 100.0}
                ],
                "mean_env_kills": 3.0,
                "num_episodes": 1,
                "num_seeds": 1,
                "episodes_per_seed": 1,
                "category": "base",
                "mods": [],
            }
        },
    }
    summary = results_to_summary_rows(payload)
    assert summary[0]["mean_env_kills"] == 3.0
    episodes = results_to_episode_rows(payload)
    assert episodes[0]["env_kills"] == 3.0
    assert episodes[0]["env_rescues"] == 1.0


def test_aggregate_episode_metrics():
    agg = aggregate_episode_metrics(
        [
            {"env_kills": 1.0, "env_rescues": 0.0},
            {"env_kills": 3.0, "env_rescues": 2.0},
        ]
    )
    assert agg["mean_env_kills"] == 2.0
    assert agg["mean_env_rescues"] == 1.0
