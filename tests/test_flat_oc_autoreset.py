"""Flat OC PQN sits on the semantic env, which has no auto-reset of its own.

Regression: without a reset in the flat wrapper every training env stayed stuck in its
finished game, reporting ``done`` on nearly every step and earning almost no reward.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concept_based_learning.baselines.flat_oc_features import (  # noqa: E402
    make_flat_oc_build_env,
)

CFG_DIR = ROOT / "configs/experiment/paper_eval/context_ablation"


def _rollout(env_name: str, *, n_envs: int, steps: int, train: bool = True):
    raw = yaml.safe_load((CFG_DIR / f"tf_{env_name}_pqn_ego_scenerel_seed44.yaml").read_text())
    cfg = {**raw, **raw.pop("alg", {})}
    env = make_flat_oc_build_env(
        mode=cfg["OC_INPUT_MODE"],
        max_objects=int(cfg.get("MAX_OBJECTS", 35)),
        env_name=env_name,
        config=cfg,
        train=train,
    )
    key = jax.random.PRNGKey(0)
    obs, state = jax.vmap(env.reset)(jax.random.split(key, n_envs))
    step = jax.jit(jax.vmap(env.step))
    n_actions = env.action_space().n
    done, reward, game_over = [], [], []
    for _ in range(steps):
        key, sub = jax.random.split(key)
        action = jax.random.randint(sub, (n_envs,), 0, n_actions)
        obs, state, r, term, trunc, info = step(state, action)
        done.append(np.asarray(term) | np.asarray(trunc))
        reward.append(np.asarray(r))
        game_over.append(np.asarray(info["env_done"]))
    return np.stack(done), np.stack(reward), np.stack(game_over)


@pytest.mark.slow
@pytest.mark.parametrize("env_name,steps", [("seaquest", 1500), ("namethisgame", 6000)])
def test_finished_games_restart(env_name, steps):
    done, reward, game_over = _rollout(env_name, n_envs=16, steps=steps)
    assert game_over.any(), "rollout too short to reach a game over"
    # Stuck envs reported done on 93.5-100% of the steps after their first done.
    rates = [
        done[int(np.argmax(done[:, i])) + 1 :, i].mean()
        for i in range(done.shape[1])
        if done[:, i].any()
    ]
    assert np.mean(rates) < 0.2
    # Every game over is a single step: the next step belongs to a new game.
    consecutive = game_over[1:] & game_over[:-1]
    assert not consecutive.any()
    # Rewards keep arriving after the first game over.
    first = int(np.argmax(game_over.any(axis=1)))
    assert reward[first + 1 :].sum() > 0
