"""Integration of focused mods with allowlist selection and semantic slots."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from concept_based_learning.paper_eval.mods import mod_specs_for_env
from concept_based_learning.environments.env_manager import EnvManager

GAMES = ("namethisgame", "timepilot", "spaceinvaders", "beamrider", "asteroids", "pong", "breakout", "freeway", "phoenix")


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("game", GAMES)
def test_focused_allowlist_selects_three_independent_conditions(game):
    specs = mod_specs_for_env(game + "_robustness_focus", paper_core_only=True)
    assert len(specs) == 3
    assert specs[0].key == "base" and not specs[0].mods
    assert [s.category for s in specs] == ["control", "layout", "simplification"]
    assert all(s.mods == (s.key,) for s in specs[1:])
    assert all(not s.relabel for s in specs)


@pytest.mark.parametrize("game,mod", [("freeway", "no_cars"), ("phoenix", "no_enemy_fire")])
def test_wrapped_simplification_has_consistent_active_objects(game, mod):
    env, _ = EnvManager().make_env(
        game,
        {"GAME_MODS": [mod], "FRAME_SKIP": 4, "FRAME_STACK_SIZE": 4, "FIRST_FIRE": False, "CLIP_REWARD": False},
        train=False,
    )
    obs, state = env.reset(jax.random.PRNGKey(2))
    obs, state, reward, done, _ = env.step(state, jnp.int32(0))
    if game == "freeway":
        origin = env.game_slot_origin
        assert not bool(jnp.any(obs.raw_observation.car.active))
        assert not bool(jnp.any(obs.objects.active[-1, origin:origin + 10]))
        assert bool(obs.objects.active[-1, origin + 10])  # crossing goal retained
    else:
        assert not bool(jnp.any(obs.raw_observation.enemy_projectiles.active))
    assert np.isfinite(float(reward))
