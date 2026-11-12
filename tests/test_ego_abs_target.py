"""Absolute-only restricted target ablation, including real game observations."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from tests.test_ego_context import env_case, _active_target, _shift, REF_DIR
from concept_based_learning.observations.ego_context import (
    EGO_ABS_TARGET, build_ego_abs_target_obs, build_ego_context_skill_obs,
    player_abs_position_features,
)
from concept_based_learning.networks.skill_network import QNetwork


def test_absolute_target_geometry(env_case):
    _, cfg, obs = env_case
    t = _active_target(obs)
    o, raw = obs.objects, obs.raw_observation
    # Explicitly exercise a fully active slot, including its older frames.
    o = o.replace(active=o.active.at[:, t].set(1))
    x = build_ego_abs_target_obs(raw, o, t)
    assert x.shape == (36,)
    assert x.dtype == jnp.float32
    assert np.all((np.asarray(x) >= 0) & (np.asarray(x) <= 1))
    frames = np.asarray(x).reshape(4, 9)
    expected = np.stack([(o.x[:, t]+o.width[:, t]/2)/160,
                         (o.y[:, t]+o.height[:, t]/2)/210,
                         o.width[:, t]/160, o.height[:, t]/210], axis=-1)
    np.testing.assert_allclose(frames[:, :4], expected, atol=1e-7)
    np.testing.assert_array_equal(frames[:, 4], 1)
    np.testing.assert_allclose(frames[:, 5:7], player_abs_position_features(raw).reshape(4, 2))
    p = raw.player
    np.testing.assert_allclose(frames[:, 7:], np.stack([p.width/160, p.height/210], axis=-1))
    dispatched = build_ego_context_skill_obs(raw, o, t, mode=EGO_ABS_TARGET, max_objects=cfg['MAX_OBJECTS'])
    np.testing.assert_array_equal(x, dispatched)


def test_inactive_mask_and_no_other_objects(env_case):
    _, _, obs = env_case
    t = _active_target(obs)
    o = obs.objects.replace(active=obs.objects.active.at[:, t].set(jnp.array([1, 0, 1, 0])))
    frames = np.asarray(build_ego_abs_target_obs(obs.raw_observation, o, t)).reshape(4, 9)
    np.testing.assert_array_equal(frames[[1, 3]], 0)
    # A non-target object cannot affect this restricted input.
    other = (t + 1) % o.x.shape[-1]
    altered = o.replace(x=o.x.at[:, other].add(50), y=o.y.at[:, other].add(30))
    np.testing.assert_array_equal(build_ego_abs_target_obs(obs.raw_observation, altered, t), frames.reshape(-1))
    inactive = o.replace(active=o.active.at[:, t].set(0))
    np.testing.assert_array_equal(build_ego_abs_target_obs(obs.raw_observation, inactive, t), 0)


def test_shift_sensitivity_and_target_selection(env_case):
    _, _, obs = env_case
    o = obs.objects.replace(active=jnp.ones_like(obs.objects.active))
    raw2, o2 = _shift(obs.replace(objects=o), 7., -5.)
    a = np.asarray(build_ego_abs_target_obs(obs.raw_observation, o, 0)).reshape(4, 9)
    b = np.asarray(build_ego_abs_target_obs(raw2, o2, 0)).reshape(4, 9)
    assert not np.allclose(a, b)
    np.testing.assert_allclose((b-a)[:, [0, 5]], 7/160, atol=1e-7)
    np.testing.assert_allclose((b-a)[:, [1, 6]], -5/210, atol=1e-7)
    o = o.replace(x=o.x.at[:, 1].set(40), width=o.width.at[:, 1].set(8))
    selected = np.asarray(build_ego_abs_target_obs(obs.raw_observation, o, 1)).reshape(4, 9)
    np.testing.assert_allclose(selected[:, 0], 44/160)
    np.testing.assert_array_equal(selected[:, 5:], a[:, 5:])


def test_jit_vmap_and_network(env_case):
    _, cfg, obs = env_case
    build = jax.jit(jax.vmap(lambda t: build_ego_context_skill_obs(
        obs.raw_observation, obs.objects, t, mode=EGO_ABS_TARGET, max_objects=cfg['MAX_OBJECTS'])))
    x = build(jnp.array([0, 1]))
    assert x.shape == (2, 36)
    net = QNetwork(action_dim=18, hidden_size=128, num_layers=3, input_mode=EGO_ABS_TARGET)
    params = net.init(jax.random.PRNGKey(0), x[0], train=False)
    assert params['params']['Dense_0']['kernel'].shape == (36, 128)
    q = jax.jit(lambda p, features: net.apply(p, features, train=False))(params, x)
    assert q.shape == (2, 18)
    assert np.isfinite(q).all()


def test_bottom_edge_coordinates():
    box = SimpleNamespace(x=jnp.full(4, 20), y=jnp.full(4, 50),
                          width=jnp.full(4, 8), height=jnp.full(4, 10))
    objects = SimpleNamespace(**{k: v[:, None] for k, v in vars(box).items()}, active=jnp.ones((4, 1)))
    x = np.asarray(build_ego_abs_target_obs(SimpleNamespace(chicken=box), objects, 0)).reshape(4, 9)
    np.testing.assert_allclose(x[:, [1, 6]], 45/210)


def test_materialized_configs_preserve_scope(tmp_path):
    from scripts.materialize_abs_target_ablation import materialize
    paths = materialize(output=tmp_path)
    assert len(paths) == 10
    for path in paths:
        cfg = yaml.safe_load(path.read_text())
        ref = yaml.safe_load((REF_DIR / path.name.replace('_abstarget', '')).read_text())
        assert cfg['SKILL_INPUT_MODE'] == cfg['alg']['SKILL_INPUT_MODE'] == EGO_ABS_TARGET
        for key in ('SKILL_INPUT_MODE', 'SAVE_PATH', 'WANDB_RUN_NAME', 'WANDB_TAGS'):
            del cfg[key]
            del ref[key]
        del cfg['alg']['SKILL_INPUT_MODE']
        del ref['alg']['SKILL_INPUT_MODE']
        assert cfg == ref
