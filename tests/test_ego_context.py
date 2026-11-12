"""Context ablations of the reference ego skill input (ego_scene_rel / ego_abs_pos)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concept_based_learning.networks.skill_network import QNetwork  # noqa: E402
from concept_based_learning.observations.ego_context import (  # noqa: E402
    EGO_ABS_POS,
    EGO_OC_NATIVE,
    EGO_SCENE_REL,
    EGO_SCENE_ROWS,
    apply_abs_pos_intervention,
    build_ego_context_skill_obs,
    build_ego_oc_native_obs,
    build_ego_scene_rows,
    ego_context_extra_dim,
)
from concept_based_learning.observations.egocentric import (  # noqa: E402
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    _get_player,
    ego_obs_channels_per_frame,
    get_ego_obs,
)

REF_DIR = ROOT / "configs" / "experiment" / "paper_eval" / "thesis_final"


def _env_obs(env_name: str, n_steps: int = 60, native_stack: bool = False):
    import main as M
    from concept_based_learning.environments.env_manager import (
        EnvManager,
        align_config_max_objects,
    )

    raw = yaml.safe_load((REF_DIR / f"tf_{env_name}_skills_ego_seed44.yaml").read_text())
    cfg = {**raw, **raw.pop("alg")}
    if native_stack:
        cfg["OC_NATIVE_STACK"] = True
    M._configure_ego_obs_format(cfg)
    env, _ = EnvManager().make_env(env_name, dict(cfg), train=True)
    align_config_max_objects(cfg, env, env_name=env_name)
    obs, st = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(1)
    n_act = env.action_space().n
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        obs, st = step(st, jax.random.randint(k, (), 0, n_act))[:2]
    return cfg, obs


def _shift(obs, dx: float, dy: float):
    """Translate the player and every object by (dx, dy) in screen pixels."""
    raw = obs.raw_observation
    name = next(n for n in ("player", "chicken", "agent", "skier") if hasattr(raw, n))
    p = getattr(raw, name)
    raw = raw.replace(**{name: p.replace(x=p.x + dx, y=p.y + dy)})
    o = obs.objects
    return raw, o.replace(x=o.x + dx, y=o.y + dy)


def _active_target(obs) -> int:
    act = np.asarray(obs.objects.active[-1])
    return int(np.flatnonzero(act > 0)[0])


@pytest.fixture(scope="module", params=["seaquest", "namethisgame"])
def env_case(request):
    return request.param, *_env_obs(request.param)


def test_reference_prefix_is_unchanged(env_case):
    _, cfg, obs = env_case
    t = _active_target(obs)
    ref = get_ego_obs(obs.raw_observation, obs.objects, t)
    assert ref.shape == (36,)
    for mode in (EGO_SCENE_REL, EGO_ABS_POS):
        x = build_ego_context_skill_obs(
            obs.raw_observation, obs.objects, t, mode=mode, max_objects=cfg["MAX_OBJECTS"]
        )
        np.testing.assert_array_equal(np.asarray(x[:36]), np.asarray(ref))
        extra = ego_context_extra_dim(
            mode,
            ego_channels_per_frame=ego_obs_channels_per_frame(),
            stack_size=cfg["FRAME_STACK_SIZE"],
            max_objects=cfg["MAX_OBJECTS"],
        )
        assert x.shape == (36 + extra,)


def test_abs_pos_tail_is_normalized_player_centre(env_case):
    _, cfg, obs = env_case
    x = build_ego_context_skill_obs(
        obs.raw_observation, obs.objects, 0, mode=EGO_ABS_POS, max_objects=cfg["MAX_OBJECTS"]
    )
    p = _get_player(obs.raw_observation)
    cx = (np.asarray(p.x) + np.asarray(p.width) / 2) / SCREEN_WIDTH
    cy = (np.asarray(p.y) + np.asarray(p.height) / 2) / SCREEN_HEIGHT
    tail = np.asarray(x[36:]).reshape(cfg["FRAME_STACK_SIZE"], 2)
    np.testing.assert_allclose(tail[:, 0], cx, rtol=1e-6)
    np.testing.assert_allclose(tail[:, 1], cy, rtol=1e-6)
    assert np.all((tail >= 0) & (tail <= 1.0))


def test_scene_rows_mask_inactive_and_mark_target(env_case):
    _, cfg, obs = env_case
    t = _active_target(obs)
    rows = np.asarray(
        build_ego_scene_rows(
            obs.raw_observation, obs.objects, t, max_objects=cfg["MAX_OBJECTS"]
        )
    )
    stack, ch = cfg["FRAME_STACK_SIZE"], ego_obs_channels_per_frame()
    assert rows.shape == (cfg["MAX_OBJECTS"], stack * (ch + 1) + 1)
    np.testing.assert_array_equal(rows[:, -1], np.eye(cfg["MAX_OBJECTS"])[t])
    framed = rows[:, :-1].reshape(cfg["MAX_OBJECTS"], stack, ch + 1)
    active = np.asarray(obs.objects.active).T > 0
    np.testing.assert_array_equal(framed[..., -1], active.astype(np.float32))
    assert np.all(framed[~active][:, :-1] == 0.0)
    # Active target row reproduces the reference per-frame encoding.
    ref = np.asarray(get_ego_obs(obs.raw_observation, obs.objects, t)).reshape(stack, ch)
    np.testing.assert_allclose(framed[t, active[t], :-1], ref[active[t]], rtol=1e-6)


def test_scene_rel_is_translation_invariant_abs_pos_is_not(env_case):
    """No explicit absolute coordinates: a rigid shift of the scene changes nothing."""
    _, cfg, obs = env_case
    t = _active_target(obs)
    raw2, obj2 = _shift(obs, 7.0, -5.0)
    kw = dict(max_objects=cfg["MAX_OBJECTS"])
    a = build_ego_context_skill_obs(obs.raw_observation, obs.objects, t, mode=EGO_SCENE_REL, **kw)
    b = build_ego_context_skill_obs(raw2, obj2, t, mode=EGO_SCENE_REL, **kw)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    a = build_ego_context_skill_obs(obs.raw_observation, obs.objects, t, mode=EGO_ABS_POS, **kw)
    b = build_ego_context_skill_obs(raw2, obj2, t, mode=EGO_ABS_POS, **kw)
    np.testing.assert_allclose(np.asarray(a[:36]), np.asarray(b[:36]), atol=1e-5)
    tail = (np.asarray(b[36:]) - np.asarray(a[36:])).reshape(-1, 2)
    np.testing.assert_allclose(tail[:, 0], 7.0 / SCREEN_WIDTH, atol=1e-5)
    np.testing.assert_allclose(tail[:, 1], -5.0 / SCREEN_HEIGHT, atol=1e-5)


@pytest.mark.parametrize("mode", [EGO_SCENE_REL, EGO_ABS_POS])
def test_qnetwork_forward_uses_reference_head(mode):
    width = 36 + ego_context_extra_dim(
        mode, ego_channels_per_frame=9, stack_size=4, max_objects=35
    )
    net = QNetwork(action_dim=18, hidden_size=128, num_layers=3, input_mode=mode)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros(width), train=False)
    q = net.apply(params, jnp.zeros((2, width)), train=False)
    assert q.shape == (2, 18)
    assert params["params"]["Dense_0"]["kernel"].shape == (width, 128)
    assert sorted(params["params"]) == [f"Dense_{i}" for i in range(4)] + [
        f"LayerNorm_{i}" for i in range(3)
    ]


def test_abs_pos_intervention_tail():
    tail = jnp.array([0.25, 0.5, 0.3, 0.6], dtype=jnp.float32)
    assert apply_abs_pos_intervention(tail, None) is tail
    np.testing.assert_allclose(
        apply_abs_pos_intervention(tail, "mirror_x"), [0.75, 0.5, 0.7, 0.6]
    )
    np.testing.assert_allclose(
        apply_abs_pos_intervention(tail, "frozen:0.1,0.2"), [0.1, 0.2, 0.1, 0.2]
    )
    with pytest.raises(ValueError):
        apply_abs_pos_intervention(tail, "nope")


@pytest.mark.parametrize("env_name", ["seaquest", "namethisgame"])
def test_ego_oc_native_is_reference_plus_native_input(env_name):
    from concept_based_learning.observations.native_oc_stack import (
        oc_native_skill_features,
    )

    cfg, obs = _env_obs(env_name, native_stack=True)
    t = _active_target(obs)
    x = build_ego_oc_native_obs(
        obs.raw_observation,
        obs.objects,
        t,
        obs.native_oc_vector,
        max_objects=cfg["MAX_OBJECTS"],
    )
    ref = get_ego_obs(obs.raw_observation, obs.objects, t)
    native = oc_native_skill_features(obs.native_oc_vector, t, cfg["MAX_OBJECTS"])
    assert x.shape == (36 + native.shape[0],)
    np.testing.assert_array_equal(np.asarray(x[:36]), np.asarray(ref))
    np.testing.assert_array_equal(np.asarray(x[36:]), np.asarray(native))


def test_ego_oc_native_requires_native_vector():
    with pytest.raises(ValueError):
        build_ego_oc_native_obs(None, None, 0, None, max_objects=35)


def test_qnetwork_accepts_ego_oc_native():
    width = 500
    net = QNetwork(action_dim=18, hidden_size=128, num_layers=3, input_mode=EGO_OC_NATIVE)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros(width), train=False)
    assert net.apply(params, jnp.zeros((2, width)), train=False).shape == (2, 18)


@pytest.mark.parametrize("env_name", ["seaquest", "namethisgame"])
def test_flat_scene_is_scene_rows_without_target_mark(env_name):
    from concept_based_learning.baselines.flat_oc_features import (
        FLAT_EGO_SCENE_REL,
        build_flat_oc_feature_vector,
    )

    cfg, obs = _env_obs(env_name)
    flat = build_flat_oc_feature_vector(
        obs.raw_observation, obs.objects, mode=FLAT_EGO_SCENE_REL, max_objects=cfg["MAX_OBJECTS"]
    )
    rows = build_ego_scene_rows(
        obs.raw_observation, obs.objects, 3, max_objects=cfg["MAX_OBJECTS"]
    )
    assert flat.shape == (cfg["MAX_OBJECTS"] * (rows.shape[1] - 1),)
    np.testing.assert_array_equal(np.asarray(flat), np.asarray(rows[:, :-1]).reshape(-1))


def test_flat_scene_env_uses_facing_actions():
    from concept_based_learning.baselines.flat_oc_features import (
        FLAT_EGO_SCENE_REL,
        make_flat_oc_build_env,
    )
    from concept_based_learning.environments.env_manager import EnvManager
    from concept_based_learning.observations.pov_relative import resolve_action_mappers

    raw = yaml.safe_load((REF_DIR / "tf_seaquest_skills_ego_seed44.yaml").read_text())
    cfg = {**raw, **raw.pop("alg")}
    env = make_flat_oc_build_env(
        mode=FLAT_EGO_SCENE_REL, max_objects=35, env_name="seaquest", config=cfg, train=False
    )
    assert env.action_space().n == 18
    obs, st = env.reset(jax.random.PRNGKey(0))
    assert obs.ndim == 1
    # Forward along the facing direction must match the skill mapper in both facings.
    mapper, _ = resolve_action_mappers(EnvManager().get_components("seaquest"), cfg)
    inner = st.atari_state
    a0 = int(mapper(jnp.int32(2), inner.obs_stack, 0, None))
    # The player only responds after the opening frames; BACKWARD then flips the facing.
    seen = {a0}
    for _ in range(40):
        obs, st, reward, term, trunc, info = env.step(st, jnp.int32(5))
        seen.add(int(mapper(jnp.int32(2), st.atari_state.obs_stack, 0, None)))
    assert len(seen) == 2


def test_scene_rows_is_scene_rel_without_prefix(env_case):
    _, cfg, obs = env_case
    t = _active_target(obs)
    kw = {"max_objects": cfg["MAX_OBJECTS"]}
    rows = build_ego_context_skill_obs(
        obs.raw_observation, obs.objects, t, mode=EGO_SCENE_ROWS, **kw
    )
    full = build_ego_context_skill_obs(
        obs.raw_observation, obs.objects, t, mode=EGO_SCENE_REL, **kw
    )
    assert rows.shape[0] == full.shape[0] - 36
    np.testing.assert_array_equal(np.asarray(rows), np.asarray(full[36:]))

