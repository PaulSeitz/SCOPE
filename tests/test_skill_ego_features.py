import jax.numpy as jnp

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
)
from concept_based_learning.observations.skill_ego_features import (
    build_skill_ego_extras_stacked,
    configure_skill_ego_features,
    skill_ego_extra_channels_per_frame,
)


class _FakeProjectiles:
    def __init__(self, active):
        self.active = active


class _FakePlayer:
    def __init__(self, n_frames: int):
        self.x = jnp.zeros((n_frames,), dtype=jnp.float32)


class _FakeRawObs:
    def __init__(self, torpedo_active):
        n = torpedo_active.shape[0]
        self.player = _FakePlayer(n)
        active = jnp.zeros((n, 5), dtype=jnp.float32)
        active = active.at[:, 0].set(torpedo_active)
        self.projectiles = _FakeProjectiles(active)


def test_skill_ego_extra_channel_count():
    configure_skill_ego_features(
        enabled=True, num_skills=3, extra_dim_per_skill=4, env_name="seaquest"
    )
    assert skill_ego_extra_channels_per_frame() == 12


def test_seaquest_can_fire_in_destroy_block():
    configure_skill_ego_features(
        enabled=True, num_skills=3, extra_dim_per_skill=4, env_name="seaquest"
    )
    raw = _FakeRawObs(jnp.array([1.0, 0.0]))  # torpedo out, then ready
    extras = build_skill_ego_extras_stacked(raw)
    assert extras.shape == (2, 12)
    assert float(extras[0, 0]) == 0.0  # cannot fire
    assert float(extras[1, 0]) == 1.0  # can fire


def test_ego_obs_channels_include_velocity_and_skill_extras():
    configure_ego_obs_from_config(
        {
            "USE_LEGACY_EGO_OBS": False,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": True,
            "USE_SKILL_EGO_FEATURES": True,
            "SKILL_EGO_EXTRA_DIM": 4,
            "NUM_SKILLS": 3,
            "ENV_NAME": "seaquest",
        }
    )
    # distance + v_r + v_t + sin/cos + 3*4 skill extras
    assert ego_obs_channels_per_frame() == 5 + 12