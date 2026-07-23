import jax.numpy as jnp

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
)
from concept_based_learning.observations.skill_ego_features import (
    WEAPON_IN_FLIGHT_AIMED,
    WEAPON_IN_FLIGHT_WASTED,
    WEAPON_READY,
    build_skill_ego_extras_stacked,
    configure_skill_ego_features,
    latch_weapon_state,
    skill_ego_extra_channels_per_frame,
)


class _FakeProjectiles:
    def __init__(self, active):
        self.active = active


class _FakePlayer:
    def __init__(self, n_frames: int, orientation=0.0, x=10.0, width=4.0):
        self.x = jnp.full((n_frames,), x, dtype=jnp.float32)
        self.width = jnp.full((n_frames,), width, dtype=jnp.float32)
        self.orientation = jnp.full((n_frames,), orientation, dtype=jnp.float32)


class _FakeRawObs:
    def __init__(self, torpedo_active, orientation=0.0, x=10.0):
        n = torpedo_active.shape[0]
        self.player = _FakePlayer(n, orientation=orientation, x=x)
        active = jnp.zeros((n, 5), dtype=jnp.float32)
        active = active.at[:, 0].set(torpedo_active)
        self.projectiles = _FakeProjectiles(active)


class _FakeObjects:
    def __init__(self, n_frames, target_x, target_w=4.0, n_objects=2):
        self.x = jnp.zeros((n_frames, n_objects), dtype=jnp.float32)
        self.x = self.x.at[:, 0].set(target_x)
        self.width = jnp.full((n_frames, n_objects), target_w, dtype=jnp.float32)
        self.y = jnp.zeros((n_frames, n_objects), dtype=jnp.float32)
        self.height = jnp.ones((n_frames, n_objects), dtype=jnp.float32)
        self.active = jnp.ones((n_frames, n_objects), dtype=jnp.int32)


class _FakeSingleProjectile:
    def __init__(self, active):
        self.active = active


class _FakeSingleProjectileObs:
    def __init__(self, projectile_active):
        n = projectile_active.shape[0]
        self.player = _FakePlayer(n)
        self.player_projectile = _FakeSingleProjectile(projectile_active)


class _FakePlayerBulletObs:
    def __init__(self, bullet_active):
        n = bullet_active.shape[0]
        self.player = _FakePlayer(n)
        self.player_bullet = _FakeSingleProjectile(bullet_active)


class _FakeGroupedProjectileObs:
    def __init__(self, attr: str, slot_active):
        n = slot_active.shape[0]
        self.player = _FakePlayer(n)
        setattr(self, attr, _FakeProjectiles(slot_active))


def test_skill_ego_extra_channel_count():
    configure_skill_ego_features(enabled=True, env_name="seaquest")
    assert skill_ego_extra_channels_per_frame() == 1
    configure_skill_ego_features(enabled=False, env_name="seaquest")
    assert skill_ego_extra_channels_per_frame() == 0


def test_latch_weapon_state_ready_aimed_wasted():
    can_fire = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    aim_ok = jnp.array([True, True, False, True])
    ws = latch_weapon_state(can_fire, aim_ok)
    assert ws.tolist() == [
        WEAPON_READY,
        WEAPON_IN_FLIGHT_AIMED,
        WEAPON_IN_FLIGHT_AIMED,
        WEAPON_READY,
    ]

    can_fire = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    aim_bad = jnp.array([False, False, True])
    ws = latch_weapon_state(can_fire, aim_bad)
    assert ws.tolist() == [
        WEAPON_READY,
        WEAPON_IN_FLIGHT_WASTED,
        WEAPON_IN_FLIGHT_WASTED,
    ]


def test_seaquest_weapon_state_ready_and_aimed():
    configure_skill_ego_features(enabled=True, env_name="seaquest")
    raw = _FakeRawObs(
        jnp.array([0.0, 1.0, 1.0]),
        orientation=0.0,
        x=10.0,
    )
    objects = _FakeObjects(3, target_x=40.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert extras.shape == (3, 1)
    assert float(extras[0, 0]) == WEAPON_READY
    assert float(extras[1, 0]) == WEAPON_IN_FLIGHT_AIMED
    assert float(extras[2, 0]) == WEAPON_IN_FLIGHT_AIMED


def test_seaquest_weapon_state_wasted_when_facing_away():
    configure_skill_ego_features(enabled=True, env_name="seaquest")
    raw = _FakeRawObs(
        jnp.array([0.0, 1.0, 1.0]),
        orientation=180.0,
        x=10.0,
    )
    objects = _FakeObjects(3, target_x=40.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert float(extras[0, 0]) == WEAPON_READY
    assert float(extras[1, 0]) == WEAPON_IN_FLIGHT_WASTED
    assert float(extras[2, 0]) == WEAPON_IN_FLIGHT_WASTED


def test_seaquest_weapon_state_ignores_target_crossing_after_fire():
    configure_skill_ego_features(enabled=True, env_name="seaquest")
    raw = _FakeRawObs(
        jnp.array([0.0, 1.0, 1.0]),
        orientation=0.0,
        x=50.0,
    )
    objects = _FakeObjects(3, target_x=80.0)
    objects.x = objects.x.at[2, 0].set(10.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert float(extras[1, 0]) == WEAPON_IN_FLIGHT_AIMED
    assert float(extras[2, 0]) == WEAPON_IN_FLIGHT_AIMED


def test_phoenix_weapon_state_in_flight_is_aimed():
    configure_skill_ego_features(enabled=True, env_name="phoenix")
    raw = _FakeSingleProjectileObs(jnp.array([1.0, 0.0]))
    extras = build_skill_ego_extras_stacked(raw)
    assert extras.shape == (2, 1)
    assert float(extras[0, 0]) == WEAPON_IN_FLIGHT_AIMED
    assert float(extras[1, 0]) == WEAPON_READY


def test_spaceinvaders_weapon_state_in_flight_is_aimed():
    configure_skill_ego_features(enabled=True, env_name="spaceinvaders")
    raw = _FakePlayerBulletObs(jnp.array([1.0, 0.0]))
    extras = build_skill_ego_extras_stacked(raw)
    assert float(extras[0, 0]) == WEAPON_IN_FLIGHT_AIMED
    assert float(extras[1, 0]) == WEAPON_READY


def test_choppercommand_weapon_state_ready():
    configure_skill_ego_features(enabled=True, env_name="choppercommand")
    raw = _FakeGroupedProjectileObs(
        "player_missiles",
        jnp.array([[1.0, 0.0], [0.0, 0.0]]),
    )
    extras = build_skill_ego_extras_stacked(raw)
    assert float(extras[0, 0]) == WEAPON_IN_FLIGHT_AIMED
    assert float(extras[1, 0]) == WEAPON_READY


def test_beamrider_weapon_state_ready():
    configure_skill_ego_features(enabled=True, env_name="beamrider")
    raw = _FakeGroupedProjectileObs(
        "player_shots",
        jnp.array([[1.0], [0.0]]),
    )
    extras = build_skill_ego_extras_stacked(raw)
    assert float(extras[0, 0]) == WEAPON_IN_FLIGHT_AIMED
    assert float(extras[1, 0]) == WEAPON_READY


def test_ego_obs_channels_include_velocity_and_weapon():
    configure_ego_obs_from_config(
        {
            "USE_LEGACY_EGO_OBS": False,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": True,
            "USE_EGO_WEAPON_FEATURES": True,
            "ENV_NAME": "seaquest",
        }
    )
    # distance + v_r + v_t + sin/cos + weapon_state
    assert ego_obs_channels_per_frame() == 5 + 1
