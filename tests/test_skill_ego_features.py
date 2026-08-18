import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_COLLECT, LABEL_DESTROY
from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    ego_obs_channels_per_frame,
    ego_pose_channels_per_frame,
    skill_ego_flat_dim,
    slice_skill_ego,
)
from concept_based_learning.observations.skill_ego_features import (
    CAN_FIRE_IN_FLIGHT,
    CAN_FIRE_READY,
    FACING_ENEMY_NO,
    FACING_ENEMY_YES,
    build_skill_ego_extras_stacked,
    configure_skill_ego_features,
    latch_weapon_state,
    skill_ego_extra_channels_per_frame,
    WEAPON_IN_FLIGHT_AIMED,
    WEAPON_IN_FLIGHT_WASTED,
    WEAPON_READY,
)


class _FakeProjectiles:
    def __init__(self, active, x=None, y=None, width=8.0, height=1.0):
        self.active = active
        n = active.shape[0]
        n_slots = active.shape[1] if active.ndim > 1 else 1
        shape = (n, n_slots) if active.ndim > 1 else (n,)
        self.x = jnp.zeros(shape, dtype=jnp.float32) if x is None else x
        self.y = jnp.zeros(shape, dtype=jnp.float32) if y is None else y
        self.width = jnp.full(shape, width, dtype=jnp.float32)
        self.height = jnp.full(shape, height, dtype=jnp.float32)


class _FakePlayer:
    def __init__(self, n_frames: int, orientation=90.0, x=10.0, y=45.0, width=4.0, height=11.0):
        self.x = jnp.full((n_frames,), x, dtype=jnp.float32)
        self.y = jnp.full((n_frames,), y, dtype=jnp.float32)
        self.width = jnp.full((n_frames,), width, dtype=jnp.float32)
        self.height = jnp.full((n_frames,), height, dtype=jnp.float32)
        self.orientation = jnp.full((n_frames,), orientation, dtype=jnp.float32)


class _FakeRawObs:
    def __init__(self, torpedo_active, orientation=90.0, x=10.0):
        n = torpedo_active.shape[0]
        self.player = _FakePlayer(n, orientation=orientation, x=x)
        active = jnp.zeros((n, 5), dtype=jnp.float32)
        active = active.at[:, 0].set(torpedo_active)
        self.projectiles = _FakeProjectiles(active)


class _FakeObjects:
    def __init__(self, n_frames, target_x, target_w=4.0, n_objects=2, label=LABEL_DESTROY):
        self.x = jnp.zeros((n_frames, n_objects), dtype=jnp.float32)
        self.x = self.x.at[:, 0].set(target_x)
        self.width = jnp.full((n_frames, n_objects), target_w, dtype=jnp.float32)
        self.y = jnp.zeros((n_frames, n_objects), dtype=jnp.float32)
        self.height = jnp.ones((n_frames, n_objects), dtype=jnp.float32)
        self.active = jnp.ones((n_frames, n_objects), dtype=jnp.int32)
        self.semantic_label = jnp.full(
            (n_frames, n_objects), label, dtype=jnp.int32
        )


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
    assert skill_ego_extra_channels_per_frame() == 2
    configure_skill_ego_features(
        enabled=True, env_name="seaquest", shot_features=True
    )
    assert skill_ego_extra_channels_per_frame() == 4
    configure_skill_ego_features(
        enabled=True, env_name="seaquest", shot_between=True
    )
    assert skill_ego_extra_channels_per_frame() == 3
    configure_skill_ego_features(enabled=False, env_name="seaquest", shot_features=True)
    assert skill_ego_extra_channels_per_frame() == 0


def test_latch_weapon_state_ready_aimed_wasted():
    # Legacy ternary helper still available for regression.
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


def test_seaquest_facing_enemy_yes_when_target_co_located_vertically():
    """Surface WP shares player x — must not invent a rightward aim bias."""
    configure_skill_ego_features(enabled=True, env_name="seaquest")
    n = 2
    # Player facing left (270) with target at same centre x → aim N/A → YES.
    raw = _FakeRawObs(
        jnp.ones((n,), dtype=jnp.float32),
        orientation=270.0,
        x=40.0,
    )
    objects = _FakeObjects(n, target_x=40.0, target_w=4.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert float(extras[0, 1]) == FACING_ENEMY_YES
    assert float(extras[1, 1]) == FACING_ENEMY_YES


def test_seaquest_can_fire_and_facing_enemy_when_aimed():
    configure_skill_ego_features(enabled=True, env_name="seaquest")
    # Real jaxatari Seaquest: 90 = facing right.
    raw = _FakeRawObs(
        jnp.array([0.0, 1.0, 1.0]),
        orientation=90.0,
        x=10.0,
    )
    objects = _FakeObjects(3, target_x=40.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert extras.shape == (3, 2)
    assert float(extras[0, 0]) == CAN_FIRE_READY
    assert float(extras[1, 0]) == CAN_FIRE_IN_FLIGHT
    assert float(extras[2, 0]) == CAN_FIRE_IN_FLIGHT
    assert float(extras[0, 1]) == FACING_ENEMY_YES
    assert float(extras[1, 1]) == FACING_ENEMY_YES
    assert float(extras[2, 1]) == FACING_ENEMY_YES


def test_seaquest_facing_enemy_no_when_facing_away():
    configure_skill_ego_features(enabled=True, env_name="seaquest")
    # 270 = facing left; target is to the right.
    raw = _FakeRawObs(
        jnp.array([0.0, 1.0, 1.0]),
        orientation=270.0,
        x=10.0,
    )
    objects = _FakeObjects(3, target_x=40.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert float(extras[0, 0]) == CAN_FIRE_READY
    assert float(extras[0, 1]) == FACING_ENEMY_NO
    assert float(extras[1, 1]) == FACING_ENEMY_NO


def test_seaquest_facing_enemy_updates_when_target_crosses():
    configure_skill_ego_features(enabled=True, env_name="seaquest")
    raw = _FakeRawObs(
        jnp.array([0.0, 1.0, 1.0]),
        orientation=90.0,
        x=50.0,
    )
    objects = _FakeObjects(3, target_x=80.0)
    objects.x = objects.x.at[2, 0].set(10.0)  # crosses behind after fire
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert float(extras[0, 1]) == FACING_ENEMY_YES
    assert float(extras[1, 1]) == FACING_ENEMY_YES
    # facing_enemy is live geometry (not fire-latched).
    assert float(extras[2, 1]) == FACING_ENEMY_NO


def test_phoenix_can_fire_facing_always_yes():
    configure_skill_ego_features(enabled=True, env_name="phoenix")
    raw = _FakeSingleProjectileObs(jnp.array([1.0, 0.0]))
    extras = build_skill_ego_extras_stacked(raw)
    assert extras.shape == (2, 2)
    assert float(extras[0, 0]) == CAN_FIRE_IN_FLIGHT
    assert float(extras[1, 0]) == CAN_FIRE_READY
    assert float(extras[0, 1]) == FACING_ENEMY_YES
    assert float(extras[1, 1]) == FACING_ENEMY_YES


def test_spaceinvaders_can_fire():
    configure_skill_ego_features(enabled=True, env_name="spaceinvaders")
    raw = _FakePlayerBulletObs(jnp.array([1.0, 0.0]))
    extras = build_skill_ego_extras_stacked(raw)
    assert float(extras[0, 0]) == CAN_FIRE_IN_FLIGHT
    assert float(extras[1, 0]) == CAN_FIRE_READY


def test_choppercommand_can_fire():
    configure_skill_ego_features(enabled=True, env_name="choppercommand")
    raw = _FakeGroupedProjectileObs(
        "player_missiles",
        jnp.array([[1.0, 0.0], [0.0, 0.0]]),
    )
    extras = build_skill_ego_extras_stacked(raw)
    assert float(extras[0, 0]) == CAN_FIRE_IN_FLIGHT
    assert float(extras[1, 0]) == CAN_FIRE_READY


def test_beamrider_can_fire():
    configure_skill_ego_features(enabled=True, env_name="beamrider")
    raw = _FakeGroupedProjectileObs(
        "player_shots",
        jnp.array([[1.0], [0.0]]),
    )
    extras = build_skill_ego_extras_stacked(raw)
    assert float(extras[0, 0]) == CAN_FIRE_IN_FLIGHT
    assert float(extras[1, 0]) == CAN_FIRE_READY


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
    # distance + v_r + v_t + sin/cos + can_fire + facing_enemy
    assert ego_obs_channels_per_frame() == 5 + 2


def test_ego_obs_channels_include_shot_features():
    configure_ego_obs_from_config(
        {
            "USE_LEGACY_EGO_OBS": False,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": True,
            "USE_EGO_WEAPON_FEATURES": True,
            "USE_EGO_SHOT_FEATURES": True,
            "ENV_NAME": "seaquest",
        }
    )
    # + shot_dx + shot_dy
    assert ego_obs_channels_per_frame() == 5 + 4


def test_shot_to_target_channels_when_in_flight():
    configure_skill_ego_features(
        enabled=True, env_name="seaquest", shot_features=True
    )
    raw = _FakeRawObs(jnp.array([0.0, 1.0]), orientation=90.0, x=10.0)
    raw.projectiles.x = raw.projectiles.x.at[1, 0].set(40.0)
    raw.projectiles.y = raw.projectiles.y.at[1, 0].set(50.0)
    objects = _FakeObjects(2, target_x=80.0)
    objects.y = objects.y.at[:, 0].set(50.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert extras.shape == (2, 4)
    assert float(extras[0, 2]) == 0.0
    assert float(extras[0, 3]) == 0.0
    # missile centre 40+4=44; target centre 80+2=82
    assert abs(float(extras[1, 2]) - (82.0 - 44.0) / 160.0) < 1e-5
    assert abs(float(extras[1, 3])) < 1e-5


def test_shot_channels_zero_on_env_without_bullets():
    """Pong/Freeway keep the same extra dim; missing bullets are zeros, not OOD."""
    configure_skill_ego_features(enabled=True, env_name="pong", shot_features=True)

    class _PongObs:
        def __init__(self):
            self.player = _FakePlayer(2)

    extras = build_skill_ego_extras_stacked(_PongObs())
    assert extras.shape == (2, 4)
    assert float(extras[0, 0]) == CAN_FIRE_READY
    assert float(extras[0, 1]) == FACING_ENEMY_YES
    assert float(extras[0, 2]) == 0.0
    assert float(extras[0, 3]) == 0.0


def test_shot_between_when_missile_on_open_segment():
    configure_skill_ego_features(enabled=True, env_name="seaquest", shot_between=True)
    raw = _FakeRawObs(jnp.array([0.0, 1.0]), orientation=90.0, x=10.0)
    raw.projectiles.x = raw.projectiles.x.at[1, 0].set(40.0)
    raw.projectiles.y = raw.projectiles.y.at[1, 0].set(50.0)
    objects = _FakeObjects(2, target_x=80.0)
    objects.y = objects.y.at[:, 0].set(50.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert extras.shape == (2, 3)
    assert float(extras[0, 2]) == 0.0  # no missile
    assert float(extras[1, 2]) == 1.0  # missile between player and enemy


def test_shot_between_zero_when_missile_goes_the_wrong_way():
    configure_skill_ego_features(enabled=True, env_name="seaquest", shot_between=True)
    raw = _FakeRawObs(jnp.array([1.0]), orientation=90.0, x=10.0)
    raw.projectiles.x = raw.projectiles.x.at[0, 0].set(0.0)
    raw.projectiles.y = raw.projectiles.y.at[0, 0].set(50.0)
    objects = _FakeObjects(1, target_x=80.0)
    objects.y = objects.y.at[:, 0].set(50.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert float(extras[0, 2]) == 0.0


def test_shot_between_zero_for_collect_target():
    configure_skill_ego_features(enabled=True, env_name="seaquest", shot_between=True)
    raw = _FakeRawObs(jnp.array([1.0]), orientation=90.0, x=10.0)
    raw.projectiles.x = raw.projectiles.x.at[0, 0].set(40.0)
    raw.projectiles.y = raw.projectiles.y.at[0, 0].set(50.0)
    objects = _FakeObjects(1, target_x=80.0, label=LABEL_COLLECT)
    objects.y = objects.y.at[:, 0].set(50.0)
    extras = build_skill_ego_extras_stacked(raw, objects=objects, target_idx=0)
    assert float(extras[0, 2]) == 0.0


def test_shot_between_zero_on_env_without_bullets():
    configure_skill_ego_features(enabled=True, env_name="pong", shot_between=True)

    class _PongObs:
        def __init__(self):
            self.player = _FakePlayer(2)

    extras = build_skill_ego_extras_stacked(_PongObs())
    assert extras.shape == (2, 3)
    assert float(extras[0, 2]) == 0.0


def _paper_eval_ego_cfg(**overrides):
    cfg = {
        "USE_LEGACY_EGO_OBS": False,
        "USE_EGO_SIZE_FEATURES": False,
        "USE_EGO_VELOCITY_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": True,
        "USE_EGO_WEAPON_FEATURES": True,
        "USE_EGO_SHOT_FEATURES": False,
        "USE_EGO_SHOT_BETWEEN": False,
        "USE_DESTROY_ONLY_EGO_EXTRAS": False,
        "ENV_NAME": "seaquest",
    }
    cfg.update(overrides)
    configure_ego_obs_from_config(cfg)
    return cfg


def test_pose_only_is_12_dim_for_every_skill():
    _paper_eval_ego_cfg(
        USE_EGO_BLOCKING_FEATURES=False,
        USE_EGO_WEAPON_FEATURES=False,
        USE_DESTROY_ONLY_EGO_EXTRAS=False,
    )
    assert ego_pose_channels_per_frame() == 3
    assert ego_obs_channels_per_frame() == 3
    for i in range(4):
        assert skill_ego_flat_dim(i, stack_size=4) == 12


def test_destroy_only_extras_shrinks_collect_avoid_navigate():
    _paper_eval_ego_cfg(USE_DESTROY_ONLY_EGO_EXTRAS=True)
    # pose 3 + los 1 + can_fire/facing 2 = 6 / frame → 24 DESTROY, 12 others
    assert ego_obs_channels_per_frame() == 6
    assert skill_ego_flat_dim(0, stack_size=4, destroy_skill_idx=0) == 24
    for i in (1, 2, 3):
        assert skill_ego_flat_dim(i, stack_size=4, destroy_skill_idx=0) == 12


def test_slice_skill_ego_drops_weapon_for_collect():
    _paper_eval_ego_cfg(USE_DESTROY_ONLY_EGO_EXTRAS=True)
    # Fake a 4-stack full vector; last frame extras are the tail of each 6-block.
    full = jnp.arange(24, dtype=jnp.float32)
    collect = slice_skill_ego(full, 1, stack_size=4, destroy_skill_idx=0)
    assert collect.shape == (12,)
    # Pose prefix per frame is 3 channels; extras (los, can_fire, facing) dropped.
    expected = jnp.concatenate([full[frame * 6 : frame * 6 + 3] for frame in range(4)])
    assert jnp.allclose(collect, expected)
    destroy = slice_skill_ego(full, 0, stack_size=4, destroy_skill_idx=0)
    assert destroy.shape == (24,)
    assert jnp.allclose(destroy, full)


def test_pong_collect_pose_has_no_can_fire_channel():
    """Pong COLLECT intercept should not see Seaquest leftover magazine bits."""
    _paper_eval_ego_cfg(USE_DESTROY_ONLY_EGO_EXTRAS=True, ENV_NAME="pong")
    assert skill_ego_flat_dim(1, stack_size=4) == 12
    # Shared (non-sliced) layout still builds the extras for DESTROY / occupancy.
    assert ego_obs_channels_per_frame() == 6


def test_flatten_packed_skill_params_roundtrip():
    from flax.traverse_util import unflatten_dict

    from concept_based_learning.training.skill_params import (
        as_per_skill_param_list,
        flatten_skill_params,
    )

    packed = tuple(
        {"Dense_0": {"kernel": jnp.zeros((12 if i else 24, 8))}} for i in range(4)
    )
    flat = flatten_skill_params(packed)
    restored = unflatten_dict(flat, sep=",")
    listed = as_per_skill_param_list(restored)
    assert listed is not None
    assert len(listed) == 4
    assert listed[0]["Dense_0"]["kernel"].shape[0] == 24
    assert listed[1]["Dense_0"]["kernel"].shape[0] == 12


def test_qnetwork_accepts_per_skill_in_dim():
    from concept_based_learning.networks.skill_network import QNetwork

    net = QNetwork(action_dim=18, hidden_size=8, num_layers=2)
    rng = jax.random.PRNGKey(0)
    v24 = net.init(rng, jnp.zeros(24), train=False)
    v12 = net.init(rng, jnp.zeros(12), train=False)
    q24 = net.apply({"params": v24["params"]}, jnp.zeros((2, 24)), train=False)
    q12 = net.apply({"params": v12["params"]}, jnp.zeros((2, 12)), train=False)
    assert q24.shape == (2, 18)
    assert q12.shape == (2, 18)
