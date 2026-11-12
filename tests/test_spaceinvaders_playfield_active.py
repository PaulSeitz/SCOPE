"""Space Invaders: only playfield-visible objects stay selectable."""

import jax.numpy as jnp

from concept_based_learning.environments.spaceinvaders.spaceinvaders_label import (
    SpaceInvadersSemanticWrapper,
)


class _Obj:
    def __init__(self, x, y, w, h, active):
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)
        self.width = jnp.asarray(w)
        self.height = jnp.asarray(h)
        self.active = jnp.asarray(active)


class _FakeObs:
    """Minimal SI obs: 36 enemies + 3 bullets + 1 ufo + player."""

    def __init__(self, *, bullet_y: int, enemy_y: int = 50, player_y: int = 185):
        n_e = 36
        self.player = _Obj(80, player_y, 7, 10, 1)
        self.enemies = _Obj(
            jnp.zeros(n_e, dtype=jnp.int32),
            jnp.full(n_e, enemy_y, dtype=jnp.int32),
            jnp.full(n_e, 8, dtype=jnp.int32),
            jnp.full(n_e, 10, dtype=jnp.int32),
            jnp.ones(n_e, dtype=jnp.int32),
        )
        self.enemy_bullets = _Obj(
            jnp.array([40, 50, 60], dtype=jnp.int32),
            jnp.array([bullet_y, 100, 100], dtype=jnp.int32),
            jnp.ones(3, dtype=jnp.int32),
            jnp.full(3, 10, dtype=jnp.int32),
            jnp.ones(3, dtype=jnp.int32),
        )
        self.ufo = _Obj(56, 11, 7, 8, 1)


def _wrapper():
    w = object.__new__(SpaceInvadersSemanticWrapper)
    w.game_max_objects = 40
    return w


def test_enemy_bullet_in_ground_band_is_inactive():
    # Ground top = player_y + player_h = 195. Bullet at y=206 is below ground.
    objs = _wrapper()._extract_frame_objects(_FakeObs(bullet_y=206))
    assert int(objs.active[36]) == 0  # bullet[0]
    assert int(objs.active[37]) == 1  # bullet[1] still mid-air


def test_enemy_bullet_just_above_ground_stays_active():
    objs = _wrapper()._extract_frame_objects(_FakeObs(bullet_y=194))
    assert int(objs.active[36]) == 1


def test_enemy_marched_into_ground_is_inactive():
    objs = _wrapper()._extract_frame_objects(
        _FakeObs(bullet_y=100, enemy_y=200)
    )
    assert int(objs.active[0]) == 0
    assert int(objs.active[39]) == 1  # ufo still visible
