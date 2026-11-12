"""Relational Seaquest mods change object-object behaviour, not player rules."""
import jax
import jax.numpy as jnp
import jaxtari
import numpy as np
import pytest


def _rollout(mods, n=1500, seed=0):
    env = jaxtari.make("seaquest", mods=mods)
    obs, st = env.reset(jax.random.PRNGKey(seed))
    n_act = env.action_space().n

    def f(c, k):
        o, s = c
        a = jax.random.randint(k, (), 0, n_act)
        o, s, r, d, i = env.step(s, a)
        return (o, s), (s.shark_positions, s.diver_positions, s.enemy_missile_positions, s.sub_positions, s.spawn_state.lane_directions, s.score)

    _, out = jax.lax.scan(f, (obs, st), jax.random.split(jax.random.PRNGKey(seed + 1), n))
    return [np.asarray(x) for x in out]


def _heading_match(sharks, divers):
    """Share of frames where a diver and an active shark in its lane share a heading."""
    t, match = 0, 0
    lanes = sharks.reshape(sharks.shape[0], 4, 3, 3)
    for lane in range(4):
        d = divers[:, lane, 2]
        for slot in range(3):
            s = lanes[:, lane, slot, 2]
            both = (d != 0) & (s != 0)
            t += both.sum()
            match += (d[both] == s[both]).sum()
    return match / max(t, 1), t


def test_decoupled_divers_oppose_their_escorts():
    base = _heading_match(*_rollout(None)[:2])
    mod = _heading_match(*_rollout(["decoupled_divers"])[:2])
    assert base[1] > 0 and mod[1] > 0
    assert base[0] > 0.9
    assert mod[0] < 0.1


def test_scrambled_relations_runs_and_keeps_spawning():
    sharks, divers, torpedoes, subs, _, _ = _rollout(["scrambled_relations"], n=4000)
    active = (sharks[:, :, 2] != 0).sum(1) + (subs[:, :, 2] != 0).sum(1)
    assert active[-1000:].mean() > 0.5
    assert (divers[:, :, 2] != 0).any()
    assert np.isfinite(sharks).all()


def test_torpedoes_leave_their_lane():
    torpedoes = _rollout(["scrambled_relations"], n=6000)[2]
    env = jaxtari.make("seaquest")
    stock = np.asarray(env.consts.ENEMY_MISSILE_Y)
    active = torpedoes[:, :, 2] != 0
    lanes = np.arange(4)[None, :].repeat(torpedoes.shape[0], 0)
    if active.any():
        assert (torpedoes[:, :, 1][active] != stock[lanes[active]]).all()


@pytest.mark.parametrize("mods", [["decoupled_divers"], ["scrambled_relations"]])
def test_player_rules_are_untouched_at_reset(mods):
    env = jaxtari.make("seaquest", mods=mods)
    base = jaxtari.make("seaquest")
    a = env.reset(jax.random.PRNGKey(3))[1]
    b = base.reset(jax.random.PRNGKey(3))[1]
    assert a.player_x == b.player_x and a.player_y == b.player_y and a.oxygen == b.oxygen


def _crossings(sharks, divers):
    """Frames with a diver and an oppositely headed shark overlapping in one lane."""
    lanes = sharks.reshape(sharks.shape[0], 4, 3, 3)
    n = 0
    for lane in range(4):
        d = divers[:, lane]
        for slot in range(3):
            s = lanes[:, lane, slot]
            n += (
                (d[:, 2] != 0) & (s[:, 2] != 0) & (d[:, 2] == -s[:, 2]) & (np.abs(d[:, 0] - s[:, 0]) < 4)
            ).sum()
    return n


def test_divers_clip_through_sharks_instead_of_locking_on():
    """A diver next to an oppositely headed shark keeps its own heading and passes it."""
    from jaxtari.games.jax_seaquest import JaxSeaquest
    from concept_based_learning.jaxtari_mods.seaquest.seaquest_mod_plugins import DecoupledDiversMod

    game = JaxSeaquest()
    _, state = game.reset(jax.random.PRNGKey(0))
    divers = jnp.zeros((4, 3), dtype=jnp.int32).at[0].set(jnp.array([80, 69, -1]))
    sharks = jnp.zeros((12, 3), dtype=jnp.int32).at[0].set(jnp.array([78, 69, 1]))
    args = (divers, sharks, state.sub_positions, 0, 150, 0, state.spawn_state, jnp.int32(2), jax.random.PRNGKey(1))

    stock = game.step_diver_movement(*args)[0]
    assert int(stock[0, 2]) == 1  # the stock game locks the diver onto the escort

    plugin = DecoupledDiversMod()
    plugin._env = game
    x, heading = 80, -1
    for step in range(40):
        moved = plugin.step_diver_movement(
            divers.at[0, 0].set(x), sharks, state.sub_positions, 0, 150, 0, state.spawn_state, jnp.int32(step), jax.random.PRNGKey(1)
        )[0]
        assert int(moved[0, 2]) == heading
        x = int(moved[0, 0])
    assert x < 78  # it has passed the shark


def _noop_rollout(mods, n=9000, seed=0):
    """An idle player lets waves swim off, so the follow-up waves appear too."""
    env = jaxtari.make("seaquest", mods=mods)
    obs, st = env.reset(jax.random.PRNGKey(seed))

    def f(c, _):
        o, s = c
        o, s, r, d, i = env.step(s, 0)
        return (o, s), (
            s.shark_positions,
            s.sub_positions,
            s.diver_positions,
            s.enemy_missile_positions,
            o.enemies.x,
            o.enemies.y,
            o.enemies.width,
            o.enemies.height,
            o.enemies.active,
        )

    _, out = jax.lax.scan(f, (obs, st), None, length=n)
    return [np.asarray(x) for x in out]


def _active(a):
    return a[:, :, 2] != 0


def test_swapped_roles_open_with_subs_and_divers():
    sharks, subs, divers, *_ = _noop_rollout(["swapped_enemy_roles"], n=1500)
    assert _active(subs).any() and _active(divers).any()
    first_sub = np.flatnonzero(_active(subs).any(1))[0]
    shark_frames = np.flatnonzero(_active(sharks).any(1))
    assert shark_frames.size == 0 or shark_frames[0] > first_sub


def test_swapped_roles_follow_up_waves_are_sharks():
    sharks, subs, *_ = _noop_rollout(["swapped_enemy_roles"])
    assert _active(sharks).any() and _active(subs).any()
    # Each wave type stays alone in its lane.
    both = _active(sharks).reshape(-1, 4, 3).any(2) & _active(subs).reshape(-1, 4, 3).any(2)
    assert both.mean() < 0.01


def test_swapped_roles_keep_each_type_in_its_own_slots():
    sharks, subs, _, _, ex, ey, ew, eh, ea = _noop_rollout(["swapped_enemy_roles"])
    on_screen = (sharks[:, :, 0] >= 0) & (sharks[:, :, 0] < 160) & _active(sharks)
    assert on_screen.any()
    assert (ex[:, :12][on_screen] == sharks[:, :, 0][on_screen]).all()
    assert (ey[:, :12][on_screen] == sharks[:, :, 1][on_screen]).all()
    assert (ew[:, :12] == 8).all() and (eh[:, :12] == 7).all()
    assert (ew[:, 12:24] == 8).all() and (eh[:, 12:24] == 11).all()
    on_screen = (subs[:, :, 0] >= 0) & (subs[:, :, 0] < 160) & _active(subs)
    assert (ex[:, 12:24][on_screen] == subs[:, :, 0][on_screen]).all()


def test_swapped_roles_keep_each_type_movement():
    sharks, subs, *_ = _noop_rollout(["swapped_enemy_roles"])
    env = jaxtari.make("seaquest")
    base = np.repeat(np.asarray(env.consts.SPAWN_POSITIONS_Y), 3)
    a = _active(subs)
    assert len(np.unique((subs[:, :, 1] - base)[a])) == 1  # subs hold a fixed height
    a = _active(sharks)
    assert len(np.unique((sharks[:, :, 1] - base)[a])) > 2  # sharks bob


def test_swapped_roles_only_sharks_fire():
    sharks, subs, _, torpedoes, *_ = _noop_rollout(["swapped_enemy_roles"])
    live = _active(torpedoes)
    assert live.any()
    new = live[1:] & ~live[:-1]
    lane_sharks = _active(sharks).reshape(-1, 4, 3).any(2)[1:]
    lane_subs = _active(subs).reshape(-1, 4, 3).any(2)[1:]
    assert lane_sharks[new].mean() > 0.95
    assert (new & lane_subs & ~lane_sharks).sum() == 0


def test_stock_game_still_opens_with_sharks():
    sharks, subs, *_ = _noop_rollout(None, n=1500)
    assert _active(sharks).any() and _active(subs).any()
    assert np.flatnonzero(_active(sharks).any(1))[0] < np.flatnonzero(_active(subs).any(1))[0]


def test_no_shark_never_stalls_in_rest_without_a_fire_press():
    env = jaxtari.make("namethisgame", mods=["no_shark"])
    obs, st = env.reset(jax.random.PRNGKey(0))

    def f(c, _):
        o, s = c
        o, s, r, d, i = env.step(s, 0)
        return (o, s), (s.resting, s.tentacle_active.any(), s.lives_remaining)

    _, (rest, tentacles, lives) = jax.lax.scan(f, (obs, st), None, length=12000)
    assert not np.asarray(rest)[10:].any()
    # The idle player keeps losing lives to the tentacles until the game can end.
    assert int(np.asarray(lives)[-1]) <= 1 or np.asarray(tentacles)[-2000:].any()
