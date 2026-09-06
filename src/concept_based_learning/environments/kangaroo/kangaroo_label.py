import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import (
    avoid_survival_reward_from_config,
    extract_skill_rewards,
    penalize_collect_death_from_config,
)
from concept_based_learning.observations.action_space import make_target_aligned_action_mappers


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info=None, config=None):
    del info
    return extract_skill_rewards(
        prev_raw_obs,
        next_raw_obs,
        prev_target,
        next_target,
        life_lost,
        avoid_survival_reward=avoid_survival_reward_from_config(config),
        penalize_collect_death=penalize_collect_death_from_config(config),
        config=config,
    )


# Kangaroo: full 18-action set. Player moves in all 8 directions + jump/punch (FIRE).
kangaroo_action_mapper, kangaroo_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [3,  8, 5,  9, 4, 7, 2, 6],   # R DR D DL L UL U UR
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)

# Each ladder produces two virtual COLLECT waypoints snapped to the
# standable platforms the ladder connects (not the ladder sprite edges):
#   bottom: player standing-centre at the ladder foot (narrow 2×1 so AABB
#           overlap implies climbable ladder-column overlap)
#   top:    standing-centre on the upper platform, raised by ``_TOP_WAYPOINT_LIFT_PX``
#           so mid-ladder approach cannot complete the stage early
# Observation pads ladders/platforms to 20 slots; inactive slots are ignored.
#
# Total objects: 20 bottom + 20 top + 3 fruits + 1 bell + 1 child
#              + 4 monkeys + 1 falling_coconut + 4 thrown_coconuts = 54
_KANGAROO_MAX_OBJECTS = 55  # 54 + 1 meta NOOP
_N_WAYPOINTS = 40  # 20 bottom + 20 top (COLLECT)
_N_COLLECTIBLE = 5  # 3 fruits + bell + child (COLLECT)
_N_DESTROY = 4  # monkeys only
_N_AVOID = 5  # 1 falling coconut + 4 thrown
_MONKEY_SPAWN_X = 152
_MONKEY_SPAWN_Y = 5
_COCO_OFFSCREEN_X = 13
_COCO_OFFSCREEN_Y = -1
_COCO_PUNCH_SCORE = 200
# Standing kangaroo height in jax_kangaroo (non-crouch); waypoint y = platform_y - half.
_PLAYER_STAND_HEIGHT = 24
_PLAYER_STAND_HALF = _PLAYER_STAND_HEIGHT // 2
# jax_kangaroo PLAYER_WIDTH is 8. Bottom WP must be narrow enough that AABB
# overlap with the WP implies ≥0.3 ladder-column overlap (climb start). A 16px
# WP let arrive fire ~10px short of the shaft (ladder_climb_probe).
_PLAYER_STAND_WIDTH = 2
_PLAYER_STAND_HALF_W = _PLAYER_STAND_WIDTH // 2
_LADDER_WAYPOINT_H = 1
# If no platform matches, lift the top waypoint by this many px (L1 ladder→floor gap).
_TOP_FALLBACK_LIFT = 8
# Pull top waypoints above standing-centre so arrive / overlap only fire once the
# player is actually on the upper platform — not while still on the ladder a few
# px below the floor (arrive_dist alone was firing early).
_TOP_WAYPOINT_LIFT_PX = 17
# Bottom WP at standing-centre (offset 0): a downward bearing made greedy FORWARD
# pick DOWNRIGHT near the foot and jump instead of entering the shaft.
_BOTTOM_WAYPOINT_Y_OFFSET_PX = 0
# Max |ladder_edge - platform.y| to count as the connected floor.
_PLATFORM_MATCH_PX = 16
# Falling coco x must lie within ladder width ± this margin to gate a ladder.
_FALLING_COCO_LADDER_X_MARGIN = 10
# Gate until the coco has fallen past the ladder foot (+ slack). Proactive: includes
# hanging cocos above the shaft, not only cocos already inside it.
_FALLING_COCO_PASSED_FOOT_MARGIN_PX = 8
# Stamped on ``raw_observation.player.state[-1]`` by ``KangarooSemanticWrapper``.
# Crashing takes priority over climbing (death animation must freeze the plan).
PLAYER_STATE_CLIMBING = 1
PLAYER_STATE_CRASHING = 2


def stamp_player_status_on_raw(raw_obs, is_climbing, is_crashing):
    """Expose climb / crash flags on the stacked player ``state`` channel."""
    player = raw_obs.player
    flag = jnp.where(
        is_crashing,
        jnp.int32(PLAYER_STATE_CRASHING),
        jnp.where(is_climbing, jnp.int32(PLAYER_STATE_CLIMBING), jnp.int32(0)),
    )
    st = player.state
    new_state = st.at[-1].set(flag) if jnp.ndim(st) > 0 else flag
    return raw_obs.replace(player=player.replace(state=new_state))


def stamp_player_climbing_on_raw(raw_obs, is_climbing):
    """Backward-compatible climb-only stamp (no crash bit)."""
    return stamp_player_status_on_raw(raw_obs, is_climbing, False)


def _scalar_to_1d(x):
    return jnp.expand_dims(x, axis=0) if x.ndim == 0 else x


def _falling_coco_active(pos):
    return jnp.logical_or(pos[0] != _COCO_OFFSCREEN_X, pos[1] != _COCO_OFFSCREEN_Y)


def _standing_center_y(platform_y):
    """Player centre-y when feet rest on ``platform_y`` (top-left player convention)."""
    return platform_y - _PLAYER_STAND_HALF


def _top_waypoint_y(platform_y):
    """Standing-centre on ``platform_y``, then raised by ``_TOP_WAYPOINT_LIFT_PX``."""
    return _standing_center_y(platform_y) - _TOP_WAYPOINT_LIFT_PX


# Player must be this close to a ladder column (centre vs shaft) to count as
# "already on it" for coco-gate bypass (keep climbing when coco hangs above).
_PLAYER_ON_LADDER_X_SLACK_PX = 14.0
_PLAYER_ON_LADDER_Y_SLACK_PX = 6.0
# Stamped on labeled falling-coco ``state`` when JAX-Atari reports dropping.
FALLING_COCO_STATE_DROPPING = 1


def _falling_coco_covers_ladders(falling_coconut, ladders):
    """Per-ladder bool: falling coco blocks the column until it passes the foot.

    Proactive gating: any live coco whose x-column overlaps the ladder and whose
    y has not yet cleared the ladder foot deactivates both waypoints. The plan
    waits at the platform (``skip_if_inactive=False``) until the column is safe.
    """
    coco_x = falling_coconut.x
    coco_y = falling_coconut.y
    coco_w = falling_coconut.width
    coco_active = falling_coconut.active > 0.5
    coco_alive = jnp.logical_and(
        coco_active,
        _falling_coco_active(jnp.stack([coco_x, coco_y])),
    )
    coco_left = coco_x - _FALLING_COCO_LADDER_X_MARGIN
    coco_right = coco_x + coco_w + _FALLING_COCO_LADDER_X_MARGIN
    lad_left = ladders.x
    lad_right = ladders.x + ladders.width
    x_overlap = jnp.logical_and(coco_left < lad_right, coco_right > lad_left)
    lad_bot = ladders.y + ladders.height
    not_passed_foot = coco_y <= lad_bot + float(_FALLING_COCO_PASSED_FOOT_MARGIN_PX)
    return jnp.logical_and(
        coco_alive,
        jnp.logical_and(x_overlap, not_passed_foot),
    )


def _player_on_ladders(player, ladders):
    """Per-ladder bool: player centre lies in the climb shaft (full column)."""
    px = player.x + player.width * 0.5
    py = player.y + player.height * 0.5
    lad_cx = ladders.x + ladders.width * 0.5
    lad_top = ladders.y
    lad_bot = ladders.y + ladders.height
    active = ladders.active > 0.5
    in_y = jnp.logical_and(
        py >= lad_top - float(_PLAYER_ON_LADDER_Y_SLACK_PX),
        py <= lad_bot + float(_PLAYER_ON_LADDER_Y_SLACK_PX),
    )
    near_x = jnp.abs(px - lad_cx) <= (ladders.width * 0.5 + float(_PLAYER_ON_LADDER_X_SLACK_PX))
    return jnp.logical_and(active, jnp.logical_and(in_y, near_x))


def _match_platform_y(edge_y, plat_y, plat_active, *, above: bool):
    """Nearest active platform to a ladder edge."""
    edge = edge_y[:, None]
    py = plat_y[None, :]
    active = plat_active[None, :] > 0.5
    if above:
        geom = jnp.logical_and(active, py <= edge)
        dist = edge - py
    else:
        geom = jnp.logical_and(active, py >= edge)
        dist = py - edge
    in_range = jnp.logical_and(geom, dist <= float(_PLATFORM_MATCH_PX))
    scored = jnp.where(in_range, dist, jnp.inf)
    best = jnp.argmin(scored, axis=1)
    found = jnp.any(in_range, axis=1)
    matched = plat_y[best]
    return jnp.where(found, matched, edge_y)


class KangarooSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_KANGAROO_MAX_OBJECTS)

    def _patch_raw_observation(self, semantic_obs, env_state):
        """Stamp climb/crash on player.state and falling-coco dropping on objects.state.

        ``falling_coco_dropping`` lives on the Atari level state (not ObjectObservation),
        so we copy it onto the labeled falling-coco slot after meta-NOOP prepend.
        """
        raw = stamp_player_status_on_raw(
            semantic_obs.raw_observation,
            env_state.player.is_climbing,
            env_state.player.is_crashing,
        )
        # Layout after meta NOOP: collect(45) | destroy(4) | falling(1) | thrown(4)
        fc_slot = 1 + _N_WAYPOINTS + _N_COLLECTIBLE + _N_DESTROY  # 50
        dropping = jnp.asarray(env_state.level.falling_coco_dropping).astype(jnp.int32)
        dropping = jnp.where(
            dropping > 0, jnp.int32(FALLING_COCO_STATE_DROPPING), jnp.int32(0)
        )
        objects = semantic_obs.objects
        new_state = objects.state.at[-1, fc_slot].set(dropping)
        return semantic_obs.replace(
            raw_observation=raw,
            objects=objects.replace(state=new_state),
        )

    def reset(self, key=None):
        semantic_obs, state = super().reset(key)
        env_state = self._unwrap_env_state(state.atari_state)
        return self._patch_raw_observation(semantic_obs, env_state), state

    def step(self, state, action):
        semantic_obs, new_state, reward, done, info = super().step(state, action)
        env_state = self._unwrap_env_state(new_state.atari_state)
        semantic_obs = self._patch_raw_observation(semantic_obs, env_state)
        return semantic_obs, new_state, reward, done, info

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Detect monkey punches from raw level state.

        Returns a 54-element was_killed mask:
          [collect(45) | destroy(4) | avoid(5)]
        """
        prev_level = prev_env_state.level
        next_level = next_env_state.level

        punched_spawn = jnp.logical_and(
            next_level.monkey_positions[:, 0] == _MONKEY_SPAWN_X,
            next_level.monkey_positions[:, 1] == _MONKEY_SPAWN_Y,
        )
        monkey_killed = jnp.logical_and(
            prev_level.monkey_states != 0,
            jnp.logical_and(next_level.monkey_states == 0, punched_spawn),
        ).astype(jnp.int32)

        collect_zeros = jnp.zeros(_N_WAYPOINTS + _N_COLLECTIBLE, dtype=jnp.int32)
        avoid_zeros = jnp.zeros(_N_AVOID, dtype=jnp.int32)
        return jnp.concatenate([collect_zeros, monkey_killed, avoid_zeros])

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        pad = jnp.zeros(self.game_max_objects - kill_mask.shape[0], dtype=jnp.int32)
        was_killed_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), kill_mask.astype(jnp.int32), pad]
        )
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        ladders = obs.ladders
        platforms = obs.platforms
        fruits = obs.fruits
        bell = obs.bell
        child = obs.child
        falling_coconut = obs.falling_coconut
        monkeys = obs.monkeys
        thrown_coconuts = obs.thrown_coconuts

        # --- Ladder waypoints (COLLECT), snapped to standable floors ---
        lad_half_w = ladders.width // 2
        lad_cx = ladders.x + lad_half_w
        wp_half_w = jnp.full_like(lad_cx, float(_PLAYER_STAND_HALF_W))
        wp_w = jnp.full_like(lad_cx, float(_PLAYER_STAND_WIDTH))
        wp_h = jnp.full_like(lad_cx, float(_LADDER_WAYPOINT_H))

        lad_top_edge = ladders.y
        lad_bot_edge = ladders.y + ladders.height

        dest_plat_y = _match_platform_y(
            lad_top_edge, platforms.y, platforms.active, above=True
        )
        dest_plat_y = jnp.where(
            dest_plat_y == lad_top_edge,
            lad_top_edge - _TOP_FALLBACK_LIFT,
            dest_plat_y,
        )
        src_plat_y = _match_platform_y(
            lad_bot_edge, platforms.y, platforms.active, above=False
        )

        coco_on_ladder = _falling_coco_covers_ladders(falling_coconut, ladders)
        # Keep waypoints live if we are already in that shaft — otherwise the
        # coco-above gate clears both WPs mid-climb and the plan gets stuck.
        already_on = _player_on_ladders(obs.player, ladders)
        coco_blocks = jnp.logical_and(coco_on_ladder, jnp.logical_not(already_on))
        ladder_climbable = jnp.logical_and(
            ladders.active > 0.5,
            jnp.logical_not(coco_blocks),
        ).astype(jnp.int32)

        bot_x = lad_cx - wp_half_w
        bot_y = _standing_center_y(src_plat_y) + float(_BOTTOM_WAYPOINT_Y_OFFSET_PX)
        bot_active = ladder_climbable

        top_x = lad_cx - wp_half_w
        top_y = _top_waypoint_y(dest_plat_y)
        top_active = ladder_climbable

        waypoint_w = wp_w
        waypoint_h = wp_h

        bell_x, bell_y = _scalar_to_1d(bell.x), _scalar_to_1d(bell.y)
        bell_w, bell_h = _scalar_to_1d(bell.width), _scalar_to_1d(bell.height)
        bell_act = _scalar_to_1d(bell.active)

        child_x, child_y = _scalar_to_1d(child.x), _scalar_to_1d(child.y)
        child_w, child_h = _scalar_to_1d(child.width), _scalar_to_1d(child.height)
        child_act = _scalar_to_1d(child.active)

        fc_x, fc_y = _scalar_to_1d(falling_coconut.x), _scalar_to_1d(falling_coconut.y)
        fc_w, fc_h = _scalar_to_1d(falling_coconut.width), _scalar_to_1d(falling_coconut.height)
        fc_alive = _scalar_to_1d(
            jnp.logical_and(
                falling_coconut.active > 0.5,
                _falling_coco_active(jnp.stack([falling_coconut.x, falling_coconut.y])),
            ).astype(jnp.int32)
        )

        wp_x = jnp.concatenate([bot_x, top_x])
        wp_y = jnp.concatenate([bot_y, top_y])
        wp_w = jnp.concatenate([waypoint_w, waypoint_w])
        wp_h = jnp.concatenate([waypoint_h, waypoint_h])
        wp_act = jnp.concatenate([bot_active, top_active])
        n_wp = wp_x.shape[0]  # 40

        collect_x = jnp.concatenate([wp_x, fruits.x, bell_x, child_x])
        collect_y = jnp.concatenate([wp_y, fruits.y, bell_y, child_y])
        collect_w = jnp.concatenate([wp_w, fruits.width, bell_w, child_w])
        collect_h = jnp.concatenate([wp_h, fruits.height, bell_h, child_h])
        collect_act = jnp.concatenate([wp_act, fruits.active, bell_act, child_act])
        n_collect = collect_x.shape[0]  # 45

        monkey_on_platform = jnp.logical_or(
            monkeys.state == 2,
            jnp.logical_or(monkeys.state == 3, monkeys.state == 4),
        )
        monkey_act = jnp.logical_and(monkeys.active > 0, monkey_on_platform).astype(
            jnp.int32
        )

        destroy_x = monkeys.x
        destroy_y = monkeys.y
        destroy_w = monkeys.width
        destroy_h = monkeys.height
        destroy_act = monkey_act
        n_destroy = destroy_x.shape[0]  # 4

        avoid_x = jnp.concatenate([fc_x, thrown_coconuts.x])
        avoid_y = jnp.concatenate([fc_y, thrown_coconuts.y])
        avoid_w = jnp.concatenate([fc_w, thrown_coconuts.width])
        avoid_h = jnp.concatenate([fc_h, thrown_coconuts.height])
        avoid_act = jnp.concatenate([fc_alive, (thrown_coconuts.active > 0.5).astype(jnp.int32)])
        n_avoid = avoid_x.shape[0]  # 5

        all_x = jnp.concatenate([collect_x, destroy_x, avoid_x])
        all_y = jnp.concatenate([collect_y, destroy_y, avoid_y])
        all_w = jnp.concatenate([collect_w, destroy_w, avoid_w])
        all_h = jnp.concatenate([collect_h, destroy_h, avoid_h])
        all_active = jnp.concatenate([collect_act, destroy_act, avoid_act])
        all_labels = jnp.concatenate([
            jnp.full(n_collect, LABEL_COLLECT, dtype=jnp.int32),
            jnp.full(n_destroy, LABEL_DESTROY, dtype=jnp.int32),
            jnp.full(n_avoid, LABEL_AVOID, dtype=jnp.int32),
        ])

        n_objects = all_x.shape[0]  # 54
        pad_size = self.game_max_objects - n_objects
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)

        all_offscreen = jnp.zeros(n_objects, dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([all_x, pad_x]),
            y=jnp.concatenate([all_y, pad_y]),
            width=jnp.concatenate([all_w, pad_w]),
            height=jnp.concatenate([all_h, pad_h]),
            active=jnp.concatenate([all_active, pad_active]),
            semantic_label=jnp.concatenate([all_labels, pad_labels]),
            state=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate([all_offscreen, pad_offscreen]),
            was_killed=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
            was_collected=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
        )
