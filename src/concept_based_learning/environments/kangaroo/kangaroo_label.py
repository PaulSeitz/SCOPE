import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NAVIGATE,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import extract_skill_rewards
from concept_based_learning.observations.action_space import make_target_aligned_action_mappers


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    return extract_skill_rewards(
        prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost
    )


# Kangaroo: full 18-action set. Player moves in all 8 directions + jump/punch (FIRE).
kangaroo_action_mapper, kangaroo_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [3,  8, 5,  9, 4, 7, 2, 6],   # R DR D DL L UL U UR
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)

# Each ladder produces two virtual 1px NAVIGATE waypoints snapped to the
# standable platforms the ladder connects (not the ladder sprite edges), so
# NavigateTo(arrive) fires when the player is actually standing on that floor:
#   bottom: centre of player AABB when standing on the platform at the ladder foot
#   top:    centre of player AABB when standing on the platform at the ladder head
# Observation pads ladders/platforms to 20 slots; inactive slots are ignored.
#
# Total objects: 20 bottom + 20 top + 3 fruits + 1 bell + 1 child
#              + 1 falling_coconut + 4 monkeys + 4 thrown_coconuts = 54
_KANGAROO_MAX_OBJECTS = 55  # 54 + 1 meta NOOP
_N_NAVIGATE = 40  # 20 bottom + 20 top
_N_COLLECT = 5    # 3 fruits + bell + child
_N_DESTROY = 5
_N_AVOID = 4
_MONKEY_SPAWN_X = 152
_MONKEY_SPAWN_Y = 5
_COCO_OFFSCREEN_X = 13
_COCO_OFFSCREEN_Y = -1
_COCO_PUNCH_SCORE = 200
# Standing kangaroo height in jax_kangaroo (non-crouch); waypoint y = platform_y - half.
_PLAYER_STAND_HEIGHT = 24
_PLAYER_STAND_HALF = _PLAYER_STAND_HEIGHT // 2
# If no platform matches, lift the top waypoint by this many px (L1 ladder→floor gap).
_TOP_FALLBACK_LIFT = 8
# Max |ladder_edge - platform.y| to count as the connected floor.
_PLATFORM_MATCH_PX = 16


def _scalar_to_1d(x):
    return jnp.expand_dims(x, axis=0) if x.ndim == 0 else x


def _falling_coco_active(pos):
    return jnp.logical_or(pos[0] != _COCO_OFFSCREEN_X, pos[1] != _COCO_OFFSCREEN_Y)


def _standing_center_y(platform_y):
    """Player centre-y when feet rest on ``platform_y`` (top-left player convention)."""
    return platform_y - _PLAYER_STAND_HALF


def _match_platform_y(edge_y, plat_y, plat_active, *, above: bool):
    """Nearest active platform to a ladder edge.

    ``above=True``: destination floor (platform.y <= edge, maximize y).
    ``above=False``: source floor (platform.y >= edge, minimize y).
    Falls back to ``edge_y`` when nothing is within ``_PLATFORM_MATCH_PX``.
    """
    # Broadcast edge (n_ladders,) against platforms (n_plats,) → (n_ladders, n_plats).
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

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Detect monkey punches and falling-coconut punches from raw level state.

        Returns a 54-element was_killed mask:
          [navigate(40) | collect(5) | destroy(5) | avoid(4)]
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

        prev_fc = prev_level.falling_coco_position
        next_fc = next_level.falling_coco_position
        fc_prev_active = _falling_coco_active(prev_fc)
        fc_next_active = _falling_coco_active(next_fc)
        score_delta = next_env_state.score - prev_env_state.score
        monkey_score = jnp.sum(monkey_killed) * _COCO_PUNCH_SCORE
        coco_killed = jnp.logical_and(
            jnp.logical_and(fc_prev_active, jnp.logical_not(fc_next_active)),
            score_delta - monkey_score >= _COCO_PUNCH_SCORE,
        ).astype(jnp.int32)

        destroy_mask = jnp.concatenate(
            [jnp.reshape(coco_killed, (1,)), monkey_killed]
        )
        navigate_zeros = jnp.zeros(_N_NAVIGATE, dtype=jnp.int32)
        collect_zeros = jnp.zeros(_N_COLLECT, dtype=jnp.int32)
        avoid_zeros = jnp.zeros(_N_AVOID, dtype=jnp.int32)
        return jnp.concatenate(
            [navigate_zeros, collect_zeros, destroy_mask, avoid_zeros]
        )

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
        ladders = obs.ladders               # n=20 (padded)
        platforms = obs.platforms           # n=20 (padded)
        fruits = obs.fruits                 # n=3
        bell = obs.bell                     # scalar
        child = obs.child                   # scalar
        falling_coconut = obs.falling_coconut  # scalar
        monkeys = obs.monkeys               # n=4
        thrown_coconuts = obs.thrown_coconuts  # n=4

        # --- Ladder waypoints (NAVIGATE), snapped to standable floors ---
        lad_half_w = ladders.width // 2
        lad_cx = ladders.x + lad_half_w

        lad_top_edge = ladders.y
        lad_bot_edge = ladders.y + ladders.height

        dest_plat_y = _match_platform_y(
            lad_top_edge, platforms.y, platforms.active, above=True
        )
        # No platform match → pull up by the usual L1 ladder→floor gap.
        dest_plat_y = jnp.where(
            dest_plat_y == lad_top_edge,
            lad_top_edge - _TOP_FALLBACK_LIFT,
            dest_plat_y,
        )
        src_plat_y = _match_platform_y(
            lad_bot_edge, platforms.y, platforms.active, above=False
        )

        bot_x = lad_cx
        bot_y = _standing_center_y(src_plat_y)
        bot_active = ladders.active

        top_x = lad_cx
        top_y = _standing_center_y(dest_plat_y)
        top_active = ladders.active

        waypoint_w = jnp.ones_like(bot_x)
        waypoint_h = jnp.ones_like(bot_y)

        # --- Flatten scalar fields ---
        bell_x, bell_y = _scalar_to_1d(bell.x), _scalar_to_1d(bell.y)
        bell_w, bell_h = _scalar_to_1d(bell.width), _scalar_to_1d(bell.height)
        bell_act = _scalar_to_1d(bell.active)

        child_x, child_y = _scalar_to_1d(child.x), _scalar_to_1d(child.y)
        child_w, child_h = _scalar_to_1d(child.width), _scalar_to_1d(child.height)
        child_act = _scalar_to_1d(child.active)

        fc_x, fc_y = _scalar_to_1d(falling_coconut.x), _scalar_to_1d(falling_coconut.y)
        fc_w, fc_h = _scalar_to_1d(falling_coconut.width), _scalar_to_1d(falling_coconut.height)
        fc_act = _scalar_to_1d(falling_coconut.active)

        nav_x = jnp.concatenate([bot_x, top_x])
        nav_y = jnp.concatenate([bot_y, top_y])
        nav_w = jnp.concatenate([waypoint_w, waypoint_w])
        nav_h = jnp.concatenate([waypoint_h, waypoint_h])
        nav_act = jnp.concatenate([bot_active, top_active])
        n_nav = nav_x.shape[0]  # 40

        collect_x = jnp.concatenate([fruits.x, bell_x, child_x])
        collect_y = jnp.concatenate([fruits.y, bell_y, child_y])
        collect_w = jnp.concatenate([fruits.width, bell_w, child_w])
        collect_h = jnp.concatenate([fruits.height, bell_h, child_h])
        collect_act = jnp.concatenate([fruits.active, bell_act, child_act])
        n_collect = collect_x.shape[0]  # 5

        # Monkeys: only targetable while on a platform (walk / throw), not while
        # climbing the wall.  jax_kangaroo states: 1=down, 2=left, 3=throw,
        # 4=right, 5=up (see ObjectObservation.state on monkeys).
        monkey_on_platform = jnp.logical_or(
            monkeys.state == 2,
            jnp.logical_or(monkeys.state == 3, monkeys.state == 4),
        )
        monkey_act = jnp.logical_and(monkeys.active > 0, monkey_on_platform).astype(
            jnp.int32
        )

        destroy_x = jnp.concatenate([fc_x, monkeys.x])
        destroy_y = jnp.concatenate([fc_y, monkeys.y])
        destroy_w = jnp.concatenate([fc_w, monkeys.width])
        destroy_h = jnp.concatenate([fc_h, monkeys.height])
        destroy_act = jnp.concatenate([fc_act, monkey_act])
        n_destroy = destroy_x.shape[0]  # 5

        avoid_x = thrown_coconuts.x
        avoid_y = thrown_coconuts.y
        avoid_w = thrown_coconuts.width
        avoid_h = thrown_coconuts.height
        avoid_act = thrown_coconuts.active
        n_avoid = avoid_x.shape[0]  # 4

        all_x = jnp.concatenate([nav_x, collect_x, destroy_x, avoid_x])
        all_y = jnp.concatenate([nav_y, collect_y, destroy_y, avoid_y])
        all_w = jnp.concatenate([nav_w, collect_w, destroy_w, avoid_w])
        all_h = jnp.concatenate([nav_h, collect_h, destroy_h, avoid_h])
        all_active = jnp.concatenate([nav_act, collect_act, destroy_act, avoid_act])
        all_labels = jnp.concatenate([
            jnp.full(n_nav, LABEL_NAVIGATE, dtype=jnp.int32),
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
