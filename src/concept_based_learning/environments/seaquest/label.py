import jax.numpy as jnp
from jaxatari.environment import JAXAtariAction as Atari

from concept_based_learning.environments.labeling_wrapper import (
    JaxatariSemanticBaseWrapper,
)
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    SemanticState,
)
from concept_based_learning.environments.seaquest.heuristic_policy import (
    SURFACE_DUMMY_IDX,
)
from concept_based_learning.environments.skill_rewards import extract_skill_rewards
from concept_based_learning.observations.action_space import (
    make_ego_action_mapper,
    make_inverse_ego_action_mapper,
)

# X boundaries where enemies scroll off — used to distinguish scroll-off from kill.
# Enemies that disappear with prev_x outside [_SCROLL_X_MIN, _SCROLL_X_MAX] were
# scrolling off the edge, not destroyed by the player.
_SCROLL_X_MIN = 5
_SCROLL_X_MAX = 155

# Surface waypoint: player-aligned in x/width, fixed y at surfacing threshold.
# JAXAtari Seaquest begins surfacing when player_y == 46 (see update_oxygen()).
# Reward overlap globally applies a 1px inward inset, so we place this band at y=44
# with h=1; after inset that yields an effective [45, 46] band and a clean
# rising-edge when moving from y=47 -> y=46.
_SURFACE_TRIGGER_PLAYER_Y = 46
_SURFACE_WAYPOINT_Y = _SURFACE_TRIGGER_PLAYER_Y - 2
_SURFACE_WAYPOINT_H = 1

_SEAQUEST_MOVE_TABLE = [3, 8, 5, 9, 4, 7, 2, 6]
_SEAQUEST_MOVE_FIRE_TABLE = [11, 16, 13, 17, 12, 15, 10, 14]

_USE_LEGACY_SEAQUEST_ACTION_MAPPER = False
_USE_LEGACY_SEAQUEST_ASCENT_BLOCK = True
_USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD = False

# Surface threshold used by the ascent-block logic (unchanged).
SURFACE_Y_THRESHOLD = 52.0


def set_seaquest_action_mapper(use_legacy: bool) -> None:
    global _USE_LEGACY_SEAQUEST_ACTION_MAPPER
    _USE_LEGACY_SEAQUEST_ACTION_MAPPER = bool(use_legacy)


def using_legacy_seaquest_action_mapper() -> bool:
    return _USE_LEGACY_SEAQUEST_ACTION_MAPPER


def set_seaquest_ascent_block(use_legacy: bool) -> None:
    global _USE_LEGACY_SEAQUEST_ASCENT_BLOCK
    _USE_LEGACY_SEAQUEST_ASCENT_BLOCK = bool(use_legacy)


def using_legacy_seaquest_ascent_block() -> bool:
    return _USE_LEGACY_SEAQUEST_ASCENT_BLOCK


def set_seaquest_surfacing_death_guard(use_legacy: bool) -> None:
    global _USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD
    _USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD = bool(use_legacy)


def using_legacy_seaquest_surfacing_death_guard() -> bool:
    return _USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD


# Target-aligned mapper: ego FORWARD always means toward θ_target.
_target_aligned_seaquest_action_mapper = make_ego_action_mapper(
    noop_action=0,
    fire_action=1,
    move_table=_SEAQUEST_MOVE_TABLE,
    move_fire_table=_SEAQUEST_MOVE_FIRE_TABLE,
)
_target_aligned_seaquest_inverse_mapper = make_inverse_ego_action_mapper(
    noop_action=0,
    fire_action=1,
    move_table=_SEAQUEST_MOVE_TABLE,
    move_fire_table=_SEAQUEST_MOVE_FIRE_TABLE,
)


def seaquest_inverse_action_mapper(game_action, raw_obs, target_idx, objects):
    return _target_aligned_seaquest_inverse_mapper(
        game_action, raw_obs, target_idx, objects
    )


def _legacy_orientation_relative_mapper(ego_action, raw_obs, target_idx, objects):
    del target_idx, objects
    player_ori = raw_obs.player.orientation[-1]
    is_facing_right = player_ori < 180.0

    map_right = jnp.array(
        [
            Atari.NOOP,              # 0: NOOP
            Atari.FIRE,              # 1: FIRE
            Atari.RIGHT,             # 2: FORWARD
            Atari.DOWN,              # 3: RIGHT
            Atari.UP,                # 4: LEFT
            Atari.LEFT,              # 5: BACKWARD
            Atari.DOWNRIGHT,         # 6: FORWARD_RIGHT
            Atari.UPRIGHT,           # 7: FORWARD_LEFT
            Atari.DOWNLEFT,          # 8: BACKWARD_RIGHT
            Atari.UPLEFT,            # 9: BACKWARD_LEFT
            Atari.RIGHTFIRE,         # 10: FORWARD_FIRE
            Atari.DOWNFIRE,          # 11: RIGHT_FIRE
            Atari.UPFIRE,            # 12: LEFT_FIRE
            Atari.LEFTFIRE,          # 13: BACKWARD_FIRE
            Atari.DOWNRIGHTFIRE,     # 14: FORWARD_RIGHT_FIRE
            Atari.UPRIGHTFIRE,       # 15: FORWARD_LEFT_FIRE
            Atari.DOWNLEFTFIRE,      # 16: BACKWARD_RIGHT_FIRE
            Atari.UPLEFTFIRE,        # 17: BACKWARD_LEFT_FIRE
        ],
        dtype=jnp.int32,
    )
    map_left = jnp.array(
        [
            Atari.NOOP,              # 0: NOOP
            Atari.FIRE,              # 1: FIRE
            Atari.LEFT,              # 2: FORWARD
            Atari.UP,                # 3: RIGHT
            Atari.DOWN,              # 4: LEFT
            Atari.RIGHT,             # 5: BACKWARD
            Atari.UPLEFT,            # 6: FORWARD_RIGHT
            Atari.DOWNLEFT,          # 7: FORWARD_LEFT
            Atari.UPRIGHT,           # 8: BACKWARD_RIGHT
            Atari.DOWNRIGHT,         # 9: BACKWARD_LEFT
            Atari.LEFTFIRE,          # 10: FORWARD_FIRE
            Atari.UPFIRE,            # 11: RIGHT_FIRE
            Atari.DOWNFIRE,          # 12: LEFT_FIRE
            Atari.RIGHTFIRE,         # 13: BACKWARD_FIRE
            Atari.UPLEFTFIRE,        # 14: FORWARD_RIGHT_FIRE
            Atari.DOWNLEFTFIRE,      # 15: FORWARD_LEFT_FIRE
            Atari.UPRIGHTFIRE,       # 16: BACKWARD_RIGHT_FIRE
            Atari.DOWNRIGHTFIRE,     # 17: BACKWARD_LEFT_FIRE
        ],
        dtype=jnp.int32,
    )
    chosen_map = jnp.where(is_facing_right, map_right, map_left)
    return chosen_map[ego_action]


def seaquest_action_mapper(ego_action, raw_obs, target_idx, objects):
    if _USE_LEGACY_SEAQUEST_ACTION_MAPPER:
        mapped_action = _legacy_orientation_relative_mapper(ego_action, raw_obs, target_idx, objects)
    else:
        mapped_action = _target_aligned_seaquest_action_mapper(ego_action, raw_obs, target_idx, objects)

    near_surface = raw_obs.player.y[-1] <= SURFACE_Y_THRESHOLD
    surface_target_exception = target_idx == SURFACE_DUMMY_IDX
    should_block_ascent = jnp.logical_and(
        _USE_LEGACY_SEAQUEST_ASCENT_BLOCK,
        jnp.logical_and(near_surface, jnp.logical_not(surface_target_exception)),
    )

    ascent_actions = jnp.array(
        [Atari.UP, Atari.UPLEFT, Atari.UPRIGHT, Atari.UPFIRE, Atari.UPLEFTFIRE, Atari.UPRIGHTFIRE],
        dtype=jnp.int32,
    )
    ascent_fire_actions = jnp.array(
        [Atari.UPFIRE, Atari.UPLEFTFIRE, Atari.UPRIGHTFIRE],
        dtype=jnp.int32,
    )

    is_ascent_action = jnp.any(mapped_action == ascent_actions)
    is_ascent_fire_action = jnp.any(mapped_action == ascent_fire_actions)
    blocked_replacement = jnp.where(is_ascent_fire_action, Atari.FIRE, Atari.NOOP)

    return jnp.where(
        jnp.logical_and(should_block_ascent, is_ascent_action),
        blocked_replacement,
        mapped_action,
    )


class SeaquestSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 4 divers + 25 enemies + 4 enemy missiles + 1 surface waypoint = 34 game slots
        # +1 meta NOOP = 35 total
        super().__init__(env, max_objects=35)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        """
        Map a single-frame Seaquest observation to a LabeledObjectObservation.

        Object dimensions match JAXAtari's native hitbox sizes (plain AABB, no insets):
          player 16×11, diver 8×11, shark 8×7, sub 8×11, missile 8×1.

        The surface waypoint tracks player x/width but uses a fixed y band near the
        top, aligned with Seaquest's surfacing trigger height.
        """
        divers = obs.divers            # n=4
        enemies = obs.enemies          # n=25 (12 sharks + 12 subs + 1 surface sub)
        projectiles = obs.projectiles  # n=5 (1 player torpedo + 4 enemy torpedoes)

        # Strip the player torpedo (index 0).
        enemy_missiles_x      = projectiles.x[1:]
        enemy_missiles_y      = projectiles.y[1:]
        enemy_missiles_w      = projectiles.width[1:]
        enemy_missiles_h      = projectiles.height[1:]
        enemy_missiles_active = projectiles.active[1:]

        # Labels
        diver_labels          = jnp.full(4,  LABEL_COLLECT, dtype=jnp.int32)
        enemy_labels          = jnp.concatenate([
            jnp.full(24, LABEL_DESTROY, dtype=jnp.int32),
            jnp.array([LABEL_AVOID],    dtype=jnp.int32),   # surface sub
        ])
        enemy_missile_labels  = jnp.full(4,  LABEL_AVOID, dtype=jnp.int32)

        # can_go_offscreen flags: enemies and missiles scroll off the screen edges.
        diver_offscreen         = jnp.zeros(4,  dtype=jnp.int32)
        enemy_offscreen         = jnp.ones(25,  dtype=jnp.int32)
        enemy_missile_offscreen = jnp.ones(4,   dtype=jnp.int32)

        # Concatenate game objects (33 slots).
        all_x          = jnp.concatenate([divers.x,      enemies.x,      enemy_missiles_x])
        all_y          = jnp.concatenate([divers.y,      enemies.y,      enemy_missiles_y])
        all_w          = jnp.concatenate([divers.width,  enemies.width,  enemy_missiles_w])
        all_h          = jnp.concatenate([divers.height, enemies.height, enemy_missiles_h])
        all_active     = jnp.concatenate([divers.active, enemies.active, enemy_missiles_active])
        all_labels     = jnp.concatenate([diver_labels,  enemy_labels,   enemy_missile_labels])
        all_offscreen  = jnp.concatenate([diver_offscreen, enemy_offscreen, enemy_missile_offscreen])

        # Surface waypoint: always active/selectable; x/width follow player while
        # y is fixed to the surfacing trigger band.
        player_x = jnp.reshape(obs.player.x, (1,))
        player_w = jnp.reshape(obs.player.width, (1,))
        surface_x      = player_x
        surface_y      = jnp.array([_SURFACE_WAYPOINT_Y], dtype=jnp.int32)
        surface_w      = player_w
        surface_h      = jnp.array([_SURFACE_WAYPOINT_H], dtype=jnp.int32)
        surface_active = jnp.array([1], dtype=jnp.int32)
        surface_label  = jnp.array([LABEL_COLLECT], dtype=jnp.int32)
        surface_offscreen = jnp.array([0], dtype=jnp.int32)

        final_x         = jnp.concatenate([all_x,         surface_x])
        final_y         = jnp.concatenate([all_y,         surface_y])
        final_w         = jnp.concatenate([all_w,         surface_w])
        final_h         = jnp.concatenate([all_h,         surface_h])
        final_active    = jnp.concatenate([all_active,    surface_active])
        final_labels    = jnp.concatenate([all_labels,    surface_label])
        final_offscreen = jnp.concatenate([all_offscreen, surface_offscreen])

        return LabeledObjectObservation(
            x=final_x,
            y=final_y,
            width=final_w,
            height=final_h,
            active=final_active,
            semantic_label=final_labels,
            can_go_offscreen=final_offscreen,
            was_killed=jnp.zeros_like(final_active),  # stamped by step() after kill detection
        )

    @staticmethod
    def _enemy_kill_mask(prev_es, next_es) -> jnp.ndarray:
        """
        Int32 mask (shape [25]) — 1 for each enemy slot destroyed by the player
        this step, 0 otherwise.  Layout: [0:12] sharks, [12:24] subs, [24] surface sub.

        Distinguishes torpedo kills from scroll-offs by checking that the enemy's
        previous x position was mid-screen (not at the scroll boundary).
        """
        def _killed(prev_pos, next_pos):
            was_active = prev_pos[:, 2] != 0
            now_gone   = next_pos[:, 2] == 0
            mid_screen = jnp.logical_and(
                prev_pos[:, 0] > _SCROLL_X_MIN,
                prev_pos[:, 0] < _SCROLL_X_MAX,
            )
            return jnp.logical_and(jnp.logical_and(was_active, now_gone), mid_screen)

        shark_killed = _killed(prev_es.shark_positions, next_es.shark_positions)  # (12,)
        sub_killed   = _killed(prev_es.sub_positions,   next_es.sub_positions)    # (12,)
        surf_killed  = jnp.zeros(1, dtype=jnp.bool_)                             # surface sub never torpedoed

        return jnp.concatenate([shark_killed, sub_killed, surf_killed]).astype(jnp.int32)  # (25,)

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """OR-accumulated across frame_skip so respawns do not erase kill signals."""
        return self._enemy_kill_mask(prev_env_state, next_env_state)

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        # Game layout (34 slots) → +1 meta-NOOP → 35 stacked object indices.
        was_killed_game = jnp.concatenate([
            jnp.zeros(4, dtype=jnp.int32),
            kill_mask.astype(jnp.int32),
            jnp.zeros(4, dtype=jnp.int32),
            jnp.zeros(1, dtype=jnp.int32),
        ])
        was_killed_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), was_killed_game]
        )
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def step(self, state: SemanticState, action: int):
        """Optionally apply the legacy surfacing-death guard after base stepping."""
        import jax
        semantic_obs, new_state, reward, done, info = super().step(state, action)

        if not _USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD:
            return semantic_obs, new_state, reward, done, info

        last_obs = jax.tree_util.tree_map(
            lambda x: x[-1] if jnp.ndim(x) > 0 else x, semantic_obs.raw_observation
        )
        near_surface   = last_obs.player.y <= SURFACE_Y_THRESHOLD
        fewer_than_six = last_obs.collected_divers < 6
        surfacing_death = jnp.logical_and(
            done,
            jnp.logical_and(near_surface, jnp.logical_and(fewer_than_six, last_obs.oxygen_level > 0.0)),
        )
        done   = jnp.where(surfacing_death, False, done)
        reward = jnp.where(surfacing_death, 0.0, reward)
        return semantic_obs, new_state, reward, done, info


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info=None, config=None):
    """
    Seaquest skill rewards.

    DESTROY success reads `next_target.was_killed` — stamped on the post-step
    observation from raw env-state kill detection (substeps OR-merged across frame_skip).
    """
    del info, config
    return extract_skill_rewards(
        prev_raw_obs,
        next_raw_obs,
        prev_target,
        next_target,
        life_lost,
    )
