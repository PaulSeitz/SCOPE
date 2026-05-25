import jax
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
    LABEL_NOOP,
    SemanticState,
)
from concept_based_learning.environments.seaquest.heuristic_policy import (
    SURFACE_DUMMY_IDX,
)
from concept_based_learning.environments.skill_rewards import (
    pack_skill_rewards,
    reward_transition_context,
    standard_avoid_outcomes,
    standard_destroy_outcomes,
)
from concept_based_learning.observations.action_space import make_ego_action_mapper

# Smaller y is nearer the top of the playfield; surface waypoint uses y≈41 (see dummy_y below).
SURFACE_Y_THRESHOLD = 52.0
# Synthetic surface waypoint (last game slot; index 34 after meta NOOP at 0).
SURFACE_WAYPOINT_Y = 41.0
AT_SURFACE_PLAYER_Y = 46.0

_USE_LEGACY_SEAQUEST_ACTION_MAPPER = False
_USE_LEGACY_SEAQUEST_ASCENT_BLOCK = True
_USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD = False

def _is_surface_waypoint_target(target):
    """True when the assigned target is the synthetic surface COLLECT waypoint (y≈41)."""
    return target.y == SURFACE_WAYPOINT_Y


def _surface_reached_success(prev_raw_obs, next_raw_obs):
    """
    True on the step the player actually surfaces in jax_seaquest.

    Mirrors update_oxygen() surfacing: player transitions to y==46 and either
    deposits a carried diver, begins oxygen refill, or completes a full rescue.
    """
    def last_scalar(obs, field):
        val = getattr(obs, field)
        return val[-1] if jnp.ndim(val) > 0 else val

    prev_y = prev_raw_obs.player.y[-1] if jnp.ndim(prev_raw_obs.player.y) > 0 else prev_raw_obs.player.y
    next_y = next_raw_obs.player.y[-1] if jnp.ndim(next_raw_obs.player.y) > 0 else next_raw_obs.player.y
    surfaced = jnp.logical_and(prev_y > AT_SURFACE_PLAYER_Y, next_y == AT_SURFACE_PLAYER_Y)

    prev_oxygen = last_scalar(prev_raw_obs, "oxygen_level")
    next_oxygen = last_scalar(next_raw_obs, "oxygen_level")
    prev_divers = last_scalar(prev_raw_obs, "collected_divers")
    next_divers = last_scalar(next_raw_obs, "collected_divers")

    divers_deposited = next_divers < prev_divers
    oxygen_refilled = next_oxygen > prev_oxygen
    full_rescue_reset = jnp.logical_and(prev_divers >= 6, next_divers == 0)

    return jnp.logical_and(
        surfaced,
        jnp.logical_or(divers_deposited, jnp.logical_or(oxygen_refilled, full_rescue_reset)),
    )


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    """
    Seaquest skill rewards — the reward contract from the strong Mar/Apr 2026 runs.

    All three skills are scored for the *selected object* regardless of its true
    semantic label (hindsight relabeling), so every skill learns off-policy from
    every transition.
    """
    (
        _prev_player,
        prev_target,
        _next_target,
        disappeared,
        hit_by_assigned,
        near_player,
        at_screen_edge,
    ) = reward_transition_context(prev_raw_obs, prev_target, next_target)

    # The surface-waypoint test is an object-identity check (y == SURFACE_WAYPOINT_Y),
    # not a label gate, so it stays: it selects which collect-success rule applies.
    is_surface_waypoint = _is_surface_waypoint_target(prev_target)
    diver_collect_success = jnp.logical_and(
        jnp.logical_not(is_surface_waypoint),
        jnp.logical_and(disappeared, near_player),
    )
    surface_collect_success = jnp.logical_and(
        is_surface_waypoint, _surface_reached_success(prev_raw_obs, next_raw_obs)
    )
    collect_success = jnp.logical_or(diver_collect_success, surface_collect_success)

    destroy_success, destroy_failure = standard_destroy_outcomes(
        prev_target,
        disappeared=disappeared,
        hit_by_assigned=hit_by_assigned,
        at_screen_edge=at_screen_edge,
        life_lost=life_lost,
    )
    avoid_success, avoid_failure = standard_avoid_outcomes(
        prev_target,
        disappeared=disappeared,
        hit_by_assigned=hit_by_assigned,
        life_lost=life_lost,
    )
    return pack_skill_rewards(
        destroy_success=destroy_success,
        destroy_failure=destroy_failure,
        collect_success=collect_success,
        avoid_success=avoid_success,
        avoid_failure=avoid_failure,
    )


def set_seaquest_action_mapper(use_legacy: bool) -> None:
    """Select legacy orientation-relative vs target-aligned Seaquest action mapper."""
    global _USE_LEGACY_SEAQUEST_ACTION_MAPPER
    _USE_LEGACY_SEAQUEST_ACTION_MAPPER = bool(use_legacy)


def using_legacy_seaquest_action_mapper() -> bool:
    return _USE_LEGACY_SEAQUEST_ACTION_MAPPER


def set_seaquest_ascent_block(use_legacy: bool) -> None:
    """Enable/disable legacy ascent blocking near the Seaquest surface."""
    global _USE_LEGACY_SEAQUEST_ASCENT_BLOCK
    _USE_LEGACY_SEAQUEST_ASCENT_BLOCK = bool(use_legacy)


def using_legacy_seaquest_ascent_block() -> bool:
    return _USE_LEGACY_SEAQUEST_ASCENT_BLOCK


def set_seaquest_surfacing_death_guard(use_legacy: bool) -> None:
    """Enable/disable legacy surfacing death guard (<6 divers near surface)."""
    global _USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD
    _USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD = bool(use_legacy)


def using_legacy_seaquest_surfacing_death_guard() -> bool:
    return _USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD


# Target-aligned mapper: ego FORWARD always means toward θ_target.
_target_aligned_seaquest_action_mapper = make_ego_action_mapper(
    noop_action=0,
    fire_action=1,
    move_table=     [3, 8, 5, 9, 4, 7, 2, 6],
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)


def _legacy_orientation_relative_mapper(ego_action, raw_obs, target_idx, objects):
    """Pre-refactor Seaquest mapping: ego FORWARD follows sprite facing direction."""
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
        [
            Atari.UP,
            Atari.UPLEFT,
            Atari.UPRIGHT,
            Atari.UPFIRE,
            Atari.UPLEFTFIRE,
            Atari.UPRIGHTFIRE,
        ],
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
        # Seaquest has roughly 33 objects (12 sharks + 12 subs + 4 divers + 4 missiles + 1 surface sub)
        super().__init__(env, max_objects=35)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        """
        Takes a single Jaxatari Seaquest observation, strips irrelevant entities (player missile),
        assigns universal labels, and flattens it into our universal object list.
        """
        # 1. Grab the natively grouped arrays
        divers = obs.divers           # n=4
        enemies = obs.enemies         # n=25 (12 Sharks + 12 Subs + 1 Surface Sub)
        projectiles = obs.projectiles # n=5 (1 Player Torpedo + 4 Enemy Torpedoes)
        
        # --- Filter out the Player Missile ---
        # The player torpedo is always at index 0
        enemy_missiles_x = projectiles.x[1:]
        enemy_missiles_y = projectiles.y[1:]
        enemy_missiles_w = projectiles.width[1:]
        enemy_missiles_h = projectiles.height[1:]
        enemy_missiles_active = projectiles.active[1:]

        # 2. Create the label arrays 
        diver_labels = jnp.full(4, LABEL_COLLECT, dtype=jnp.int32)
        
        # Enemies: Sharks (0-11) and Subs (12-23) are DESTROY, Surface Sub (24) is AVOID
        enemy_labels = jnp.concatenate([
            jnp.full(24, LABEL_DESTROY, dtype=jnp.int32), 
            jnp.array([LABEL_AVOID], dtype=jnp.int32)
        ])
        
        # Projectiles: Remaining 4 are Enemy Missiles (AVOID)
        enemy_missile_labels = jnp.full(4, LABEL_AVOID, dtype=jnp.int32)

        # 3. Concatenate all attributes into flat arrays
        # Total objects: 4 (divers) + 25 (enemies) + 4 (enemy missiles) = 33 active slots
        all_x = jnp.concatenate([divers.x, enemies.x, enemy_missiles_x])
        all_y = jnp.concatenate([divers.y, enemies.y, enemy_missiles_y])
        all_w = jnp.concatenate([divers.width, enemies.width, enemy_missiles_w])
        all_h = jnp.concatenate([divers.height, enemies.height, enemy_missiles_h])
        all_active = jnp.concatenate([divers.active, enemies.active, enemy_missiles_active])
        all_labels = jnp.concatenate([diver_labels, enemy_labels, enemy_missile_labels])

        # 4. Pad with inactive dummy objects up to `self.game_max_objects` (34)
        # Make the dummy object (last game slot) act as our "Surface Waypoint"
        pad_size = self.game_max_objects - all_x.shape[0]  # 34 - 33 = 1

        dummy_x = jnp.array([obs.player.x])  # Track player's X to go straight up!
        dummy_y = jnp.array([SURFACE_WAYPOINT_Y])  # Slightly above the clamped top-of-water Y
        dummy_w = jnp.array([10])
        dummy_h = jnp.array([10])
        dummy_active = jnp.array([1])         # Make it active so the skill pursues it
        dummy_label = jnp.array([LABEL_COLLECT])  # Route to the Collect Skill!

        final_x = jnp.concatenate([all_x, dummy_x])
        final_y = jnp.concatenate([all_y, dummy_y])
        final_w = jnp.concatenate([all_w, dummy_w])
        final_h = jnp.concatenate([all_h, dummy_h])
        final_active = jnp.concatenate([all_active, dummy_active])
        final_labels = jnp.concatenate([all_labels, dummy_label])

        # 5. Return our universal struct
        return LabeledObjectObservation(
            x=final_x,
            y=final_y,
            width=final_w,
            height=final_h,
            active=final_active,
            semantic_label=final_labels,
        )

    def step(self, state: SemanticState, action: int):
        """Optional legacy guard against immediate surfacing-death terminations."""
        semantic_obs, new_state, reward, done, info = super().step(state, action)

        if not _USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD:
            return semantic_obs, new_state, reward, done, info

        last_obs = jax.tree_util.tree_map(
            lambda x: x[-1] if jnp.ndim(x) > 0 else x, semantic_obs.raw_observation
        )
        player_y = last_obs.player.y
        oxygen = last_obs.oxygen_level
        collected_divers = last_obs.collected_divers

        near_surface = player_y <= SURFACE_Y_THRESHOLD
        fewer_than_six_divers = collected_divers < 6
        surfacing_death = jnp.logical_and(
            done,
            jnp.logical_and(
                near_surface,
                jnp.logical_and(fewer_than_six_divers, oxygen > 0.0),
            ),
        )

        done = jnp.where(surfacing_death, False, done)
        reward = jnp.where(surfacing_death, 0.0, reward)
        return semantic_obs, new_state, reward, done, info
