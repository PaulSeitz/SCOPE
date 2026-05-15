import functools
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any
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
from concept_based_learning.observations.action_space import make_ego_action_mapper

# Smaller y is nearer the top of the playfield; surface waypoint uses y≈41 (see dummy_y below).
SURFACE_Y_THRESHOLD = 48.0

_USE_LEGACY_SEAQUEST_ACTION_MAPPER = False

# TODO: we can probably rewrite this to just check if enemies / divers went off screen and if yes not award any reward
def _bbox_overlap(player, target):
    """True if player and target bounding boxes overlap (e.g. collision)."""
    p_left = player.x
    p_right = player.x + player.width
    p_top = player.y
    p_bottom = player.y + player.height
    t_left = target.x
    t_right = target.x + target.width
    t_top = target.y
    t_bottom = target.y + target.height
    horiz = jnp.logical_and(p_left <= t_right, p_right >= t_left)
    vert = jnp.logical_and(p_top <= t_bottom, p_bottom >= t_top)
    return jnp.logical_and(horiz, vert)


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    def last_frame(e):
        return jax.tree_util.tree_map(lambda x: x[-1] if jnp.ndim(x) > 0 else x, e)

    prev_player = last_frame(prev_raw_obs.player)
    prev_target = last_frame(prev_target)
    next_target = last_frame(next_target)

    # Check if the target disappeared
    was_active = prev_target.active == 1
    is_active = next_target.active == 1
    disappeared = jnp.logical_and(was_active, jnp.logical_not(is_active))

    # Collision and Edge checks
    hit_by_assigned = _bbox_overlap(prev_player, prev_target)
    at_screen_edge = jnp.logical_or(prev_target.x <= 15.0, prev_target.x >= 145.0)

    p_left, p_right = prev_player.x, prev_player.x + prev_player.width
    p_top, p_bottom = prev_player.y, prev_player.y + prev_player.height
    t_left, t_right = prev_target.x, prev_target.x + prev_target.width
    t_top, t_bottom = prev_target.y, prev_target.y + prev_target.height
    tolerance = 4.0
    horizontal_overlap = jnp.logical_and(p_left - tolerance <= t_right, p_right + tolerance >= t_left)
    vertical_overlap = jnp.logical_and(p_top - tolerance <= t_bottom, p_bottom + tolerance >= t_top)
    near_player = jnp.logical_and(horizontal_overlap, vertical_overlap)

    # --- COLLECT ---
    is_collect = prev_target.semantic_label == LABEL_COLLECT
    collect_success = jnp.logical_and(is_collect, jnp.logical_and(was_active, near_player))
    collect_done = collect_success
    collect_reward = jnp.where(collect_success, 1.0, 0.0)

    # --- DESTROY ---
    is_destroy = prev_target.semantic_label == LABEL_DESTROY
    destroy_success = jnp.logical_and(
        is_destroy,
        jnp.logical_and(
            disappeared,
            jnp.logical_and(jnp.logical_not(hit_by_assigned), jnp.logical_not(at_screen_edge)),
        ),
    )
    destroy_failure = jnp.logical_and(is_destroy, jnp.logical_and(life_lost > 0, hit_by_assigned))
    destroy_done = jnp.logical_or(destroy_success, destroy_failure)
    destroy_reward = jnp.where(destroy_success, 1.0, 0)

    # --- AVOID ---
    is_avoid = prev_target.semantic_label == LABEL_AVOID
    # Success for Avoid: The hazard disappeared off screen, and it didn't hit the player
    avoid_success = jnp.logical_and(is_avoid, jnp.logical_and(disappeared, jnp.logical_not(hit_by_assigned)))
    avoid_failure = jnp.logical_and(is_avoid, jnp.logical_and(life_lost > 0, hit_by_assigned))
    avoid_done = jnp.logical_or(avoid_success, avoid_failure)
    # +1 for successfully outlasting it, -10 for dying to it
    avoid_reward = jnp.where(avoid_success, 1.0, jnp.where(avoid_failure, -10.0, 0.0))

    rewards = jnp.array([destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32)
    skill_done = jnp.array([destroy_done, collect_done, avoid_done], dtype=jnp.bool_)

    return rewards, skill_done

def set_seaquest_action_mapper(use_legacy: bool) -> None:
    """Select legacy orientation-relative vs target-aligned Seaquest action mapper."""
    global _USE_LEGACY_SEAQUEST_ACTION_MAPPER
    _USE_LEGACY_SEAQUEST_ACTION_MAPPER = bool(use_legacy)


def using_legacy_seaquest_action_mapper() -> bool:
    return _USE_LEGACY_SEAQUEST_ACTION_MAPPER


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
        return _legacy_orientation_relative_mapper(ego_action, raw_obs, target_idx, objects)
    return _target_aligned_seaquest_action_mapper(ego_action, raw_obs, target_idx, objects)

class SeaquestSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # Seaquest has roughly 33 objects (12 sharks + 12 subs + 4 divers + 4 missiles + 1 surface sub)
        super().__init__(env, max_objects=34)

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

        # 4. Pad with inactive dummy objects up to `self.max_objects` (34)
        # Make the dummy object (Index 33) act as our "Surface Waypoint"
        pad_size = self.max_objects - all_x.shape[0]  # 34 - 33 = 1

        dummy_x = jnp.array([obs.player.x])  # Track player's X to go straight up!
        dummy_y = jnp.array([41])            # Slightly above the clamped top-of-water Y
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

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SemanticState, action: int
    ):
        """
        Override the base step to neutralize deaths caused purely by surfacing
        with fewer than 6 divers. This prevents the game-over signal from
        resetting the episode in those cases while still allowing normal deaths.
        """
        semantic_obs, new_state, reward, done, info = super().step(state, action)

        # Inspect the most recent native Seaquest frame to detect surfacing deaths.
        # `obs_stack` is a stacked SeaquestObservation PyTree; take the last frame.
        last_obs = jax.tree_util.tree_map(
            lambda x: x[-1] if jnp.ndim(x) > 0 else x, semantic_obs.raw_observation
        )
        player_y = last_obs.player.y
        oxygen = last_obs.oxygen_level
        collected_divers = last_obs.collected_divers

        near_surface = player_y <= SURFACE_Y_THRESHOLD
        fewer_than_six_divers = collected_divers < 6

        # Heuristic: a "surfacing death" is a terminal transition that occurs
        # near the surface, with oxygen remaining and fewer than six divers.
        surfacing_death = jnp.logical_and(
            done,
            jnp.logical_and(
                near_surface,
                jnp.logical_and(fewer_than_six_divers, oxygen > 0.0),
            ),
        )

        # Neutralize the death signal and any associated reward spike.
        done = jnp.where(surfacing_death, False, done)
        reward = jnp.where(surfacing_death, 0.0, reward)

        return semantic_obs, new_state, reward, done, info