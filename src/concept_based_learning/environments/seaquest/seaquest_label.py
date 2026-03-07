from flax.linen.fp8_ops import Fp8DotGeneralBase
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any
from jaxatari.environment import JAXAtariAction as Atari

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NOOP,
)
from concept_based_learning.environments.seaquest.seaquest_policy import (
    SURFACE_DUMMY_IDX,
)

# TODO: we can probably rewrite this to just check if enemies / divers went off screen and if yes not award any reward
def extract_all_skills_rewards(prev_player, next_player, prev_target, next_target, prev_projectiles, next_projectiles):
    def last_frame(e):
        return jax.tree_util.tree_map(lambda x: x[-1] if x.ndim > 0 else x, e)

    prev_player = last_frame(prev_player)
    next_player = last_frame(next_player)
    prev_target = last_frame(prev_target)
    next_target = last_frame(next_target)
    prev_projectiles = last_frame(prev_projectiles)
    next_projectiles = last_frame(next_projectiles)

    # A "success" event is when the current target was active and becomes inactive
    was_active = prev_target.active == 1
    is_active = next_target.active == 1
    success = jnp.logical_and(was_active, jnp.logical_not(is_active))

    # Require that the player is spatially close to the target when it disappears.
    # This avoids rewarding off-screen despawns.
    #
    # We approximate "close" via bounding box overlap on the previous frame.
    p_left = prev_player.x
    p_right = prev_player.x + prev_player.width
    p_top = prev_player.y
    p_bottom = prev_player.y + prev_player.height

    t_left = prev_target.x
    t_right = prev_target.x + prev_target.width
    t_top = prev_target.y
    t_bottom = prev_target.y + prev_target.height

    horizontal_overlap = jnp.logical_and(p_left <= t_right, p_right >= t_left)
    vertical_overlap = jnp.logical_and(p_top <= t_bottom, p_bottom >= t_top)
    near_player = jnp.logical_and(horizontal_overlap, vertical_overlap)

    close_success = jnp.logical_and(success, near_player)

    # [collect] +1 reward for actually collecting (player close when target disappears)
    collect_reward = jnp.where(close_success, 1.0, 0.0)

    # For destroy, require that the player missile is the cause:
    # - target becomes inactive (success)
    # - the player missile is active
    # - missile bounding box overlaps the target bounding box
    #
    # In the native SeaquestObservation, `projectiles` contains:
    # index 0 → player torpedo, indices 1–4 → enemy torpedoes.
    def select_player_missile(obj_obs):
        # Take projectile index 0 along the object axis.
        return jax.tree_util.tree_map(lambda x: x[0] if x.ndim > 0 else x, obj_obs)

    prev_player_missile = select_player_missile(prev_projectiles)

    m_left = prev_player_missile.x
    m_right = prev_player_missile.x + prev_player_missile.width
    m_top = prev_player_missile.y
    m_bottom = prev_player_missile.y + prev_player_missile.height

    # Reuse target bounds from above
    missile_horizontal_overlap = jnp.logical_and(m_left <= t_right, m_right >= t_left)
    missile_vertical_overlap = jnp.logical_and(m_top <= t_bottom, m_bottom >= t_top)
    missile_overlap = jnp.logical_and(missile_horizontal_overlap, missile_vertical_overlap)

    missile_active = prev_player_missile.active == 1
    destroy_success = jnp.logical_and(success, jnp.logical_and(missile_active, missile_overlap))

    # [destroy] +1 reward for destroying via missile hit
    destroy_reward = jnp.where(destroy_success, 1.0, 0.0)

    # [avoid] base reward comes purely from being alive (handled in the trainer)
    avoid_reward = jnp.zeros_like(destroy_reward)

    return jnp.array([destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32)


# TODO: might be more fitting in the policy?
BLOCK_SURFACE_ASCENT: bool = True
SURFACE_Y_THRESHOLD: float = 50.0


def seaquest_action_mapper(
    ego_action: jnp.ndarray,
    global_features: Any,
    target_idx: jnp.ndarray,
) -> jnp.ndarray:
    """
    Translates the full 18-space Egocentric Action to Seaquest Atari Actions.
    global_features: The unflattened SeaquestObservation PyTree stack.
    """
    # 1. Extract player facing from the native PyTree (grab the most recent frame [-1])
    # 90.0 is Right, 270.0 is Left
    player_ori = global_features.player.orientation[-1] 
    
    # Check if facing right (allowing a small tolerance for floating point)
    is_facing_right = player_ori < 180.0
    
    # 2. Map for when facing RIGHT
    map_right = jnp.array([
        Atari.NOOP,              # 0: NOOP
        Atari.FIRE,              # 1: FIRE
        Atari.RIGHT,             # 2: FORWARD
        Atari.DOWN,              # 3: RIGHT (Strafe Right = Descend)
        Atari.UP,                # 4: LEFT (Strafe Left = Ascend)
        Atari.LEFT,              # 5: BACKWARD
        Atari.DOWNRIGHT,         # 6: FORWARD_RIGHT (Descend + Move Right)
        Atari.UPRIGHT,           # 7: FORWARD_LEFT (Ascend + Move Right)
        Atari.DOWNLEFT,          # 8: BACKWARD_RIGHT (Descend + Move Left)
        Atari.UPLEFT,            # 9: BACKWARD_LEFT (Ascend + Move Left)
        Atari.RIGHTFIRE,         # 10: FORWARD_FIRE
        Atari.DOWNFIRE,          # 11: RIGHT_FIRE
        Atari.UPFIRE,            # 12: LEFT_FIRE
        Atari.LEFTFIRE,          # 13: BACKWARD_FIRE
        Atari.DOWNRIGHTFIRE,     # 14: FORWARD_RIGHT_FIRE
        Atari.UPRIGHTFIRE,       # 15: FORWARD_LEFT_FIRE
        Atari.DOWNLEFTFIRE,      # 16: BACKWARD_RIGHT_FIRE
        Atari.UPLEFTFIRE         # 17: BACKWARD_LEFT_FIRE
    ], dtype=jnp.int32)
    
    # 3. Map for when facing LEFT
    map_left = jnp.array([
        Atari.NOOP,              # 0: NOOP
        Atari.FIRE,              # 1: FIRE
        Atari.LEFT,              # 2: FORWARD
        Atari.UP,                # 3: RIGHT (Strafe Right = Ascend)
        Atari.DOWN,              # 4: LEFT (Strafe Left = Descend)
        Atari.RIGHT,             # 5: BACKWARD
        Atari.UPLEFT,            # 6: FORWARD_RIGHT (Ascend + Move Left)
        Atari.DOWNLEFT,          # 7: FORWARD_LEFT (Descend + Move Left)
        Atari.UPRIGHT,           # 8: BACKWARD_RIGHT (Ascend + Move Right)
        Atari.DOWNRIGHT,         # 9: BACKWARD_LEFT (Descend + Move Right)
        Atari.LEFTFIRE,          # 10: FORWARD_FIRE
        Atari.UPFIRE,            # 11: RIGHT_FIRE
        Atari.DOWNFIRE,          # 12: LEFT_FIRE
        Atari.RIGHTFIRE,         # 13: BACKWARD_FIRE
        Atari.UPLEFTFIRE,        # 14: FORWARD_RIGHT_FIRE
        Atari.DOWNLEFTFIRE,      # 15: FORWARD_LEFT_FIRE
        Atari.UPRIGHTFIRE,       # 16: BACKWARD_RIGHT_FIRE
        Atari.DOWNRIGHTFIRE      # 17: BACKWARD_LEFT_FIRE
    ], dtype=jnp.int32)

    # 4. Dynamically select the correct map and propose an action
    chosen_map = jnp.where(is_facing_right, map_right, map_left)
    proposed_action = chosen_map[ego_action]

    # 5. Optional: block further ascent when close to the surface
    if not BLOCK_SURFACE_ASCENT:
        return proposed_action

    # Player's vertical position on the most recent frame (smaller y = higher up)
    player_y = global_features.player.y[-1]
    near_surface = player_y <= SURFACE_Y_THRESHOLD

    # Actions that move the player upward in screen space
    up_like_actions = jnp.array(
        [
            Atari.UP,
            Atari.UPFIRE,
            Atari.UPRIGHT,
            Atari.UPLEFT,
            Atari.UPRIGHTFIRE,
            Atari.UPLEFTFIRE,
        ],
        dtype=jnp.int32,
    )
    is_up_like = jnp.any(proposed_action == up_like_actions)

    # Allow surfacing (going above the threshold) only when the current
    # target is the generic surface-waypoint COLLECT object.
    surfacing_target = target_idx == SURFACE_DUMMY_IDX

    block_up = jnp.logical_and(near_surface, jnp.logical_and(is_up_like, jnp.logical_not(surfacing_target)))
    safe_action = Atari.NOOP

    return jnp.where(block_up, safe_action, proposed_action)

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
        dummy_y = jnp.array([46])            # Top of the playable water area (clamped player min Y)
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
            semantic_label=final_labels
        )