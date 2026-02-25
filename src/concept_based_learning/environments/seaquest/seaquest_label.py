import jax
import jax.numpy as jnp
from flax import struct
from typing import Any
from jaxatari.environment import JAXAtariAction as Atari

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import LabeledObjectObservation, LABEL_DESTROY, LABEL_COLLECT, LABEL_AVOID, LABEL_NOOP


def extract_all_skills_rewards(prev_player, next_player, prev_target, next_target):
    """
    Evaluates pure mechanical transitions for the single selected target object.
    Uses only the last frame of any stacked entity (transition outcome).
    Returns: jnp.array([destroy_reward, collect_reward, avoid_reward]) with shape (3,).
    """
    def last_frame(e):
        return jax.tree_util.tree_map(lambda x: x[-1] if x.ndim > 0 else x, e)

    prev_player = last_frame(prev_player)
    next_player = last_frame(next_player)
    prev_target = last_frame(prev_target)
    next_target = last_frame(next_target)

    # --- Helper Functions ---
    def get_center(entity):
        return entity.x + (entity.width / 2.0), entity.y + (entity.height / 2.0)

    def check_collision(p, t):
        overlap_x = jnp.maximum(0, jnp.minimum(p.x + p.width, t.x + t.width) - jnp.maximum(p.x, t.x))
        overlap_y = jnp.maximum(0, jnp.minimum(p.y + p.height, t.y + t.height) - jnp.maximum(p.y, t.y))
        return jnp.logical_and(overlap_x > 0, overlap_y > 0)

    # Calculate states
    px, py = get_center(prev_player)
    tx, ty = get_center(prev_target)
    npx, npy = get_center(next_player)
    ntx, nty = get_center(next_target)

    d_prev = jnp.sqrt((px - tx)**2 + (py - ty)**2)
    d_next = jnp.sqrt((npx - ntx)**2 + (npy - nty)**2)
    delta_d = d_prev - d_next  # Positive if getting closer

    is_colliding = check_collision(next_player, next_target)
    was_active = prev_target.active == 1
    is_active = next_target.active == 1
    deactivated = jnp.logical_and(was_active, jnp.logical_not(is_active))
    
    # Catch if the agent rammed the target this frame
    rammed = jnp.logical_and(was_active, is_colliding)

    # ==========================================
    # 1. DESTROY: "Sniper" Shape (Strict Anti-Ram)
    # Goal: Align Y, face target, shoot, do NOT touch.
    # ==========================================
    # Sparse objective: ONLY reward if it died AND we didn't ram it
    destroy_sparse = jnp.where(jnp.logical_and(deactivated, jnp.logical_not(rammed)), 1.0, 0.0)
    
    # Massive anti-ramming penalty to override any perceived benefit of colliding
    destroy_anti_ram = jnp.where(rammed, -5.0, 0.0)
    
    # Dense Y-alignment (reward for reducing delta_y)
    dy_prev = jnp.abs(py - ty)
    dy_next = jnp.abs(npy - nty)
    y_alignment_reward = jnp.where(was_active, (dy_prev - dy_next) * 0.05, 0.0)
    
    # Dense Facing bonus
    is_facing_right = prev_player.orientation < 180.0
    target_is_right = tx > px
    correct_facing = is_facing_right == target_is_right
    facing_bonus = jnp.where(
        jnp.logical_and(jnp.logical_and(was_active, correct_facing), dy_prev < 10.0), 
        0.02, 
        0.0
    )
    
    destroy_reward = destroy_sparse + destroy_anti_ram + y_alignment_reward + facing_bonus

    # ==========================================
    # 2. COLLECT: "Breadcrumb + Timer" Shape
    # Goal: Overlap bounding boxes as fast as possible.
    # ==========================================
    collect_sparse = jnp.where(rammed, 1.0, 0.0)
    
    # Dense breadcrumb trail so it knows *how* to find the target
    collect_dense = jnp.where(was_active, delta_d * 0.1, 0.0)
    
    # Time penalty to create urgency
    collect_step_penalty = jnp.where(was_active, -0.01, 0.0)
    
    collect_reward = collect_sparse + collect_dense + collect_step_penalty

    # ==========================================
    # 3. AVOID: "Potential Field" Shape
    # Goal: Bouncing the agent away without farming.
    # ==========================================
    SAFE_DISTANCE = jnp.maximum(prev_player.width, prev_target.width) * 1.5 
    
    def phi(distance):
        return jnp.where(
            distance < SAFE_DISTANCE, 
            -((SAFE_DISTANCE - distance) / SAFE_DISTANCE), 
            0.0
        )
    
    avoid_potential_reward = jnp.where(was_active, phi(d_next) - phi(d_prev), 0.0)
    avoid_sparse = jnp.where(rammed, -1.0, 0.0)
    
    avoid_reward = avoid_potential_reward + avoid_sparse

    return jnp.array([destroy_reward, collect_reward, avoid_reward], dtype=jnp.float32)


def seaquest_action_mapper(ego_action: jnp.ndarray, global_features: Any) -> jnp.ndarray:
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

    # 4. Dynamically select the correct map
    chosen_map = jnp.where(is_facing_right, map_right, map_left)
    
    return chosen_map[ego_action]

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
        dummy_y = jnp.array([20])             # Top of the playable water area
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