from typing import Tuple
import jax
from concept_based_learning.environments.datatypes import LabeledObjectObservation
import jax.numpy as jnp

def get_ego_obs(raw_observation, objects, target_idx):
    """
    Computes [distance, angle] for ALL frames in the stack and flattens them.
    Distances and angles are computed between object centers (not top-left corners).
    """
    def get_single_frame_ego(p_x, p_y, p_w, p_h, p_ori, t_x, t_y, t_w, t_h):
        # Convert top-left coordinates to centers
        p_cx = p_x + (p_w / 2.0)
        p_cy = p_y + (p_h / 2.0)
        t_cx = t_x + (t_w / 2.0)
        t_cy = t_y + (t_h / 2.0)

        dx = t_cx - p_cx
        dy = t_cy - p_cy
        distance = jnp.sqrt(dx**2 + dy**2)
        
        absolute_angle = jnp.arctan2(dy, dx)
        p_ori_rad = p_ori * (jnp.pi / 180.0)
        relative_angle = absolute_angle - p_ori_rad
        relative_angle = jnp.arctan2(jnp.sin(relative_angle), jnp.cos(relative_angle))
        
        return jnp.array([distance / 264.0, relative_angle / jnp.pi], dtype=jnp.float32)

    # objects.x is [stack_size, max_objects]. We slice out the specific target's history.
    t_x_history = objects.x[:, target_idx]
    t_y_history = objects.y[:, target_idx]
    t_w_history = objects.width[:, target_idx]
    t_h_history = objects.height[:, target_idx]
    
    # Map the math across the `stack_size` dimension
    stacked_ego = jax.vmap(get_single_frame_ego)(
        raw_observation.player.x,            # Shape: [stack_size]
        raw_observation.player.y,            # Shape: [stack_size]
        raw_observation.player.width,        # Shape: [stack_size]
        raw_observation.player.height,       # Shape: [stack_size]
        raw_observation.player.orientation,  # Shape: [stack_size]
        t_x_history,                         # Shape: [stack_size]
        t_y_history,                         # Shape: [stack_size]
        t_w_history,                         # Shape: [stack_size]
        t_h_history                          # Shape: [stack_size]
    ) # Resulting shape: [stack_size, 2]
    
    # Flatten it into a 1D vector for the MLP (e.g., size 8)
    return stacked_ego.flatten()