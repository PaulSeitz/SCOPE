from typing import Tuple
import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LabeledObjectObservation

def get_ego_obs(raw_observation, objects, target_idx):
    """
    Computes [dx_ego, dy_ego, sin(ori), cos(ori)] for ALL frames in the stack.
    Universally compatible with both 360-degree top-down games and 2D side-scrollers.
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

        # 1. Calculate Cartesian absolute angle
        cartesian_angle = jnp.arctan2(dy, dx)
        
        # 2. Shift to Jaxatari's system (where Up is 0, Right is pi/2)
        jaxatari_absolute_angle = cartesian_angle + (jnp.pi / 2.0)
        p_ori_rad = p_ori * (jnp.pi / 180.0)
        
        # 3. Calculate relative angle safely
        relative_angle = jaxatari_absolute_angle - p_ori_rad
        relative_angle = jnp.arctan2(jnp.sin(relative_angle), jnp.cos(relative_angle))
        
        # 4. Convert to Egocentric Cartesian Coordinates
        # relative_angle = 0 is straight ahead, so cos() gives forward/backward axis
        # sin() gives the lateral (left/right or up/down relative to facing) axis
        dx_ego = distance * jnp.cos(relative_angle)
        dy_ego = distance * jnp.sin(relative_angle)
        
        # Output 4 features, normalizing by rough screen dimensions
        return jnp.array([
            dx_ego / 160.0, 
            dy_ego / 210.0,
            jnp.sin(p_ori_rad),
            jnp.cos(p_ori_rad)
        ], dtype=jnp.float32)

    
    # [The rest of the slice and vmap code remains exactly the same]
    t_x_history = objects.x[:, target_idx]
    t_y_history = objects.y[:, target_idx]
    t_w_history = objects.width[:, target_idx]
    t_h_history = objects.height[:, target_idx]
    
    stacked_ego = jax.vmap(get_single_frame_ego)(
        raw_observation.player.x,            
        raw_observation.player.y,            
        raw_observation.player.width,        
        raw_observation.player.height,       
        raw_observation.player.orientation,  
        t_x_history,                         
        t_y_history,                         
        t_w_history,                         
        t_h_history                          
    ) 
    
    # Flattens into a 1D vector (e.g., size 16 for a stack of 4 frames)
    return stacked_ego.flatten()