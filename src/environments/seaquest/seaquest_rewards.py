import jax
import jax.numpy as jnp
from functools import partial
from src.environments.seaquest.datatypes import MultiReward
import src.environments.seaquest.seaquest_constants as consts
from src.config import Config


@partial(jax.jit, static_argnames=('reward_scale', 'bonus_success', 'penalty_failure', 'aim_bonus_scale', 'global_score_scale'))
def _compute_rewards_jitted(obs_st, next_obs_st, state, next_state, 
                             reward_scale, bonus_success, penalty_failure, 
                             aim_bonus_scale, global_score_scale):
    """
    JIT-compiled version of compute_rewards with static config values extracted.
    """
    # --- 0. Global Safety Checks ---
    # Detect Level Resets or Death Animations to prevent global teleport penalties
    prev_steps = state.env_state.step_counter
    curr_steps = next_state.env_state.step_counter
    
    # If steps reset, or death animation is playing -> invalid step for shaping
    is_reset = curr_steps < prev_steps
    is_dying = next_state.env_state.death_counter > 0
    
    # 1.0 if safe to calculate distance, 0.0 if global reset/death
    valid_global_mask = jnp.logical_and(~is_reset, ~is_dying).astype(jnp.float32)

    # --- 1. Pre-computation ---
    # Extract Player Logic
    # Player is no longer Object 0. Player Pos is in Global Features.
    # Index 5=X, 6=Y (normalized)
    p_x = obs_st.global_features[5]
    p_y = obs_st.global_features[6]
    p_pos = jnp.array([p_x, p_y])

    next_p_x = next_obs_st.global_features[5]
    next_p_y = next_obs_st.global_features[6]
    p_next_pos = jnp.array([next_p_x, next_p_y])
    
    # --- TELEPORT GUARD ---
    # Calculate the raw distance moved in this frame
    delta_x = jnp.abs(p_x - next_p_x)
    delta_y = jnp.abs(p_y - next_p_y)
    
    # Seaquest sub moves slowly. Any jump > 0.1 (10% of screen) is a glitch/respawn.
    is_teleport = (delta_x > 0.1) | (delta_y > 0.1)
    
    # Update the mask to invalidate this frame if a teleport occurred
    valid_global_mask = valid_global_mask * (1.0 - is_teleport.astype(jnp.float32))
    # ----------------------------
    
    # Extract Shooting Vector from Capabilities
    facing_vec = obs_st.capabilities[consts.CAP_FACING_X:consts.CAP_FACING_Y+1]
    
    # --- Targets (Objects 0...N are now ALL external entities) ---
    t_pos = obs_st.objects[:, -4:-2]  # [N_objs, 2] - x, y from last 4 features
    t_next_pos = next_obs_st.objects[:, -4:-2]
    
    # Events: [Batch, 35]
    events = next_obs_st.object_events
    # 1.0 if event happened, 0.0 otherwise
    is_triggered = (events == consts.EVENT_TRIGGERED).astype(jnp.float32)
    
    # TODO: maybe pull this into the event detection step and only give the reward event once in cases where its really necessary! 
    # --- Local Shaping Mask ---
    # If an event triggered (Object Died/Collected), its next position is garbage (teleported).
    # We must mask shaping for THAT specific object on THIS specific frame.
    # mask = 1.0 (Normal), 0.0 (Event happened, ignore distance)
    valid_shaping_mask = valid_global_mask * (1.0 - is_triggered)

    # --- 2. Calculate Deltas (Masked) ---
    
    # A. Euclidean Delta (for Collect/Avoid)
    dists_curr = jnp.sqrt(jnp.sum((t_pos - p_pos)**2, axis=-1))
    dists_next = jnp.sqrt(jnp.sum((t_next_pos - p_next_pos)**2, axis=-1))
    
    # A. Distance Logic (For Collect/Avoid)
    # dist_delta is [Max_Objs] or [Batch, Max_Objs]
    # (Positive = Moved Closer)
    dist_delta = (dists_curr - dists_next) * reward_scale * valid_shaping_mask

    # B. Cosine Similarity Alignment (for Destroy)
    def get_cosine_similarity(target_pos, p_pos, aim_vec):
        """
        Calculates alignment between (Target - Player) and dynamic Shooting_Vector.
        """
        rel_vec = target_pos - p_pos
        # Avoid division by zero with a small epsilon
        dist = jnp.linalg.norm(rel_vec) + 1e-6
        rel_vec_norm = rel_vec / dist
        
        # USE DYNAMIC AIM VEC
        alignment = jnp.dot(rel_vec_norm, aim_vec)
        return alignment
    
    # Pass the dynamic facing_vec
    alignments = jax.vmap(lambda t: get_cosine_similarity(t, p_pos, facing_vec))(t_pos)
    # Range [-1, 1]. We clamp negative values so we don't punish for being behind,
    # just reward for being in front.
    alignments = jnp.maximum(0.0, alignments)
    
    # B. Aiming Logic (For Destroy)
    # align_delta is [Max_Objs] or [Batch, Max_Objs]
    # (Positive = Aiming at object)
    align_delta = alignments * aim_bonus_scale * valid_shaping_mask

    # C. Events (Did we succeed?)
    # is_triggered is [Max_Objs] or [Batch, Max_Objs] (1.0 if event happened)
    
    # C. Surface Delta (for Monitor)
    # Player movement is global, not per-object, so use global mask only
    p_y = p_pos[1]
    p_y_next = p_next_pos[1]
    move_up_delta = (p_y - p_y_next) * reward_scale * valid_global_mask

    # --- 3. CONCEPT REWARD ROWS (The "Vectors") ---
    # We construct the rows of the matrix. 
    # Each row assumes the agent selected THAT concept.
    
    # Row: COLLECT
    # Rule: Reward for getting closer + Big Bonus for success
    r_collect = dist_delta + (is_triggered * bonus_success)
    
    # Row: AVOID
    # Rule: PENALTY for getting closer + Big Penalty for triggering (collision)
    # Note: We invert dist_delta (getting closer is bad)
    r_avoid = (-dist_delta) + (is_triggered * penalty_failure)
    
    # Row: DESTROY
    # Rule: Reward for Aiming + Big Bonus for triggering (kill)
    r_destroy = align_delta + (is_triggered * bonus_success)
    
    # Row: MONITOR / HIGH_LEVEL (Self-centric) TODO: we might need to give reward for moving up for monitor. But maybe it already works, if it does we can just remove one of the two labels.
    # These usually don't depend on the target object index and only rely on global features like the player position.
    # We broadcast the global result to all objects for simplicity, or zero out others.
    r_monitor = jnp.full_like(dist_delta, move_up_delta)

    # --- 4. STACK INTO TENSOR ---
    # Stack shape: [Num_Concepts, Max_Objs] or [Num_Concepts, Batch, Max_Objs] -> Transpose if needed
    # We assume Num_Concepts = 7
    
    # Handle both batched and unbatched cases
    if dist_delta.ndim == 1:
        # Unbatched: [Max_Objs]
        max_objs = dist_delta.shape[0]
        num_concepts = 7
        reward_potentials = jnp.zeros((num_concepts, max_objs))
        
        reward_potentials = reward_potentials.at[consts.LABEL_COLLECT].set(r_collect)
        reward_potentials = reward_potentials.at[consts.LABEL_AVOID].set(r_avoid)
        reward_potentials = reward_potentials.at[consts.LABEL_DESTROY].set(r_destroy)
        reward_potentials = reward_potentials.at[consts.LABEL_MONITOR_STATISTIC].set(r_monitor)
        
        # object_labels is [Max_Objs, Num_Concepts], transpose to [Num_Concepts, Max_Objs]
        label_mask = jnp.transpose(obs_st.object_labels, (1, 0))
        
        # Element-wise multiplication: [Num_Concepts, Max_Objs]
        final_concept_rewards = reward_potentials * label_mask
    else:
        # Batched: [Batch, Max_Objs]
        batch_size = dist_delta.shape[0]
        max_objs = dist_delta.shape[1]
        num_concepts = 7
        
        # Initialize with zeros (for concepts we didn't define like Player/Tool)
        reward_potentials = jnp.zeros((num_concepts, batch_size, max_objs))
        
        reward_potentials = reward_potentials.at[consts.LABEL_COLLECT].set(r_collect)
        reward_potentials = reward_potentials.at[consts.LABEL_AVOID].set(r_avoid)
        reward_potentials = reward_potentials.at[consts.LABEL_DESTROY].set(r_destroy)
        reward_potentials = reward_potentials.at[consts.LABEL_MONITOR_STATISTIC].set(r_monitor)
        
        # Transpose to [Batch, Num_Concepts, Max_Objs]
        reward_potentials = jnp.transpose(reward_potentials, (1, 0, 2))
        
        # object_labels is [Batch, Max_Objs, Num_Concepts], transpose to [Batch, Num_Concepts, Max_Objs]
        label_mask = jnp.transpose(obs_st.object_labels, (0, 2, 1))
        
        # Element-wise multiplication: [Batch, Num_Concepts, Max_Objs]
        final_concept_rewards = reward_potentials * label_mask

    # --- 5. Global Score ---
    # Extract SeaquestState from AtariState wrapper
    prev_seaquest = state.env_state
    curr_seaquest = next_state.env_state
    score_diff = (curr_seaquest.score - prev_seaquest.score).astype(jnp.float32)
    global_reward = score_diff * global_score_scale

    return MultiReward(
        global_reward=global_reward,
        concept_rewards=final_concept_rewards
    )