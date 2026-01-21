import jax
import jax.numpy as jnp
from functools import partial
from src.environments.seaquest.datatypes import MultiReward
import src.environments.seaquest.seaquest_constants as consts
from src.config import Config
# TODO: DESTROY is not working properly, its jittering like crazy and shooting in the wrong direction
# TODO: AVOID is permanently chilling at the top of the screen (reasonable I guess)
# TODO: Collect is a bit weird, its not b-lining to the divers and therefore often dies to the sharks.. might be fine? But its a bit fickle.

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
    
    # "Blind" Avoidance: Only punish distance if within "Panic Radius" (e.g., 0.2 screen units)
    # This allows the agent to swim freely in "Open Water" without constant negative feedback,
    # only reacting when a shark is actively biting it.
    # We'll apply this mask only to AVOID rewards below
    panic_radius = 0.2
    is_close = (dists_curr < panic_radius).astype(jnp.float32)

    # --- Global Avoidance Shaping ---
    # To fix "Tunnel Vision", we also compute reward based on the NEAREST threat.
    # labels are in obs_st.object_labels: [Max_Objs, Num_Concepts] or [Batch, Max_Objs, Num_Concepts]
    if dist_delta.ndim == 1:
        is_avoid_label = obs_st.object_labels[:, consts.LABEL_AVOID] > 0.5
        is_threat = is_avoid_label & (obs_st.object_mask > 0.5)
        
        # Distance to nearest threat
        min_dist_curr = jnp.min(jnp.where(is_threat, dists_curr, 1e9))
        min_dist_next = jnp.min(jnp.where(is_threat, dists_next, 1e9))
        
        # If no threats, delta is 0
        global_avoid_delta = jnp.where(jnp.any(is_threat), 
                                       (min_dist_curr - min_dist_next), 
                                       0.0) * reward_scale * valid_global_mask
    else:
        is_avoid_label = obs_st.object_labels[:, :, consts.LABEL_AVOID] > 0.5
        is_threat = is_avoid_label & (obs_st.object_mask > 0.5)
        
        # Distance to nearest threat (vmap-friendly min)
        min_dist_curr = jnp.min(jnp.where(is_threat, dists_curr, 1e9), axis=-1)
        min_dist_next = jnp.min(jnp.where(is_threat, dists_next, 1e9), axis=-1)
        
        global_avoid_delta = jnp.where(jnp.any(is_threat, axis=-1), 
                                       (min_dist_curr - min_dist_next), 
                                       0.0) * reward_scale * valid_global_mask

    # B. Aiming Logic (Fixed: Potential-Based)
    # Calculate alignment quality for Current and Next state
    # We want to reward INCREASING alignment, not MAINTAINING it.
    
    def get_alignment_potential(target_pos, p_pos, aim_vec):
        rel_vec = target_pos - p_pos
        # Use simple Dot Product or Cosine Similarity for smoother gradients
        # Normalized vectors are better for "potential" logic
        rel_norm = rel_vec / (jnp.linalg.norm(rel_vec) + 1e-6)
        # alignment is cosine similarity
        alignment = jnp.dot(rel_norm, aim_vec)
        # Clip to 0 (we only care if it's in front)
        return jnp.maximum(alignment, 0.0)

    # Calculate potentials
    align_curr = jax.vmap(lambda t: get_alignment_potential(t, p_pos, facing_vec))(t_pos)
    align_next = jax.vmap(lambda t: get_alignment_potential(t, p_next_pos, facing_vec))(t_next_pos)

    # Reward: Improvement (Shaping)
    align_delta = (align_next - align_curr)
    align_shaping = align_delta * aim_bonus_scale * valid_shaping_mask
    
    # --- Holding Bonus ---
    # Reward for simply BEING aimed correctly. 
    # This prevents the "Aim Away -> Snap Back" jitter loop.
    # We give a bonus if alignment is > 0.9 (roughly 25 degrees cone)
    is_aligned = (align_curr > 0.9).astype(jnp.float32)
    align_holding = is_aligned * (aim_bonus_scale * 0.1) * valid_shaping_mask

    # C. Events (Did we succeed?)
    # is_triggered is [Max_Objs] or [Batch, Max_Objs] (1.0 if event happened)
    
    # C. Surface Delta (for Monitor)
    # Player movement is global, not per-object, so use global mask only
    # Now: Only weak shaping for moving up. The Global Reward will handle actual scoring.
    p_y = p_pos[1]
    p_y_next = p_next_pos[1]
    # Seaquest: Y=0 is top. Moving UP means Y decreases.
    # So (p_y - p_y_next) is positive when moving UP.
    move_up_delta = (p_y - p_y_next) * reward_scale * valid_global_mask
    
    # C.1. Oxygen Refill Reward (for Monitor)
    # Reward for being at the top while oxygen is refilling
    # The event is already calculated in the wrapper with all conditions:
    # - Oxygen increased
    # - Player is at the top
    # - Oxygen is not full
    # - Player is playing (not dead, not in init state)
    # Handle both batched and unbatched cases
    if events.ndim == 1:
        # Unbatched: events is [Max_Objs]
        oxygen_refill_event = (events[32] == consts.EVENT_TRIGGERED).astype(jnp.float32)
    else:
        # Batched: events is [Batch, Max_Objs]
        oxygen_refill_event = (events[:, 32] == consts.EVENT_TRIGGERED).astype(jnp.float32)
    
    # Reward should be similar to move_up_delta in magnitude
    # The event already includes all necessary conditions, so we just use it directly
    oxygen_refill_reward = oxygen_refill_event * reward_scale * valid_global_mask
    
    # Weak shaping only - no state-based bonuses
    # The Global Reward (from scoring) will be the main signal
    r_monitor = move_up_delta * 0.5 + oxygen_refill_reward  # Weak shaping + oxygen refill reward

    # --- 3. CONCEPT REWARD ROWS (The "Vectors") ---
    # We construct the rows of the matrix. 
    # Each row assumes the agent selected THAT concept.
    
    # Row: COLLECT
    # "Fickleness" is often due to inertia (drifting back creates neg reward).
    # We add a small binary bonus just for "Closing the Gap" regardless of speed.
    is_closing = (dist_delta > 0).astype(jnp.float32)
    closing_bonus = is_closing * (reward_scale * 0.1) 
    
    # We keep COLLECT reckless. It is the Meta-Policy's job to switch if a Shark is in the way.
    r_collect = dist_delta + closing_bonus + (is_triggered * bonus_success)
    
    # Row: AVOID
    # Rule: PENALTY for getting closer + Big Penalty for triggering (collision)
    # We now incorporate the global_avoid_delta to punish getting close to ANY enemy.
    # "Blind" Avoidance: Only apply distance penalty when enemy is within panic radius
    # This prevents constant negative feedback in open water
    dist_delta_avoid = dist_delta * is_close  # Only punish when close
    if dist_delta.ndim == 1:
        r_avoid = (-dist_delta_avoid) - global_avoid_delta + (is_triggered * penalty_failure)
    else:
        r_avoid = (-dist_delta_avoid) - global_avoid_delta[:, None] + (is_triggered * penalty_failure)
    
    # Row: DESTROY
    # Note: We keep `align_shaping` for the gradient signal to GET there.
    r_destroy = align_shaping + align_holding + (is_triggered * bonus_success)
    
    # Row: MONITOR / HIGH_LEVEL (Self-centric)
    # These usually don't depend on the target object index and only rely on global features like the player position.
    # We broadcast the global result to all objects for simplicity, or zero out others.
    r_monitor_row = jnp.full_like(dist_delta, r_monitor)

    # --- 4. STACK INTO TENSOR ---
    # Stack shape: [Num_Concepts, Max_Objs] or [Num_Concepts, Batch, Max_Objs] -> Transpose if needed
    # We assume Num_Concepts = 7
    
    # Handle both batched and unbatched cases
    # num_concepts is now 4 (COLLECT, AVOID, DESTROY, DEFAULT) - removed PLAYER_TOOL
    if dist_delta.ndim == 1:
        # Unbatched: [Max_Objs]
        max_objs = dist_delta.shape[0]
        num_concepts = 4
        reward_potentials = jnp.zeros((num_concepts, max_objs))
        
        reward_potentials = reward_potentials.at[consts.LABEL_COLLECT].set(r_collect)
        reward_potentials = reward_potentials.at[consts.LABEL_AVOID].set(r_avoid)
        reward_potentials = reward_potentials.at[consts.LABEL_DESTROY].set(r_destroy)
        reward_potentials = reward_potentials.at[consts.LABEL_DEFAULT].set(r_monitor_row)
        
        # object_labels is [Max_Objs, Num_Concepts], transpose to [Num_Concepts, Max_Objs]
        label_mask = jnp.transpose(obs_st.object_labels, (1, 0))
        
        # Override mask for Monitor/Default concept so it's always rewardable
        label_mask = label_mask.at[consts.LABEL_DEFAULT, :].set(1.0)
        
        # Element-wise multiplication: [Num_Concepts, Max_Objs]
        final_concept_rewards = reward_potentials * label_mask
    else:
        # Batched: [Batch, Max_Objs]
        batch_size = dist_delta.shape[0]
        max_objs = dist_delta.shape[1]
        num_concepts = 4
        
        # Initialize with zeros (now only 4 concepts: COLLECT, AVOID, DESTROY, DEFAULT)
        reward_potentials = jnp.zeros((num_concepts, batch_size, max_objs))
        
        reward_potentials = reward_potentials.at[consts.LABEL_COLLECT].set(r_collect)
        reward_potentials = reward_potentials.at[consts.LABEL_AVOID].set(r_avoid)
        reward_potentials = reward_potentials.at[consts.LABEL_DESTROY].set(r_destroy)
        reward_potentials = reward_potentials.at[consts.LABEL_DEFAULT].set(r_monitor_row)
        
        # Transpose to [Batch, Num_Concepts, Max_Objs]
        reward_potentials = jnp.transpose(reward_potentials, (1, 0, 2))
        
        # object_labels is [Batch, Max_Objs, Num_Concepts], transpose to [Batch, Num_Concepts, Max_Objs]
        label_mask = jnp.transpose(obs_st.object_labels, (0, 2, 1))

        # Override mask for Monitor/Default concept so it's always rewardable
        label_mask = label_mask.at[:, consts.LABEL_DEFAULT, :].set(1.0)
        
        # Element-wise multiplication: [Batch, Num_Concepts, Max_Objs]
        final_concept_rewards = reward_potentials * label_mask

    # --- 5. Global Score & Life Penalty ---
    # Extract SeaquestState from AtariState wrapper
    prev_seaquest = state.env_state
    curr_seaquest = next_state.env_state
    
    # A. Score Reward
    score_diff = (curr_seaquest.score - prev_seaquest.score).astype(jnp.float32)
    
    # B. Death Penalty (Global - Shared by all concepts)
    # penalty_failure is usually negative (e.g. -1 or -50)
    # Detect the transition from alive (death_counter == 0) to dead (death_counter > 0)
    just_died = jnp.logical_and(prev_seaquest.death_counter == 0, curr_seaquest.death_counter > 0)
    death_penalty = just_died.astype(jnp.float32) * penalty_failure
    
    # --- 6. GATED GLOBAL REWARDS ---
    # "No Work, No Pay" - Only the relevant concept gets the Global Score reward
    # This prevents "Default" from being a "Free Rider" that gets rewards for unintended actions
    
    # A. Define Event Masks based on what actually happened
    divers_prev = prev_seaquest.divers_collected.astype(jnp.float32)
    divers_curr = curr_seaquest.divers_collected.astype(jnp.float32)
    
    # Diver count increased = Collection
    is_collect_event = (divers_curr > divers_prev).astype(jnp.float32)
    # Score went up but NOT via collection = Combat Kill
    is_combat_event = (score_diff > 0) & (divers_curr == divers_prev)
    
    # B. Define Concept Masks (Who deserves credit?)
    # Concept indices: 0=Collect, 1=Avoid, 2=Destroy, 3=Monitor/Default
    num_concepts = 4
    gate_collect = jnp.array([1.0, 0.0, 0.0, 0.0])  # Only Collect gets paid
    gate_destroy = jnp.array([0.0, 0.0, 1.0, 0.0])  # Only Destroy gets paid
    gate_shared = jnp.ones(num_concepts)  # Default: Everyone shares (for death penalty or no event)
    
    # C. Select the gate based on the event type
    # Convert boolean to float for arithmetic
    is_combat_event_float = is_combat_event.astype(jnp.float32)
    
    # Handle batched vs unbatched
    if score_diff.ndim == 0:  # Unbatched (scalar)
        # Select gate based on event
        current_gate = (
            is_collect_event * gate_collect +
            is_combat_event_float * gate_destroy +
            (1.0 - is_collect_event - is_combat_event_float) * gate_shared
        )
        
        # Apply the Gate: Gated Score + Shared Pain
        reward_score_gated = score_diff * global_score_scale * current_gate  # [Concepts]
        reward_combined = reward_score_gated + death_penalty * gate_shared  # Death shared by all
        
        # Broadcast to [Num_Concepts, Max_Objs]
        final_concept_rewards = final_concept_rewards + reward_combined[:, None]
    else:  # Batched
        batch_size = score_diff.shape[0]
        # Expand gates for broadcasting
        gate_collect_b = jnp.tile(gate_collect[None, :], (batch_size, 1))  # [Batch, Concepts]
        gate_destroy_b = jnp.tile(gate_destroy[None, :], (batch_size, 1))
        gate_shared_b = jnp.tile(gate_shared[None, :], (batch_size, 1))
        
        # Expand event flags for broadcasting
        is_collect_b = is_collect_event[:, None] if is_collect_event.ndim > 0 else jnp.full((batch_size, 1), is_collect_event)  # [Batch, 1]
        is_combat_b = is_combat_event_float[:, None] if is_combat_event_float.ndim > 0 else jnp.full((batch_size, 1), is_combat_event_float)  # [Batch, 1]
        
        current_gate = (
            is_collect_b * gate_collect_b +
            is_combat_b * gate_destroy_b +
            (1.0 - is_collect_b - is_combat_b) * gate_shared_b
        )
        
        # Apply the Gate: Gated Score + Shared Pain
        reward_score_gated = score_diff[:, None] * global_score_scale * current_gate  # [Batch, Concepts]
        reward_combined = reward_score_gated + death_penalty[:, None] * gate_shared_b  # [Batch, Concepts]
        
        # Broadcast to [Batch, Num_Concepts, Max_Objs]
        final_concept_rewards = final_concept_rewards + reward_combined[:, :, None]
    
    # Store global reward for logging (sum of all concept rewards or original calculation)
    global_reward = (score_diff * global_score_scale) + death_penalty

    return MultiReward(
        global_reward=global_reward,
        concept_rewards=final_concept_rewards
    )