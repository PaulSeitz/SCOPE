# TODO: add a proper config file for this
import jax
import jax.numpy as jnp
from src.environments.seaquest.datatypes import MultiReward
import src.environments.seaquest.seaquest_constants as consts


# --- Index Ranges (Must match SeaquestConceptWrapper) ---
# 0: Player
# 1-12: Sharks
# 13-24: Subs
# 25-28: Divers
# 29-32: Enemy Missiles
# 33: Surface Sub
# 34: Player Missile
IDX_SHARKS_START = 1
IDX_SUBS_START = 13
IDX_DIVERS_START = 25
IDX_EMISSILES_START = 29
IDX_SSUB = 33

# --- Agnostic Config ---
REWARD_SCALE = 10.0       # High scale because inputs are normalized small numbers
BONUS_SUCCESS = 1.0       # Collection / Kill / Refill
PENALTY_FAILURE = -1.0    # Collision
AIM_BONUS_SCALE = 0.1    # Small shaping reward for good aim

def compute_rewards(obs_st, next_obs_st, state, next_state):
    """
    The Pure Symbolic Reward Kernel.
    Inputs: ConceptObs (with Events & Globals)
    Outputs: Matrix [Num_Concepts, Max_Objs]
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
    # Positions: [Batch, 2]
    p_pos = obs_st.objects[0, -4:-2]       
    p_next_pos = next_obs_st.objects[0, -4:-2]
    
    t_pos = obs_st.objects[:, -4:-2]
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
    dists_curr = jnp.sqrt(jnp.sum((t_pos - p_pos)**2, axis=1))
    dists_next = jnp.sqrt(jnp.sum((t_next_pos - p_next_pos)**2, axis=1))
    
    # Apply Mask HERE
    dist_delta = (dists_curr - dists_next) * REWARD_SCALE * valid_shaping_mask
    
    # B. Cosine Similarity Alignment (for Destroy)
    def get_cosine_similarity(target_pos, p_pos):
        """
        Calculates alignment between (Target - Player) and Shooting_Vector.
        """
        rel_vec = target_pos - p_pos
        # Avoid division by zero with a small epsilon
        dist = jnp.linalg.norm(rel_vec) + 1e-6
        rel_vec_norm = rel_vec / dist
        
        # Dot product with the game-specific shooting vector
        # Seaquest: [1, 0], Phoenix: [0, -1]
        alignment = jnp.dot(rel_vec_norm, consts.SHOOTING_VECTOR)
        return alignment
    
    # Calculate alignment for all targets
    alignments = jax.vmap(lambda t_pos: get_cosine_similarity(t_pos, p_pos))(t_pos)
    # Range [-1, 1]. We clamp negative values so we don't punish for being behind,
    # just reward for being in front.
    alignments = jnp.maximum(0.0, alignments)
    # Apply Mask HERE
    align_delta = alignments * AIM_BONUS_SCALE * valid_shaping_mask

    # C. Surface Delta (for Monitor)
    # Player movement is global, not per-object, so use global mask only
    p_y = p_pos[1]
    p_y_next = p_next_pos[1]
    move_up_delta = (p_y - p_y_next) * REWARD_SCALE * valid_global_mask

    # --- 3. Define Kernels ---

    def reward_collect(label, idx):
        # Delta is now 0.0 if triggered, so we only get the Bonus
        return dist_delta[idx] + (is_triggered[idx] * BONUS_SUCCESS)

    def reward_avoid(label, idx):
        return (-dist_delta[idx]) + (is_triggered[idx] * PENALTY_FAILURE)

    def reward_destroy(label, idx):
        """
        Generalized Shooting Reward.
        Rewards:
        1. Alignment (Cosine Sim): Keep target in the line of fire.
        2. Destruction: Event triggered.
        """
        return align_delta[idx] + (is_triggered[idx] * BONUS_SUCCESS)

    def reward_monitor(label, idx):
        return move_up_delta + (is_triggered[idx] * BONUS_SUCCESS)

    def reward_high_level(label, idx):
        return 0.0

    # --- 4. Dispatch & Assemble ---
    
    labels = obs_st.object_labels
    indices = jnp.arange(35)

    r_c = jax.vmap(reward_collect)(labels, indices)
    r_a = jax.vmap(reward_avoid)(labels, indices)
    r_d = jax.vmap(reward_destroy)(labels, indices)
    r_m = jax.vmap(reward_monitor)(labels, indices)
    r_h = jax.vmap(reward_high_level)(labels, indices)

    rewards = jnp.zeros((7, 35))
    
    rewards = rewards.at[consts.LABEL_COLLECT, :].set(
        jnp.where(labels == consts.LABEL_COLLECT, r_c, 0.0)
    )
    rewards = rewards.at[consts.LABEL_AVOID, :].set(
        jnp.where(labels == consts.LABEL_AVOID, r_a, 0.0)
    )
    rewards = rewards.at[consts.LABEL_DESTROY, :].set(
        jnp.where(labels == consts.LABEL_DESTROY, r_d, 0.0)
    )
    rewards = rewards.at[consts.LABEL_MONITOR_STATISTIC, :].set(
        jnp.where(labels == consts.LABEL_MONITOR_STATISTIC, r_m, 0.0)
    )
    rewards = rewards.at[consts.LABEL_HIGH_LEVEL_CONCEPT, :].set(
        jnp.where(labels == consts.LABEL_HIGH_LEVEL_CONCEPT, r_h, 0.0)
    )

    # --- 5. Global Score ---
    # Extract SeaquestState from AtariState wrapper
    prev_seaquest = state.env_state
    curr_seaquest = next_state.env_state
    score_diff = (curr_seaquest.score - prev_seaquest.score).astype(jnp.float32)
    global_reward = score_diff * 0.01

    return MultiReward(
        global_reward=global_reward,
        concept_rewards=rewards
    )