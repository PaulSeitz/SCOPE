# src/envs/schemas.py
import chex
from flax import struct
import jax.numpy as jnp

@struct.dataclass
class ConceptObs:
    """
    The Single Source of Truth for the Agent's observations.
    Modules will cherry-pick what they need from this struct.
    """
    # --- 1. Strategic Dashboard (For Meta-Policy) ---
    # Hand-crafted features: [Oxygen, Held_Divers, Lives, Closest_Enemy_Dist, Enemy_Density, Diver_Density, Player_X, Player_Y, PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_ORIENTATION]
    global_features: chex.Array  # Shape: [Batch, N_Globals]

    # --- 2. Capabilities (For Sub-Policies) ---
    # Standardized vector defining what the agent CAN do right now. TODO: this might be an issue with transferability, tbd.
    # [Can_Fire, Facing_X, Facing_Y]
    capabilities: chex.Array # Shape: [Batch, N_Caps]

    # --- 3. Tactical Detail ---
    # The raw object list. 
    # Shape: [Batch, Max_Objs, Obj_Feats] (x, y, h, w, multi-hot encoding of labels)
    objects: chex.Array 
    
    # Shape: [Batch, Max_Objs] (Boolean)
    # 1.0 if object exists, 0.0 if padding. -> used for existence masking in various components
    object_mask: chex.Array
    
    # Multi-Hot Encoding of Labels
    # Shape: [Batch, Max_Objs, Max_Labels] (Integer) -> can contain multiple labels per object!!
    # Semantic Labels: 0=COLLECT, 1=AVOID, 2=DESTROY, 3=DEFAULT (catch all for high level concepts, no object should have this label as this is in the global features!)
    object_labels: chex.Array 

    # Shape: [Batch, Max_Objs] (Integer)
    # Event Flags: 0=Not Triggered, 1=Triggered (e.g. collected a diver, got hit by a shark, etc.) meaning is label specific
    object_events: chex.Array 

    flat_raw_observations: chex.Array # Shape: [Batch, Flat_Obs_Size]

    # --- 3. Internal Agent State (The "Bus") ---
    # These are filled dynamically during the agent's forward pass.
    
    # The concept chosen by the Meta-Policy (i.e. 0=Collect)
    active_concept: chex.Array = None # Shape: [Batch] (Int)
    
    # The specific object index chosen by the Instance Selector
    active_target_idx: chex.Array = None # Shape: [Batch] (Int)

@struct.dataclass
class MultiReward:
    """
    Holds rewards for every possible target to allow instance-specific updates.
    """
    # Global Game Score (for Meta-Policy)
    global_reward: chex.Array # Shape: [Batch]
    
    # Per-Instance Rewards (for Sub-Policies & Selector)
    # Shape: [Batch, Num_Concepts, Max_Objs]
    # e.g., concept_rewards[0, 25] is the reward for collecting Diver #1.
    concept_rewards: chex.Array

@struct.dataclass
class AgentState:
    """
    Standardizes what the agent returns to the buffer/training loop.
    Needed because we need to store more than just 'action' for HRL updates. 
    """
    action: chex.Array
    selected_concept: chex.Array    # The output of Meta-Policy
    selected_target_idx: chex.Array # The output of Instance Selector
    log_probs: chex.Array # Q-values for the selected action