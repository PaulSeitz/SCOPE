# src/train_pqn.py
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import optax
import wandb
import time
import numpy as np
from tqdm import tqdm
from flax import struct
import chex
from functools import partial
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

from src.config import Config, get_default_config, load_config_from_file
from src.environments.seaquest.seaquest_wrapper import SeaquestConceptWrapper
from src.environments.seaquest.seaquest_rewards import _compute_rewards_jitted
from src.agent.orchestrator import RelationalAgent
from src.environments.seaquest.datatypes import ConceptObs, AgentState, MultiReward
import src.environments.seaquest.seaquest_constants as consts
from src.rl_utils import double_q_learning_loss, compute_q_lambda_returns
from jaxatari.wrappers import AtariWrapper
from jaxatari import make as make_atari_env


# Helper to log from inside JIT without blocking
def wandb_log_callback(metrics, step):
    # metrics is a dictionary of JAX values, converted to numpy by the callback
    log_dict = {k: float(v) for k, v in metrics.items()}
    log_dict["global_step"] = int(step)
    wandb.log(log_dict)


@struct.dataclass
class Transition:
    obs: ConceptObs
    action: chex.Array
    reward: MultiReward
    next_obs: ConceptObs
    done: chex.Array
    agent_state: AgentState
    q_lambda_target: chex.Array = None  # Q(λ) return target (optional, computed after rollout)

def get_final_layer_l2(params):
    """
    Extract final layer kernel weights for L2 regularization.
    PQN requires L2 regularization specifically on the final layer kernel (not bias).
    """
    # Flatten the parameter tree to get all leaves
    leaves = jax.tree_util.tree_leaves(params)
    # Iterate backwards to find the first 2D array (kernel matrices are 2D, biases are 1D)
    for leaf in reversed(leaves):
        if leaf.ndim == 2:  # Kernels are matrices, biases are vectors
            return jnp.sum(leaf ** 2)
    # Fallback: if no 2D array found, return 0.0 (should not happen if model is correct)
    return 0.0

class AgentUpdater:
    """Stateless update logic adapted for PQN batches."""
    
    @staticmethod
    def update_grad_masking(agent, batch_flat: Transition, global_step, phase_1_steps, phase_2_steps, gamma, lambda_):
        """
        Update all components with gradient masking based on training phase.
        
        Phase 1 (0 to phase_1_steps): Only Sub-Policy trains
        Phase 2 (phase_1_steps to phase_2_steps): Sub-Policy + Selector train
        Phase 3 (after phase_2_steps): All components train
        """
        batch_size = batch_flat.action.shape[0]
        num_concepts = agent.config.concepts.label_high_level_concept + 1
        concept_ids = batch_flat.agent_state.selected_concept
        is_avoid = (concept_ids == consts.LABEL_AVOID)
        target_idx = batch_flat.agent_state.selected_target_idx
        
        # --- 1. Compute Sub-Policy Loss & Gradients ---
        # Reward Extraction
        r_concept = jnp.take_along_axis(batch_flat.reward.concept_rewards, concept_ids[:, None, None], axis=1).squeeze(1)
        reward_std = jnp.take_along_axis(r_concept, target_idx[:, None], axis=1).squeeze(1)
        # Aggregated reward (for AVOID, we average over all threats)
        # Only count objects that are actually threats (have LABEL_AVOID)
        is_threat = (batch_flat.obs.object_labels[:, :, consts.LABEL_AVOID] > 0.5) & (batch_flat.obs.object_mask > 0.5)
        threat_count = jnp.sum(is_threat, axis=1) + 1e-8
        reward_agg = jnp.sum(r_concept * is_threat, axis=1) / threat_count
        specific_reward = jnp.where(is_avoid, reward_agg, reward_std)
        
        # Q-Value Helper (same as in update_sub)
        def get_all_q(params, obs, c_ids, t_indices):
            p_pos = obs.global_features[:, 5:7]
            target_obj = jnp.take_along_axis(obs.objects, t_indices[:, None, None], axis=1).squeeze(1)
            rel_vec_std = target_obj[:, 0:2] - p_pos
            
            concept_one_hot = jax.nn.one_hot(c_ids, num_classes=num_concepts)
            ctx_std = jnp.concatenate([obs.capabilities, concept_one_hot], axis=-1)
            q_std = agent.sub_state.apply_fn(params, obs.global_features, rel_vec_std, ctx_std)
            
            def get_agg_q_single(b_idx):
                labels = obs.object_labels[b_idx]
                is_threat = (labels[:, consts.LABEL_AVOID] > 0.5) & (obs.object_mask[b_idx] > 0.5)
                all_rel_vecs = obs.objects[b_idx, :, 0:2] - p_pos[b_idx]
                avoid_one_hot = jax.nn.one_hot(consts.LABEL_AVOID, num_classes=num_concepts)
                avoid_ctx = jnp.concatenate([obs.capabilities[b_idx], avoid_one_hot], axis=-1)
                n_objs = obs.objects.shape[1]
                glob_tile = jnp.tile(obs.global_features[b_idx][None, :], (n_objs, 1))
                ctx_tile = jnp.tile(avoid_ctx[None, :], (n_objs, 1))
                all_q = agent.sub_state.apply_fn(params, glob_tile, all_rel_vecs, ctx_tile)
                valid_threats = jnp.sum(is_threat) + 1e-8
                return jnp.sum(all_q * is_threat[:, None], axis=0) / valid_threats
            
            q_agg = jax.vmap(get_agg_q_single)(jnp.arange(batch_size))
            return jnp.where(is_avoid[:, None], q_agg, q_std)
        
        def sub_loss_fn(params):
            # A. Online Q (Current State)
            q_online = get_all_q(params, batch_flat.obs, concept_ids, target_idx)
            q_pred = jnp.take_along_axis(q_online, batch_flat.action[:, None], axis=-1).squeeze(-1)
            
            # B. Target Calculation (NO TARGET NETWORK - PQN Style)
            # Use current params, but Stop Gradient
            q_next = get_all_q(params, batch_flat.next_obs, concept_ids, target_idx)
            q_next = jax.lax.stop_gradient(q_next)  # <--- CRITICAL PQN CHANGE
            
            # For 1-step TD fallback (if Q(λ) not computed)
            next_actions = jnp.argmax(q_next, axis=-1)
            next_q_val = jnp.take_along_axis(q_next, next_actions[:, None], axis=-1).squeeze(-1)
            
            # Use Q(λ) target if provided, otherwise fall back to 1-step TD
            if batch_flat.q_lambda_target is not None:
                target = batch_flat.q_lambda_target
            else:
                # Standard 1-step TD target (equivalent to λ=0)
                target = specific_reward + (1.0 - batch_flat.done) * gamma * next_q_val
            
            # C. TD Loss
            td_loss = optax.huber_loss(q_pred, target, delta=1.0)
            
            # D. PQN Regularization: L2 on final layer
            # Paper Theorem 2: The coefficient must be 0.5 * (gamma/2)^2 to cancel bootstrap expansion
            pqn_l2_coeff = 0.5 * ((gamma / 2.0) ** 2)
            l2_loss = pqn_l2_coeff * get_final_layer_l2(params)
            
            return td_loss + l2_loss
        
        sub_grad_fn = jax.value_and_grad(lambda p: jnp.mean(sub_loss_fn(p)))
        loss_per_sample = sub_loss_fn(agent.sub_state.params)
        loss_sub, grads_sub = sub_grad_fn(agent.sub_state.params)
        
        # Compute losses per concept label
        concept_one_hot = jax.nn.one_hot(concept_ids, num_classes=num_concepts)
        concept_counts = jnp.sum(concept_one_hot, axis=0)  # [Num_Concepts]
        loss_per_concept = jnp.sum(loss_per_sample[:, None] * concept_one_hot, axis=0) / (concept_counts + 1e-8)
        # Set to NaN for concepts that were never selected in this batch (for cleaner logging)
        loss_per_concept = jnp.where(concept_counts > 0, loss_per_concept, jnp.nan)
        
        # Compute average rewards per concept label (for logging)
        reward_per_concept = jnp.sum(specific_reward[:, None] * concept_one_hot, axis=0) / (concept_counts + 1e-8)
        # Set to NaN for concepts that were never selected in this batch (for cleaner logging)
        reward_per_concept = jnp.where(concept_counts > 0, reward_per_concept, jnp.nan)
        
        # --- 2. Compute Meta-Policy Loss & Gradients ---
        def meta_loss_fn(params):
            # Labels are now part of objects, so no separate object_labels parameter needed
            q_online = agent.meta_state.apply_fn(params, batch_flat.obs.global_features, batch_flat.obs.objects, batch_flat.obs.object_mask)
            q_pred = jnp.take_along_axis(q_online, batch_flat.agent_state.selected_concept[:, None], axis=-1).squeeze(-1)
            
            # PQN: Use current params with stop_gradient
            q_next = agent.meta_state.apply_fn(params, batch_flat.next_obs.global_features, batch_flat.next_obs.objects, batch_flat.next_obs.object_mask)
            q_next = jax.lax.stop_gradient(q_next)
            
            next_action = jnp.argmax(q_next, axis=-1)
            next_q_val = jnp.take_along_axis(q_next, next_action[:, None], axis=-1).squeeze(-1)
            target = batch_flat.reward.global_reward + (1.0 - batch_flat.done) * gamma * next_q_val
            
            td_loss = optax.huber_loss(q_pred, target, delta=1.0)
            # PQN Regularization: L2 on final layer with paper-specified coefficient
            pqn_l2_coeff = 0.5 * ((gamma / 2.0) ** 2)
            l2_loss = pqn_l2_coeff * get_final_layer_l2(params)
            return td_loss + l2_loss
        
        meta_grad_fn = jax.value_and_grad(lambda p: jnp.mean(meta_loss_fn(p)))
        loss_meta, grads_meta = meta_grad_fn(agent.meta_state.params)
        
        # --- 3. Compute Selector Loss & Gradients (PQN-based) ---
        def sel_loss_fn(params):
            # Convert the Recorded Concept to One-Hot
            cmd_one_hot = jax.nn.one_hot(
                batch_flat.agent_state.selected_concept,
                num_classes=num_concepts
            )
            
            # 1. Compute Q_online(s, a)
            # Returns: [Batch, Max_Objs] with -1e9 in invalid slots
            q_values = agent.selector_state.apply_fn(
                params,
                batch_flat.obs.global_features,
                batch_flat.obs.objects,
                batch_flat.obs.object_mask,
                cmd_one_hot
            )
            
            # Gather Q-value of the Action Taken
            # If selected_target_idx points to an invalid object, this is -1e9
            q_pred = jnp.take_along_axis(q_values, batch_flat.agent_state.selected_target_idx[:, None], axis=1).squeeze(1)
            
            # --- SANITY CHECK 1: Was the selected target actually valid? ---
            # If we picked a 'Ghost' (Index 0 fallback when nothing existed), we can't train on it.
            # Check validity of the selected index
            valid_mask_batch = (batch_flat.obs.object_mask > 0.5).astype(jnp.float32)
            is_target_valid = jnp.take_along_axis(valid_mask_batch, batch_flat.agent_state.selected_target_idx[:, None], axis=1).squeeze(1)
            
            # 2. Compute Target (PQN: Use current params with stop_gradient)
            reward = batch_flat.reward.global_reward
            
            # Next State Q (PQN: current params with stop_gradient)
            q_next_all = agent.selector_state.apply_fn(
                params,
                batch_flat.next_obs.global_features,
                batch_flat.next_obs.objects,
                batch_flat.next_obs.object_mask,
                cmd_one_hot
            )
            q_next_all = jax.lax.stop_gradient(q_next_all)  # <--- CRITICAL PQN CHANGE
            
            # Next Concept Mask
            next_concept_mask = jnp.take_along_axis(
                batch_flat.next_obs.object_labels,
                batch_flat.agent_state.selected_concept[:, None, None],
                axis=2
            ).squeeze(2)
            
            # Next Valid Mask (Existence AND Concept Match)
            large_neg = -1e9
            next_existence_mask = (batch_flat.next_obs.object_mask > 0.5).astype(jnp.float32)
            next_valid_mask = (next_concept_mask * next_existence_mask).astype(jnp.float32)
            
            # Apply mask to Q-values
            q_next_masked = q_next_all + (1.0 - next_valid_mask) * large_neg
            
            # --- SANITY CHECK 2: Does ANY valid object exist next step? ---
            any_valid_next = jnp.max(next_valid_mask, axis=1) > 0.5
            
            # If valid objects exist, take Max. 
            # If NOT, return 0.0 (Terminal-like state for this decision). 
            # Do NOT propagate -1e9.
            next_val_raw = jnp.max(q_next_masked, axis=1)
            next_val = jnp.where(any_valid_next, next_val_raw, 0.0)
            
            target = reward + (1.0 - batch_flat.done) * gamma * next_val
            
            # 3. Calculate Loss
            td_loss = optax.huber_loss(q_pred, target, delta=1.0)
            
            # 4. Mask the Loss
            # If we selected an invalid target, the Q-value is garbage (-1e9).
            # Zero out the loss for this specific sample so it doesn't explode the gradients.
            masked_loss = td_loss * is_target_valid
            
            # 5. PQN Regularization: L2 on final layer with paper-specified coefficient
            pqn_l2_coeff = 0.5 * ((gamma / 2.0) ** 2)
            l2_loss = pqn_l2_coeff * get_final_layer_l2(params)
            
            return jnp.mean(masked_loss) + l2_loss
        
        sel_grad_fn = jax.value_and_grad(sel_loss_fn)
        loss_sel, grads_sel = sel_grad_fn(agent.selector_state.params)
        
        # --- 4. Create Freeze Masks (0.0 = Frozen, 1.0 = Active) ---
        is_phase_2 = global_step > phase_1_steps
        is_phase_3 = global_step > phase_2_steps
        
        # If forced concept mode is enabled, freeze meta and selector (only train sub-policy)
        force_concept_mode = agent.config.training.force_concept_mode
        force_concept_mask = jnp.where(force_concept_mode, 0.0, 1.0).astype(jnp.float32)
        
        mask_sel = (is_phase_2.astype(jnp.float32) * force_concept_mask)
        mask_meta = (is_phase_3.astype(jnp.float32) * force_concept_mask)
        
        # --- 5. Apply Masks to Gradients ---
        # Zero out gradients for frozen components
        grads_sel = jax.tree.map(lambda g: g * mask_sel, grads_sel)
        grads_meta = jax.tree.map(lambda g: g * mask_meta, grads_meta)
        
        # --- 6. Mask Losses for Frozen Components (for cleaner logging) ---
        # Set loss to NaN when frozen so it's clear the component isn't training
        # This helps distinguish between "loss is 0" and "component is frozen"
        loss_sel_masked = jnp.where(mask_sel > 0.5, loss_sel, jnp.nan)
        loss_meta_masked = jnp.where(mask_meta > 0.5, loss_meta, jnp.nan)
        
        # --- 7. Apply Updates ---
        # The optimizers will receive 0.0 gradients for frozen weights,
        # effectively acting as "no update" (momentum might drift slightly, but negligible).
        # PQN: No target network updates needed!
        new_sub = agent.sub_state.apply_gradients(grads=grads_sub)
        new_sel = agent.selector_state.apply_gradients(grads=grads_sel)
        new_meta = agent.meta_state.apply_gradients(grads=grads_meta)
        
        new_agent = agent.replace(sub_state=new_sub, selector_state=new_sel, meta_state=new_meta)
        return new_agent, loss_sub, loss_meta_masked, loss_sel_masked, loss_per_concept, reward_per_concept

def rollout_step(carry, unused, NUM_ENVS, env_step_vmap, compute_rewards_vmap, 
                 reward_scale, bonus_success, penalty_failure, aim_bonus_scale, global_score_scale, global_step):
    agent_c, env_state_c, obs_c, c_concept, c_target, rng, eps_sub_c, eps_meta_c = carry
    
    rng, rng_act = jax.random.split(rng)
    keys_act = jax.random.split(rng_act, NUM_ENVS)
    
    # A. Action Selection (Batched)
    agent_select_vmap = jax.vmap(agent_c.select_action, in_axes=(0, 0, 0, 0, None, None, None))
    agent_state = agent_select_vmap(obs_c, c_concept, c_target, keys_act, eps_meta_c, eps_sub_c, global_step)
    
    # B. Env Step
    action_idx = agent_state.action.astype(jnp.int32)
    next_obs, next_env_state, r_raw, done, info = env_step_vmap(env_state_c, action_idx)
    
    # C. Rewards
    multi_reward = compute_rewards_vmap(
        obs_c, next_obs, env_state_c, next_env_state,
        reward_scale, bonus_success, penalty_failure, aim_bonus_scale, global_score_scale
    )
    
    # D. Reset Logic
    next_concept = jnp.where(done, 0, agent_state.selected_concept)
    next_target = jnp.where(done, 0, agent_state.selected_target_idx)
    
    transition = Transition(
        obs=obs_c, action=action_idx, reward=multi_reward, 
        next_obs=next_obs, done=done, agent_state=agent_state,
        q_lambda_target=None  # Will be computed after rollout
    )
    
    new_carry = (agent_c, next_env_state, next_obs, next_concept, next_target, rng, eps_sub_c, eps_meta_c)
    return new_carry, (transition, r_raw, done)


@partial(jax.jit, static_argnames=['NUM_ENVS', 'ROLLOUT_LEN', 'EPOCHS', 'NUM_MINIBATCHES', 'gamma', 'lambda_', 
                                   'reward_scale', 'bonus_success', 'penalty_failure', 'aim_bonus_scale', 'global_score_scale',
                                   'phase_1_steps', 'phase_2_steps',
                                   'env_step_vmap', 'compute_rewards_vmap'])
def train_step(agent_carry, env_carry, global_step, NUM_ENVS, ROLLOUT_LEN, EPOCHS, NUM_MINIBATCHES, gamma, lambda_,
               reward_scale, bonus_success, penalty_failure, aim_bonus_scale, global_score_scale,
               phase_1_steps, phase_2_steps,
               env_step_vmap, compute_rewards_vmap):
    # Unpack
    env_state, obs, c_concept, c_target, rng, eps_sub, eps_meta = env_carry
    
    # 1. Rollout
    scan_carry = (agent_carry, env_state, obs, c_concept, c_target, rng, eps_sub, eps_meta)
    
    rollout_fn = partial(rollout_step, NUM_ENVS=NUM_ENVS, env_step_vmap=env_step_vmap, 
                         compute_rewards_vmap=compute_rewards_vmap,
                         reward_scale=reward_scale, bonus_success=bonus_success, 
                         penalty_failure=penalty_failure, aim_bonus_scale=aim_bonus_scale, 
                         global_score_scale=global_score_scale, global_step=global_step)
    
    final_carry, (batch, rewards, dones) = jax.lax.scan(
        rollout_fn, scan_carry, None, length=ROLLOUT_LEN
    )
    
    # Unpack final state
    _, next_env_state, next_obs, next_concept, next_target, next_rng, _, _ = final_carry
    
    # 1.5. Compute Q(λ) returns for the rollout
    # PQN relies on Q(λ) returns to compensate for the lack of a replay buffer.
    # We must compute bootstrap Q-values for the entire rollout using the CURRENT network.
    
    # Extract rewards and dones from rollout: [ROLLOUT_LEN, NUM_ENVS]
    rollout_dones = dones  # Already in correct shape
    
    # Flatten batch for Q-value computation
    flat_next_obs = jax.tree.map(lambda x: jnp.reshape(x, (ROLLOUT_LEN * NUM_ENVS, *x.shape[2:])), batch.next_obs)
    flat_concept = jnp.reshape(batch.agent_state.selected_concept, (ROLLOUT_LEN * NUM_ENVS,))
    flat_target_idx = jnp.reshape(batch.agent_state.selected_target_idx, (ROLLOUT_LEN * NUM_ENVS,))
    
    # Helper to get max Q-values for bootstrapping (reuse logic from AgentUpdater)
    def get_batch_max_q(obs_batch, concept_batch, target_batch):
        batch_size = obs_batch.global_features.shape[0]
        num_concepts = agent_carry.config.concepts.label_high_level_concept + 1
        is_avoid = (concept_batch == consts.LABEL_AVOID)
        
        # Standard Q computation
        p_pos = obs_batch.global_features[:, 5:7]
        target_obj = jnp.take_along_axis(obs_batch.objects, target_batch[:, None, None], axis=1).squeeze(1)
        rel_vec_std = target_obj[:, 0:2] - p_pos
        concept_one_hot = jax.nn.one_hot(concept_batch, num_classes=num_concepts)
        ctx_std = jnp.concatenate([obs_batch.capabilities, concept_one_hot], axis=-1)
        q_std = agent_carry.sub_state.apply_fn(agent_carry.sub_state.params, obs_batch.global_features, rel_vec_std, ctx_std)
        
        # Aggregated Q for AVOID
        def get_agg_q_single(b_idx):
            labels = obs_batch.object_labels[b_idx]
            is_threat = (labels[:, consts.LABEL_AVOID] > 0.5) & (obs_batch.object_mask[b_idx] > 0.5)
            all_rel_vecs = obs_batch.objects[b_idx, :, 0:2] - p_pos[b_idx]
            avoid_one_hot = jax.nn.one_hot(consts.LABEL_AVOID, num_classes=num_concepts)
            avoid_ctx = jnp.concatenate([obs_batch.capabilities[b_idx], avoid_one_hot], axis=-1)
            n_objs = obs_batch.objects.shape[1]
            glob_tile = jnp.tile(obs_batch.global_features[b_idx][None, :], (n_objs, 1))
            ctx_tile = jnp.tile(avoid_ctx[None, :], (n_objs, 1))
            all_q = agent_carry.sub_state.apply_fn(agent_carry.sub_state.params, glob_tile, all_rel_vecs, ctx_tile)
            valid_threats = jnp.sum(is_threat) + 1e-8
            return jnp.sum(all_q * is_threat[:, None], axis=0) / valid_threats
        
        q_agg = jax.vmap(get_agg_q_single)(jnp.arange(batch_size))
        q_all = jnp.where(is_avoid[:, None], q_agg, q_std)
        
        # Return max Q-value for each sample (for bootstrapping)
        return jnp.max(q_all, axis=-1)
    
    # Compute Q-values for all next states and apply stop_gradient (PQN requirement)
    next_q_values_flat = jax.lax.stop_gradient(get_batch_max_q(flat_next_obs, flat_concept, flat_target_idx))
    next_q_values = jnp.reshape(next_q_values_flat, (ROLLOUT_LEN, NUM_ENVS))
    
    # Extract concept-specific rewards (same logic as in update_grad_masking)
    # Use CURRENT obs and rewards (not next_obs) - rewards correspond to transitions from obs to next_obs
    flat_obs = jax.tree.map(lambda x: jnp.reshape(x, (ROLLOUT_LEN * NUM_ENVS, *x.shape[2:])), batch.obs)
    flat_reward = jax.tree.map(lambda x: jnp.reshape(x, (ROLLOUT_LEN * NUM_ENVS, *x.shape[2:])), batch.reward)
    
    def extract_specific_rewards(obs_batch, reward_batch, concept_batch, target_batch):
        batch_size = obs_batch.global_features.shape[0]
        num_concepts = agent_carry.config.concepts.label_high_level_concept + 1
        is_avoid = (concept_batch == consts.LABEL_AVOID)
        
        r_concept = jnp.take_along_axis(reward_batch.concept_rewards, concept_batch[:, None, None], axis=1).squeeze(1)
        reward_std = jnp.take_along_axis(r_concept, target_batch[:, None], axis=1).squeeze(1)
        is_threat = (obs_batch.object_labels[:, :, consts.LABEL_AVOID] > 0.5) & (obs_batch.object_mask > 0.5)
        threat_count = jnp.sum(is_threat, axis=1) + 1e-8
        reward_agg = jnp.sum(r_concept * is_threat, axis=1) / threat_count
        return jnp.where(is_avoid, reward_agg, reward_std)
    
    specific_rewards_flat = extract_specific_rewards(flat_obs, flat_reward, flat_concept, flat_target_idx)
    specific_rewards = jnp.reshape(specific_rewards_flat, (ROLLOUT_LEN, NUM_ENVS))
    
    # Compute Q(λ) returns using the helper function
    q_lambda_targets = compute_q_lambda_returns(
        rewards=specific_rewards,
        next_q_values=next_q_values,
        dones=rollout_dones,
        gamma=gamma,
        lambda_=lambda_
    )
    
    # Assign computed Q(λ) targets back to the batch
    q_lambda_flat = jnp.reshape(q_lambda_targets, (ROLLOUT_LEN * NUM_ENVS,))
    batch = batch.replace(q_lambda_target=q_lambda_flat)
    
    # 2. Updates
    # PQN convention: compute batch_size from num_minibatches
    total_steps = NUM_ENVS * ROLLOUT_LEN
    batch_size = total_steps // NUM_MINIBATCHES
    
    def update_epoch(carry, _):
        ag, rng_epoch = carry
        rng_epoch, rng_perm = jax.random.split(rng_epoch)
        perm = jax.random.permutation(rng_perm, total_steps)
        
        def get_minibatch(x):
            flat = jnp.reshape(x, (total_steps, *x.shape[2:]))
            shuffled = flat[perm]
            return jnp.reshape(shuffled, (NUM_MINIBATCHES, batch_size, *x.shape[2:]))

        shuffled_batch = jax.tree.map(get_minibatch, batch)
        
        def update_minibatch(ag_curr, minibatch):
            new_ag, l_sub, l_meta, l_sel, l_sub_per_concept, r_sub_per_concept = AgentUpdater.update_grad_masking(
                ag_curr, minibatch, global_step, phase_1_steps, phase_2_steps, gamma, lambda_
            )
            return new_ag, (l_sub, l_meta, l_sel, l_sub_per_concept, r_sub_per_concept)

        new_ag, (l_sub_batch, l_meta_batch, l_sel_batch, l_sub_per_concept_batch, r_sub_per_concept_batch) = jax.lax.scan(
            update_minibatch, ag, shuffled_batch
        )
        # Average losses and rewards per concept across minibatches
        avg_loss_per_concept = jnp.mean(l_sub_per_concept_batch, axis=0)  # [Num_Concepts]
        avg_reward_per_concept = jnp.mean(r_sub_per_concept_batch, axis=0)  # [Num_Concepts]
        return (new_ag, rng_epoch), (jnp.mean(l_sub_batch), jnp.mean(l_meta_batch), jnp.mean(l_sel_batch), avg_loss_per_concept, avg_reward_per_concept)

    epoch_carry = (agent_carry, next_rng)
    (new_agent, _), (loss_sub, loss_meta, loss_sel, loss_sub_per_concept, reward_sub_per_concept) = jax.lax.scan(update_epoch, epoch_carry, None, length=EPOCHS)
    
    avg_reward = jnp.mean(jnp.sum(rewards, axis=0))
    new_env_carry = (next_env_state, next_obs, next_concept, next_target, next_rng, eps_sub, eps_meta)
    
    # Return metrics as a dict with fixed keys + raw arrays for per-concept losses and rewards
    # We'll expand the per-concept metrics in Python code outside JIT
    return new_agent, new_env_carry, {
        "loss_sub": loss_sub[-1],
        "loss_meta": loss_meta[-1],
        "loss_sel": loss_sel[-1],
        "avg_episodic_return": avg_reward,
        "loss_sub_per_concept": loss_sub_per_concept[-1],  # [Num_Concepts] array
        "reward_sub_per_concept": reward_sub_per_concept[-1]  # [Num_Concepts] array
    }


def create_train_scan(total_updates, config, NUM_ENVS, ROLLOUT_LEN, env_step_vmap, compute_rewards_vmap):
    # Extract static config values
    eps_start = config.training.epsilon_start
    eps_end = config.training.epsilon_end
    steps_per_update = NUM_ENVS * ROLLOUT_LEN
    log_freq = config.wandb.log_interval
    
    # Update parameters
    EPOCHS = config.training.epochs
    NUM_MINIBATCHES = config.training.num_minibatches
    gamma = config.training.gamma
    lambda_ = config.training.lambda_
    reward_scale = config.reward.reward_scale
    bonus_success = config.reward.bonus_success
    penalty_failure = config.reward.penalty_failure
    aim_bonus_scale = config.reward.aim_bonus_scale
    global_score_scale = config.reward.global_score_scale
    phase_1_steps = config.training.phase_1_steps
    phase_2_steps = config.training.phase_2_steps
    total_steps = config.training.total_steps
    
    # Concept labels for logging
    # Must match the order in seaquest_constants.py:
    # LABEL_COLLECT = 0, LABEL_AVOID = 1, LABEL_DESTROY = 2, LABEL_DEFAULT = 3
    num_concepts = config.concepts.label_high_level_concept + 1
    concept_labels = ["COLLECT", "AVOID", "DESTROY", "DEFAULT"]

    @jax.jit
    def train_scan_step(carry, iteration_idx):
        agent_c, env_carry, global_step, rng = carry

        # --- DYNAMIC EPSILON SCHEDULES ---
        # Decoupled schedules for Sub-Policy (muscles) and Meta/Selector (brain)
        
        # 1. Sub-Policy Schedule (Fast Decay during Phase 1)
        # We want it to be fully trained by end of Phase 1 so the "muscles" are competent
        # before the "brain" tries to use them.
        progress_sub = jnp.minimum(1.0, global_step / phase_1_steps)
        current_eps_sub = jnp.maximum(eps_end, eps_start - progress_sub * (eps_start - eps_end))
        
        # 2. Meta/Selector Schedule (Delayed Start)
        # We hold it high until Phase 2 starts, then decay slowly.
        # This covers both Phase 2 (Selector) and Phase 3 (Meta).
        # This prevents the "brain" from being "born senile" - it needs to explore
        # when it wakes up, not exploit random initial weights.
        
        # Calculate progress starting ONLY after Phase 1 finishes
        steps_since_phase2 = jnp.maximum(0.0, global_step - phase_1_steps)
        
        # Decay over the remainder of training (from Phase 2 start to end)
        decay_duration_meta = total_steps - phase_1_steps
        progress_meta = jnp.minimum(1.0, steps_since_phase2 / decay_duration_meta)
        
        # Before Phase 2, keep epsilon high (at eps_start)
        # After Phase 2 starts, decay from eps_start to eps_end
        current_eps_meta = jnp.where(
            global_step < phase_1_steps,
            eps_start,  # Stay high during Phase 1
            jnp.maximum(eps_end, eps_start - progress_meta * (eps_start - eps_end))  # Decay after Phase 2 starts
        )
        
        # 3. Update Env Carry with SEPARATE epsilons
        env_state, obs, cc, ct, k, _, _ = env_carry
        env_carry_update = (env_state, obs, cc, ct, k, current_eps_sub, current_eps_meta)

        # 2. Run train step
        new_agent, new_env_carry, metrics_raw = train_step(
            agent_c, env_carry_update, global_step,
            NUM_ENVS=NUM_ENVS, ROLLOUT_LEN=ROLLOUT_LEN, EPOCHS=EPOCHS, 
            NUM_MINIBATCHES=NUM_MINIBATCHES, gamma=gamma, lambda_=lambda_,
            reward_scale=reward_scale, bonus_success=bonus_success, 
            penalty_failure=penalty_failure, aim_bonus_scale=aim_bonus_scale, 
            global_score_scale=global_score_scale,
            phase_1_steps=phase_1_steps, phase_2_steps=phase_2_steps,
            env_step_vmap=env_step_vmap, compute_rewards_vmap=compute_rewards_vmap
        )
        
        new_global_step = global_step + steps_per_update

        # 3. Asynchronous Logging with expanded per-concept losses
        # Capture num_concepts and concept_labels in closure
        def do_log(m_raw, s):
            # The callback will expand per-concept losses
            # num_concepts and concept_labels are captured from outer scope
            def expand_and_log(metrics_dict, step_val):
                log_dict = {
                    "loss_sub": float(metrics_dict["loss_sub"]),
                    "avg_episodic_return": float(metrics_dict["avg_episodic_return"]),
                    "global_step": int(step_val)
                }
                # Handle NaN for meta and selector losses (frozen components)
                # Only log if not NaN (wandb handles NaN, but skipping makes plots cleaner)
                loss_meta_val = metrics_dict["loss_meta"]
                loss_sel_val = metrics_dict["loss_sel"]
                if not (np.isnan(loss_meta_val) or np.isinf(loss_meta_val)):
                    log_dict["loss_meta"] = float(loss_meta_val)
                if not (np.isnan(loss_sel_val) or np.isinf(loss_sel_val)):
                    log_dict["loss_sel"] = float(loss_sel_val)
                
                # Expand per-concept losses
                # metrics_dict values are already numpy arrays from jax.debug.callback
                loss_per_concept = metrics_dict["loss_sub_per_concept"]
                reward_per_concept = metrics_dict["reward_sub_per_concept"]
                for i in range(num_concepts):
                    label_name = concept_labels[i] if i < len(concept_labels) else f"CONCEPT_{i}"
                    # Handle NaN values (concepts not selected in batch)
                    loss_val = loss_per_concept[i]
                    reward_val = reward_per_concept[i]
                    # Only log if not NaN (wandb will skip NaN automatically, but being explicit is clearer)
                    if not (np.isnan(loss_val) or np.isinf(loss_val)):
                        log_dict[f"loss_sub_{label_name}"] = float(loss_val)
                    if not (np.isnan(reward_val) or np.isinf(reward_val)):
                        log_dict[f"reward_sub_{label_name}"] = float(reward_val)
                wandb.log(log_dict)
            
            jax.debug.callback(expand_and_log, m_raw, s)

        jax.lax.cond(
            iteration_idx % log_freq == 0,
            do_log,
            lambda m, s: None,
            metrics_raw, new_global_step
        )
        
        return (new_agent, new_env_carry, new_global_step, rng), metrics_raw

    return train_scan_step


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Concept-Based Learning Agent')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (default: use default config)')
    args = parser.parse_args()
    
    # Resolve config path relative to project root if it's a relative path
    config_path = args.config
    if config_path is not None and not Path(config_path).is_absolute():
        config_path = str(project_root / config_path)
    
    # Load config from file if provided, otherwise use default
    config = load_config_from_file(config_path)
    
    # Print config source for debugging
    if args.config:
        print(f"Loaded config from: {config_path}")
    else:
        print("Using default config")
    
    # Extract training parameters from config
    NUM_ENVS = config.training.num_envs
    TOTAL_TIMESTEPS = config.training.total_steps
    ROLLOUT_LEN = config.training.rollout_len
    EPOCHS = config.training.epochs
    
    NUM_MINIBATCHES = config.training.num_minibatches
    
    print(f"Training configuration: {NUM_ENVS} environments, {TOTAL_TIMESTEPS:,} total steps, rollout_len={ROLLOUT_LEN}, epochs={EPOCHS}, num_minibatches={NUM_MINIBATCHES}")
    
    # Validate num_minibatches divides evenly (PQN convention)
    total_steps_per_update = NUM_ENVS * ROLLOUT_LEN
    if total_steps_per_update % NUM_MINIBATCHES != 0:
        raise ValueError(
            f"num_minibatches {NUM_MINIBATCHES} must divide total steps per update "
            f"({total_steps_per_update} = {NUM_ENVS} envs * {ROLLOUT_LEN} rollout_len). "
            f"Computed batch_size would be {total_steps_per_update // NUM_MINIBATCHES}"
        )
    
    # 1. Init WandB
    wandb.init(
        project=config.wandb.project,
        name=f"pqn-{config.wandb.name}" if config.wandb.name else "pqn-run",
        config=config.to_dict(),
        mode=config.wandb.mode
    )

    # 2. Setup Environment
    print(f"Initializing {NUM_ENVS} PQN environments...")
    base_env = make_atari_env(config.environment.game_name, mods_config=config.environment.game_mods)
    env_base = AtariWrapper(base_env, frame_skip=config.environment.frame_skip, frame_stack_size=config.environment.frame_stack_size)
    env = SeaquestConceptWrapper(env_base, config)
    
    # Reset
    key = jax.random.PRNGKey(config.training.seed)
    key_env, key_agent, key_train = jax.random.split(key, 3)
    
    # Vectorized Reset
    env_reset_vmap = jax.jit(jax.vmap(env.reset))
    env_step_vmap = jax.jit(jax.vmap(env.step))
    
    # Use the JIT-compiled version with static argnames marked
    # The static args (reward_scale, etc.) are already marked as static in _compute_rewards_jitted
    # but we also need to mark them as static in the outer jit when vmap is applied
    compute_rewards_vmap = jax.jit(
        jax.vmap(_compute_rewards_jitted, in_axes=(0, 0, 0, 0, None, None, None, None, None)),
        static_argnames=['reward_scale', 'bonus_success', 'penalty_failure', 'aim_bonus_scale', 'global_score_scale']
    )
    
    keys_env = jax.random.split(key_env, NUM_ENVS)
    obs, env_state = env_reset_vmap(keys_env)
    
    # 3. Setup Agent
    obs_sample = jax.tree.map(lambda x: x[0], obs)
    agent = RelationalAgent.create(key_agent, config, obs_sample)
    
    # State Carry for Scan
    current_concept = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    current_target_idx = jnp.zeros(NUM_ENVS, dtype=jnp.int32)

    # --- 4. THE MAIN SCAN LOOP ---
    global_step = 0
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_LEN)
    train_scan_fn = create_train_scan(num_updates, config, NUM_ENVS, ROLLOUT_LEN, env_step_vmap, compute_rewards_vmap)

    # Chunking Parameters
    chunk_size = 50 
    num_chunks = num_updates // chunk_size
    remainder = num_updates % chunk_size

    # Initial Carry
    train_env_carry = (env_state, obs, current_concept, current_target_idx, key_train, config.training.epsilon_start, config.training.epsilon_start)
    scan_carry = (agent, train_env_carry, global_step, key_train)

    print(f"Starting PQN Training: {num_chunks} chunks of {chunk_size} updates...")
    start_time = time.time()

    # Progress Bar (Total Updates)
    pbar = tqdm(total=num_updates)

    for chunk in range(num_chunks + (1 if remainder > 0 else 0)):
        current_chunk_size = chunk_size if chunk < num_chunks else remainder
        if current_chunk_size == 0: break
        
        # Calculate iteration indices for this chunk
        chunk_start = chunk * chunk_size
        iteration_indices = jnp.arange(chunk_start, chunk_start + current_chunk_size)

        # --- RUN ON GPU ---
        # This blocks only once per chunk!
        scan_carry, stacked_metrics = jax.lax.scan(
            train_scan_fn, 
            scan_carry, 
            iteration_indices, 
            length=current_chunk_size
        )
        
        # Update Python Variables
        agent, train_env_carry, global_step, _ = scan_carry
        
        # Update Progress Bar (SPS calculation) - avoid accessing stacked_metrics to prevent CPU-GPU transfer
        elapsed = time.time() - start_time
        sps = int(global_step) / (elapsed + 1e-8)
        pbar.update(current_chunk_size)
        pbar.set_description(f"SPS: {int(sps)} | Step: {int(global_step)}")

    pbar.close()
    wandb.finish()
    
    # --- 7. SAVE MODEL ---
    print("\nSaving trained model...")
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"agent_{timestamp}.pkl"
    
    # Convert JAX arrays to numpy for pickle compatibility
    # Note: This is the only place we need CPU-GPU transfer, and it's only at the end
    def to_numpy(x):
        if isinstance(x, jnp.ndarray):
            return np.array(x)
        return x
    
    model_data = {
        "meta_params": jax.tree_util.tree_map(to_numpy, agent.meta_state.params),
        "selector_params": jax.tree_util.tree_map(to_numpy, agent.selector_state.params),
        "sub_params": jax.tree_util.tree_map(to_numpy, agent.sub_state.params),
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {model_path}")
    print("\nTraining complete! View training progress in wandb.")

if __name__ == "__main__":
    main()