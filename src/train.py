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
from src.rl_utils import double_q_learning_loss, soft_update_params
from jaxatari.wrappers import AtariWrapper
from jaxatari import make as make_atari_env



@struct.dataclass
class Transition:
    obs: ConceptObs
    action: chex.Array
    reward: MultiReward
    next_obs: ConceptObs
    done: chex.Array
    agent_state: AgentState

class AgentUpdater:
    """Stateless update logic adapted for PQN batches."""
    
    @staticmethod
    def update_sub(agent, batch: Transition, gamma, tau):
        # Flatten batch: [Rollout, Envs] -> [Batch]
        def flatten(x): return jnp.reshape(x, (-1, *x.shape[2:]))
        
        batch_flat = jax.tree.map(flatten, batch)
        
        # Prepare Inputs (Same logic as before)
        def prepare_inputs(obs, target_idx, concept_id):
            target_obj = jnp.take_along_axis(obs.objects, target_idx[:, None, None], axis=1).squeeze(1)
            p_pos = obs.objects[:, 0, 0:2]
            t_pos = target_obj[:, 0:2]
            rel_vec = t_pos - p_pos
            num_concepts = agent.config.concepts.label_high_level_concept + 1
            concept_one_hot = jax.nn.one_hot(concept_id, num_classes=num_concepts)
            context = jnp.concatenate([obs.capabilities, concept_one_hot], axis=-1)
            return rel_vec, context

        curr_rel, curr_ctx = prepare_inputs(batch_flat.obs, batch_flat.agent_state.selected_target_idx, batch_flat.agent_state.selected_concept)
        next_rel, next_ctx = prepare_inputs(batch_flat.next_obs, batch_flat.agent_state.selected_target_idx, batch_flat.agent_state.selected_concept)
        
        # Specific Reward Extraction
        r_concept = jnp.take_along_axis(batch_flat.reward.concept_rewards, batch_flat.agent_state.selected_concept[:, None, None], axis=1).squeeze(1)
        specific_reward = jnp.take_along_axis(r_concept, batch_flat.agent_state.selected_target_idx[:, None], axis=1).squeeze(1)

        # Loss Function
        def loss_fn(params, target_params):
            q_online = agent.sub_state.apply_fn(params, curr_rel, curr_ctx)
            q_target = agent.sub_state.apply_fn(target_params, next_rel, next_ctx)
            
            # Double Q Logic
            q_pred = jnp.take_along_axis(q_online, batch_flat.action[:, None], axis=-1).squeeze(-1)
            next_action = jnp.argmax(agent.sub_state.apply_fn(params, next_rel, next_ctx), axis=-1)
            next_q_val = jnp.take_along_axis(q_target, next_action[:, None], axis=-1).squeeze(-1)
            
            target = specific_reward + (1.0 - batch_flat.done) * gamma * next_q_val
            return jnp.mean(optax.huber_loss(q_pred, target, delta=1.0))

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(agent.sub_state.params, agent.sub_state.target_params)
        
        new_state = agent.sub_state.apply_gradients(grads=grads)
        new_state = soft_update_params(new_state, tau)
        return new_state, loss

    @staticmethod
    def update_meta(agent, batch: Transition, gamma, tau):
        def flatten(x): return jnp.reshape(x, (-1, *x.shape[2:]))
        batch_flat = jax.tree.map(flatten, batch)
        
        def loss_fn(params, target_params):
            q_online = agent.meta_state.apply_fn(params, batch_flat.obs.global_features, batch_flat.obs.objects, batch_flat.obs.object_labels, batch_flat.obs.object_mask)
            q_target = agent.meta_state.apply_fn(target_params, batch_flat.next_obs.global_features, batch_flat.next_obs.objects, batch_flat.next_obs.object_labels, batch_flat.next_obs.object_mask)
            
            q_pred = jnp.take_along_axis(q_online, batch_flat.agent_state.selected_concept[:, None], axis=-1).squeeze(-1)
            next_action = jnp.argmax(agent.meta_state.apply_fn(params, batch_flat.next_obs.global_features, batch_flat.next_obs.objects, batch_flat.next_obs.object_labels, batch_flat.next_obs.object_mask), axis=-1)
            next_q_val = jnp.take_along_axis(q_target, next_action[:, None], axis=-1).squeeze(-1)
            
            target = batch_flat.reward.global_reward + (1.0 - batch_flat.done) * gamma * next_q_val
            return jnp.mean(optax.huber_loss(q_pred, target, delta=1.0))

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(agent.meta_state.params, agent.meta_state.target_params)
        new_state = agent.meta_state.apply_gradients(grads=grads)
        new_state = soft_update_params(new_state, tau)
        return new_state, loss

    @staticmethod
    def update_selector(agent, batch: Transition):
        def flatten(x): return jnp.reshape(x, (-1, *x.shape[2:]))
        batch_flat = jax.tree.map(flatten, batch)

        def loss_fn(params):
            attn_weights = agent.selector_state.apply_fn(
                params, 
                batch_flat.obs.global_features, 
                batch_flat.obs.objects, 
                batch_flat.obs.object_labels,
                batch_flat.obs.object_mask
            )
            
            active_probs = jnp.take_along_axis(attn_weights, batch_flat.agent_state.selected_concept[:, None, None], axis=1).squeeze(1)
            chosen_prob = jnp.take_along_axis(active_probs, batch_flat.agent_state.selected_target_idx[:, None], axis=1).squeeze(1)
            
            # Hybrid Reward
            r_concept = jnp.take_along_axis(batch_flat.reward.concept_rewards, batch_flat.agent_state.selected_concept[:, None, None], axis=1).squeeze(1)
            hybrid_return = jnp.take_along_axis(r_concept, batch_flat.agent_state.selected_target_idx[:, None], axis=1).squeeze(1)
            
            # Standardize
            hybrid_return = (hybrid_return - jnp.mean(hybrid_return)) / (jnp.std(hybrid_return) + 1e-8)
            
            log_prob = jnp.log(chosen_prob + 1e-6)
            loss = -jnp.mean(log_prob * hybrid_return)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(agent.selector_state.params)
        new_state = agent.selector_state.apply_gradients(grads=grads)
        return new_state, loss

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
    
    print(f"Training configuration: {NUM_ENVS} environments, {TOTAL_TIMESTEPS:,} total steps, rollout_len={ROLLOUT_LEN}, epochs={EPOCHS}")
    
    # 1. Init WandB
    wandb.init(
        project=config.wandb.project,
        name=f"pqn-{config.wandb.name}" if config.wandb.name else "pqn-run",
        config=config.to_dict(),
        mode=config.wandb.mode
    )

    # 2. Setup Environment
    print(f"Initializing {NUM_ENVS} PQN environments...")
    base_env = make_atari_env(config.environment.game_name)
    env_base = AtariWrapper(base_env, frame_skip=config.environment.frame_skip, frame_stack_size=config.environment.frame_stack_size)
    env = SeaquestConceptWrapper(env_base, config)
    
    # Reset
    key = jax.random.PRNGKey(config.training.seed)
    key_env, key_agent, key_train = jax.random.split(key, 3)
    
    # Vectorized Reset
    env_reset_vmap = jax.jit(jax.vmap(env.reset))
    env_step_vmap = jax.jit(jax.vmap(env.step))
    
    # Extract config values for rewards (they're static/hashable)
    reward_scale = config.reward.reward_scale
    bonus_success = config.reward.bonus_success
    penalty_failure = config.reward.penalty_failure
    aim_bonus_scale = config.reward.aim_bonus_scale
    global_score_scale = config.reward.global_score_scale
    
    # Use the JIT-compiled version directly with extracted static values
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

    # --- 4. THE PQN ROLLOUT (Compiled Loop) ---
    def rollout_step(carry, unused):
        agent_c, env_state_c, obs_c, c_concept, c_target, rng, eps_sub_c, eps_meta_c = carry
        
        rng, rng_act = jax.random.split(rng)
        keys_act = jax.random.split(rng_act, NUM_ENVS)
        
        # A. Action Selection (Batched)
        # Note: We must vmap the method correctly
        agent_select_vmap = jax.vmap(agent_c.select_action, in_axes=(0, 0, 0, 0, None, None))
        agent_state = agent_select_vmap(obs_c, c_concept, c_target, keys_act, eps_meta_c, eps_sub_c)
        
        # B. Env Step
        action_idx = agent_state.action.astype(jnp.int32)
        next_obs, next_env_state, r_raw, done, info = env_step_vmap(env_state_c, action_idx)
        
        # C. Rewards
        multi_reward = compute_rewards_vmap(
            obs_c, next_obs, env_state_c, next_env_state,
            reward_scale, bonus_success, penalty_failure, aim_bonus_scale, global_score_scale
        )
        
        # D. Reset Logic
        # Stickiness reset on done
        next_concept = jnp.where(done, 0, agent_state.selected_concept)
        next_target = jnp.where(done, 0, agent_state.selected_target_idx)
        
        transition = Transition(
            obs=obs_c, action=action_idx, reward=multi_reward, 
            next_obs=next_obs, done=done, agent_state=agent_state
        )
        
        new_carry = (agent_c, next_env_state, next_obs, next_concept, next_target, rng, eps_sub_c, eps_meta_c)
        return new_carry, (transition, r_raw, done)

    # --- 5. THE TRAIN STEP (Compiled Update) ---
    @jax.jit
    def train_step(agent_carry, env_carry):
        # Unpack
        env_state, obs, c_concept, c_target, rng, eps_sub, eps_meta = env_carry
        
        # 1. Rollout
        scan_carry = (agent_carry, env_state, obs, c_concept, c_target, rng, eps_sub, eps_meta)
        final_carry, (batch, rewards, dones) = jax.lax.scan(
            rollout_step, scan_carry, None, length=ROLLOUT_LEN
        )
        
        # Unpack final state
        _, next_env_state, next_obs, next_concept, next_target, next_rng, _, _ = final_carry
        
        # 2. Updates (Repeated K times on the batch)
        # PQN/PPO Style: we train on the collected data multiple times
        # Note: PQN typically uses minibatches, but for simplicity/speed we take full batch first
        
        def update_epoch(ag, _):
            # 1. Update Sub
            # Note: update_sub returns (new_sub_state, loss), NOT the agent
            new_sub_state, l_sub = AgentUpdater.update_sub(ag, batch, config.training.gamma, config.training.tau)
            
            # Use .replace() to create a new agent with the updated state
            ag_sub = ag.replace(sub_state=new_sub_state)
            
            # 2. Update Meta (Use the agent with updated sub-policy)
            new_meta_state, l_meta = AgentUpdater.update_meta(ag_sub, batch, config.training.gamma, config.training.tau)
            ag_meta = ag_sub.replace(meta_state=new_meta_state)
            
            # 3. Update Selector (Use the agent with updated meta)
            new_sel_state, l_sel = AgentUpdater.update_selector(ag_meta, batch)
            ag_final = ag_meta.replace(selector_state=new_sel_state)
            
            return ag_final, (l_sub, l_meta, l_sel)

        # Run Epochs
        new_agent, (loss_sub, loss_meta, loss_sel) = jax.lax.scan(update_epoch, agent_carry, None, length=EPOCHS)
        
        # Metrics
        avg_reward = jnp.mean(jnp.sum(rewards, axis=0)) # approx sum over rollout
        
        new_env_carry = (next_env_state, next_obs, next_concept, next_target, next_rng, eps_sub, eps_meta)
        
        # Return last losses
        return new_agent, new_env_carry, {
            "loss_sub": loss_sub[-1],
            "loss_meta": loss_meta[-1],
            "loss_sel": loss_sel[-1],
            "reward": avg_reward
        }

    # --- 6. MAIN LOOP ---
    print(f"Starting PQN Training: {TOTAL_TIMESTEPS} steps with {NUM_ENVS} envs...")
    
    # Init Epsilon values
    eps_sub = config.training.epsilon_start
    eps_meta = config.training.epsilon_start
    
    # Init Carry
    train_env_carry = (env_state, obs, current_concept, current_target_idx, key_train, eps_sub, eps_meta)
    
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_LEN)
    pbar = tqdm(range(num_updates))
    
    # Collect metrics for visualization
    metrics_history = {
        'rewards': [],
        'loss_sub': [],
        'loss_meta': [],
        'loss_sel': [],
        'steps': []
    }
    
    for i in pbar:
        # Decay Epsilon manually here (outside JIT)
        frac = i / num_updates
        current_eps_sub = max(config.training.epsilon_end, config.training.epsilon_start - frac * (config.training.epsilon_start - config.training.epsilon_end))
        current_eps_meta = max(config.training.epsilon_end, config.training.epsilon_start - frac * (config.training.epsilon_start - config.training.epsilon_end))
        
        # Update carry with new epsilons
        env_s, o, cc, ct, k, _, _ = train_env_carry
        train_env_carry = (env_s, o, cc, ct, k, current_eps_sub, current_eps_meta)
        
        start_t = time.time()
        agent, train_env_carry, metrics = train_step(agent, train_env_carry)
        jax.block_until_ready(metrics) # Wait for GPU
        
        # Logging
        sps = (NUM_ENVS * ROLLOUT_LEN) / (time.time() - start_t)
        pbar.set_description(f"SPS: {int(sps)} | R: {metrics['reward']:.2f}")
        
        wandb.log({
            "train/sps": sps,
            "train/reward_batch_mean": metrics['reward'],
            "train/loss_sub": metrics['loss_sub'],
            "train/loss_meta": metrics['loss_meta'],
            "train/loss_sel": metrics['loss_sel'],
            "train/epsilon": current_eps_sub
        })
        
        # Store metrics for visualization
        step = i * NUM_ENVS * ROLLOUT_LEN
        metrics_history['steps'].append(step)
        metrics_history['rewards'].append(float(metrics['reward']))
        metrics_history['loss_sub'].append(float(metrics['loss_sub']))
        metrics_history['loss_meta'].append(float(metrics['loss_meta']))
        metrics_history['loss_sel'].append(float(metrics['loss_sel']))

    wandb.finish()
    
    # --- 7. SAVE MODEL ---
    print("\nSaving trained model...")
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"agent_{timestamp}.pkl"
    
    # Convert JAX arrays to numpy for pickle compatibility
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
    
    # --- 8. VISUALIZE METRICS ---
    print("\nGenerating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot Rewards
    axes[0, 0].plot(metrics_history['steps'], metrics_history['rewards'], alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Reward per Update', fontweight='bold')
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Sub Policy Loss
    axes[0, 1].plot(metrics_history['steps'], metrics_history['loss_sub'], alpha=0.7, linewidth=1, color='orange')
    axes[0, 1].set_title('Sub Policy Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Meta Policy Loss
    axes[1, 0].plot(metrics_history['steps'], metrics_history['loss_meta'], alpha=0.7, linewidth=1, color='green')
    axes[1, 0].set_title('Meta Policy Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot Selector Loss
    axes[1, 1].plot(metrics_history['steps'], metrics_history['loss_sel'], alpha=0.7, linewidth=1, color='red')
    axes[1, 1].set_title('Selector Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = models_dir / f"training_metrics_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()