# src/rl_utils.py
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax import struct
import optax
import chex
from typing import Callable, Any, Optional

@struct.dataclass
class PQNTrainState(TrainState):
    """
    PQN State: No Target Params.
    PQN uses the current parameters with stop_gradient for bootstrapping.
    """
    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        return super().create(apply_fn=apply_fn, params=params, tx=tx, **kwargs)

# Backward compatibility alias (deprecated - use PQNTrainState)
RLTrainState = PQNTrainState


def create_optimizer(learning_rate: float, 
                     max_grad_norm: Optional[float] = None,
                     lr_linear_decay: bool = False,
                     total_steps: Optional[int] = None) -> optax.GradientTransformation:
    """
    Create PQN optimizer using Rectified Adam (RAdam) as per paper Appendix D.
    
    Args:
        learning_rate: Base learning rate
        max_grad_norm: Maximum gradient norm for clipping (PQN parameter). If None, no clipping.
        lr_linear_decay: Whether to linearly decay learning rate over training (PQN parameter)
        total_steps: Total training steps (required if lr_linear_decay is True)
    
    Returns:
        optax.GradientTransformation optimizer
    """
    # Create optimizer chain
    chain = []
    
    # Add gradient clipping first (before RAdam)
    if max_grad_norm is not None and max_grad_norm > 0:
        chain.append(optax.clip_by_global_norm(max_grad_norm))
    
    # PQN uses Rectified Adam (RAdam) as per Appendix D
    # Paper: "All experiments used Rectified Adam optimizer"
    if lr_linear_decay and total_steps is not None:
        # Linear decay from learning_rate to 0 over total_steps
        lr_schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0.0,
            transition_steps=total_steps
        )
        chain.append(optax.scale_by_radam())
        chain.append(optax.scale_by_schedule(lr_schedule))
    else:
        # Fixed learning rate - use radam with learning_rate directly
        chain.append(optax.radam(learning_rate=learning_rate))
    
    return optax.chain(*chain) if len(chain) > 1 else chain[0]

def compute_q_lambda_returns(rewards: chex.Array, 
                             next_q_values: chex.Array,
                             dones: chex.Array,
                             gamma: float,
                             lambda_: float) -> chex.Array:
    """
    Compute Q(λ) returns using Backward Recursion (Algorithm 2 from PQN paper).
    
    PQN Algorithm 2: Recursive backward computation
    R[t] = r[t] + gamma * (1-d[t]) * (lambda * R[t+1] + (1-lambda) * max Q(s'))
    
    This is more efficient and numerically stable than the forward-view loop.
    
    Args:
        rewards: [T, Batch] - rewards at each step
        next_q_values: [T, Batch] - max Q-values of next states (from current network, with stop_gradient)
        dones: [T, Batch] - done flags
        gamma: Discount factor
        lambda_: Q(λ) blending parameter (0 = 1-step TD, 1 = Monte Carlo)
    
    Returns:
        q_lambda_targets: [T, Batch] - Q(λ) return targets for each transition
    """
    T, batch_size = rewards.shape
    
    # We scan backwards from T-1 down to 0
    # The 'carry' is the return from the next step (R[t+1])
    # Initialize carry with the bootstrap value of the final state
    # For the last step, there's no future return, so we use the Q-value directly
    last_val = next_q_values[-1]
    
    def scan_body(next_return, inputs):
        """
        Backward scan body: compute return for current step given next step's return.
        
        Args:
            next_return: R[t+1] - return from next step [Batch]
            inputs: tuple of (r[t], Q(s_{t+1}), done[t])
        
        Returns:
            current_return: R[t] - return for current step [Batch]
            current_return: (also returned as carry for next iteration)
        """
        r, q_next, done = inputs
        
        # PQN Recursive Formula (Algorithm 2)
        # Target = r + gamma * [ lambda * R_next + (1-lambda) * Q_next ]
        # This blends the recursive return R[t+1] with the bootstrap Q(s_{t+1})
        bootstrap = lambda_ * next_return + (1.0 - lambda_) * q_next
        current_return = r + gamma * (1.0 - done) * bootstrap
        
        return current_return, current_return
    
    # Stack inputs for scan: (rewards, next_q_values, dones)
    # Note: We need to reverse the order since scan with reverse=True processes from end to start
    xs = (rewards, next_q_values, dones)
    
    # Scan backwards: from T-1 down to 0
    # The scan processes elements in reverse order when reverse=True
    _, returns = jax.lax.scan(
        scan_body, 
        last_val,  # Initial carry: bootstrap value for final state
        xs, 
        reverse=True  # Process from T-1 down to 0
    )
    
    return returns


def double_q_learning_loss(q_online_fn: Callable, 
                           q_target_fn: Callable, 
                           params: chex.ArrayTree, 
                           target_params: chex.ArrayTree, 
                           obs: Any, 
                           next_obs: Any, 
                           actions: chex.Array, 
                           rewards: chex.Array, 
                           dones: chex.Array, 
                           gamma: float):
    """
    Generic JAX-compatible Double DQN Loss.
    
    Args:
        q_online_fn: Function(params, obs) -> [Batch, Action_Dim]
        q_target_fn: Function(params, obs) -> [Batch, Action_Dim]
        obs, next_obs: Any structure (Pytree) your network accepts
        actions: [Batch] (Indices of actions taken)
        rewards: [Batch]
        dones: [Batch] (0.0 or 1.0)
    """
    # 1. Online Q-Values for Current State
    q_values = q_online_fn(params, obs) 
    # Extract Q-value for the specific action taken
    q_pred = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)

    # 2. Target Calculation (Double DQN)
    # A. Selection: Use Online Params to pick best NEXT action
    next_q_online = q_online_fn(params, next_obs)
    next_best_actions = jnp.argmax(next_q_online, axis=-1)

    # B. Evaluation: Use Target Params to evaluate that action
    next_q_target = q_target_fn(target_params, next_obs)
    next_q_value = jnp.take_along_axis(next_q_target, next_best_actions[:, None], axis=-1).squeeze(-1)

    # 3. Bellman Target
    target = rewards + (1.0 - dones) * gamma * next_q_value
    
    # 4. Huber Loss (Robust regression)
    loss = optax.huber_loss(q_pred, target, delta=1.0).mean()
    
    return loss, {"q_mean": q_pred.mean(), "target_mean": target.mean(), "loss": loss}
