# src/rl_utils.py
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax import struct
import optax
import chex
from typing import Callable, Any

@struct.dataclass
class RLTrainState(TrainState):
    """
    Extended TrainState that maintains a Target Network for Double DQN.
    """
    target_params: chex.ArrayTree
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, target_params=None, **kwargs):
        # If no specific target params provided, clone the online params
        if target_params is None:
            target_params = jax.tree_util.tree_map(lambda x: x, params)
            
        return super().create(apply_fn=apply_fn, params=params, tx=tx, target_params=target_params, **kwargs)

@jax.jit
def soft_update_params(state: RLTrainState, tau: float) -> RLTrainState:
    """
    Polyak Averaging: target = (1-tau)*target + tau*online
    """
    new_target_params = optax.incremental_update(
        new_tensors=state.params, 
        old_tensors=state.target_params, 
        step_size=tau
    )
    return state.replace(target_params=new_target_params)

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
