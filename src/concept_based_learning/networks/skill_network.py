import jax.numpy as jnp
from typing import Any

import chex
import flax.linen as nn
from flax.training.train_state import TrainState


class QNetwork(nn.Module):
    """Simple MLP Q-network for skills (System 1). Maps ego-centric features → action-values."""

    action_dim: int
    hidden_size: int = 64
    num_layers: int = 3
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda v: nn.LayerNorm()(v)
        else:
            normalize = lambda v: v

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Transition:
    """Single transition in hierarchical replay rollouts. Rewards stored for all skills."""

    target_idx: chex.Array
    ego_obs: chex.Array
    next_ego_obs: chex.Array
    ego_action: chex.Array
    active_skill: chex.Array
    q_val: chex.Array       # all skills' Q-values [num_skills, num_envs, ego_action_dim]
    rewards: chex.Array     # rewards for ALL skills [num_envs, num_skills]
    skill_done: chex.Array  # [num_envs, num_skills]
    done: chex.Array
    env_reward: chex.Array
    # Q-decomposition composition slots (K axis). None when composition disabled
    # or for tests/legacy paths; populated by rollouts that engage composition.
    # See environments.datatypes.CompositionSlots for field details.
    composition: Any = None


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
