import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import Dict, List, Any
import chex

# --- PLACEHOLDERS ---
# env = YourVectorizedEnvWrapper(...)
# meta_policy = YourMetaPolicyLogic(...)
# --------------------

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    policy_id: chex.Array  # Which sub-policy acted at time t
    next_policy_id: chex.Array # Which sub-policy is selected for t+1
    q_val_matrix: chex.Array # [Num_Policies, Num_Actions] Q-values for all sub-policies

def make_multi_policy_train(config):
    # Initialize a Bank of sub-policies (e.g., Collect, Destroy, Avoid)
    sub_policy_ids = jnp.arange(config["NUM_SUB_POLICIES"])

    def _learn_phase(states: Dict[int, TrainState], traj_batch: Transition):
        """PQN synchronous update for all sub-policies in the bank."""

        # 1. Calculate Multi-Step Targets recursively (Point 4)
        def _get_target(carry, transition):
            lambda_returns, _ = carry

            # Select max Q from the policy that WILL be active at t+1
            next_q_vals = transition.q_val_matrix[transition.next_policy_id]
            next_max_q = jnp.max(next_q_vals, axis=-1)

            # PQN target with Q-lambda
            target = transition.reward + config["GAMMA"] * (1 - transition.done) * next_max_q
            delta = lambda_returns - next_max_q
            lambda_returns = target + config["GAMMA"] * config["LAMBDA"] * delta
            return (lambda_returns, next_max_q), lambda_returns

        _, targets = jax.lax.scan(_get_target, (traj_batch.reward[-1], 0.0), traj_batch, reverse=True)

        # 2. Update each policy using its specific transitions
        new_states = {}
        metrics = {}

        for pid in sub_policy_ids:
            state = states[pid]

            def _loss_fn(params):
                # Predict Q-values for this specific policy
                q_vals = state.apply_fn(params, traj_batch.obs)
                chosen_q = jnp.take_along_axis(q_vals, traj_batch.action[..., None], axis=-1).squeeze()

                # MASKING: Only learn from transitions belonging to this PID
                mask = (traj_batch.policy_id == pid).astype(jnp.float32)
                loss = jnp.sum(jnp.square(chosen_q - targets) * mask) / (jnp.sum(mask) + 1e-8)
                return loss

            grads = jax.grad(_loss_fn)(state.params)
            new_states[pid] = state.apply_gradients(grads=grads)
            metrics[f"loss_policy_{pid}"] = _loss_fn(state.params)

        return new_states, metrics

    def _update_step(runner_state, _):
        # runner_state contains the states for all policies in the bank
        states, env_state, last_obs, rng = runner_state

        # 1. Collect trajectory using Meta-Policy Selection
        def _env_step(carry, _):
            env_state, last_obs, rng = carry

            # --- PLACEHOLDER: META-POLICY SELECTION ---
            # current_policy_id = meta_policy(last_obs)
            # ------------------------------------------

            # Get action from the selected sub-policy
            active_state = states[current_policy_id]
            q_values = active_state.apply_fn(active_state.params, last_obs)
            action = jnp.argmax(q_values, axis=-1) # E-greedy logic omitted for brevity

            # Step environment
            next_obs, next_env_state, reward, done, info = env.step(env_state, action)

            # Pre-calculate Q-matrix for targeting (all policies evaluate current obs)
            # Efficiently done via vmap or list comprehension in JAX
            q_matrix = jnp.stack([states[p].apply_fn(states[p].params, next_obs) for p in sub_policy_ids])

            transition = Transition(
                obs=last_obs, action=action, reward=reward, done=done,
                policy_id=current_policy_id, next_policy_id=..., # predicted by meta
                q_val_matrix=q_matrix
            )

            return (next_env_state, next_obs, rng), transition

        runner_state_step, traj_batch = jax.lax.scan(_env_step, (env_state, last_obs, rng), None, length=config["NUM_STEPS"])

        # 2. Learn from the batch
        new_states, metrics = _learn_phase(states, traj_batch)

        return (new_states, *runner_state_step), metrics

    return _update_step
