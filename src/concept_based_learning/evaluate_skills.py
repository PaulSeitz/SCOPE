import argparse
from typing import Tuple, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from concept_based_learning.egocentric_transformation import get_ego_obs
from concept_based_learning.environments.env_manager import EnvManager
from concept_based_learning.pqn_visualizer import (
    load_base_config,
    load_params_from_file,
    create_eval_env,
    label_to_skill_idx,
)
from concept_based_learning.train_agent import QNetwork


def evaluate_skills(
    config: dict,
    params_tree,
    episodes: int = 5,
    max_frames: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    Run greedy evaluation episodes with a fixed meta-policy and skill Q-networks.

    Only logs the raw environment return per episode (no training, no videos).
    """
    max_frames = max_frames or config.get("VIDEO_MAX_FRAMES", 2000)

    # Construct environment for evaluation.
    env, _renderer = create_eval_env(config)

    # Environment-specific hooks
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    components = manager.get_components(env_name)
    heuristic_target = components.get_heuristic_target
    no_target_idx = components.no_target_idx
    action_mapper = components.action_mapper

    # Shared Q-network (same architecture as during training/visualization)
    num_skills = config["NUM_SKILLS"]
    ego_action_dim = config["EGO_ACTION_DIM"]
    hidden_size = config["HIDDEN_SIZE"]
    num_layers = config["NUM_LAYERS"]
    seed = config["SEED"]
    restrict_fire_for_non_destroy = bool(
        config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False)
    )

    skill_network = QNetwork(
        action_dim=ego_action_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    rng = jax.random.PRNGKey(seed)
    returns: List[float] = []

    for ep in range(episodes):
        rng, rng_reset = jax.random.split(rng)
        obs, state = env.reset(rng_reset)

        # Infer ego feature dimension from a dummy egocentric observation.
        dummy_ego = get_ego_obs(
            obs.raw_observation,
            obs.objects,
            jnp.array(0, dtype=jnp.int32),
        )
        ego_feat_dim = int(dummy_ego.shape[-1])

        # Initialize parameters if none were provided (for completeness).
        local_params_tree = params_tree
        if local_params_tree is None:
            init_x = jnp.zeros(ego_feat_dim)
            network_vars = skill_network.init(
                jax.random.PRNGKey(seed + 1),
                init_x,
                train=False,
            )
            base_params = network_vars["params"]
            local_params_tree = jax.tree_util.tree_map(
                lambda p: jnp.stack([p] * num_skills, axis=0),
                base_params,
            )

        ep_return = 0.0
        prev_target_idx = no_target_idx

        for _ in range(max_frames):
            rng, step_key = jax.random.split(rng)

            # Heuristic meta-policy selects a target object.
            target_idx = heuristic_target(
                obs.raw_observation,
                obs.objects,
                prev_target_idx,
                step_key,
            )
            target_idx = int(jnp.ravel(target_idx)[0])
            prev_target_idx = target_idx

            if target_idx == no_target_idx:
                # No valid target on screen → NOOP.
                global_action = 0
            else:
                # Choose skill from the semantic label of the current target object.
                raw_label = obs.objects.semantic_label[-1, target_idx]
                active_skill_idx = int(label_to_skill_idx(raw_label, num_skills))

                ego_obs = get_ego_obs(
                    obs.raw_observation,
                    obs.objects,
                    jnp.array(target_idx, dtype=jnp.int32),
                )
                ego_obs_batch = ego_obs[jnp.newaxis, ...]

                def apply_one(params_for_skill):
                    return skill_network.apply(
                        {"params": params_for_skill},
                        ego_obs_batch,
                        train=False,
                    )

                q_vals = jax.vmap(apply_one)(local_params_tree)  # [num_skills, 1, A]
                q_vals = jnp.squeeze(q_vals, axis=1)  # [num_skills, A]

                # FIRE-related ego action indices in the shared 18D space.
                fire_indices = jnp.array(
                    [1, 10, 11, 12, 13, 14, 15, 16, 17], dtype=jnp.int32
                )

                q_active = q_vals[active_skill_idx]
                if restrict_fire_for_non_destroy and active_skill_idx != 0:
                    q_active = q_active.at[fire_indices].set(-jnp.inf)

                chosen_ego = int(jnp.argmax(q_active))
                global_action = int(
                    action_mapper(
                        chosen_ego,
                        obs.raw_observation,
                        jnp.array(target_idx, dtype=jnp.int32),
                    )
                )

            # Step environment.
            obs, state, reward, done, _info = env.step(state, global_action)
            ep_return += float(reward)

            if done:
                break

        print(f"Episode {ep + 1}: return = {ep_return:.2f}")
        returns.append(ep_return)

    mean_ret = float(np.mean(returns)) if returns else 0.0
    print(f"Mean return over {len(returns)} episodes: {mean_ret:.2f}")
    return returns, mean_ret


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained hierarchical skills and log episode returns."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Path to a .safetensors file containing skill Q-network parameters.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames per episode (defaults to VIDEO_MAX_FRAMES from config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (defaults to SEED from config).",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help=(
            "Override ENV_NAME from the loaded config. "
            "Selects which environment wrapper, meta-policy, and action mapping are used."
        ),
    )
    args = parser.parse_args()

    # Load base experiment configuration (global + experiment-specific).
    base_config = load_base_config()

    # Optional overrides from CLI.
    if args.env_name is not None:
        base_config["ENV_NAME"] = args.env_name
    if args.seed is not None:
        base_config["SEED"] = args.seed

    # Load parameters if a model path is provided.
    params_tree = None
    if args.model is not None:
        params_tree = load_params_from_file(args.model)

    evaluate_skills(
        base_config,
        params_tree=params_tree,
        episodes=args.episodes,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

