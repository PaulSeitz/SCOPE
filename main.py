import os
import time
import copy
import jax
import jax.numpy as jnp
import wandb
import hydra
from omegaconf import OmegaConf

from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer
from jaxatari import make as make_atari_env
from jaxatari.wrappers import AtariWrapper

# Import your custom architecture components
# (Adjust the import paths if your folder structure differs slightly)
from src.concept_based_learning.environments.seaquest.seaquest_label import SeaquestSemanticWrapper
from src.concept_based_learning.train_agent import make_train
from src.concept_based_learning.video_recorder import video_callback, wait_for_video_finish


def outer_make_train(config):
    """
    Constructs the Seaquest environments with the custom semantic wrappers
    and passes them to the training factory.
    """
    if config.get("ENV_NAME", "Seaquest") != "Seaquest":
        raise NotImplementedError("This pipeline is currently optimized for Seaquest.")

    sticky_actions = config.get("STICKY_ACTIONS", False)
    episodic_life = config.get("EPISODIC_LIFE", False)

    def create_env(train: bool = False):
        # 1. Base Environment
        env = make_atari_env('seaquest')
        
        # 2. Standard Atari Wrapper
        if train:
            env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life, max_episode_length=27000)
        else:
            env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            
        # 3. Custom Semantic Wrapper
        # This replaces the ObjectCentricWrapper and FlattenObservationWrapper
        # by directly injecting the DESTROY, COLLECT, and AVOID labels.
        env = SeaquestSemanticWrapper(env)
        
        return env

    env = create_env(train=True)
    test_env = create_env(train=False)
    renderer = SeaquestRenderer()

    # Call the hierarchical training loop; returns (train_fn, eval_rollout_fn)
    train_fn, eval_rollout_fn = make_train(config, env, test_env, renderer)
    return train_fn, eval_rollout_fn, renderer


def single_run(config):
    """
    Executes a single training run with Weights & Biases logging.
    """
    # Flatten: Hydra puts experiment config under "experiment", alg params under "experiment.alg"
    exp = config.get("experiment", {})
    config = {**config, **exp, **exp.get("alg", {})}

    alg_name = config.get("ALG_NAME", "hierarchical_pqn")
    env_name = config.get("ENV_NAME", "Seaquest")

    if config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            entity=config.get("ENTITY", None),
            project=config.get("PROJECT", "concept_based_learning"),
            tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
            name=f'{alg_name}_{env_name}',
            config=config,
            mode=config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(config["SEED"])
    t0 = time.time()
    rngs = jax.random.split(rng, config.get("NUM_SEEDS", 1))
    
    # Compile the training step
    print(f"Compiling training loop for {env_name}...")
    start_compile = time.time()
    train_fn, eval_rollout_fn, renderer = outer_make_train(config)
    train_vjit = jax.jit(jax.vmap(train_fn))

    # Provide empty initial params/batch_stats to match your train signature
    params, batch_stats = None, None
    train_vjit.lower(rngs, params, batch_stats).compile()
    print(f"Compilation took {time.time() - start_compile:.2f} seconds.")

    # Execute training
    outs = jax.block_until_ready(train_vjit(rngs, params, batch_stats))
    print(f"Training completed in {time.time() - t0:.2f} seconds.")

    # Optionally record a video at the end of training (e.g. for wandb)
    if config.get("RECORD_VIDEO") and config.get("VIDEO_AT_END", True) and eval_rollout_fn is not None:
        if config.get("WANDB_MODE", "disabled") != "disabled" and wandb.run is not None:
            print("Recording end-of-training video...")
            runner_state = outs["runner_state"]
            step = config.get("NUM_UPDATES", 0)
            states, active_agents, combined_qs, dones, target_obj_idxs, objects_bbox = eval_rollout_fn(
                runner_state, step, config.get("VIDEO_MAX_FRAMES", 2000)
            )
            video_callback(states, active_agents, combined_qs, dones, step, renderer,
                          target_obj_idxs=target_obj_idxs, objects_bbox=objects_bbox)
            wait_for_video_finish()
        else:
            print("RECORD_VIDEO/VIDEO_AT_END set but wandb disabled; skipping video.")

    # Optional: Save model parameters
    if config.get("SAVE_PATH", None) is not None:
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'),
        )
        print(f"Config saved to {save_dir}")


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config):
    # Convert OmegaConf to standard dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    print("Initializing training with configuration:\n", OmegaConf.to_yaml(config_dict))
    
    single_run(config_dict)


if __name__ == "__main__":
    main()