import os
import time
import copy
import jax
import jax.numpy as jnp
import wandb
from flax.traverse_util import flatten_dict
from safetensors.flax import save_file
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
    game_mods = config.get("GAME_MODS", None)

    def create_env(train: bool = False):
        # 1. Base Environment
        if game_mods:
            # Mods are configured as a list in Hydra; JaxAtari expects a set.
            env = make_atari_env('seaquest', mods=set(game_mods))
        else:
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
    base_config = {**config, **exp, **exp.get("alg", {})}

    alg_name = base_config.get("ALG_NAME", "hierarchical_pqn")
    env_name = base_config.get("ENV_NAME", "Seaquest")

    if base_config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            entity=base_config.get("ENTITY", None),
            project=base_config.get("PROJECT", "concept_based_learning"),
            tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
            name=f'{alg_name}_{env_name}',
            config=base_config,
            mode=base_config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(base_config["SEED"])
    num_seeds = base_config.get("NUM_SEEDS", 1)

    # --------------------------------------------------------------
    # Curriculum setup: decide which phases to run and their mods
    # --------------------------------------------------------------
    curriculum_enabled = base_config.get("CURRICULUM_ENABLED", False)
    if curriculum_enabled:
        phase_names = list(base_config.get("CURRICULUM_PHASES", ["collect", "destroy", "avoid"]))
        curriculum_mods = base_config.get("CURRICULUM_MODS", {})
        phases = [(name, curriculum_mods.get(name, [])) for name in phase_names]
    else:
        # Single phase with default environment (no explicit mods)
        phases = [("full", None)]

    total_timesteps = int(base_config.get("TOTAL_TIMESTEPS", 0))
    n_phases = len(phases)
    if n_phases > 1 and total_timesteps > 0:
        base_per_phase = total_timesteps // n_phases
        remainder = total_timesteps % n_phases
        phase_timesteps = [
            base_per_phase + (1 if i < remainder else 0) for i in range(n_phases)
        ]
    else:
        phase_timesteps = [total_timesteps] * n_phases

    global_t0 = time.time()
    prev_params = None  # Shared skill network weights across phases

    for phase_idx, ((phase_name, phase_mods), phase_steps) in enumerate(zip(phases, phase_timesteps)):
        print(f"\n=== Starting phase {phase_idx + 1}/{n_phases}: {phase_name} ===")

        phase_config = copy.deepcopy(base_config)
        if phase_steps > 0:
            phase_config["TOTAL_TIMESTEPS"] = phase_steps
        if phase_mods:
            phase_config["GAME_MODS"] = list(phase_mods)
        else:
            phase_config.pop("GAME_MODS", None)

        # Seed splitting per phase for reproducibility
        rng, rng_phase = jax.random.split(rng)
        rngs = jax.random.split(rng_phase, num_seeds)

        # Compile the training step for this phase
        print(f"Compiling training loop for {env_name} (phase: {phase_name})...")
        start_compile = time.time()
        train_fn, eval_rollout_fn, renderer = outer_make_train(phase_config)
        # Vectorize over seeds only; params/batch_stats are shared across seeds.
        train_vjit = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None)))

        params, batch_stats = prev_params, None
        train_vjit.lower(rngs, params, batch_stats).compile()
        print(f"Compilation for phase '{phase_name}' took {time.time() - start_compile:.2f} seconds.")

        # Execute training for this phase
        phase_t0 = time.time()
        outs = jax.block_until_ready(train_vjit(rngs, params, batch_stats))
        print(f"Phase '{phase_name}' training completed in {time.time() - phase_t0:.2f} seconds.")

        # Extract skill parameters from first seed to carry over to the next phase
        runner_state = outs["runner_state"]
        train_states, *_ = runner_state
        # train_states is batched over seeds on axis 0 when using vmap; take first seed
        first_train_states = jax.tree_util.tree_map(lambda x: x[0], train_states)
        prev_params = first_train_states.params

        # Optionally record a video at the end of each phase (e.g. for wandb)
        if (
            phase_config.get("RECORD_VIDEO")
            and phase_config.get("VIDEO_AT_END", True)
            and eval_rollout_fn is not None
        ):
            if phase_config.get("WANDB_MODE", "disabled") != "disabled" and wandb.run is not None:
                print(f"Recording end-of-phase video for '{phase_name}'...")
                step = phase_config.get("NUM_UPDATES", 0)
                states, active_agents, combined_qs, dones, target_obj_idxs, objects_bbox = eval_rollout_fn(
                    runner_state, step, phase_config.get("VIDEO_MAX_FRAMES", 2000)
                )
                video_key = f"video/{phase_idx + 1:02d}_{phase_name}"
                video_callback(
                    states,
                    active_agents,
                    combined_qs,
                    dones,
                    step,
                    renderer,
                    target_obj_idxs=target_obj_idxs,
                    objects_bbox=objects_bbox,
                    video_key=video_key,
                )
                wait_for_video_finish()
            else:
                print(
                    f"RECORD_VIDEO/VIDEO_AT_END set for phase '{phase_name}' "
                    "but wandb disabled; skipping video."
                )

    print(f"\nAll phases completed in {time.time() - global_t0:.2f} seconds.")

    # Optional: Save final config (hyperparameters) and model parameters to disk
    save_path = base_config.get("SAVE_PATH", None)
    if save_path is not None:
        save_dir = os.path.join(save_path, env_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        OmegaConf.save(
            base_config,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{base_config["SEED"]}_config.yaml'),
        )
        print(f"Config saved to {save_dir}")

        # Save final model parameters for visualization / reuse (one params tree per skill)
        if prev_params is not None:
            flat_params = flatten_dict(prev_params, sep=",")
            model_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{base_config["SEED"]}.safetensors',
            )
            save_file(flat_params, model_path)
            print(f"Model parameters saved to {model_path}")


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config):
    # Convert OmegaConf to standard dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    print("Initializing training with configuration:\n", OmegaConf.to_yaml(config_dict))
    
    single_run(config_dict)


if __name__ == "__main__":
    main()