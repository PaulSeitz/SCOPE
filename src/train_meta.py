# type: ignore
import os
import time
import copy
import jax
import jax.numpy as jnp
import wandb
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from environments.seaquest.seaquest_wrapper import SeaquestConceptWrapper
from pqn_meta import make_train as make_train_pqn_meta
from video_recorder import wait_for_video_finish

import jaxatari
from jaxatari.wrappers import AtariWrapper, LogWrapper

import os
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Dict, Union
from tqdm import tqdm

# --- UTILS ---
def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")

def load_subpolicy_params(config, task_name):
    """
    Helper to load specific sub-policy parameters.
    Assumes standard file naming from previous runs.
    """
    alg_name = config.get("SUB_ALG_NAME", "pqn") # Allow different alg name for subs
    env_name = config["ENV_NAME"]
    
    # Construct path (Adjust this logic to match where you actually saved them)
    # Example: checkpoints/pqn_Seaquest_collect/pqn_Seaquest_collect_seed0_vmap0.safetensors
    # You might need to provide a specific SUBPOLICY_PATH in config if this automatic logic fails
    base_path = config.get("SUBPOLICY_PATH", config["SAVE_PATH"])
    
    # We try to find the folder that contains the specific task
    # This is a bit of a hack, in production you'd point to exact files
    search_dir = os.path.join(base_path, f"{alg_name}_{env_name}_{task_name}")
    
    # Fallback: if specific folder doesn't exist, try looking in base
    if not os.path.exists(search_dir):
        print(f"Warning: Specific folder {search_dir} not found, looking in {base_path}")
        search_dir = base_path

    filename = f'{alg_name}_{env_name}_{task_name}_seed0_vmap0.safetensors'
    path = os.path.join(search_dir, filename)
    
    if not os.path.exists(path):
        print(f"WARNING: Could not find subpolicy for {task_name} at {path}. Using Random Init.")
        return None
        
    print(f"Loading {task_name} subpolicy from {path}")
    return load_params(path)

# --- ENVIRONMENT WRAPPER ---
class MultiRewardSeaquestWrapper(object):
    """
    Exposes ALL rewards (Collect, Destroy, Avoid, Game) simultaneously.
    Returns observation as [Raw_Obs | Global_Feats | Object_Feats].
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.stack_size = 4 
        
        # Get Raw Size from Config (injected by outer_make_train)
        self.raw_size = self.config.get("RAW_OFFSET", 0)
        
        # Calculate Concept Size
        num_concepts = self.config["concepts"]["label_high_level_concept"] + 1
        n_obj_feats = self.config["environment"]["n_obj_feats"] + num_concepts
        concept_size = (self.config["environment"]["n_global_feats"] * self.stack_size + 
                        self.config["environment"]["max_objs"] * self.stack_size * n_obj_feats)
        
        # Total Shape = Raw + Concepts
        self.meta_obs_shape = (self.raw_size + concept_size, )

    def reset(self, rng):
        concept_obs, env_state = self.env.reset(rng)
        
        # We return the ConceptObs object directly if we can, but gym expects arrays.
        # We will return the Flattened Meta Observation here.
        obs = self._get_meta_obs(concept_obs)
        return obs, env_state

    def step(self, env_state, action):
        # 'action' is the primitive joystick action from the Sub-Policy
        concept_obs, next_state, env_reward, done, info = self.env.step(env_state, action)
        
        # 1. Calculate Specific Skill Rewards
        # Destroy: Enemy event (0-24)
        rew_destroy = jnp.max(concept_obs.object_events[0:24]).astype(jnp.float32)
        
        # Collect: Diver event (24-28)
        rew_collect = jnp.max(concept_obs.object_events[24:28]).astype(jnp.float32)
        
        # Avoid: Negative life lost
        life_lost = env_state.env_state.lives - next_state.env_state.lives
        # Standard avoid reward: -1 if hit, small positive survival bonus otherwise
        rew_avoid = jnp.where(life_lost > 0, -1.0, 0.001).astype(jnp.float32)

        # 2. Meta Reward (Actual Game Score)
        rew_meta = jnp.clip(env_reward, -1.0, 1.0) # Standard Atari practice

        # 3. Concatenate: [Collect, Destroy, Avoid, Meta]
        # We pack this into the 'reward' field. 
        # The training loop will unpack it.
        rewards = jnp.stack([rew_collect, rew_destroy, rew_avoid, rew_meta])
        
        obs = self._get_meta_obs(concept_obs)
        
        return obs, next_state, rewards, done, info

    def _get_meta_obs(self, concept_obs):
        """Returns [Raw_Obs | Global_Feats | Object_Feats]"""
        return jnp.concatenate([
            concept_obs.flat_raw_observations,
            concept_obs.global_features.flatten(),
            concept_obs.objects.flatten()
        ])

    def observation_space(self):
        class Space: shape = self.meta_obs_shape
        return Space()
    def action_space(self):
        return self.env.action_space()

# --- MAIN SETUP ---
def outer_make_train(config):
    sticky_actions = config.get("STICKY_ACTIONS", False)
    episodic_life = config.get("EPISODIC_LIFE", False)
    
    # Calculate RAW_OFFSET automatically
    # Create a dummy env to measure the flattened pixel size
    def calculate_offset():
        dummy_env = jaxatari.make("seaquest")
        dummy_env = AtariWrapper(dummy_env, sticky_actions=False, episodic_life=False)
        dummy_env = SeaquestConceptWrapper(dummy_env)
        
        rng = jax.random.PRNGKey(0)
        concept_obs, _ = dummy_env.reset(rng)
        return int(concept_obs.flat_raw_observations.shape[0])
    
    config["RAW_OFFSET"] = calculate_offset()
    print(f"Meta-Policy Input Size: {config['RAW_OFFSET']} (Pixels Only)")
    
    # Load Sub-Policies
    # We attempt to load them. If not found, we will initialize random ones in make_train.
    p_collect = load_subpolicy_params(config, "collect")
    p_destroy = load_subpolicy_params(config, "destroy")
    p_avoid   = load_subpolicy_params(config, "avoid")
    
    pretrained_params = None
    if p_collect and p_destroy and p_avoid:
        # Stack them: [3, ...]
        # This requires them to have identical architecture keys
        pretrained_params = jax.tree.map(
            lambda x, y, z: jnp.stack([x, y, z], axis=0), 
            p_collect, p_destroy, p_avoid
        )
        print("Successfully stacked pre-trained parameters.")
    else:
        print("WARNING: Could not load all sub-policies. Training skills from scratch.")

    def calculate_offset():
        dummy_env = jaxatari.make("seaquest")
        dummy_env = AtariWrapper(dummy_env, sticky_actions=False, episodic_life=False)
        dummy_env = SeaquestConceptWrapper(dummy_env)
        
        rng = jax.random.PRNGKey(0)
        concept_obs, _ = dummy_env.reset(rng)
        return concept_obs.flat_raw_observations.shape[0]

    config["RAW_OFFSET"] = int(calculate_offset())
    print(f"Detected Raw Observation Size: {config['RAW_OFFSET']}")

    def create_env(train: bool = False):
        env = jaxatari.make("seaquest")
        if train:
            env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life)
        else:
            env = AtariWrapper(env, sticky_actions=False, episodic_life=False)

        env = SeaquestConceptWrapper(env)
        # Use our new Multi-Reward Wrapper
        env = MultiRewardSeaquestWrapper(env, config) 
        env = LogWrapper(env)
        return env

    env = create_env(True)
    renderer = env._env.env._env.renderer
    test_env = create_env(False)

    return make_train_pqn_meta(
        config,
        env,
        test_env,
        test_env,
        pretrained_params, # Pass params here
        renderer,
    )
        
def single_run(config):
    alg_name = config.get("ALG_NAME", "hierarchical_pqn")
    env_name = config["ENV_NAME"]
    
    wandb.init(
        entity=config["ENTITY"], project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper()],
        name=f'{alg_name}_{env_name}_meta', config=config, mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng_init = jax.random.split(rng, config["NUM_SEEDS"])
    
    # 1. Get Functions
    # CALL outer_make_train TO GET THE COMPILED FUNCTIONS
    init_fn, update_fn, eval_fn = outer_make_train(config)
    
    # 2. JIT Compile
    print("Compiling Init...")
    init_vjit = jax.jit(jax.vmap(init_fn))
    runner_state = init_vjit(rng_init)
    
    print("Compiling Update...")
    update_vjit = jax.jit(jax.vmap(update_fn))
    # Warmup to force compilation
    runner_state, metrics = update_vjit(runner_state, None) 
    
    print("Compiling Evaluation...")
    eval_vjit = jax.jit(jax.vmap(eval_fn, in_axes=(0, None, 0)))

    # 3. Python Training Loop
    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    eval_interval = max(1, int(num_updates * config.get("TEST_INTERVAL", 0.05)))
    # If interval >= num_updates, only evaluate at the end
    if eval_interval >= num_updates:
        eval_interval = num_updates

    print(f"Starting Training ({num_updates} updates)...")
    with tqdm(total=num_updates) as pbar:
        for update_i in range(num_updates):
            # This runs 1 update step (fast). Ctrl+C works in between these calls.
            runner_state, metrics = update_vjit(runner_state, None)
            
            # WandB Log
            log_metrics = {k: jnp.mean(v).item() for k, v in metrics.items()}
            if config["WANDB_MODE"] != "disabled":
                wandb.log(log_metrics, step=int(log_metrics["timesteps"]))
                
            # Evaluation & Video
            should_eval = (update_i % eval_interval == 0 and update_i > 0) or update_i == num_updates - 1
            if config.get("TEST_DURING_TRAINING", False) and should_eval:
                rng, _rng = jax.random.split(rng)
                _rngs = jax.random.split(_rng, config["NUM_SEEDS"])
                test_metrics = eval_vjit(runner_state, False, _rngs)
                
                test_log = {f"test/{k}": jnp.mean(v).item() for k, v in test_metrics.items()}
                if config["WANDB_MODE"] != "disabled":
                    wandb.log(test_log, step=int(log_metrics["timesteps"]))
            
            pbar.update(1)
            pbar.set_postfix(score=log_metrics["game_score"])

    # 4. Save Logic
    if config.get("SAVE_PATH", None) is not None:
        save_dir = os.path.join(config["SAVE_PATH"], f"{alg_name}_{env_name}_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(save_dir, 'config.yaml'))
        
        meta_state, skills_state = runner_state[0]
        
        for i in range(config["NUM_SEEDS"]):
            meta_params = jax.tree_util.tree_map(lambda x: x[i], meta_state.params)
            save_params(meta_params, os.path.join(save_dir, f'meta_seed{config["SEED"]}_vmap{i}.safetensors'))
            
            # Save Skills
            skill_params = jax.tree_util.tree_map(lambda x: x[i], skills_state.params)
            save_params(skill_params, os.path.join(save_dir, f'skills_seed{config["SEED"]}_vmap{i}.safetensors'))
            print(f"Saved params to {save_dir}")
            
    wait_for_video_finish()

@hydra.main(version_base=None, config_path=".", config_name="config_meta")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)

if __name__ == "__main__":
    main(None)