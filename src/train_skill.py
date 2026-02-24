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
from pqn_skills import make_train as make_train_pqn_jaxatari
from video_recorder import wait_for_video_finish

import jaxatari
from jaxatari.wrappers import AtariWrapper, LogWrapper

import os
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Dict, Union

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")

# NOTE: temporary, just to test the sub-policy training
class SeaquestAdapter(object):
    def __init__(self, env, task_name, config):
        self.env = env
        self.task_name = task_name
        self.config = config

        # Logic matches SeaquestConceptWrapper
        self.stack_size = 4 # Standard Atari stack # TODO: should normally be taken from the env but fine for the test
        # This entire calculation still uses the original setup because I was lazy
        num_concepts = self.config["concepts"]["label_high_level_concept"] + 1
        self.n_obj_feats = self.config["environment"]["n_obj_feats"] + num_concepts
        
        # The flattened object vector is [Stack * Feats]. The last frame is at the end.
        self.latest_start_idx = (self.stack_size - 1) * self.n_obj_feats

        self.obs_shape = ((2 * self.stack_size) + (5 * self.stack_size),) 

    def reset(self, rng):
        # Reset the base environment
        concept_obs, env_state = self.env.reset(rng)
        obs = self._process_obs(concept_obs)  # Unpack obs and distance_reward
        return obs, env_state

    def step(self, env_state, action):
        concept_obs, next_state, _, done, info = self.env.step(env_state, action)

        # 1. FILTER OBSERVATION
        obs = self._process_obs(concept_obs)

        # 2. EXTRACT REWARD based on task (first try does not use custom rewards but only the object_events)
        if self.task_name == "collect":
            # Reward is 1.0 if any diver event (indices 24-28) is triggered
            event_reward = jnp.max(concept_obs.object_events[24:28]).astype(jnp.float32)
            reward = event_reward
        elif self.task_name == "destroy":
            # Reward is 1.0 if any enemy event (indices 0-24) is triggered
            reward = jnp.max(concept_obs.object_events[0:24]).astype(jnp.float32)
        elif self.task_name == "avoid":
            # calculate the reward based on if you lost a life this step
            life_lost = env_state.env_state.lives - next_state.env_state.lives # TODO: pull into wrapper
            reward = jnp.where(life_lost > 0, -1.0, 0.001).astype(jnp.float32)
        else:
            # crash for now
            raise ValueError(f"Unknown task name: {self.task_name}")
        
        return obs, next_state, reward, done, info

    def _process_obs(self, concept_obs):
        # 1. Get Player Position (History)
        globals_stack = concept_obs.global_features.reshape(self.stack_size, -1)
        p_x_seq = globals_stack[:, 5]
        p_y_seq = globals_stack[:, 6]
        
        p_x = p_x_seq[-1] 
        p_y = p_y_seq[-1]

        # 2. Reshape Objects
        n_objs = concept_obs.objects.shape[0]
        objs_reshaped = concept_obs.objects.reshape(n_objs, self.stack_size, -1)
        
        # 3. Identify Target Logic
        latest_objs = objs_reshaped[:, -1, :]
        obj_x_latest = latest_objs[:, 0]
        obj_y_latest = latest_objs[:, 1]
        
        # [Selection Logic]
        if self.task_name == "collect": start, end = 24, 28
        elif self.task_name == "destroy": start, end = 0, 24
        else: start, end = 0, 32 # avoid (Divers included per your request)

        target_indices = jnp.arange(35)
        type_mask = (target_indices >= start) & (target_indices < end)
        
        valid_mask = concept_obs.object_mask * type_mask

        dist_sq = (obj_x_latest - p_x)**2 + (obj_y_latest - p_y)**2
        masked_dist = jnp.where(valid_mask > 0.5, dist_sq, 1000.0)
        nearest_idx = jnp.argmin(masked_dist)
        is_valid = valid_mask[nearest_idx]

        # 4. Extract History
        target_history = objs_reshaped[nearest_idx] 
        
        # 5. Calculate Relative History
        rel_x_seq = (target_history[:, 0] - p_x_seq) * is_valid
        rel_y_seq = (target_history[:, 1] - p_y_seq) * is_valid
        
        # --- NEW: Extract Width and Height ---
        # Indices 2 and 3 in SeaquestConceptWrapper are w/SCREEN_WIDTH and h/SCREEN_HEIGHT
        w_seq = target_history[:, 2] * is_valid
        h_seq = target_history[:, 3] * is_valid
        
        valid_seq = jnp.full((self.stack_size,), is_valid)

        # Zero out absolute player pos
        p_x_seq = jnp.full_like(p_x_seq, 0.0)
        
        # 6. Flatten
        player_history = jnp.stack([p_x_seq, p_y_seq], axis=-1).flatten()

        # 7. Flatten Object History (Now includes W and H)
        # Input features increase from 3 (x,y,valid) to 5 (x,y,w,h,valid)
        stack_feats = jnp.stack([rel_x_seq, rel_y_seq, w_seq, h_seq, valid_seq], axis=-1).flatten()

        # 8. Concatenate
        return jnp.concatenate([player_history, stack_feats])
    
    # Forward other required methods
    def observation_space(self):
        class Space: shape = self.obs_shape
        return Space()
    def action_space(self):
        return self.env.action_space()


def outer_make_train(config):
    sticky_actions = config.get("STICKY_ACTIONS", False)
    episodic_life = config.get("EPISODIC_LIFE", False)
    def create_env(train: bool = False):
        env = jaxatari.make("seaquest")
        if train:
            env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life)
        else:
            env = AtariWrapper(env, sticky_actions=False, episodic_life=False)

        env = SeaquestConceptWrapper(env)

        env = SeaquestAdapter(env, task_name=config.get("SUB_POLICY_TASK", "collect"), config=config)
        
        env = LogWrapper(env)

        return env

    env = create_env(True)
    renderer = env._env.env._env.renderer
    test_env = create_env(False)

    return make_train_pqn_jaxatari(
        config,
        env,
        test_env,
        test_env,
        None,
        renderer,
    )
        
def load_network_params(config):
    if config.get("LOAD_PARAMS", False) and config.get("LOAD_PATH", None) is not None:
        import os
        alg_name = config["ALG_NAME"]
        env_name = config["ENV_NAME"]
        load_dir = os.path.join(config["LOAD_PATH"], env_name)
        params = []
        batch_stats = []
        for i in range(config["NUM_SEEDS"]):
            load_path = os.path.join(
                load_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            inner_params = None
            inner_bs = None
            # check if file exists
            if os.path.exists(load_path):
                print(f"Loading params from {load_path}")
                inner_params = load_params(load_path)
                inner_bs = load_params(load_path.replace(".safetensors", "_bs.safetensors"))
            else:
                print(f"File {load_path} does not exist, using random params")
            params.append(inner_params)
            batch_stats.append(inner_bs)

        # transform to pytree
        params = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *params
        )
        batch_stats = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *batch_stats
        )

        return params, batch_stats

    return None, None

def single_run(config):

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]
    subpolicy_task = config.get("SUB_POLICY_TASK", "")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}_{subpolicy_task}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    params, batch_stats = load_network_params(config)
    train_vjit = jax.jit(jax.vmap(outer_make_train(config)))
    # time compilation
    print("Compiling...")
    start = time.time()
    # with jax.profiler.trace("outputs/jax-trace", create_perfetto_link=True):
    train_vjit.lower(rngs, params, batch_stats).compile()
    print(f"Compilation took {time.time()-start} seconds.")
    outs = jax.block_until_ready(train_vjit(rngs, params, batch_stats))
    print(f"Took {time.time()-t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        model_state = outs["runner_state"][0]
        
        # --- NEW SAVE LOGIC ---
        # 1. Create a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 2. Define the folder name: PQN_Seaquest_[concept]_[timestamp]
        folder_name = f"{alg_name}_{env_name}_{subpolicy_task}_{timestamp}"
        
        # 3. Create the full directory path inside SAVE_PATH
        save_dir = os.path.join(config["SAVE_PATH"], folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving model to {save_dir}")

        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            batch_stats = jax.tree_util.tree_map(lambda x: x[i], model_state.batch_stats)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_{subpolicy_task}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_path_bs = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_{subpolicy_task}_seed{config["SEED"]}_vmap{i}_bs.safetensors',
            )
            save_params(params, save_path)
            save_params(batch_stats, save_path_bs)
            print(f"Saved params to {save_path}")

    wait_for_video_finish()


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    print("Running hyperparameter tuning with wandb...")
    default_config = dict(default_config)
    print(default_config)
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        params, batch_stats = load_network_params(config)
        train_vjit = jax.jit(jax.vmap(outer_make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs, params, batch_stats))


    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test/returned_episode_env_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "min": 0.00001,
                "max": 0.001,
            },
            "EPS_FINISH": {
                "min": 0.001,
                "max": 0.1,
            },
            "EPS_DECAY": {
                "min": 0.1,
                "max": 0.7,
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=100)


@hydra.main(version_base=None, config_path=".", config_name="config_skills")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main(None)