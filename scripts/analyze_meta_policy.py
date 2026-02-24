"""
Analyze Meta Policy Q-values & Visual Debugging
Loads a trained meta agent, visualizes Q-values, and renders a video with
bounding boxes around the selected targets.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict
import cv2  # For video generation and drawing
from tqdm import tqdm

# --- Import your framework components ---
from src.pqn_meta import make_train, extract_relative_obs, get_skill_id_from_target
from src.environments.seaquest.seaquest_wrapper import SeaquestConceptWrapper
import jaxatari
from jaxatari.wrappers import AtariWrapper, LogWrapper

# --- Wrapper Definition (Must match train_meta.py) ---
class MultiRewardSeaquestWrapper(object):
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.stack_size = 4 
        num_concepts = self.config["concepts"]["label_high_level_concept"] + 1
        self.n_obj_feats = self.config["environment"]["n_obj_feats"] + num_concepts
        self.obs_shape = ((2 * self.stack_size) + (5 * self.stack_size),) 
        self.meta_obs_shape = (self.config["environment"]["n_global_feats"] * self.stack_size + 
                          self.config["environment"]["max_objs"] * self.stack_size * self.n_obj_feats, )

    def reset(self, rng):
        concept_obs, env_state = self.env.reset(rng)
        obs = self._get_meta_obs(concept_obs)
        # Return concept_obs in info so we can access positions later
        return obs, env_state

    def step(self, env_state, action):
        concept_obs, next_state, env_reward, done, info = self.env.step(env_state, action)
        
        rew_destroy = jnp.max(concept_obs.object_events[0:24]).astype(jnp.float32)
        rew_collect = jnp.max(concept_obs.object_events[24:28]).astype(jnp.float32)
        life_lost = env_state.env_state.lives - next_state.env_state.lives
        rew_avoid = jnp.where(life_lost > 0, -1.0, 0.001).astype(jnp.float32)
        rew_meta = jnp.clip(env_reward, -1.0, 1.0)

        rewards = jnp.stack([rew_collect, rew_destroy, rew_avoid, rew_meta])
        obs = self._get_meta_obs(concept_obs)
        
        # HACK: Pass concept observation data in info for visualization
        info["concept_objects"] = concept_obs.objects
        info["global_features"] = concept_obs.global_features
        
        return obs, next_state, rewards, done, info

    def _get_meta_obs(self, concept_obs):
        return jnp.concatenate([concept_obs.global_features.flatten(), concept_obs.objects.flatten()])

    def observation_space(self):
        class Space: shape = self.meta_obs_shape
        return Space()
    def action_space(self):
        return self.env.action_space()

# --- Helper Functions ---

def load_params_from_safetensors(filename):
    """Load parameters from safetensors file."""
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")

def create_env(config, train: bool = False):
    """Recreate the exact environment stack."""
    sticky_actions = config.get("STICKY_ACTIONS", False)
    episodic_life = config.get("EPISODIC_LIFE", False)
    
    env = jaxatari.make("seaquest")
    # For analysis, force clean evaluation (no sticky actions/episodic life)
    env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
    env = SeaquestConceptWrapper(env)
    env = MultiRewardSeaquestWrapper(env, config) 
    env = LogWrapper(env)
    return env

def find_files_in_folder(folder_path):
    """Automatically find config and checkpoints in the folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Config
    config_path = folder / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {folder}")

    # Meta Checkpoint (look for *meta*seed0_vmap0.safetensors)
    meta_files = list(folder.glob("*meta*seed0_vmap0.safetensors"))
    if not meta_files:
        raise FileNotFoundError(f"Could not find meta checkpoint in {folder}")
    meta_ckpt = meta_files[0]

    # Skills Checkpoint (look for *skills*seed0_vmap0.safetensors)
    skills_files = list(folder.glob("*skills*seed0_vmap0.safetensors"))
    skills_ckpt = skills_files[0] if skills_files else None
    
    return str(config_path), str(meta_ckpt), str(skills_ckpt)

def analyze_meta_policy(folder_path, num_steps=600, visualize=True):
    
    # 1. Auto-detect paths
    config_path, meta_checkpoint_path, skills_checkpoint_path = find_files_in_folder(folder_path)
    print(f"--- Paths Detected ---")
    print(f"Config: {config_path}")
    print(f"Meta:   {meta_checkpoint_path}")
    print(f"Skills: {skills_checkpoint_path}")

    # 2. Load Configuration
    conf = OmegaConf.load(config_path)
    config = OmegaConf.to_container(conf)
    
    # Force single env for analysis
    config["TEST_NUM_ENVS"] = 1
    config["TEST_NUM_STEPS"] = num_steps
    config["EPS_TEST"] = 0.0  # Greedy
    
    # 3. Setup Environment
    print("Creating environment...")
    env = create_env(config, train=True) # Dummy for init
    test_env = create_env(config, train=False)
    renderer = env._env.env._env.renderer

    # 4. Load Parameters
    print(f"Loading Parameters...")
    meta_params = load_params_from_safetensors(meta_checkpoint_path)
    
    skill_params = None
    if skills_checkpoint_path:
        skill_params = load_params_from_safetensors(skills_checkpoint_path)
    else:
        print("WARNING: No skill checkpoint found. Skills will be random.")

    # 5. Initialize Agent
    init_agent, _, _ = make_train(
        config, env, test_env, test_env, 
        skill_params, 
        renderer
    )

    rng = jax.random.PRNGKey(config["SEED"])
    runner_state = init_agent(rng)
    (meta_state, skills_state), _, _, rng = runner_state
    
    meta_state = meta_state.replace(params=meta_params)
    if skill_params is not None:
        skills_state = skills_state.replace(params=skill_params)
        
    print("Agent initialized.")

    # 6. Define Analysis Loop
    n_global = config["environment"]["n_global_feats"]
    base_obj_feats = config["environment"]["n_obj_feats"]
    num_concepts = config["concepts"]["label_high_level_concept"] + 1

    test_vmap_reset = lambda n_envs: lambda rng: jax.vmap(test_env.reset)(jax.random.split(rng, n_envs))
    test_vmap_step = lambda env_state, action: jax.vmap(test_env.step)(env_state, action)

    def analysis_step(carry, _):
        env_state, last_obs, rng = carry
        rng, rng_meta = jax.random.split(rng)

        # Meta Pass
        meta_q = meta_state.apply_fn(
            {"params": meta_state.params, "batch_stats": meta_state.batch_stats}, 
            last_obs, train=False
        ) 
        meta_action = jnp.argmax(meta_q, axis=-1)

        # Router
        skill_indices = jax.vmap(get_skill_id_from_target)(meta_action)
        extract_fn = lambda o, t: extract_relative_obs(o, t, 4, n_global, base_obj_feats, num_concepts)
        sub_obs = jax.vmap(extract_fn)(last_obs, meta_action)

        # Skill Pass
        def apply_skill(params, obs): 
            return skills_state.apply_fn({"params": params}, obs, train=False)
        all_skill_q = jax.vmap(apply_skill, in_axes=(0, None))(skills_state.params, sub_obs)
        all_skill_q = jnp.transpose(all_skill_q, (1, 0, 2))
        active_q = jnp.take_along_axis(all_skill_q, skill_indices[:, None, None], axis=1).squeeze(1)
        primitive_action = jnp.argmax(active_q, axis=-1)

        # Execute
        new_obs, new_env_state, _, done, info = test_vmap_step(env_state, primitive_action)
        
        # Render Frame (Raw pixel data from Atari)
        # Note: jaxatari wrapper stores the underlying env in a complex way.
        # We rely on video_recorder's logic or renderer.render(env_state).
        # But we need to extract it per step here to draw on it.
        # We will assume 'env_state' can be rendered by the renderer object passed in.
        
        return (new_env_state, new_obs, rng), (
            info, 
            new_env_state, # Store full state for rendering
            skill_indices[0], 
            done[0], 
            meta_q[0],
            meta_action[0]
        )

    # 7. Run Evaluation
    print(f"Running {num_steps} steps...")
    rng, _rng = jax.random.split(rng)
    init_obs, env_state = test_vmap_reset(1)(_rng)
    
    scan_fn = jax.jit(lambda c: jax.lax.scan(analysis_step, c, None, num_steps))
    _, output = scan_fn((env_state, init_obs, _rng))
    infos, states, active_agents, dones, meta_qs, meta_actions = output

    # 8. Render Video with Annotations
    print("Generating Annotated Video...")
    
    # Process JAX arrays to Numpy
    meta_actions = np.array(meta_actions)
    active_agents = np.array(active_agents)
    dones = np.array(dones)
    
    # We need to extract object positions. We hacked 'info' to contain concept_objects.
    # Dimensions: [Steps, Batch, MaxObjs=35, Feats]
    concept_objects = np.array(infos["concept_objects"]) 
    
    # Setup Video Writer
    fps = 5
    height, width = 210, 160 # Atari standard
    video_path = os.path.join(folder_path, "meta_agent_debug.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height * 2)) # 2x scale

    for i in tqdm(range(len(meta_actions))):
        if dones[i]: break
        
        # 1. Render Base Frame
        # We use the renderer to get the image from the state
        # State is likely [Batch=1, ...]. We take [0].
        state_i = jax.tree.map(lambda x: x[i][0], states) 
        
        # Unpack the wrapper stack to get the raw EnvState
        # LogWrapper returns LogState with field 'atari_state'
        if hasattr(state_i, 'atari_state'):
            state_i = state_i.atari_state
        
        # Continue unpacking through wrapper chain
        # AtariWrapper returns AtariState with field 'env_state'
        if hasattr(state_i, 'env_state'):
            state_i = state_i.env_state
        
        frame = renderer.render(state_i) # Returns (H, W, 3) or (H, W)
        frame = np.array(frame)
        
        # Normalize/Convert to uint8 RGB
        if frame.max() <= 1.0: frame = (frame * 255).astype(np.uint8)
        else: frame = frame.astype(np.uint8)
        
        if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Scale up for visibility
        frame = cv2.resize(frame, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)
        
        # 2. Get Target Info
        target_idx = meta_actions[i]
        skill_id = active_agents[i] # 0=Collect, 1=Destroy, 2=Avoid
        
        # Get coordinates of the target
        # Objects shape after vmap/scan: [Steps, Batch, MaxObjs=35, Feats]
        # We already extracted batch=0, so it's just [MaxObjs, Feats]
        target_obj = concept_objects[i, 0, target_idx, :] # Get target object features
        
        # Coords are normalized [0, 1]. Indices 0,1 are x, y
        # Convert to pixel space (Original 160x210)
        t_x = int(target_obj[0] * 160)
        t_y = int(target_obj[1] * 210)
        
        # Get width/height if available (Indices 2, 3)
        # If not sure, use fixed size.
        t_w = int(target_obj[2] * 160) if target_obj[2] > 0 else 10
        t_h = int(target_obj[3] * 210) if target_obj[3] > 0 else 10
        
        # Scale to video size (2x)
        x1 = (t_x - t_w//2) * 2
        y1 = (t_y - t_h//2) * 2
        x2 = (t_x + t_w//2) * 2
        y2 = (t_y + t_h//2) * 2
        
        # 3. Draw Bounding Box
        # Colors (BGR for OpenCV): 
        # Collect (0) -> Green (0, 255, 0)
        # Destroy (1) -> Red (0, 0, 255)
        # Avoid (2)   -> Blue (255, 0, 0)
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        color = colors[skill_id] if skill_id < 3 else (255, 255, 255)
        
        # Only draw if valid active object
        is_active = target_obj[4] # Index 4 is usually valid/active flag
        if is_active > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"T:{target_idx}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 4. Draw HUD
        skill_names = ["COLLECT", "DESTROY", "AVOID"]
        skill_name = skill_names[skill_id] if skill_id < 3 else "UNKNOWN"
        
        cv2.putText(frame, f"Step: {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Skill: {skill_name}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        out.write(frame)

    out.release()
    print(f"Video saved to {video_path}")
    
    # 9. Plot Statistics
    if visualize:
        create_visualizations(np.array(meta_qs), meta_actions, active_agents, folder_path)


def create_visualizations(meta_qs, meta_actions, active_agents, folder_path):
    skill_names = ["Collect", "Destroy", "Avoid"]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Actions
    ax = axes[0]
    ax.plot(meta_actions, drawstyle='steps-post', linewidth=1.5)
    ax.set_title("Meta Action Selection (Object Index)")
    ax.set_ylabel("Target Index (0-34)")
    ax.grid(True, alpha=0.3)
    
    # 2. Skills
    ax = axes[1]
    colors = ['tab:blue', 'tab:red', 'tab:green'] # Matplotlib RGB
    c_map = [colors[i] for i in active_agents]
    ax.scatter(range(len(active_agents)), active_agents, c=c_map, s=20)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(skill_names)
    ax.set_title("Active Sub-Policy")
    ax.grid(True, axis='x', alpha=0.3)
    
    # 3. Q-Values
    ax = axes[2]
    im = ax.imshow(meta_qs.T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title("Q-Values for all 35 Targets")
    ax.set_ylabel("Target Index")
    ax.set_xlabel("Timestep")
    plt.colorbar(im, ax=ax)
    
    output_path = os.path.join(folder_path, "meta_analysis.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing config and checkpoints")
    args = parser.parse_args()

    analyze_meta_policy(args.folder)