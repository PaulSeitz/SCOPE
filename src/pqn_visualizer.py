import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import pygame
import cv2
from datetime import datetime
import hydra
from omegaconf import OmegaConf
from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict

from jaxatari.core import make
from jaxatari.wrappers import AtariWrapper
from environments.seaquest.seaquest_wrapper import SeaquestConceptWrapper
from environments.pong.pong_wrapper import PongConceptWrapper
from src.pqn_skills import QNetwork
from src.train_skill import SeaquestAdapter

def load_params_from_file(filename):
    print(f"Loading weights from: {filename}")
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")

def load_config(model_path):
    # Expects config.yaml in the same folder as the model
    config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
    if os.path.exists(config_path):
        print(f"Loading Config from: {config_path}")
        return OmegaConf.load(config_path)
    else:
        print("WARNING: config.yaml not found in model folder! Using defaults.")
        return None

def make_env_and_renderer(game_name, config, seed=0):
    rng = jax.random.PRNGKey(seed)
    
    # Extract Seaquest specific config if available
    # This ensures the Wrapper extracts the exact same features as training
    sq_config = None
    if config:
        # Reconstruct Config object structure expected by wrapper
        # We try to map the loaded dict back to the Config object
        # Or pass it directly if wrapper accepts dict (Wrapper expects Config object usually)
        # For safety, we rely on the Wrapper's internal default handling if explicit config fails
        pass 

    if game_name.lower() == "seaquest":
        # 1. Base
        env = make("seaquest")
        renderer = make("seaquest")
        env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
        # env = SeaquestConceptWrapper(env, config=sq_config) 
        env = SeaquestConceptWrapper(env)
        
        # Determine task from config or default to collect
        task = config.get("SUB_POLICY_TASK", "collect") if config else "collect"
        env = SeaquestAdapter(env, task_name=task, config=config)
        print(f"Created Seaquest Environment ({task})")
        
    elif game_name.lower() == "pong":
        # 1. Base
        env = make("pong", mods={'linear_movement'})
        renderer = make("pong", mods={'linear_movement'})
        #env = make("pong")
        #renderer = make("pong")
        # 2. Atari Wrappers
        env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
        # 3. Pong Transfer Adapter
        env = PongConceptWrapper(env)
        print("Created Pong Environment (Transfer Task)")
        
    else:
        raise ValueError(f"Unknown game: {game_name}")
        
    return env, renderer, rng

# ==========================================
# 3. MAIN LOOP
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="PQN Transfer Visualizer")
    parser.add_argument("--game", type=str, required=True, choices=["seaquest", "pong"], help="Game to run")
    parser.add_argument("--model", type=str, required=True, help="Path to .safetensors file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # 1. Load Config & Params
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    config = load_config(args.model)
    params = load_params_from_file(args.model)
    
    bs_path = args.model.replace(".safetensors", "_bs.safetensors")
    if os.path.exists(bs_path):
        batch_stats = load_params_from_file(bs_path)
    else:
        print("Warning: No batch stats found.")
        batch_stats = {}

    # 2. Setup Env
    pygame.init()
    scale = 3
    screen = pygame.display.set_mode((160 * scale, 210 * scale))
    pygame.display.set_caption(f"PQN Transfer: {args.game.upper()}")

    env, renderer, rng = make_env_and_renderer(args.game, config, args.seed)

    # 3. Init Network (Using Config for Architecture Consistency)
    # Default to 64/3 if config missing
    h_size = config.get("HIDDEN_SIZE", 64) if config else 64
    n_layers = config.get("NUM_LAYERS", 3) if config else 3
    
    print(f"Initializing Network: Hidden={h_size}, Layers={n_layers}")
    
    network = QNetwork(
        action_dim=18, # force full action space for transfer experiments
        hidden_size=h_size,
        num_layers=n_layers,
        norm_type="layer_norm",
        norm_input=False
    )

    # 4. JIT Step
    print("Compiling...")
    @jax.jit
    def policy_step(obs, env_state, rng, params, batch_stats):
        # Pad observation to match model's expected input size (28)
        obs_padded = jnp.pad(obs, (0, max(0, 28 - obs.shape[-1])), mode='constant', constant_values=0)
        
        # Forward
        q_vals = network.apply({"params": params, "batch_stats": batch_stats}, obs_padded, train=False)
        action = jnp.argmax(q_vals, axis=-1)
        
        # Step
        next_obs, next_state, _, done, _ = env.step(env_state, action)
        
        # Auto-Reset
        rng, rng_r = jax.random.split(rng)
        
        def do_reset(_): return env.reset(rng_r)
        def do_step(_):  return next_obs, next_state

        final_obs, final_state = jax.lax.cond(done, do_reset, do_step, operand=None)
        
        return final_obs, final_state, action, q_vals

    # 5. Run
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset)
    
    clock = pygame.time.Clock()
    running = True
    
    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"playback_{args.game}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (160 * scale, 210 * scale))

    def get_render_state(state):
        s = state
        while hasattr(s, "env_state"): s = s.env_state
        return s

    print("\nVisualizer Running. Press ESC to quit.")
    print(f"Video will be saved to: {video_filename}\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # Step
        rng, rng_s = jax.random.split(rng)
        obs, env_state, action, q_vals = policy_step(obs, env_state, rng_s, params, batch_stats)
        
        # Log observation and action
        print(f"Obs: {np.array(obs)}")
        print(f"Action: {action} | Q-values: {np.array(q_vals)}\n")

        # Render
        base_state = get_render_state(env_state)
        frame = renderer.render(base_state) # (H, W, 3)
        
        # Blit
        frame_np = np.array(frame, dtype=np.uint8)
        surface = pygame.surfarray.make_surface(frame_np.transpose(1, 0, 2))
        surface = pygame.transform.scale(surface, (160 * scale, 210 * scale))
        screen.blit(surface, (0, 0))
        
        # Write frame to video
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, (160 * scale, 210 * scale))
        video_writer.write(frame_resized)

        # Debug Text
        font = pygame.font.SysFont("Arial", 18)
        text = f"Action: {action} | Q: {float(jnp.max(q_vals)):.2f}"
        screen.blit(font.render(text, True, (255, 255, 255)), (10, 10))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    video_writer.release()
    print(f"\nVideo saved to: {video_filename}")

if __name__ == "__main__":
    main()