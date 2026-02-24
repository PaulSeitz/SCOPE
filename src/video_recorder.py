import os
import threading
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxatari.renderers import JAXGameRenderer
import wandb

CURRENT_VIDEO_THREAD = None

class SidebarRenderer:
    def __init__(self, frame_shape, sidebar_width=50, font_size=15, text_color=(255, 255, 255), bg_color=(0, 0, 0), highlight_color=(50, 50, 200)):
        """
        frame_shape: (3, H, W)
        """
        _, self.h, self.w = frame_shape
        self.sidebar_width = sidebar_width
        self.text_color = text_color
        self.bg_color = bg_color
        self.highlight_color = highlight_color

        pygame.init()
        pygame.font.init()

        self.font = pygame.font.SysFont("Arial", font_size)
        self.surface = pygame.Surface((self.w + sidebar_width, self.h))
        self.left_surface = pygame.Surface((self.w, self.h))

    def render(self, frame, texts, active_cell):
        """
        Add sidebar with text to a single frame.

        frame: (3, H, W) RGB numpy array
        texts: list of tuple of strings
        """

        self.surface.fill(self.bg_color)

        # Draw the frame on the left
        pygame.surfarray.blit_array(self.left_surface, frame.transpose(2, 1, 0))
        self.surface.blit(self.left_surface, (0, 0))

        # Draw text on the right
        cell_height = self.h // len(texts)
        for i, text in enumerate(texts):
            y = i * cell_height
            if i == active_cell:
                pygame.draw.rect(
                    self.surface,
                    self.highlight_color,
                    pygame.Rect(self.w, y, self.sidebar_width, cell_height)
                )
            t0_surface = self.font.render(text[0], True, self.text_color)
            t1_surface = self.font.render(text[1], True, self.text_color)
            t0_x = self.w + (self.sidebar_width- t0_surface.get_width()) // 2
            t1_x = self.w + (self.sidebar_width- t1_surface.get_width()) // 2
            total_height = t0_surface.get_height() + t1_surface.get_height() + 4 # 4 pixels of padding 
            start_y = y + (cell_height - total_height) // 2
            t0_y = start_y
            t1_y = start_y + t0_surface.get_height() + 4

            self.surface.blit(t0_surface, (t0_x, t0_y))
            self.surface.blit(t1_surface, (t1_x, t1_y))

        # Convert back to NumPy
        # Note: pygame.surfarray.array3d returns (W, H, 3) but we need (3, H, W)
        output_frame = pygame.surfarray.array3d(self.surface).transpose(2, 1, 0)
        return output_frame

    def close(self):
        pygame.quit()


def video_callback(states, active_agents, combined_qs, dones, step, renderer, modif=False):
    global CURRENT_VIDEO_THREAD
    
    if renderer is None:
        return

    # If a previous video is still saving, wait for it (optional, avoids overlapping threads)
    if CURRENT_VIDEO_THREAD is not None and CURRENT_VIDEO_THREAD.is_alive():
        try:
            CURRENT_VIDEO_THREAD.join(timeout=10)
        except Exception as e:
            print(f"Warning: Error waiting for previous video thread: {e}")

    # Start new thread as daemon so it doesn't block shutdown
    CURRENT_VIDEO_THREAD = threading.Thread(
        target=collect_video, 
        args=(states, active_agents, combined_qs, dones, step, renderer, modif),
        daemon=False  # Keep as non-daemon but with proper cleanup
    )
    CURRENT_VIDEO_THREAD.start()

def add_active_agent(screen, active_agent_num: int):
    font = pygame.font.Font(None, 50)
    text_surface = font.render(f"active agent: {active_agent_num}", True, (255, 255, 255)) 
    screen.blit(text_surface, (300, 300)) 

def collect_video(states, active_agents, combined_qs, dones, step, renderer, modif=False):
    try:
        if wandb.run is None:
            print("WandB not initialized, skipping video logging.")
            return
            
        print("Rendering video...")
        video_folder = f"{wandb.run.dir}/media/videos/"
        os.makedirs(video_folder, exist_ok=True)
        
        # Unwrap the exact wrapper chain: LogWrapper -> SubPolicyAdapter -> SeaquestConceptWrapper -> AtariWrapper -> base state
        # LogWrapper stores the next env's state in .env_state
        states = states.atari_state
        # AtariWrapper (nested after SeaquestConceptWrapper) stores the base state in .env_state
        states = states.env_state

        # Handle 'dones' to determine video length
        # If dones is (Time, Envs), take the first environment
        if len(dones.shape) > 1:
            dones_col = dones[:, 0]
        else:
            dones_col = dones

        num_states = jnp.argmax(dones_col)
        
        # If no 'done' was found (argmax returns 0), use the full length
        if num_states == 0:
            # Get the time dimension from the first leaf of the state tree
            leaves = jax.tree_util.tree_leaves(states)
            if leaves:
                num_states = leaves[0].shape[0]
            else:
                print("Warning: State tree empty, cannot determine video length.")
                return

        # Slice the states to the correct length
        states_reduced = jax.tree_util.tree_map(lambda x: x[:num_states], states)

        if isinstance(renderer, JAXGameRenderer):
            # Render using JAX renderer (vmap over time dimension)
            rasters = jax.vmap(renderer.render)(states_reduced)
            frames = np.array(rasters, dtype=np.uint8)
        else:
            print(f"Unsupported renderer type: {type(renderer)}")
            return

        # Transpose for WandB: (Time, H, W, C) -> (Time, C, H, W)
        # JAX Atari renderer outputs (H, W, C) per frame
        if frames.shape[-1] == 3: 
            frames = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, 3) -> (T, 3, H, W)
            # Note: If the video is rotated/flipped, the transpose may need adjustment.

        # [Optional: Sidebar Logic]
        if active_agents is not None and combined_qs is not None:
            sidebar_renderer = SidebarRenderer(frames[0].shape)
            
            # New frame buffer with sidebar space
            current_w = frames.shape[3]
            new_w = current_w + sidebar_renderer.sidebar_width
            
            new_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2], new_w), dtype=np.uint8)
            
            for i in range(len(frames)):
                texts = [(f"Option {t_i}:", f"{t:.2f}") for t_i, t in enumerate(combined_qs[i])]
                new_frames[i] = sidebar_renderer.render(frames[i], texts, active_agents[i])
            
            frames = new_frames

        fps = 30
        name = f"video_{step}"
        if modif:
            name = f"video_{step}_modif"
        
        # Check if wandb is actually active before logging
        if wandb.run is not None:
            try:
                video = wandb.Video(frames, fps=fps, format="mp4")
                wandb.log({name: video}, step=wandb.run.step)
            except (RuntimeError, Exception) as e:
                print(f"Warning: Failed to log video to WandB: {e}")
        else:
            print("WandB run not active, video generated but not logged.")

        print("Video done.")
    except Exception as e:
        print(f"Error in collect_video: {e}")
        import traceback
        traceback.print_exc()


def wait_for_video_finish():
    """Call this at the end of your script to prevent crashing."""
    global CURRENT_VIDEO_THREAD
    if CURRENT_VIDEO_THREAD is not None and CURRENT_VIDEO_THREAD.is_alive():
        print("Waiting for background video rendering to finish...")
        try:
            # Use timeout to prevent hanging
            CURRENT_VIDEO_THREAD.join(timeout=30)
            if CURRENT_VIDEO_THREAD.is_alive():
                print("Warning: Video rendering still in progress but timeout reached.")
            else:
                print("Video rendering done.")
        except Exception as e:
            print(f"Warning: Error waiting for video: {e}")