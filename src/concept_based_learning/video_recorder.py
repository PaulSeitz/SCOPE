import os
import threading
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jaxatari.renderers import JAXGameRenderer

from concept_based_learning.environments.skill_rewards import SKILL_REWARD_ORDER
from concept_based_learning.training.episode_stats import unwrap_base_env_state

# Force pygame into headless mode before the first pygame.init() call so the
# video renderer works on cluster nodes that have no display server.
# SDL_VIDEODRIVER=dummy uses a null video driver; SDL_AUDIODRIVER=dummy
# suppresses the matching audio warning. Both are no-ops if a real display
# is already available.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402  (must come after the env vars are set)

pygame.init()
pygame.font.init()

CURRENT_VIDEO_THREAD = None

# Skill index = label - 1 (matches label_to_skill_idx in train_agent).
# Derived from the global skill registry so new skills appear automatically.
SKILL_NAMES = [name.upper() for name in SKILL_REWARD_ORDER]
TARGET_OUTLINE_COLOR = (255, 50, 50)  # Red, same as label_visualizer LABEL_DESTROY
TARGET_OUTLINE_WIDTH = 3

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


def video_callback(states, active_agents, combined_qs, dones, step, renderer, modif=False,
                   target_obj_idxs=None, objects_bbox=None, video_key: str | None = None,
                   wandb_step: int | None = None, distances=None, angles=None,
                   video_output_path: str | None = None):
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
        kwargs={
            "target_obj_idxs": target_obj_idxs,
            "objects_bbox": objects_bbox,
            "video_key": video_key,
            "wandb_step": wandb_step,
            "distances": distances,
            "angles": angles,
            "video_output_path": video_output_path,
        },
        daemon=False  # Keep as non-daemon but with proper cleanup
    )
    CURRENT_VIDEO_THREAD.start()

def add_active_agent(screen, active_agent_num: int):
    font = pygame.font.Font(None, 50)
    text_surface = font.render(f"active agent: {active_agent_num}", True, (255, 255, 255)) 
    screen.blit(text_surface, (300, 300))


def draw_target_outline(frame, target_idx, obj_x, obj_y, obj_w, obj_h, active):
    """
    Draw a red outline around the current target object on a single frame.
    frame: (3, H, W) numpy uint8 RGB
    target_idx: int index into object arrays
    obj_x, obj_y, obj_w, obj_h, active: 1D arrays of length max_objects
    """
    if target_idx < 0 or target_idx >= len(active) or not active[target_idx]:
        return frame
    x = int(obj_x[target_idx])
    y = int(obj_y[target_idx])
    w = int(obj_w[target_idx])
    h = int(obj_h[target_idx])
    if w <= 0 or h <= 0:
        return frame
    _, H, W = frame.shape
    # Clip to frame bounds
    x1 = max(0, min(x, W - 1))
    y1 = max(0, min(y, H - 1))
    x2 = max(0, min(x + w, W))
    y2 = max(0, min(y + h, H))
    if x1 >= x2 or y1 >= y2:
        return frame
    # Use pygame to draw on the frame: (3, H, W) -> (W, H, 3) for pygame
    surf = pygame.surfarray.make_surface(frame.transpose(2, 1, 0))
    pygame.draw.rect(surf, TARGET_OUTLINE_COLOR, (x1, y1, x2 - x1, y2 - y1), TARGET_OUTLINE_WIDTH)
    out = np.transpose(pygame.surfarray.array3d(surf), (2, 1, 0))
    return out


def add_distance_angle_text(frame, distance, angle):
    """
    Overlay the current skill input (distance, angle) at the bottom-center of the frame.
    frame: (3, H, W) numpy uint8 RGB
    distance: float (pixels)
    angle: float (radians)
    """
    if distance is None or np.isnan(distance):
        return frame

    _, H, W = frame.shape
    surf = pygame.surfarray.make_surface(frame.transpose(2, 1, 0))

    # Slightly smaller font so it stays within the frame borders.
    font = pygame.font.Font(None, 20)
    text = f"skill: d={distance:.1f}, a={angle:.2f}"
    text_surface = font.render(text, True, (255, 255, 255))

    x = (W - text_surface.get_width()) // 2
    y = H - text_surface.get_height() - 10
    surf.blit(text_surface, (x, y))

    out = np.transpose(pygame.surfarray.array3d(surf), (2, 1, 0))
    return out


def _write_frames_mp4(frames_chw: np.ndarray, path: str, fps: int = 30) -> None:
    """frames_chw: (T, C, H, W) uint8 RGB."""
    try:
        import cv2
    except ImportError:
        print(
            "Warning: OpenCV (cv2) not installed; cannot save video to disk. "
            "Install opencv-python or enable wandb logging."
        )
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _, _, h, w = frames_chw.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        print(f"Warning: cv2.VideoWriter could not open path {path!r}")
        return
    for i in range(len(frames_chw)):
        fr = np.transpose(frames_chw[i], (1, 2, 0))
        fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        writer.write(fr_bgr)
    writer.release()
    print(f"Saved video to {path}")


def collect_video(states, active_agents, combined_qs, dones, step, renderer, modif=False,
                  target_obj_idxs=None, objects_bbox=None, video_key: str | None = None,
                  wandb_step: int | None = None, distances=None, angles=None,
                  video_output_path: str | None = None):
    try:
        use_wandb = wandb.run is not None
        if not use_wandb and not video_output_path:
            return
        print("Rendering video...")
        if use_wandb:
            video_folder = f"{wandb.run.dir}/media/videos/"
            os.makedirs(video_folder, exist_ok=True)
        
        # SemanticState.atari_state may be AtariState or PixelAndObjectCentricState
        # (when pixel_embed adds PixelAndObjectObsWrapper). Unwrap recursively.
        atari_state = states.atari_state if hasattr(states, "atari_state") else states
        states = unwrap_base_env_state(atari_state)

        # Handle 'dones' to determine video length
        # If dones is (Time, Envs), take the first environment
        if len(dones.shape) > 1:
            dones_col = dones[:, 0]
        else:
            dones_col = dones

        leaves = jax.tree_util.tree_leaves(states)
        full_length = leaves[0].shape[0] if leaves else 0
        if full_length == 0:
            print("Warning: State tree empty, cannot determine video length.")
            return

        # If any step had done=True, show frames [0 .. first_done_idx] inclusive (include terminal frame)
        # Otherwise show full rollout (hit VIDEO_MAX_FRAMES without done)
        has_done = jnp.any(dones_col > 0.5)
        first_done_idx = jnp.argmax(dones_col)
        num_states = int(jnp.where(has_done, first_done_idx + 1, full_length))

        if num_states < 1:
            print("Warning: No frames to render (empty rollout or done on step 0). Skipping video.")
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

        # [Optional: Target outline] Draw red outline around current target object per frame
        if target_obj_idxs is not None and objects_bbox is not None:
            ox, oy, ow, oh, oactive = objects_bbox
            for i in range(len(frames)):
                idx = int(target_obj_idxs[i])
                frames[i] = draw_target_outline(
                    frames[i], idx,
                    ox[i], oy[i], ow[i], oh[i], oactive[i],
                )

        # [Optional: Skill input overlay] Show distance and angle at bottom-center
        if distances is not None and angles is not None:
            for i in range(len(frames)):
                # distances/angles expected as 1D arrays of length T
                d = float(distances[i])
                a = float(angles[i])
                frames[i] = add_distance_angle_text(frames[i], d, a)

        # [Optional: Sidebar Logic] Skill names + Q-values
        if active_agents is not None and combined_qs is not None:
            sidebar_renderer = SidebarRenderer(frames[0].shape)
            current_w = frames.shape[3]
            new_w = current_w + sidebar_renderer.sidebar_width
            new_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2], new_w), dtype=np.uint8)
            for i in range(len(frames)):
                # Use skill names (DESTROY, COLLECT, AVOID) instead of "Option 0", "Option 1", ...
                skill_labels = [SKILL_NAMES[t_i] if t_i < len(SKILL_NAMES) else f"Option {t_i}" for t_i in range(len(combined_qs[i]))]
                texts = [(f"{name}:", f"{t:.2f}") for name, t in zip(skill_labels, combined_qs[i])]
                new_frames[i] = sidebar_renderer.render(frames[i], texts, active_agents[i])
            frames = new_frames

        name = video_key or (f"video_{step}_modif" if modif else f"video_{step}")
        if use_wandb:
            try:
                log_kwargs = {}
                if wandb_step is not None:
                    log_kwargs["step"] = int(wandb_step)
                wandb.log({name: wandb.Video(frames, fps=30, format="mp4")}, **log_kwargs)
            except Exception as e:
                print(f"Warning: Failed to log video: {e}")
        if video_output_path:
            _write_frames_mp4(frames, video_output_path, fps=30)
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