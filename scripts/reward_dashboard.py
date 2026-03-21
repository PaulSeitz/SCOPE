#!/usr/bin/env python3
"""
Interactive Reward Dashboard for Atari environments with semantic wrappers.

By default this is wired to Seaquest via EnvManager, but you can also import
this module and construct a `RewardDashboard` manually by passing your own
environment, renderer, and env-specific hooks.
"""
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from concept_based_learning.environments.datatypes import (
    LABEL_NOOP,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
)
from concept_based_learning.environments.env_manager import EnvManager

# Color constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

TARGET_OUTLINE_COLOR = (255, 50, 50)
TARGET_OUTLINE_WIDTH = 3

# Label names (Player Tool removed, concepts renumbered)
LABEL_NAMES = {
    LABEL_NOOP: "Noop",
    LABEL_DESTROY: "Destroy",
    LABEL_COLLECT: "Collect",
    LABEL_AVOID: "Avoid",
}

# Object type names for display.
# For non-Seaquest environments this will gracefully fall back to "Obj <idx>".
OBJECT_NAMES = [
    "Obj 0",
    "Obj 1",
    "Obj 2",
    "Obj 3",
    "Obj 4",
    "Obj 5",
    "Obj 6",
    "Obj 7",
    "Obj 8",
    "Obj 9",
    "Obj 10",
    "Obj 11",
    "Obj 12",
    "Obj 13",
    "Obj 14",
    "Obj 15",
    "Obj 16",
    "Obj 17",
    "Obj 18",
    "Obj 19",
    "Obj 20",
    "Obj 21",
    "Obj 22",
    "Obj 23",
    "Obj 24",
    "Obj 25",
    "Obj 26",
    "Obj 27",
    "Obj 28",
    "Obj 29",
    "Obj 30",
    "Obj 31",
    "Obj 32",
]

@dataclass
class MultiReward:
    """Simple container for concept-wise rewards and global reward."""

    concept_rewards: np.ndarray  # shape: [num_concepts, max_objects]
    global_reward: float


# Action mapping (Atari actions)
ACTION_NOOP = 0
ACTION_FIRE = 1
ACTION_UP = 2
ACTION_RIGHT = 3
ACTION_LEFT = 4
ACTION_DOWN = 5
ACTION_UPRIGHT = 6
ACTION_UPLEFT = 7
ACTION_DOWNRIGHT = 8
ACTION_DOWNLEFT = 9
ACTION_UPFIRE = 10
ACTION_RIGHTFIRE = 11
ACTION_LEFTFIRE = 12
ACTION_DOWNFIRE = 13
ACTION_UPRIGHTFIRE = 14
ACTION_UPLEFTFIRE = 15
ACTION_DOWNRIGHTFIRE = 16
ACTION_DOWNLEFTFIRE = 17


class RewardDashboard:
    def __init__(
        self,
        env,
        renderer,
        *,
        env_name: str,
        get_heuristic_target,
        no_target_idx,
        reward_extractor,
        screen_width=800,
        screen_height=600,
    ):
        self.env = env
        self.renderer = renderer
        self.env_name = env_name
        self.get_heuristic_target = get_heuristic_target
        self.no_target_idx = no_target_idx
        self.reward_extractor = reward_extractor

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"{self.env_name.title()} Reward Dashboard")
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.bold_font = pygame.font.Font(None, 24)

        # Game state
        self.obs = None
        self.state = None
        self.prev_obs = None
        self.prev_state = None
        self.multi_reward = None
        self.step_count = 0
        self.total_global_reward = 0.0
        self.rng_key = jax.random.key(42)
        self.last_object_events = None  # Boolean mask of objects that produced reward last step
        self.prev_target_idx = self.no_target_idx
        self.current_target_idx = self.no_target_idx

        # Freeze/step controls
        self.frozen = False
        self.step_mode = False
        self.target_fps = 30
        self.actual_fps = 0.0
        self.fps_history = []

        # Reward history buffer (to show rewards for objects that just disappeared)
        self.reward_history = []  # List of (step_count, obj_idx, label_name, reward_val, event_triggered)
        self.history_duration = 120  # Show rewards for 5 steps after event

        # Display settings
        self.game_scale = 2  # Scale factor for game rendering
        self.game_width = 160 * self.game_scale
        self.game_height = 210 * self.game_scale
        self.panel_width = screen_width - self.game_width - 20
        self.panel_start_x = self.game_width + 20

        # Column layout for horizontal display
        self.num_columns = 3
        self.column_width = self.panel_width // self.num_columns
        self.column_spacing = 10
        
    def reset(self):
        """Reset the environment"""
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.obs, self.state = self.env.reset(subkey)
        self.prev_obs = None
        self.prev_state = None
        self.multi_reward = None
        self.step_count = 0
        self.total_global_reward = 0.0
        self.reward_history = []  # Clear history on reset
        self.last_object_events = None
        self.prev_target_idx = self.no_target_idx
        self.current_target_idx = self.no_target_idx
        
    def step(self, action):
        """Step the environment and compute rewards"""
        self.prev_obs = self.obs
        self.prev_state = self.state
        self.obs, self.state, reward, done, info = self.env.step(self.state, action)

        # Compute rewards if we have previous state
        if self.prev_obs is not None:
            # First, ask the environment-specific heuristic which object is the target.
            try:
                self.rng_key, key_policy = jax.random.split(self.rng_key)
                prev_idx = jnp.array(self.prev_target_idx, dtype=jnp.int32)
                target_idx = self.get_heuristic_target(
                    self.obs.raw_observation,
                    self.obs.objects,
                    prev_idx,
                    key_policy,
                )
                target_idx_int = int(jnp.ravel(target_idx)[0])
            except Exception:
                target_idx_int = self.no_target_idx

            self.prev_target_idx = target_idx_int
            self.current_target_idx = target_idx_int

            # Compute concept rewards only for the currently selected target.
            objects = self.obs.objects
            prev_objects = self.prev_obs.objects

            # objects.x has shape [stack_size, max_objects]
            num_objects = int(objects.x.shape[1])
            concept_rewards = np.zeros((3, num_objects), dtype=np.float32)

            def slice_target(obj_obs, obj_idx: int):
                """
                Extract one object's trajectory across the frame stack as a PyTree
                matching the semantic wrapper's object structure.

                Handles both stacked arrays (shape [T, N]) and per-frame arrays
                (shape [N]), while leaving scalars / non-array leaves untouched.
                """

                def _slice_leaf(x):
                    # Non-array leaves (e.g., ints) are returned unchanged.
                    if not hasattr(x, "ndim"):
                        return x
                    # Scalars have no object axis.
                    if x.ndim == 0:
                        return x
                    # Per-frame object arrays: [N] -> scalar for this object.
                    if x.ndim == 1:
                        if 0 <= obj_idx < x.shape[0]:
                            return x[obj_idx]
                        return x
                    # Stacked arrays: [T, N] (or [..., N]) -> slice along last axis.
                    if 0 <= obj_idx < x.shape[-1]:
                        return x[..., obj_idx]
                    return x

                return jax.tree_util.tree_map(_slice_leaf, obj_obs)

            if 0 <= target_idx_int < num_objects:
                prev_target = slice_target(prev_objects, target_idx_int)
                next_target = slice_target(objects, target_idx_int)
                rewards_vec = self.reward_extractor(
                    self.prev_obs.raw_observation,
                    self.obs.raw_observation,
                    prev_target,
                    next_target,
                )
                concept_rewards[:, target_idx_int] = np.asarray(
                    jax.device_get(rewards_vec)
                )
            
            # Global reward: use the raw environment reward (game score delta)
            self.multi_reward = MultiReward(
                concept_rewards=concept_rewards,
                global_reward=float(reward),
            )
            self.total_global_reward += float(self.multi_reward.global_reward)
            
            # Mark which objects produced any non-zero concept reward this step
            self.last_object_events = np.any(np.abs(concept_rewards) > 1e-6, axis=0)
            
            # Update reward history - add events that just occurred
            if self.multi_reward is not None:
                object_labels = np.array(self.obs.objects.semantic_label[-1])
                concept_rewards_np = np.array(self.multi_reward.concept_rewards)
                
                newly_triggered = np.where(self.last_object_events)[0]
                for obj_idx in newly_triggered:
                    label_id = int(object_labels[obj_idx])
                    # Map label_id (1=DESTROY,2=COLLECT,3=AVOID) to skill index (0..2)
                    skill_idx = max(0, label_id - 1)
                    reward_val = float(concept_rewards_np[skill_idx, obj_idx])
                    if abs(reward_val) <= 1e-6:
                        continue
                    obj_name = OBJECT_NAMES[obj_idx] if obj_idx < len(OBJECT_NAMES) else f"Obj {obj_idx}"
                    label_name = LABEL_NAMES.get(label_id, f"Label {label_id}")
                    
                    # Check if this event is already in history (avoid duplicates)
                    is_new = True
                    for hist_entry in self.reward_history:
                        if (
                            hist_entry['obj_idx'] == obj_idx
                            and hist_entry['step'] == self.step_count
                        ):
                            is_new = False
                            break
                    
                    if is_new:
                        self.reward_history.append({
                            'step': self.step_count,
                            'obj_idx': obj_idx,
                            'obj_name': obj_name,
                            'label_name': label_name,
                            'reward': reward_val,
                            'event': True
                        })

        # Clean up old history entries
        self.reward_history = [
            h for h in self.reward_history 
            if (self.step_count - h['step']) <= self.history_duration
        ]
        
        self.step_count += 1
        return done
        
    def render_game(self, state):
        """Render the game screen using jaxatari rendering approach"""
        frame = None
        
        # Use the dedicated Seaquest renderer on the underlying Atari state
        try:
            if state is not None and self.renderer is not None:
                # Unwrap to the base Atari game state if needed
                atari_state = getattr(state, "atari_state", state)
                game_state = getattr(atari_state, "env_state", atari_state)
                frame = self.renderer.render(game_state)
        except Exception:
            frame = None
        
        # Convert JAX array to NumPy array properly
        if frame is not None:
            try:
                # Use jax.device_get() to properly convert JAX arrays to NumPy
                # This handles device arrays (CPU/GPU/TPU) correctly
                frame = jax.device_get(frame)
                
                # Ensure it's a numpy array
                frame = np.asarray(frame)
                
                # Handle different dtypes and shapes
                if frame.dtype != np.uint8:
                    if np.issubdtype(frame.dtype, np.floating):
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                # Handle shape: expect (H, W, 3) or (H, W)
                if len(frame.shape) == 2:
                    # Grayscale, convert to RGB
                    frame = np.stack([frame] * 3, axis=-1)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        # RGBA, convert to RGB
                        frame = frame[:, :, :3]
                    elif frame.shape[2] != 3:
                        # Unexpected channel count
                        frame = None
                
                # Create pygame surface
                if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Transpose (H, W, C) to (W, H, C) for pygame
                    frame_surface = pygame.surfarray.make_surface(
                        np.transpose(frame, (1, 0, 2))
                    )
                    # Scale up
                    frame_surface = pygame.transform.scale(
                        frame_surface, (self.game_width, self.game_height)
                    )
                    self.screen.blit(frame_surface, (10, 10))
                    # After drawing the game frame, overlay current target outline if any
                    self._draw_target_outline_on_screen()
                    return
            except Exception as e:
                # If anything fails, fall through to placeholder
                pass
        
        # Fallback: Draw placeholder with game info
        pygame.draw.rect(self.screen, DARK_GRAY, (10, 10, self.game_width, self.game_height))
        pygame.draw.rect(self.screen, GRAY, (10, 10, self.game_width, self.game_height), 2)
        
        # Show game state info
        if self.obs is not None:
            info_lines = [f"{self.env_name.title()} Game", ""]

            raw = getattr(self.obs, "raw_observation", None)
            if raw is not None:
                if hasattr(raw, "oxygen_level"):
                    info_lines.append(
                        f"Oxygen: {float(raw.oxygen_level[-1]) * 64:.0f}/64"
                    )
                if hasattr(raw, "collected_divers"):
                    info_lines.append(
                        f"Divers: {float(raw.collected_divers[-1]):.0f}/6"
                    )

            if self.state is not None and hasattr(self.state, "atari_state"):
                try:
                    lives = int(self.state.atari_state.env_state.lives)
                    info_lines.append(f"Lives: {lives}")
                except Exception:
                    pass

            info_lines.extend(
                [
                    "",
                    "Rendering not available",
                    "Check game state in panel",
                ]
            )
            y_offset = 20
            for line in info_lines:
                text = self.small_font.render(line, True, WHITE)
                self.screen.blit(text, (20, y_offset))
                y_offset += 20

        # Also overlay target outline relative to placeholder bounds
        self._draw_target_outline_on_screen()

    def _draw_target_outline_on_screen(self):
        """Draw a red outline around the current heuristic target object."""
        if (
            self.obs is None
            or self.current_target_idx is None
            or self.current_target_idx == self.no_target_idx
        ):
            return

        idx = int(self.current_target_idx)
        # Use latest frame in the stack for geometry
        try:
            x_arr = np.array(self.obs.objects.x[-1])
            y_arr = np.array(self.obs.objects.y[-1])
            w_arr = np.array(self.obs.objects.width[-1])
            h_arr = np.array(self.obs.objects.height[-1])
            active_arr = np.array(self.obs.objects.active[-1])
        except Exception:
            return

        if idx < 0 or idx >= len(active_arr) or active_arr[idx] <= 0:
            return

        x = int(x_arr[idx])
        y = int(y_arr[idx])
        w = int(w_arr[idx])
        h = int(h_arr[idx])
        # Skip degenerate or 1-pixel bounding boxes which are often invisible anyway.
        if w <= 1 or h <= 1:
            return

        # Game frame is drawn at (10, 10) scaled by game_scale
        sx = 10 + int(x * self.game_scale)
        sy = 10 + int(y * self.game_scale)
        sw = int(w * self.game_scale)
        sh = int(h * self.game_scale)

        pygame.draw.rect(
            self.screen,
            TARGET_OUTLINE_COLOR,
            (sx, sy, sw, sh),
            TARGET_OUTLINE_WIDTH,
        )
    
    def draw_reward_panel(self):
        """Draw the reward information panel in horizontal columns"""
        panel_y = 10
        
        # Column 1: Stats and Recent Events
        col1_x = self.panel_start_x
        col1_y = self._draw_column1(col1_x, panel_y)
        
        # Column 2: Active Objects & Rewards
        col2_x = self.panel_start_x + self.column_width + self.column_spacing
        col2_y = self._draw_column2(col2_x, panel_y)
        
        # Column 3: Controls (at bottom)
        col3_x = self.panel_start_x + 2 * (self.column_width + self.column_spacing)
        self._draw_column3(col3_x, panel_y)
    
    def _draw_column1(self, x, y):
        """Draw column 1: Stats and Recent Events"""
        current_y = y
        
        # Title
        title = self.bold_font.render("Stats & Events", True, WHITE)
        self.screen.blit(title, (x, current_y))
        current_y += 30
        
        # Global stats
        stats_text = [
            f"Step: {self.step_count}",
            f"Global: {self.total_global_reward:.3f}",
            f"FPS: {self.actual_fps:.1f}/{self.target_fps}",
        ]
        if self.multi_reward is not None:
            stats_text.append(f"Last: {float(self.multi_reward.global_reward):.3f}")
        
        # Event detection (objects that produced any reward last step)
        if self.last_object_events is not None:
            num_events = int(np.sum(self.last_object_events))
            if num_events > 0:
                stats_text.append(f"Events: {num_events}")
        
        for text in stats_text:
            surface = self.small_font.render(text, True, WHITE)
            self.screen.blit(surface, (x, current_y))
            current_y += 18
        
        current_y += 10
        
        # Recent Events
        if len(self.reward_history) > 0:
            header = self.small_font.render("Recent Events:", True, YELLOW)
            self.screen.blit(header, (x, current_y))
            current_y += 20
            
            # Show most recent events (up to 8)
            for hist_entry in self.reward_history[-8:]:
                age = self.step_count - hist_entry['step']
                obj_text = f"{hist_entry['obj_name']}"
                reward_text = f"{hist_entry['reward']:+.2f} ({age}s ago)"
                
                # Fade color based on age
                if age <= 2:
                    color = GREEN
                elif age <= 5:
                    color = YELLOW
                else:
                    color = GRAY
                
                surface = self.small_font.render(obj_text, True, color)
                self.screen.blit(surface, (x + 5, current_y))
                current_y += 14
                
                surface = self.small_font.render(reward_text, True, color)
                self.screen.blit(surface, (x + 5, current_y))
                current_y += 14
        
        return current_y
    
    def _draw_column2(self, x, y):
        """Draw column 2: Active Objects & Rewards"""
        current_y = y
        
        # Header
        header = self.bold_font.render("Active Objects", True, YELLOW)
        self.screen.blit(header, (x, current_y))
        current_y += 30
        
        # Reward matrix display
        if self.multi_reward is not None:
            # Show rewards for active objects only
            concept_rewards = np.array(self.multi_reward.concept_rewards)
            active_flags = np.array(self.obs.objects.active[-1])
            labels = np.array(self.obs.objects.semantic_label[-1])
            events = (
                np.array(self.last_object_events)
                if self.last_object_events is not None
                else np.zeros_like(active_flags, dtype=bool)
            )
            
            # Show each active object (compact horizontal format)
            active_count = 0
            max_display = 25  # Can show more in horizontal layout
            
            for obj_idx in range(len(active_flags)):
                if active_flags[obj_idx] > 0.5 and active_count < max_display:
                    obj_name = OBJECT_NAMES[obj_idx] if obj_idx < len(OBJECT_NAMES) else f"Obj {obj_idx}"
                    
                    label_id = int(labels[obj_idx])
                    label_name = LABEL_NAMES.get(label_id, f"Label {label_id}")
                    # Map label to skill index (0..2) for reward lookup
                    skill_idx = max(0, label_id - 1)
                    reward_val = float(concept_rewards[skill_idx, obj_idx])
                    
                    # Compact format: Name | Label | Reward
                    obj_text = f"{obj_name}"
                    label_text = f"[{label_name}]"
                    event_triggered = bool(events[obj_idx]) and abs(reward_val) > 1e-6
                    if event_triggered:
                        label_text += " [E]"
                    
                    color = GREEN if event_triggered else WHITE
                    
                    # Render in compact format
                    surface = self.small_font.render(obj_text, True, color)
                    self.screen.blit(surface, (x, current_y))
                    
                    surface = self.small_font.render(label_text, True, color)
                    self.screen.blit(surface, (x + 70, current_y))
                    
                    # Show reward
                    if abs(reward_val) > 1e-6:
                        reward_color = GREEN if event_triggered and reward_val > 0 else CYAN
                        reward_str = f"{reward_val:+.2f}"
                    else:
                        reward_color = GRAY
                        reward_str = "0.00"
                    
                    surface = self.small_font.render(reward_str, True, reward_color)
                    self.screen.blit(surface, (x + 140, current_y))
                    
                    current_y += 14
                    active_count += 1
            
            if active_count >= max_display:
                more_text = self.small_font.render(
                    f"... +{np.sum(active_flags > 0.5) - max_display} more",
                    True,
                    GRAY,
                )
                self.screen.blit(more_text, (x, current_y))
        
        return current_y
    
    def _draw_column3(self, x, y):
        """Draw column 3: Controls and Info"""
        current_y = y
        
        # Title
        title = self.bold_font.render("Controls", True, WHITE)
        self.screen.blit(title, (x, current_y))
        current_y += 30
        
        # Controls help
        help_text = [
            "Arrow Keys: Move",
            "Space: Fire",
            "R: Reset",
            "F: Freeze/Unfreeze",
            "N: Step Forward",
            "Q/ESC: Quit"
        ]
        if self.frozen:
            help_text.append("")
            help_text.append("*** FROZEN ***")
        
        for text in help_text:
            color = YELLOW if "***" in text else GRAY
            surface = self.small_font.render(text, True, color)
            self.screen.blit(surface, (x, current_y))
            current_y += 18
        
        current_y += 20
        
        # Game info
        if self.obs is not None:
            info_title = self.small_font.render("Game State:", True, WHITE)
            self.screen.blit(info_title, (x, current_y))
            current_y += 20

            info_lines = []
            raw = getattr(self.obs, "raw_observation", None)
            if raw is not None:
                if hasattr(raw, "oxygen_level"):
                    info_lines.append(
                        f"Oxygen: {float(raw.oxygen_level[-1]) * 64:.0f}/64"
                    )
                if hasattr(raw, "collected_divers"):
                    info_lines.append(
                        f"Divers: {float(raw.collected_divers[-1]):.0f}/6"
                    )

            if self.state is not None and hasattr(self.state, "atari_state"):
                try:
                    lives = int(self.state.atari_state.env_state.lives)
                    info_lines.append(f"Lives: {lives}")
                except Exception:
                    pass

            for line in info_lines:
                surface = self.small_font.render(line, True, WHITE)
                self.screen.blit(surface, (x, current_y))
                current_y += 18
    
    def handle_keyboard(self) -> Tuple[int, bool, bool, bool, bool]:
        """Handle keyboard input and return (action, quit_flag, reset_flag, freeze_toggle, step_forward)"""
        keys = pygame.key.get_pressed()
        action = ACTION_NOOP
        quit_flag = False
        reset_flag = False
        freeze_toggle = False
        step_forward = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    quit_flag = True
                elif event.key == pygame.K_r:
                    reset_flag = True
                elif event.key == pygame.K_f:
                    freeze_toggle = True
                elif event.key == pygame.K_n:
                    step_forward = True
        
        # Movement and firing
        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            action = ACTION_UPRIGHTFIRE if keys[pygame.K_SPACE] else ACTION_UPRIGHT
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            action = ACTION_UPLEFTFIRE if keys[pygame.K_SPACE] else ACTION_UPLEFT
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
            action = ACTION_DOWNRIGHTFIRE if keys[pygame.K_SPACE] else ACTION_DOWNRIGHT
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
            action = ACTION_DOWNLEFTFIRE if keys[pygame.K_SPACE] else ACTION_DOWNLEFT
        elif keys[pygame.K_UP]:
            action = ACTION_UPFIRE if keys[pygame.K_SPACE] else ACTION_UP
        elif keys[pygame.K_DOWN]:
            action = ACTION_DOWNFIRE if keys[pygame.K_SPACE] else ACTION_DOWN
        elif keys[pygame.K_RIGHT]:
            action = ACTION_RIGHTFIRE if keys[pygame.K_SPACE] else ACTION_RIGHT
        elif keys[pygame.K_LEFT]:
            action = ACTION_LEFTFIRE if keys[pygame.K_SPACE] else ACTION_LEFT
        elif keys[pygame.K_SPACE]:
            action = ACTION_FIRE
        
        return action, quit_flag, reset_flag, freeze_toggle, step_forward
    
    def run(self):
        """Main game loop"""
        self.reset()
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle input
            action, quit_flag, reset_flag, freeze_toggle, step_forward = self.handle_keyboard()
            if quit_flag:
                running = False
                break
            if reset_flag:
                self.reset()
                self.frozen = False
                self.step_mode = False
                continue
            
            # Handle freeze toggle
            if freeze_toggle:
                self.frozen = not self.frozen
                self.step_mode = False
            
            # Step environment (only if not frozen, or if step forward requested)
            should_step = not self.frozen or step_forward
            if should_step:
                done = self.step(action)
                if done:
                    self.reset()
                    self.frozen = False
                    self.step_mode = False
            
            # Render
            self.screen.fill(BLACK)
            self.render_game(self.state)
            self.draw_reward_panel()
            pygame.display.flip()
            
            # Cap FPS (lower when frozen for easier inspection)
            target_fps = 5 if self.frozen else self.target_fps
            elapsed_ms = clock.tick(target_fps)
            # Calculate actual FPS
            if elapsed_ms > 0:
                current_fps = 1000.0 / elapsed_ms
                self.fps_history.append(current_fps)
                # Keep only last 10 frames for smoothing
                if len(self.fps_history) > 10:
                    self.fps_history.pop(0)
                self.actual_fps = sum(self.fps_history) / len(self.fps_history)
        
        pygame.quit()


def main():
    """Main entry point"""
    # Default environment name; can be changed by editing this constant
    # or by importing this module and constructing your own dashboard.
    env_name = "phoenix"

    # Use EnvManager to construct the environment and renderer in the same way
    # as the training / visualization pipelines.
    manager = EnvManager()
    env, renderer = manager.make_env(env_name, {}, train=False)
    components = manager.get_components(env_name)

    # Create dashboard
    dashboard = RewardDashboard(
        env,
        renderer,
        env_name=env_name,
        get_heuristic_target=components.get_heuristic_target,
        no_target_idx=components.no_target_idx,
        reward_extractor=components.reward_extractor,
        screen_width=1200,
        screen_height=600,
    )
    
    # Run
    try:
        dashboard.run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
