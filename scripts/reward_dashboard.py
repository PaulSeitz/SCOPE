#!/usr/bin/env python3
"""
Interactive Reward Dashboard for Seaquest
Displays the game with real-time reward information while playing.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from typing import Dict, List, Tuple

import src.environments.seaquest.seaquest_wrapper as wrapper
from jaxatari.wrappers import AtariWrapper
from jaxatari import make as make_atari_env
import src.environments.seaquest.seaquest_rewards as rewards
import src.environments.seaquest.seaquest_constants as consts
from src.config import get_default_config

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

# Label names (Player is no longer a label, removed from objects)
LABEL_NAMES = {
    consts.LABEL_PLAYER_TOOL: "Player Tool",
    consts.LABEL_COLLECT: "Collect",
    consts.LABEL_AVOID: "Avoid",
    consts.LABEL_DESTROY: "Destroy",
    consts.LABEL_MONITOR_STATISTIC: "Monitor",
    consts.LABEL_HIGH_LEVEL_CONCEPT: "High Level",
}

# Object type names (Player removed, indices shifted down by 1)
OBJECT_NAMES = [
    "Shark 1", "Shark 2", "Shark 3", "Shark 4", "Shark 5", "Shark 6",  # 0-5
    "Shark 7", "Shark 8", "Shark 9", "Shark 10", "Shark 11", "Shark 12",  # 6-11
    "Sub 1", "Sub 2", "Sub 3", "Sub 4", "Sub 5", "Sub 6",  # 12-17
    "Sub 7", "Sub 8", "Sub 9", "Sub 10", "Sub 11", "Sub 12",  # 18-23
    "Diver 1", "Diver 2", "Diver 3", "Diver 4",  # 24-27
    "Enemy Missile 1", "Enemy Missile 2", "Enemy Missile 3", "Enemy Missile 4",  # 28-31
    "Surface Sub",  # 32
    "Player Missile",  # 33
]

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
    def __init__(self, env, screen_width=800, screen_height=600, config=None):
        self.env = env
        self.config = config if config is not None else get_default_config()
        # Try to get base environment for rendering
        try:
            self.base_env = env._env._env  # Unwrap to get base environment for rendering
        except:
            self.base_env = None
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Seaquest Reward Dashboard")
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
        
    def step(self, action):
        """Step the environment and compute rewards"""
        self.prev_obs = self.obs
        self.prev_state = self.state
        self.obs, self.state, reward, done, info = self.env.step(self.state, action)
        
        # Compute rewards if we have previous state
        if self.prev_obs is not None:
            self.multi_reward = rewards.compute_rewards(
                self.prev_obs, self.obs, self.prev_state, self.state, self.config
            )
            self.total_global_reward += float(self.multi_reward.global_reward)
            
            # Update reward history - add events that just occurred
            # Events are in curr_obs (next_obs_st from reward computation perspective)
            if self.multi_reward is not None:
                # Events are stored in the current observation (which represents the "next" state)
                curr_events = np.array(self.obs.object_events)
                object_labels = np.array(self.obs.object_labels)
                concept_rewards = np.array(self.multi_reward.concept_rewards)
                
                # Find objects that just triggered events (events are set in current obs)
                newly_triggered = np.where(curr_events == consts.EVENT_TRIGGERED)[0]
                for obj_idx in newly_triggered:
                    # object_labels is now multi-hot [Max_Objs, Num_Concepts]
                    # Get the primary label (first non-zero label, or use argmax)
                    if object_labels.ndim == 2:
                        label_vector = object_labels[obj_idx]
                        # Get the primary label (first concept with value > 0.5)
                        label_indices = np.where(label_vector > 0.5)[0]
                        if len(label_indices) > 0:
                            label_id = int(label_indices[0])  # Use first matching label
                        else:
                            label_id = 0  # Fallback
                    else:
                        # Fallback for old format (shouldn't happen)
                        label_id = int(object_labels[obj_idx])
                    
                    reward_val = float(concept_rewards[label_id, obj_idx])
                    obj_name = OBJECT_NAMES[obj_idx] if obj_idx < len(OBJECT_NAMES) else f"Obj {obj_idx}"
                    label_name = LABEL_NAMES.get(label_id, f"Label {label_id}")
                    
                    # Check if this event is already in history (avoid duplicates)
                    is_new = True
                    for hist_entry in self.reward_history:
                        if (hist_entry['obj_idx'] == obj_idx and 
                            hist_entry['step'] == self.step_count):
                            is_new = False
                            break
                    
                    if is_new and abs(reward_val) > 1e-6:  # Only add if reward is significant
                        # Add to history
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
        
        # Use the base environment's render method (takes SeaquestState, returns jax.Array)
        try:
            if self.base_env is not None and hasattr(self.base_env, 'render'):
                # render(state) returns a JAX array
                frame = self.base_env.render(state)
        except Exception:
            pass
        
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
                    return
            except Exception as e:
                # If anything fails, fall through to placeholder
                pass
        
        # Fallback: Draw placeholder with game info
        pygame.draw.rect(self.screen, DARK_GRAY, (10, 10, self.game_width, self.game_height))
        pygame.draw.rect(self.screen, GRAY, (10, 10, self.game_width, self.game_height), 2)
        
        # Show game state info
        if self.obs is not None:
            info_lines = [
                "Seaquest Game",
                "",
                f"Oxygen: {float(self.obs.global_features[0]) * 64:.0f}/64",
                f"Divers: {float(self.obs.global_features[1]) * 6:.0f}/6",
                f"Lives: {float(self.obs.global_features[2]) * 3:.0f}",
                "",
                "Rendering not available",
                "Check game state in panel"
            ]
            y_offset = 20
            for line in info_lines:
                text = self.small_font.render(line, True, WHITE)
                self.screen.blit(text, (20, y_offset))
                y_offset += 20
    
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
        
        # Event detection
        if self.obs is not None:
            object_events = np.array(self.obs.object_events)
            num_events = np.sum(object_events == consts.EVENT_TRIGGERED)
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
            object_mask = np.array(self.obs.object_mask)
            object_labels = np.array(self.obs.object_labels)
            object_events = np.array(self.obs.object_events)
            
            # Show each active object (compact horizontal format)
            active_count = 0
            max_display = 25  # Can show more in horizontal layout
            
            for obj_idx in range(len(object_mask)):
                if object_mask[obj_idx] > 0.5 and active_count < max_display:
                    obj_name = OBJECT_NAMES[obj_idx] if obj_idx < len(OBJECT_NAMES) else f"Obj {obj_idx}"
                    
                    # object_labels is now multi-hot [Max_Objs, Num_Concepts]
                    # Get the primary label (first non-zero label)
                    if object_labels.ndim == 2:
                        label_vector = object_labels[obj_idx]
                        label_indices = np.where(label_vector > 0.5)[0]
                        if len(label_indices) > 0:
                            label_id = int(label_indices[0])  # Use first matching label
                            # For multi-label objects, show all labels
                            label_names = [LABEL_NAMES.get(int(lid), f"Label {lid}") for lid in label_indices]
                            label_name = "/".join(label_names)
                        else:
                            label_id = 0
                            label_name = "Unknown"
                    else:
                        # Fallback for old format
                        label_id = int(object_labels[obj_idx])
                        label_name = LABEL_NAMES.get(label_id, f"Label {label_id}")
                    
                    # Get reward for this object from the concept row matching its primary label
                    reward_val = float(concept_rewards[label_id, obj_idx])
                    
                    # Compact format: Name | Label | Reward
                    obj_text = f"{obj_name}"
                    label_text = f"[{label_name}]"
                    event_triggered = object_events[obj_idx] == consts.EVENT_TRIGGERED
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
                more_text = self.small_font.render(f"... +{np.sum(object_mask > 0.5) - max_display} more", True, GRAY)
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
            
            info_lines = [
                f"Oxygen: {float(self.obs.global_features[0]) * 64:.0f}/64",
                f"Divers: {float(self.obs.global_features[1]) * 6:.0f}/6",
                f"Lives: {float(self.obs.global_features[2]) * 3:.0f}",
            ]
            
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
            self.render_game(self.state.env_state)
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
    # Get config
    config = get_default_config()
    
    # Create environment with settings for manual play
    base_env = make_atari_env('seaquest')
    atari_wrapper = AtariWrapper(
        base_env,
        frame_stack_size=1,
        frame_skip=1,  # No frame skipping for manual play
        sticky_actions=False,  # No sticky actions for manual control
        episodic_life=False,  # Don't end episode on life loss for manual play
        first_fire=False,  # Don't auto-fire on reset
        noop_reset=0,  # No noop actions on reset
        clip_reward=False,
        max_pooling=False,
    )
    env = wrapper.SeaquestConceptWrapper(atari_wrapper, config=config)
    
    # Create dashboard
    dashboard = RewardDashboard(env, screen_width=1200, screen_height=600, config=config)
    
    # Run
    try:
        dashboard.run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
