#!/usr/bin/env python3
"""
Interactive Agent Visualization for Seaquest
Displays the game with real-time agent decision information (concept, instance, Q-values).
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
import pickle
import argparse
from typing import Dict, List, Tuple, Optional

import src.environments.seaquest.seaquest_wrapper as wrapper
from jaxatari.wrappers import AtariWrapper
from jaxatari import make as make_atari_env
from src.environments.seaquest.seaquest_rewards import _compute_rewards_jitted
import src.environments.seaquest.seaquest_constants as consts
from src.config import get_default_config, Config, load_config_from_file
from src.agent.orchestrator import RelationalAgent
from src.environments.seaquest.datatypes import ConceptObs, MultiReward

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
ORANGE = (255, 165, 0)

# Concept names (Renumbered: Player Tool removed, concepts renumbered)
CONCEPT_NAMES = {
    consts.LABEL_COLLECT: "Collect",      # 0
    consts.LABEL_AVOID: "Avoid",         # 1
    consts.LABEL_DESTROY: "Destroy",     # 2
    consts.LABEL_DEFAULT: "Monitor",    # 3 (was 4, now 3)
}

# Object type names (Player Missile removed from object list)
# Total: 33 objects (12 sharks + 12 subs + 4 divers + 4 enemy missiles + 1 surface sub)
# But max_objs is 34, so index 33 is padding
OBJECT_NAMES = [
    "Shark 1", "Shark 2", "Shark 3", "Shark 4", "Shark 5", "Shark 6",  # 0-5
    "Shark 7", "Shark 8", "Shark 9", "Shark 10", "Shark 11", "Shark 12",  # 6-11
    "Sub 1", "Sub 2", "Sub 3", "Sub 4", "Sub 5", "Sub 6",  # 12-17
    "Sub 7", "Sub 8", "Sub 9", "Sub 10", "Sub 11", "Sub 12",  # 18-23
    "Diver 1", "Diver 2", "Diver 3", "Diver 4",  # 24-27
    "Enemy Missile 1", "Enemy Missile 2", "Enemy Missile 3", "Enemy Missile 4",  # 28-31
    "Surface Sub",  # 32
    # Index 33 is padding (Player Missile removed, handled via capabilities)
]


class AgentVisualization:
    def __init__(self, env, agent: RelationalAgent, screen_width=1400, screen_height=700, config=None, phase=3, force_concept_id=None):
        self.env = env
        self.agent = agent
        self.config = config if config is not None else get_default_config()
        self.phase = phase  # Training phase: 1, 2, or 3
        self.force_concept_id = force_concept_id  # Optional: force a specific concept (overrides phase)
        
        # Temporarily modify agent's config for forced concept mode
        if force_concept_id is not None:
            self.agent.config.training.force_concept_mode = True
            self.agent.config.training.force_concept_id = force_concept_id
        
        # Store expected dimensions from agent's config
        self.expected_global_feats = self.agent.config.environment.n_global_feats
        
        # Infer expected obj_dim from agent's meta policy params
        # This handles cases where the model was trained with different obj_dim than config
        def infer_obj_dim_from_params(params):
            """Infer obj_dim from Dense_1 layer which processes objects"""
            def find_dense1_kernel(d, path=""):
                if isinstance(d, dict):
                    for k, v in d.items():
                        result = find_dense1_kernel(v, f"{path}.{k}" if path else k)
                        if result is not None:
                            return result
                elif hasattr(d, 'shape') and len(d.shape) == 2:
                    if 'Dense_1' in path and 'kernel' in path:
                        obj_dim, hidden_dim = d.shape
                        return int(obj_dim)
                return None
            return find_dense1_kernel(params)
        
        inferred_obj_dim = infer_obj_dim_from_params(self.agent.meta_state.params)
        if inferred_obj_dim is not None:
            self.expected_obj_dim = inferred_obj_dim
            print(f"Info: Inferred expected_obj_dim={inferred_obj_dim} from loaded model.")
        else:
            # Fallback to config
            num_concepts = self.config.concepts.label_high_level_concept + 1
            n_obj_feats_with_labels = self.config.environment.n_obj_feats + num_concepts
            self.expected_obj_dim = self.config.environment.frame_stack_size * n_obj_feats_with_labels
            print(f"Warning: Could not infer obj_dim from model, using config value: {self.expected_obj_dim}")
        
        # Try to get base environment for rendering
        try:
            self.base_env = env._env._env  # Unwrap to get base environment for rendering
        except:
            self.base_env = None
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Agent Visualization - Seaquest")
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
        self.bold_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 32)
        
        # Game state
        self.obs = None
        self.state = None
        self.prev_obs = None
        self.prev_state = None
        self.multi_reward = None
        self.step_count = 0
        self.total_global_reward = 0.0
        self.rng_key = jax.random.key(42)
        
        # Agent decision state
        self.agent_state = None
        self.meta_q_values = None
        self.sub_q_values = None
        self.selector_weights = None
        self.valid_concepts_mask = None # Store the mask
        
        # Freeze/step controls
        self.frozen = False
        self.step_mode = False
        self.target_fps = 30
        self.actual_fps = 0.0
        self.fps_history = []
        
        # Display settings
        self.game_scale = 2
        self.game_width = 160 * self.game_scale
        self.game_height = 210 * self.game_scale
        self.panel_width = screen_width - self.game_width - 20
        self.panel_start_x = self.game_width + 20
        
        # Column layout
        self.num_columns = 3
        self.column_width = self.panel_width // self.num_columns
        self.column_spacing = 10
        
        # Track concept/instance history
        self.concept_history = []
        self.instance_history = []
        self.history_length = 50
        
        # Set global_step based on phase for phase simulation
        # Phase 1: Use a step value < phase_1_steps (e.g., 0)
        # Phase 2: Use a step value between phase_1_steps and phase_2_steps
        # Phase 3: Use a step value > phase_2_steps (full model)
        phase_1_steps = self.config.training.phase_1_steps
        phase_2_steps = self.config.training.phase_2_steps
        
        if self.phase == 1:
            self.simulated_global_step = 0
        elif self.phase == 2:
            # Set to a value in phase 2 range
            self.simulated_global_step = phase_1_steps + (phase_2_steps - phase_1_steps) // 2
        elif self.phase == 3:
            # Set to a value after phase 2 (full model)
            self.simulated_global_step = phase_2_steps + 1_000_000  # Well into phase 3
        else:
            # Default to phase 3 (full model)
            self.simulated_global_step = phase_2_steps + 1_000_000
        
    def reset(self):
        """Reset the environment"""
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.obs, self.state = self.env.reset(subkey)
        self.prev_obs = None
        self.prev_state = None
        self.multi_reward = None
        self.step_count = 0
        self.total_global_reward = 0.0
        self.agent_state = None
        # Track global step for curriculum phases in the agent
        # Use simulated_global_step instead of incrementing, to maintain phase behavior
        self.concept_history = []
        self.instance_history = []
        
    def _adapt_observation(self, obs: ConceptObs) -> ConceptObs:
        """Adapt observation to match model's expected input dimensions"""
        # Pad or truncate global_features if needed
        current_global_feats = obs.global_features.shape[-1] if obs.global_features.ndim > 0 else 0
        
        if current_global_feats < self.expected_global_feats:
            # Pad with zeros
            if obs.global_features.ndim == 1:
                padding = jnp.zeros(self.expected_global_feats - current_global_feats)
                adapted_global = jnp.concatenate([obs.global_features, padding])
            else:
                padding = jnp.zeros((obs.global_features.shape[0], self.expected_global_feats - current_global_feats))
                adapted_global = jnp.concatenate([obs.global_features, padding], axis=-1)
        elif current_global_feats > self.expected_global_feats:
            # Truncate
            if obs.global_features.ndim == 1:
                adapted_global = obs.global_features[:self.expected_global_feats]
            else:
                adapted_global = obs.global_features[..., :self.expected_global_feats]
        else:
            adapted_global = obs.global_features
        
        # Pad or truncate objects array to match max_objs (required for reward computation)
        expected_max_objs = self.config.environment.max_objs
        # Handle both batched and unbatched observations
        if obs.objects.ndim == 2:
            # Unbatched: shape is (num_objs, features)
            current_max_objs = obs.objects.shape[0]
        elif obs.objects.ndim == 3:
            # Batched: shape is (batch, num_objs, features)
            current_max_objs = obs.objects.shape[1]
        else:
            current_max_objs = 0
        
        if current_max_objs < expected_max_objs:
            # Pad objects with zeros
            if obs.objects.ndim == 2:
                # Shape: (num_objs, features) -> (max_objs, features)
                num_to_pad = expected_max_objs - current_max_objs
                obj_features = obs.objects.shape[-1]
                padding = jnp.zeros((num_to_pad, obj_features))
                adapted_objects = jnp.concatenate([obs.objects, padding], axis=0)
            else:
                # Shape: (batch, num_objs, features) -> (batch, max_objs, features)
                num_to_pad = expected_max_objs - current_max_objs
                obj_features = obs.objects.shape[-1]
                padding = jnp.zeros((obs.objects.shape[0], num_to_pad, obj_features))
                adapted_objects = jnp.concatenate([obs.objects, padding], axis=1)
            
            # Pad object_mask, object_labels, and object_events to match
            if obs.object_mask.ndim == 1:
                mask_padding = jnp.zeros(num_to_pad, dtype=obs.object_mask.dtype)
                adapted_mask = jnp.concatenate([obs.object_mask, mask_padding])
            else:
                mask_padding = jnp.zeros((obs.object_mask.shape[0], num_to_pad), dtype=obs.object_mask.dtype)
                adapted_mask = jnp.concatenate([obs.object_mask, mask_padding], axis=1)
            
            # object_labels is now multi-hot [Max_Objs, Num_Concepts] or [Batch, Max_Objs, Num_Concepts]
            if obs.object_labels.ndim == 2:
                # Unbatched: [Max_Objs, Num_Concepts]
                labels_padding = jnp.zeros((num_to_pad, obs.object_labels.shape[1]), dtype=obs.object_labels.dtype)
                adapted_labels = jnp.concatenate([obs.object_labels, labels_padding], axis=0)
            else:
                # Batched: [Batch, Max_Objs, Num_Concepts]
                labels_padding = jnp.zeros((obs.object_labels.shape[0], num_to_pad, obs.object_labels.shape[2]), dtype=obs.object_labels.dtype)
                adapted_labels = jnp.concatenate([obs.object_labels, labels_padding], axis=1)
            
            if obs.object_events.ndim == 1:
                events_padding = jnp.zeros(num_to_pad, dtype=obs.object_events.dtype)
                adapted_events = jnp.concatenate([obs.object_events, events_padding])
            else:
                events_padding = jnp.zeros((obs.object_events.shape[0], num_to_pad), dtype=obs.object_events.dtype)
                adapted_events = jnp.concatenate([obs.object_events, events_padding], axis=1)
        elif current_max_objs > expected_max_objs:
            # Truncate objects
            if obs.objects.ndim == 2:
                adapted_objects = obs.objects[:expected_max_objs, :]
            else:
                adapted_objects = obs.objects[:, :expected_max_objs, :]
            
            if obs.object_mask.ndim == 1:
                adapted_mask = obs.object_mask[:expected_max_objs]
            else:
                adapted_mask = obs.object_mask[:, :expected_max_objs]
            
            # object_labels is now multi-hot [Max_Objs, Num_Concepts] or [Batch, Max_Objs, Num_Concepts]
            if obs.object_labels.ndim == 2:
                # Unbatched: [Max_Objs, Num_Concepts]
                adapted_labels = obs.object_labels[:expected_max_objs, :]
            else:
                # Batched: [Batch, Max_Objs, Num_Concepts]
                adapted_labels = obs.object_labels[:, :expected_max_objs, :]
            
            if obs.object_events.ndim == 1:
                adapted_events = obs.object_events[:expected_max_objs]
            else:
                adapted_events = obs.object_events[:, :expected_max_objs]
        else:
            # Already correct size
            adapted_objects = obs.objects
            adapted_mask = obs.object_mask
            adapted_labels = obs.object_labels
            adapted_events = obs.object_events
        
        # Adapt object feature dimension (obj_dim) to match model's expected dimension
        current_obj_dim = adapted_objects.shape[-1]
        if current_obj_dim < self.expected_obj_dim:
            # Pad object features with zeros
            if adapted_objects.ndim == 2:
                # Unbatched: [max_objs, features] -> [max_objs, expected_features]
                padding = jnp.zeros((adapted_objects.shape[0], self.expected_obj_dim - current_obj_dim))
                adapted_objects = jnp.concatenate([adapted_objects, padding], axis=-1)
            else:
                # Batched: [batch, max_objs, features] -> [batch, max_objs, expected_features]
                padding = jnp.zeros((adapted_objects.shape[0], adapted_objects.shape[1], self.expected_obj_dim - current_obj_dim))
                adapted_objects = jnp.concatenate([adapted_objects, padding], axis=-1)
        elif current_obj_dim > self.expected_obj_dim:
            # Truncate object features
            if adapted_objects.ndim == 2:
                adapted_objects = adapted_objects[:, :self.expected_obj_dim]
            else:
                adapted_objects = adapted_objects[:, :, :self.expected_obj_dim]
        
        # Return adapted observation
        return ConceptObs(
            global_features=adapted_global,
            capabilities=obs.capabilities,
            objects=adapted_objects,
            object_mask=adapted_mask,
            object_labels=adapted_labels,
            object_events=adapted_events,
            active_concept=obs.active_concept if hasattr(obs, 'active_concept') else None,
            active_target_idx=obs.active_target_idx if hasattr(obs, 'active_target_idx') else None
        )
    
    def step(self):
        """Step the environment using agent's action"""
        self.prev_obs = self.obs
        self.prev_state = self.state
        
        # Get agent's action
        self.rng_key, key_act = jax.random.split(self.rng_key)
        
        # Get previous concept/target for stickiness
        prev_concept = jnp.array(0, dtype=jnp.int32)
        prev_target_idx = jnp.array(0, dtype=jnp.int32)
        if self.agent_state is not None:
            prev_concept = self.agent_state.selected_concept
            prev_target_idx = self.agent_state.selected_target_idx
        
        # Adapt observation to match model's expected dimensions
        adapted_obs = self._adapt_observation(self.obs)
        
        # Select action
        # Use simulated_global_step to maintain phase behavior (doesn't increment)
        self.agent_state = self.agent.select_action(
            adapted_obs,
            prev_concept=prev_concept,
            prev_target_idx=prev_target_idx,
            key=key_act,
            epsilon_meta=0.0,  # No exploration during visualization
            epsilon_sub=0.0,
            global_step=self.simulated_global_step,
        )
        
        # Get Q-values for display
        # Use adapted observation for Q-value computations
        adapted_obs_for_q = self._adapt_observation(self.obs)
        
        # Ensure observations are batched for Q-value computations
        is_unbatched = adapted_obs_for_q.global_features.ndim == 1
        if is_unbatched:
            # Add batch dimension
            obs_batched = jax.tree.map(lambda x: x[None, ...], adapted_obs_for_q)
            agent_state_batched = jax.tree.map(
                lambda x: x[None, ...] if x.ndim == 0 else x,
                self.agent_state
            )
        else:
            obs_batched = adapted_obs_for_q
            agent_state_batched = self.agent_state
        
        # Meta Q-values
        # Labels are now part of objects, so no separate object_labels parameter needed
        self.meta_q_values = self.agent.meta_state.apply_fn(
            self.agent.meta_state.params,
            obs_batched.global_features,
            obs_batched.objects,
            obs_batched.object_mask
        )
        
        # --- REPLICATE MASKING LOGIC FOR VISUALIZATION ---
        batch_size = self.meta_q_values.shape[0]
        num_concepts = self.meta_q_values.shape[1]
        
        # object_labels is now multi-hot [Batch, Max_Objs, Num_Concepts]
        # Transpose to [Batch, Num_Concepts, Max_Objs] to check label matches
        labels_T = jnp.transpose(obs_batched.object_labels, (0, 2, 1))  # [Batch, Num_Concepts, Max_Objs]
        existence = (obs_batched.object_mask[:, None, :] > 0.5).astype(jnp.float32)  # [Batch, 1, Max_Objs]
        
        # Check if any object has each concept label
        # Use multiplication instead of & since both arrays are float32
        has_label = labels_T * existence  # [Batch, Num_Concepts, Max_Objs]
        label_exists = jnp.any(has_label, axis=-1)  # [Batch, Num_Concepts]
        
        valid_mask = label_exists
        # Note: LABEL_PLAYER no longer exists (player removed from objects)
        valid_mask = valid_mask.at[:, consts.LABEL_DEFAULT].set(True)
        # valid_mask = valid_mask.at[:, consts.LABEL_HIGH_LEVEL_CONCEPT].set(True)

        any_valid = jnp.any(valid_mask, axis=-1, keepdims=True)
        fallback_mask = jax.nn.one_hot(jnp.full((batch_size,), consts.LABEL_DEFAULT), num_concepts)
        self.valid_concepts_mask = jnp.where(any_valid, valid_mask, fallback_mask)
        # ------------------------------------------------
        
        # Selector weights (conditioned on currently selected concept, matching training call signature)
        selector_num_concepts = num_concepts  # Same as meta policy concepts
        selector_cmd_one_hot = jax.nn.one_hot(
            agent_state_batched.selected_concept,
            num_classes=selector_num_concepts,
        )
        self.selector_weights = self.agent.selector_state.apply_fn(
            self.agent.selector_state.params,
            obs_batched.global_features,
            obs_batched.objects,
            obs_batched.object_mask,
            selector_cmd_one_hot,
        )
        
        # Sub Q-values (need to compute relative vector)
        target_obj = jnp.take_along_axis(
            obs_batched.objects,
            agent_state_batched.selected_target_idx[:, None, None],
            axis=1
        ).squeeze(1)
        p_pos = obs_batched.objects[:, 0, 0:2]
        t_pos = target_obj[:, 0:2]
        rel_vec = t_pos - p_pos
        
        num_concepts = self.config.concepts.label_high_level_concept + 1
        concept_one_hot = jax.nn.one_hot(agent_state_batched.selected_concept, num_classes=num_concepts)
        context = jnp.concatenate([obs_batched.capabilities, concept_one_hot], axis=-1)
        
        self.sub_q_values = self.agent.sub_state.apply_fn(
            self.agent.sub_state.params,
            obs_batched.global_features,
            rel_vec,
            context
        )
        
        # Execute action
        action_idx = int(self.agent_state.action)
        self.obs, self.state, reward, done, info = self.env.step(self.state, action_idx)
        
        # Compute rewards
        if self.prev_obs is not None:
            # Extract reward config values
            reward_scale = self.config.reward.reward_scale
            bonus_success = self.config.reward.bonus_success
            penalty_failure = self.config.reward.penalty_failure
            aim_bonus_scale = self.config.reward.aim_bonus_scale
            global_score_scale = self.config.reward.global_score_scale
            
            # Adapt observations (rewards don't need the full global features, but we adapt for consistency)
            # IMPORTANT: Adapt BEFORE batching to ensure objects are padded to max_objs
            adapted_prev_obs = self._adapt_observation(self.prev_obs)
            adapted_obs_for_reward = self._adapt_observation(self.obs)
            
            # The reward computation expects UNBATCHED observations (original design)
            # So we need to ensure observations are unbatched before calling it
            # Check if observations are batched and unbatch them if needed
            prev_obs_is_batched = adapted_prev_obs.global_features.ndim == 2
            
            if prev_obs_is_batched:
                # Unbatch: remove the batch dimension
                prev_obs_unbatched = jax.tree.map(lambda x: x[0, ...] if x.ndim > 0 else x, adapted_prev_obs)
                obs_unbatched = jax.tree.map(lambda x: x[0, ...] if x.ndim > 0 else x, adapted_obs_for_reward)
                prev_state_unbatched = jax.tree.map(lambda x: x[0, ...] if x.ndim > 0 else x, self.prev_state)
                state_unbatched = jax.tree.map(lambda x: x[0, ...] if x.ndim > 0 else x, self.state)
            else:
                # Already unbatched
                prev_obs_unbatched = adapted_prev_obs
                obs_unbatched = adapted_obs_for_reward
                prev_state_unbatched = self.prev_state
                state_unbatched = self.state
            
            # Compute rewards (function expects UNBATCHED inputs - original design)
            self.multi_reward = _compute_rewards_jitted(
                prev_obs_unbatched, obs_unbatched,
                prev_state_unbatched, state_unbatched,
                reward_scale, bonus_success, penalty_failure,
                aim_bonus_scale, global_score_scale
            )
            
            self.total_global_reward += float(self.multi_reward.global_reward)
        
        # Update history
        concept_id = int(self.agent_state.selected_concept)
        instance_id = int(self.agent_state.selected_target_idx)
        self.concept_history.append((self.step_count, concept_id))
        self.instance_history.append((self.step_count, instance_id))
        if len(self.concept_history) > self.history_length:
            self.concept_history.pop(0)
            self.instance_history.pop(0)
        
        self.step_count += 1
        return done
        
    def render_game(self, state):
        """Render the game screen"""
        if state is None:
            # Draw placeholder if state not available yet
            pygame.draw.rect(self.screen, DARK_GRAY, (10, 10, self.game_width, self.game_height))
            pygame.draw.rect(self.screen, GRAY, (10, 10, self.game_width, self.game_height), 2)
            return
        
        frame = None
        
        try:
            if self.base_env is not None and hasattr(self.base_env, 'render'):
                # Extract env_state from wrapped state if needed
                render_state = state.env_state if hasattr(state, 'env_state') else state
                frame = self.base_env.render(render_state)
        except Exception as e:
            # Print error for debugging (only first time to avoid spam)
            if not hasattr(self, '_render_error_logged'):
                import traceback
                print(f"Render error: {e}")
                traceback.print_exc()
                self._render_error_logged = True
            pass
        
        if frame is not None:
            try:
                frame = jax.device_get(frame)
                frame = np.asarray(frame)
                
                if frame.dtype != np.uint8:
                    if np.issubdtype(frame.dtype, np.floating):
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                if len(frame.shape) == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    elif frame.shape[2] != 3:
                        frame = None
                
                if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_surface = pygame.surfarray.make_surface(
                        np.transpose(frame, (1, 0, 2))
                    )
                    frame_surface = pygame.transform.scale(
                        frame_surface, (self.game_width, self.game_height)
                    )
                    self.screen.blit(frame_surface, (10, 10))
                    return
            except Exception as e:
                # Print error for debugging rendering issues
                import traceback
                traceback.print_exc()
                pass
        
        # Fallback placeholder
        pygame.draw.rect(self.screen, DARK_GRAY, (10, 10, self.game_width, self.game_height))
        pygame.draw.rect(self.screen, GRAY, (10, 10, self.game_width, self.game_height), 2)
        
        if self.obs is not None:
            info_lines = [
                "Seaquest Game",
                "",
                f"Oxygen: {float(self.obs.global_features[0]) * 64:.0f}/64",
                f"Divers: {float(self.obs.global_features[1]) * 6:.0f}/6",
                f"Lives: {float(self.obs.global_features[2]) * 3:.0f}",
            ]
            y_offset = 20
            for line in info_lines:
                text = self.small_font.render(line, True, WHITE)
                self.screen.blit(text, (20, y_offset))
                y_offset += 20
    
    def draw_panel(self):
        """Draw the information panel"""
        panel_y = 10
        
        # Column 1: Agent Decisions
        col1_x = self.panel_start_x
        self._draw_column1(col1_x, panel_y)
        
        # Column 2: Q-Values & Statistics
        col2_x = self.panel_start_x + self.column_width + self.column_spacing
        self._draw_column2(col2_x, panel_y)
        
        # Column 3: History & Controls
        col3_x = self.panel_start_x + 2 * (self.column_width + self.column_spacing)
        self._draw_column3(col3_x, panel_y)
    
    def _draw_column1(self, x, y):
        """Draw column 1: Agent Decisions"""
        current_y = y
        
        # Title
        title = self.bold_font.render("Agent Decisions", True, WHITE)
        self.screen.blit(title, (x, current_y))
        current_y += 30
        
        if self.agent_state is not None:
            # Selected Concept
            concept_id = int(self.agent_state.selected_concept)
            concept_name = CONCEPT_NAMES.get(concept_id, f"Concept {concept_id}")
            concept_text = self.font.render(f"Concept: {concept_name}", True, YELLOW)
            self.screen.blit(concept_text, (x, current_y))
            current_y += 25
            
            # Selected Instance
            instance_id = int(self.agent_state.selected_target_idx)
            instance_name = OBJECT_NAMES[instance_id] if instance_id < len(OBJECT_NAMES) else f"Obj {instance_id}"
            instance_text = self.font.render(f"Instance: {instance_name}", True, CYAN)
            self.screen.blit(instance_text, (x, current_y))
            current_y += 25
            
            # Action
            action_text = self.font.render(f"Action: {int(self.agent_state.action)}", True, GREEN)
            self.screen.blit(action_text, (x, current_y))
            current_y += 30
            
            # --- UPDATED META Q-VALUES DISPLAY ---
            if self.meta_q_values is not None:
                header = self.small_font.render("Meta Q-Values (M=Masked):", True, WHITE)
                self.screen.blit(header, (x, current_y))
                current_y += 20
                
                meta_q = np.array(self.meta_q_values[0])
                # Get the mask (convert from JAX)
                mask = np.array(self.valid_concepts_mask[0]) if self.valid_concepts_mask is not None else np.ones_like(meta_q, dtype=bool)
                
                for i, q_val in enumerate(meta_q):
                    concept_name = CONCEPT_NAMES.get(i, f"C{i}")
                    color = YELLOW if i == concept_id else GRAY
                    
                    # Check validity
                    if not mask[i]:
                        q_str = "BLOCKED"
                        color = RED
                    else:
                        q_str = f"{q_val:.3f}"
                        
                    q_text = self.small_font.render(f"{concept_name}: {q_str}", True, color)
                    self.screen.blit(q_text, (x + 5, current_y))
                    current_y += 16
            # -------------------------------------
            
            current_y += 10
            
            # Selector Weights (for selected concept)
            if self.selector_weights is not None:
                header = self.small_font.render("Selector Weights:", True, WHITE)
                self.screen.blit(header, (x, current_y))
                current_y += 20
                
                # selector_weights has shape [Batch, Num_Objects] for the currently selected concept
                sel_weights = np.array(self.selector_weights[0])
                object_mask = np.array(self.obs.object_mask[0])
                
                # Show top 5 objects
                # Handle both scalar and array object_mask
                if np.isscalar(object_mask) or (hasattr(object_mask, 'ndim') and object_mask.ndim == 0):
                    object_mask = np.atleast_1d(object_mask)
                active_indices = np.where(object_mask > 0.5)[0]
                if len(active_indices) > 0:
                    top_indices = active_indices[np.argsort(sel_weights[active_indices])[-5:]][::-1]
                    for idx in top_indices:
                        obj_name = OBJECT_NAMES[idx] if idx < len(OBJECT_NAMES) else f"Obj {idx}"
                        weight = sel_weights[idx]
                        color = GREEN if idx == instance_id else WHITE
                        weight_text = self.small_font.render(f"{obj_name}: {weight:.3f}", True, color)
                        self.screen.blit(weight_text, (x + 5, current_y))
                        current_y += 16
    
    def _draw_column2(self, x, y):
        """Draw column 2: Q-Values & Detailed Reward Breakdown"""
        current_y = y
        
        # --- 1. Sub-Policy Q-Values (Action Confidence) ---
        if self.sub_q_values is not None and self.agent_state is not None:
            header = self.small_font.render("Sub-Policy Q-Values (Action):", True, WHITE)
            self.screen.blit(header, (x, current_y))
            current_y += 20
            
            sub_q = np.array(self.sub_q_values[0])
            action_id = int(self.agent_state.action)
            
            # Show top 5 actions
            top_actions = np.argsort(sub_q)[-5:][::-1]
            for act_id in top_actions:
                q_val = sub_q[act_id]
                color = GREEN if act_id == action_id else GRAY
                # Simple mapping for Seaquest actions (approximate)
                if act_id == 0: act_name = "NOOP"
                elif act_id == 1: act_name = "FIRE"
                elif 2 <= act_id <= 5: act_name = f"MOVE {act_id}"
                else: act_name = f"ACT {act_id}"
                
                q_text = self.small_font.render(f"{act_name}: {q_val:.3f}", True, color)
                self.screen.blit(q_text, (x + 5, current_y))
                current_y += 16
            
            current_y += 15
        
        # --- 2. THE REWARD DASHBOARD (NEW) ---
        # This section visualizes exactly what each "Brain" is optimizing for.
        
        if self.multi_reward is not None and self.agent_state is not None:
            title = self.bold_font.render("Reward Signals (Last Step)", True, WHITE)
            self.screen.blit(title, (x, current_y))
            current_y += 25
            
            # Extract indices
            c_idx = int(self.agent_state.selected_concept)
            t_idx = int(self.agent_state.selected_target_idx)
            
            # Helper to extract value from batched/unbatched array
            def get_val(arr, *indices):
                if arr.ndim == len(indices) + 1: # Batched [B, ...]
                    return float(arr[0, *indices])
                return float(arr[*indices])

            # A. META-POLICY REWARD (Strategic)
            # Optimizes: Global Score + Survival
            meta_rew = get_val(self.multi_reward.global_reward)
            
            m_color = GREEN if meta_rew > 0 else (RED if meta_rew < 0 else WHITE)
            self.screen.blit(self.small_font.render("Meta-Policy (Global):", True, YELLOW), (x, current_y))
            self.screen.blit(self.small_font.render(f"{meta_rew:.4f}", True, m_color), (x + 120, current_y))
            current_y += 16
            
            # B. SUB-POLICY REWARD (Execution)
            # Optimizes: Shaping + Global Shared Fate
            # This is the specific scalar reward the active sub-policy received this step.
            sub_rew = get_val(self.multi_reward.concept_rewards, c_idx, t_idx)
            
            s_color = CYAN if sub_rew > 0 else (ORANGE if sub_rew < 0 else WHITE)
            self.screen.blit(self.small_font.render("Sub-Policy (Hybrid):", True, YELLOW), (x, current_y))
            self.screen.blit(self.small_font.render(f"{sub_rew:.4f}", True, s_color), (x + 120, current_y))
            current_y += 16
            
            # C. COMPONENT ANALYSIS (Why is Sub-Policy happy/sad?)
            # Calculate the "Pure Shaping" by removing the Global component
            # (Assuming sub_rew = shaping + meta_rew)
            pure_shaping = sub_rew - meta_rew
            
            p_color = GRAY
            if abs(pure_shaping) > 0.001:
                p_color = GREEN if pure_shaping > 0 else RED
                
            self.screen.blit(self.small_font.render("  L Shaping (Local):", True, GRAY), (x, current_y))
            self.screen.blit(self.small_font.render(f"{pure_shaping:.4f}", True, p_color), (x + 120, current_y))
            current_y += 25

            # --- D. INSTANCE SELECTOR INSIGHT ---
            # Show rewards for ALTERNATIVE targets to see if the selector made the right choice.
            # We scan the rewards for the *active concept* across all objects.
            
            self.screen.blit(self.small_font.render("Instance Potentials:", True, WHITE), (x, current_y))
            current_y += 16
            
            # Get rewards for all objects under current concept
            # Shape: [Max_Objs]
            if self.multi_reward.concept_rewards.ndim == 3:
                all_obj_rewards = self.multi_reward.concept_rewards[0, c_idx, :]
            else:
                all_obj_rewards = self.multi_reward.concept_rewards[c_idx, :]
                
            # Filter for objects that actually exist
            obj_mask = self.obs.object_mask[0] if self.obs.object_mask.ndim == 2 else self.obs.object_mask
            
            # Find best alternative
            valid_indices = np.where(obj_mask > 0.5)[0]
            if len(valid_indices) > 0:
                # Sort by reward
                best_idx = valid_indices[np.argmax(all_obj_rewards[valid_indices])]
                best_val = float(all_obj_rewards[best_idx])
                
                # Did we pick the best?
                is_optimal = (best_idx == t_idx) or (abs(best_val - sub_rew) < 0.001)
                
                # Current Selection
                sel_name = OBJECT_NAMES[t_idx] if t_idx < len(OBJECT_NAMES) else f"Obj {t_idx}"
                self.screen.blit(self.small_font.render(f"Selected [{sel_name}]:", True, GRAY), (x + 10, current_y))
                self.screen.blit(self.small_font.render(f"{sub_rew:.2f}", True, s_color), (x + 130, current_y))
                current_y += 16
                
                # Best Alternative (if different)
                if not is_optimal:
                    alt_name = OBJECT_NAMES[best_idx] if best_idx < len(OBJECT_NAMES) else f"Obj {best_idx}"
                    self.screen.blit(self.small_font.render(f"Better   [{alt_name}]:", True, GRAY), (x + 10, current_y))
                    self.screen.blit(self.small_font.render(f"{best_val:.2f}", True, MAGENTA), (x + 130, current_y))
                    current_y += 16
            
            current_y += 10

        # --- 3. Global Statistics ---
        stats_title = self.small_font.render("Global Stats:", True, WHITE)
        self.screen.blit(stats_title, (x, current_y))
        current_y += 20
        
        # Build mode description
        if self.force_concept_id is not None:
            concept_names = {0: "COLLECT", 1: "AVOID", 2: "DESTROY", 3: "DEFAULT"}
            concept_name = concept_names.get(self.force_concept_id, f"Concept {self.force_concept_id}")
            phase_name = f"Forced: {concept_name}"
        elif self.phase == 1:
            phase_name = "Phase 1 (Heuristic Meta + Heuristic Selector)"
        elif self.phase == 2:
            phase_name = "Phase 2 (Heuristic Meta + Network Selector)"
        else:  # phase == 3
            phase_name = "Phase 3 (Full Model: Network Meta + Network Selector)"
        
        stats = [
            f"Step: {self.step_count}",
            f"FPS: {self.actual_fps:.1f}/{self.target_fps}",
            f"Mode: {phase_name}",
        ]
        
        if self.obs is not None:
            stats.extend([
                f"Oxygen: {float(self.obs.global_features[0]) * 64:.0f}/64",
                f"Divers: {float(self.obs.global_features[1]) * 6:.0f}/6",
                f"Lives: {float(self.obs.global_features[2]) * 3:.0f}",
            ])
        
        for stat in stats:
            text = self.small_font.render(stat, True, WHITE)
            self.screen.blit(text, (x, current_y))
            current_y += 18
    
    def _draw_column3(self, x, y):
        """Draw column 3: History & Controls"""
        current_y = y
        
        # Title
        title = self.bold_font.render("History & Controls", True, WHITE)
        self.screen.blit(title, (x, current_y))
        current_y += 30
        
        # Concept History
        if len(self.concept_history) > 0:
            header = self.small_font.render("Recent Concepts:", True, YELLOW)
            self.screen.blit(header, (x, current_y))
            current_y += 20
            
            # Show last 10
            for step, concept_id in self.concept_history[-10:]:
                concept_name = CONCEPT_NAMES.get(concept_id, f"C{concept_id}")
                age = self.step_count - step
                color = YELLOW if age <= 5 else GRAY
                hist_text = self.small_font.render(f"{concept_name} ({age}s ago)", True, color)
                self.screen.blit(hist_text, (x + 5, current_y))
                current_y += 14
            
            current_y += 10
        
        # Instance History
        if len(self.instance_history) > 0:
            header = self.small_font.render("Recent Instances:", True, CYAN)
            self.screen.blit(header, (x, current_y))
            current_y += 20
            
            # Show last 10
            for step, instance_id in self.instance_history[-10:]:
                instance_name = OBJECT_NAMES[instance_id] if instance_id < len(OBJECT_NAMES) else f"Obj {instance_id}"
                age = self.step_count - step
                color = CYAN if age <= 5 else GRAY
                hist_text = self.small_font.render(f"{instance_name} ({age}s ago)", True, color)
                self.screen.blit(hist_text, (x + 5, current_y))
                current_y += 14
        
        current_y += 20
        
        # Controls
        controls_title = self.small_font.render("Controls:", True, WHITE)
        self.screen.blit(controls_title, (x, current_y))
        current_y += 20
        
        help_text = [
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
    
    def handle_keyboard(self) -> Tuple[bool, bool, bool, bool]:
        """Handle keyboard input"""
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
        
        return quit_flag, reset_flag, freeze_toggle, step_forward
    
    def run(self):
        """Main game loop"""
        self.reset()
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle input
            quit_flag, reset_flag, freeze_toggle, step_forward = self.handle_keyboard()
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
                done = self.step()
                if done:
                    self.reset()
                    self.frozen = False
                    self.step_mode = False
            
            # Render
            self.screen.fill(BLACK)
            self.render_game(self.state)
            self.draw_panel()
            pygame.display.flip()
            
            # Cap FPS
            target_fps = 5 if self.frozen else self.target_fps
            elapsed_ms = clock.tick(target_fps)
            if elapsed_ms > 0:
                current_fps = 1000.0 / elapsed_ms
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 10:
                    self.fps_history.pop(0)
                self.actual_fps = sum(self.fps_history) / len(self.fps_history)
        
        pygame.quit()


def load_agent(model_path: str, config: Config) -> RelationalAgent:
    """Load a trained agent from a saved model file"""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    # Convert numpy arrays back to JAX arrays
    def to_jax(x):
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        return x
    
    meta_params = jax.tree_util.tree_map(to_jax, model_data["meta_params"])
    selector_params = jax.tree_util.tree_map(to_jax, model_data["selector_params"])
    sub_params = jax.tree_util.tree_map(to_jax, model_data["sub_params"])
    
    # Infer meta policy dimensions from loaded params
    def infer_meta_dims(params):
        """Infer global_dim and hidden_dim from meta policy params"""
        def get_all_kernels(d, path=""):
            """Recursively extract all kernel arrays with their paths"""
            kernels = []
            if isinstance(d, dict):
                for k, v in d.items():
                    kernels.extend(get_all_kernels(v, f"{path}.{k}" if path else k))
            elif hasattr(d, 'shape') and len(d.shape) == 2:
                kernels.append((path, d.shape))
            return kernels
        
        kernels = get_all_kernels(params)
        
        # Find the first Dense layer (Dense_0) which processes global features
        # It should have shape (global_dim, hidden_dim)
        for path, shape in kernels:
            if 'Dense_0' in path and 'kernel' in path:
                global_dim, hidden_dim = shape
                return int(global_dim), int(hidden_dim)
        
        return None, None
    
    inferred_global_dim, inferred_hidden_dim = infer_meta_dims(meta_params)
    if inferred_global_dim is not None and inferred_global_dim != config.environment.n_global_feats:
        print(f"Warning: Config has n_global_feats={config.environment.n_global_feats}, but model was trained with {inferred_global_dim}.")
        print(f"Updating config to use n_global_feats={inferred_global_dim}")
        config.environment.n_global_feats = inferred_global_dim
    
    # Store inferred hidden_dim for later use when creating the network
    meta_hidden_dim = inferred_hidden_dim if inferred_hidden_dim is not None else 128
    if inferred_hidden_dim is not None and inferred_hidden_dim != 128:
        print(f"Warning: Model was trained with hidden_dim={inferred_hidden_dim}, but MetaPolicyQNetwork defaults to hidden_dim=128.")
        print(f"Will use hidden_dim={inferred_hidden_dim} to match loaded model.")
    
    # Infer num_heads from attention layer parameters
    def infer_num_heads(params, hidden_dim):
        """Infer num_heads from MultiHeadDotProductAttention query kernel shape"""
        def find_attention_kernel(d, path=""):
            """Recursively find the attention query kernel"""
            if isinstance(d, dict):
                for k, v in d.items():
                    result = find_attention_kernel(v, f"{path}.{k}" if path else k)
                    if result is not None:
                        return result
            elif hasattr(d, 'shape') and len(d.shape) == 2:
                # Check if this looks like an attention query kernel
                # Shape should be [input_dim, num_heads * qkv_features]
                # In our code, qkv_features = hidden_dim, so shape is [hidden_dim, num_heads * hidden_dim]
                if 'query' in path.lower() and 'MultiHeadDotProductAttention' in path:
                    input_dim, output_dim = d.shape
                    # If input_dim matches hidden_dim and output_dim is divisible by hidden_dim
                    if input_dim == hidden_dim and output_dim % hidden_dim == 0:
                        num_heads = output_dim // hidden_dim
                        return num_heads
            return None
        
        num_heads = find_attention_kernel(params)
        return num_heads if num_heads is not None else 2  # Default to 2 if not found
    
    inferred_num_heads = infer_num_heads(meta_params, meta_hidden_dim)
    if inferred_num_heads != 2:
        print(f"Info: Inferred num_heads={inferred_num_heads} from saved model parameters.")
    
    # Infer obj_dim from meta policy params (Dense_1 processes objects)
    def infer_obj_dim(params):
        """Infer obj_dim from Dense_1 layer which processes objects"""
        def find_dense1_kernel(d, path=""):
            """Recursively find Dense_1 kernel"""
            if isinstance(d, dict):
                for k, v in d.items():
                    result = find_dense1_kernel(v, f"{path}.{k}" if path else k)
                    if result is not None:
                        return result
            elif hasattr(d, 'shape') and len(d.shape) == 2:
                # Dense_1 processes objects: shape is [obj_dim, hidden_dim]
                if 'Dense_1' in path and 'kernel' in path:
                    obj_dim, hidden_dim = d.shape
                    return int(obj_dim)
            return None
        
        return find_dense1_kernel(params)
    
    inferred_obj_dim = infer_obj_dim(meta_params)
    if inferred_obj_dim is None:
        # Fallback: compute from config
        num_concepts = config.concepts.label_high_level_concept + 1
        n_obj_feats_with_labels = config.environment.n_obj_feats + num_concepts
        inferred_obj_dim = config.environment.frame_stack_size * n_obj_feats_with_labels
        print(f"Warning: Could not infer obj_dim from model, using config value: {inferred_obj_dim}")
    else:
        # Compute what config would give us
        num_concepts = config.concepts.label_high_level_concept + 1
        n_obj_feats_with_labels = config.environment.n_obj_feats + num_concepts
        config_obj_dim = config.environment.frame_stack_size * n_obj_feats_with_labels
        if inferred_obj_dim != config_obj_dim:
            print(f"Warning: Config would compute obj_dim={config_obj_dim}, but model was trained with obj_dim={inferred_obj_dim}.")
            print(f"Using inferred obj_dim={inferred_obj_dim} to match loaded model.")
    
    # Infer number of actions from loaded sub_params if needed
    # The last layer of sub_policy has shape [hidden_dim, num_actions]
    def find_num_actions(params):
        """Find num_actions from the shape of the output layer kernel"""
        # Flax params structure: nested dict with 'params' key containing layer dicts
        # Each Dense layer has 'kernel' with shape [input_dim, output_dim]
        # We need to find all kernels and identify the output layer (smallest output dim)
        
        def get_all_kernels(d):
            """Recursively extract all kernel arrays from params dict"""
            kernels = []
            if isinstance(d, dict):
                for v in d.values():
                    kernels.extend(get_all_kernels(v))
            elif hasattr(d, 'shape') and len(d.shape) == 2:
                # This looks like a kernel (2D array)
                kernels.append(d.shape[1])  # Store output dimension
            return kernels
        
        kernels = get_all_kernels(params)
        
        if kernels:
            # The output layer should have the smallest output dimension
            # (hidden layers are typically 64, output is num_actions which is usually smaller)
            return min(kernels)
        return None
    
    inferred_actions = find_num_actions(sub_params)
    if inferred_actions is not None and inferred_actions != config.environment.n_actions:
        print(f"Warning: Config has n_actions={config.environment.n_actions}, but model was trained with {inferred_actions} actions.")
        print(f"Updating config to use n_actions={inferred_actions}")
        config.environment.n_actions = inferred_actions
    
    # Create a dummy observation to initialize the agent
    num_concepts = config.concepts.label_high_level_concept + 1
    # Use inferred obj_dim instead of computing from config
    obj_dim_flat = inferred_obj_dim
    dummy_obs = ConceptObs(
        global_features=jnp.zeros((1, config.environment.n_global_feats)),
        capabilities=jnp.zeros((1, config.concepts.n_capabilities)),
        objects=jnp.zeros((1, config.environment.max_objs, obj_dim_flat)),
        object_mask=jnp.zeros((1, config.environment.max_objs)),
        object_labels=jnp.zeros((1, config.environment.max_objs, num_concepts), dtype=jnp.float32),  # Multi-hot encoding
        object_events=jnp.zeros((1, config.environment.max_objs), dtype=jnp.int32),
    )
    
    # Create agent with correct architecture matching loaded params
    key = jax.random.PRNGKey(42)
    key_m, key_s, key_sub = jax.random.split(key, 3)
    
    # 1. Meta Policy - create with inferred hidden_dim
    from src.agent.components.meta_policy import MetaPolicyQNetwork
    num_concepts = config.concepts.label_high_level_concept + 1
    # Use inferred obj_dim to match the loaded model
    obj_dim = inferred_obj_dim
    
    # Create network with the inferred hidden_dim and num_heads
    meta_net = MetaPolicyQNetwork(
        num_concepts=num_concepts,
        hidden_dim=meta_hidden_dim,
        num_heads=inferred_num_heads
    )
    # Note: We don't need to call init() since we're using loaded params directly
    # The apply_fn works with any params that match the network structure
    
    # Create train state directly with loaded params
    from src.rl_utils import RLTrainState
    import optax
    meta_tx = optax.adam(config.training.meta_policy_lr)
    meta_state = RLTrainState.create(
        apply_fn=meta_net.apply,
        params=meta_params,
        tx=meta_tx
    )
    
    # 2. Instance Selector - create normally (should match)
    from src.agent.components.instance_selector import InstanceSelector
    selector_hidden_dim = config.model.selector_hidden_dims[0]
    selector_net = InstanceSelector(
        num_concepts=num_concepts,
        hidden_dim=selector_hidden_dim,
    )
    # Note: We don't need to call init() since we're using loaded params directly
    
    from flax.training.train_state import TrainState
    sel_tx = optax.adam(config.training.selector_lr)
    selector_state = TrainState.create(
        apply_fn=selector_net.apply,
        params=selector_params,
        tx=sel_tx
    )
    
    # 3. Sub Policy - create normally (should match after n_actions fix)
    from src.agent.components.sub_policy import SubPolicyQNetwork
    sub_net = SubPolicyQNetwork(
        num_actions=config.environment.n_actions,
        hidden_dims=config.model.hidden_dims
    )
    # Note: We don't need to call init() since we're using loaded params directly
    
    sub_tx = optax.adam(config.training.sub_policy_lr)
    sub_state = RLTrainState.create(
        apply_fn=sub_net.apply,
        params=sub_params,
        tx=sub_tx
    )
    
    # Create agent directly with constructed states
    agent = RelationalAgent(
        config=config,
        meta_state=meta_state,
        selector_state=selector_state,
        sub_state=sub_state
    )
    
    return agent


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Visualize trained agent playing Seaquest")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model file (.pkl)")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=3, 
                        help="Training phase to simulate: 1 (heuristic meta + heuristic selector), 2 (heuristic meta + network selector), or 3 (full model: network meta + network selector)")
    parser.add_argument("--force-concept", type=int, choices=[0, 1, 2, 3], default=None,
                        help="Force a specific concept (overrides phase): 0=COLLECT, 1=AVOID, 2=DESTROY, 3=DEFAULT")
    args = parser.parse_args()
    
    # Get config
    if args.config:
        config = load_config_from_file(args.config)
    else:
        # Default to configs/default.yaml if it exists, otherwise use default config
        default_config_path = project_root / "configs" / "default.yaml"
        if default_config_path.exists():
            print(f"Loading config from {default_config_path}")
            config = load_config_from_file(str(default_config_path))
        else:
            print("Using default config (configs/default.yaml not found)")
            config = get_default_config()
    
    # Load agent
    print(f"Loading agent from {args.model}...")
    agent = load_agent(args.model, config)
    print("Agent loaded successfully!")
    
    # Create environment
    base_env = make_atari_env('seaquest')
    atari_wrapper = AtariWrapper(
        base_env,
        frame_stack_size=config.environment.frame_stack_size,
        frame_skip=1,
        sticky_actions=config.environment.sticky_actions,
        episodic_life=config.environment.episodic_life,
        first_fire=config.environment.first_fire,
        noop_reset=config.environment.noop_reset,
        clip_reward=config.environment.clip_reward,
        max_pooling=config.environment.max_pooling,
    )
    env = wrapper.SeaquestConceptWrapper(atari_wrapper, config=config)
    
    # Create visualization
    visualization = AgentVisualization(env, agent, screen_width=1400, screen_height=700, config=config, phase=args.phase, force_concept_id=args.force_concept)
    
    # Print mode information
    if args.force_concept is not None:
        concept_names = {0: "COLLECT", 1: "AVOID", 2: "DESTROY", 3: "DEFAULT"}
        concept_name = concept_names.get(args.force_concept, f"Concept {args.force_concept}")
        print(f"Running visualization with FORCED CONCEPT: {concept_name}")
    else:
        phase_names = {
            1: "Phase 1 (Heuristic Meta + Heuristic Selector)",
            2: "Phase 2 (Heuristic Meta + Network Selector)",
            3: "Phase 3 (Full Model: Network Meta + Network Selector)"
        }
        phase_name = phase_names.get(args.phase, "Phase 3 (Full Model)")
        print(f"Running visualization in {phase_name}")
    
    # Run
    try:
        visualization.run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
