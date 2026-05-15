import functools
from typing import Any, Dict, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
import pygame
import sys
from flax import struct
import chex

import jaxatari.spaces as spaces
from jaxatari.wrappers import JaxatariWrapper, AtariState, AtariWrapper
from jaxatari.environment import ObjectObservation
from jaxatari import make as make_atari_env

from src.concept_based_learning.environments.datatypes import (
    LABEL_NOOP,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LabeledObjectObservation,
)
from src.concept_based_learning.environments.seaquest.label import (
    SeaquestSemanticWrapper,
)


# ==============================================================================
# 1. UNIVERSAL DATATYPES & LABELS
# ==============================================================================

LABEL_COLORS = {
    LABEL_NOOP: (255, 255, 255),   # White
    LABEL_DESTROY: (255, 50, 50),  # Red
    LABEL_COLLECT: (50, 255, 50),  # Green
    LABEL_AVOID: (255, 200, 50)    # Orange
}

LABEL_NAMES = {
    LABEL_NOOP: "NOOP",
    LABEL_DESTROY: "DESTROY",
    LABEL_COLLECT: "COLLECT",
    LABEL_AVOID: "AVOID"
}

class SemanticVisualizer:
    def __init__(self, env):
        self.env = env
        self.base_env = env._env._env  # For raw RGB rendering
        
        # PyGame Init
        pygame.init()
        self.scale = 4
        self.game_w, self.game_h = 160 * self.scale, 210 * self.scale
        self.screen = pygame.display.set_mode((self.game_w, self.game_h))
        pygame.display.set_caption("System 2: Semantic View")
        self.font = pygame.font.Font(None, 24)
        
        self.rng_key = jax.random.PRNGKey(42)
        self.obs = None
        self.state = None

    def handle_input(self) -> Tuple[int, bool]:
        """Maps arrow keys & spacebar to Atari actions."""
        keys = pygame.key.get_pressed()
        action = 0 # NOOP
        quit_flag = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit_flag = True

        if keys[pygame.K_UP] and keys[pygame.K_RIGHT]: action = 14 if keys[pygame.K_SPACE] else 6
        elif keys[pygame.K_UP] and keys[pygame.K_LEFT]: action = 15 if keys[pygame.K_SPACE] else 7
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]: action = 16 if keys[pygame.K_SPACE] else 8
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]: action = 17 if keys[pygame.K_SPACE] else 9
        elif keys[pygame.K_UP]: action = 10 if keys[pygame.K_SPACE] else 2
        elif keys[pygame.K_DOWN]: action = 13 if keys[pygame.K_SPACE] else 5
        elif keys[pygame.K_RIGHT]: action = 11 if keys[pygame.K_SPACE] else 3
        elif keys[pygame.K_LEFT]: action = 12 if keys[pygame.K_SPACE] else 4
        elif keys[pygame.K_SPACE]: action = 1
        
        return action, quit_flag

    def draw_overlays(self, obs: LabeledObjectObservation):
        """Draws bounding boxes and labels based purely on LabeledObjectObservation."""
        # The AtariWrapper stacks frames. We want the latest frame [-1].
        # Convert JAX arrays to NumPy for easy iteration.
        x = np.array(obs.objects.x[-1])
        y = np.array(obs.objects.y[-1])
        w = np.array(obs.objects.width[-1])
        h = np.array(obs.objects.height[-1])
        active = np.array(obs.objects.active[-1])
        labels = np.array(obs.objects.semantic_label[-1])

        for i in range(len(active)):
            if active[i]:
                label_id = int(labels[i])
                color = LABEL_COLORS.get(label_id, (255, 255, 255))
                name = LABEL_NAMES.get(label_id, "UNK")

                # Scale coordinates for PyGame
                px, py = x[i] * self.scale, y[i] * self.scale
                pw, ph = w[i] * self.scale, h[i] * self.scale

                # Draw Bounding Box
                pygame.draw.rect(self.screen, color, (px, py, pw, ph), 2)
                
                # Draw Text Label
                text_surf = self.font.render(name, True, color)
                # Position text slightly above the box
                self.screen.blit(text_surf, (px, py - 18))

    def run(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.obs, self.state = self.env.reset(subkey)
        clock = pygame.time.Clock()

        running = True
        while running:
            action, quit_flag = self.handle_input()
            if quit_flag:
                break

            # Step Environment
            self.obs, self.state, reward, done, info = self.env.step(self.state, action)
            
            if done:
                self.rng_key, subkey = jax.random.split(self.rng_key)
                self.obs, self.state = self.env.reset(subkey)

            # 1. Render Raw Game Frame
            raw_frame = self.base_env.render(self.state.atari_state.env_state)
            frame_np = np.asarray(jax.device_get(raw_frame))
            
            # Convert (H, W, C) to (W, H, C) for PyGame
            frame_surface = pygame.surfarray.make_surface(np.transpose(frame_np, (1, 0, 2)))
            frame_surface = pygame.transform.scale(frame_surface, (self.game_w, self.game_h))
            self.screen.blit(frame_surface, (0, 0))

            # 2. Draw Semantic Overlays
            self.draw_overlays(self.obs)

            pygame.display.flip()
            clock.tick(30) # 30 FPS for comfortable manual play

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Create the base environment WITHOUT frame skipping for smooth manual play
    base_env = make_atari_env('seaquest')
    atari_wrapper = AtariWrapper(
        base_env,
        frame_stack_size=4,
        frame_skip=1,       # Important: 1 for manual play!
        sticky_actions=False,
        episodic_life=False
    )
    
    # Wrap it in our new Semantic Wrapper
    semantic_env = SeaquestSemanticWrapper(atari_wrapper)
    
    # Run Visualizer
    visualizer = SemanticVisualizer(semantic_env)
    visualizer.run()