# Note: this is a very, very rudimentary implementation
# its only usage is to see if the 'collect' skill is already transferable
import functools
import jax
import jax.numpy as jnp
from jaxatari.wrappers import AtariWrapper, JaxatariWrapper
import jaxatari.spaces as spaces
from jaxatari.games import jax_seaquest as jaxatari_seaquest
from typing import Optional
from flax import struct
import chex
from config import Config, get_default_config


class PongConceptWrapper(JaxatariWrapper):

    def __init__(self, env, config=None):
        super().__init__(env)
        
        if config is None:
            config = get_default_config()
        self.config = config

        self.env = env
        self.stack_size = 4
        self.SCREEN_WIDTH = config.environment.screen_width
        self.SCREEN_HEIGHT = config.environment.screen_height

        self.max_objs = 3 # paddle1, paddle2, ball
        self.num_concepts = 1 # you can only collect

        self.obs_shape = (2 + (3 * self.stack_size),)
        
        # MAPPING TABLE
        # Seaquest (18) -> Pong (6)
        # Pong Actions: [NOOP(0), FIRE(1), RIGHT(2/Down), LEFT(3/Up), R_FIRE(4), L_FIRE(5)]
        # Map: Horizontal->NOOP, Vertical->Up/Down, Diagonals->Vertical component
        self.action_map = jnp.array([
            0, # 0 NOOP -> NOOP
            1, # 1 FIRE -> FIRE
            2, # 2 UP -> RIGHT (Paddle Up)
            0, # 3 RIGHT -> NOOP
            0, # 4 LEFT -> NOOP
            3, # 5 DOWN -> LEFT (Paddle Down)
            2, # 6 UPRIGHT -> RIGHT
            2, # 7 UPLEFT -> RIGHT
            3, # 8 DOWNRIGHT -> LEFT
            3, # 9 DOWNLEFT -> LEFT
            4, # 10 UPFIRE -> RIGHTFIRE
            1, # 11 RIGHTFIRE -> FIRE
            1, # 12 LEFTFIRE -> FIRE
            5, # 13 DOWNFIRE -> LEFTFIRE
            4, # 14 UPRIGHTFIRE -> RIGHTFIRE
            4, # 15 UPLEFTFIRE -> RIGHTFIRE
            5, # 16 DOWNRIGHTFIRE -> LEFTFIRE
            5, # 17 DOWNLEFTFIRE -> LEFTFIRE
        ], dtype=jnp.int32)

    def reset(self, rng):
        # Reset the base environment
        concept_obs, env_state = self.env.reset(rng)
        return self._process_obs(concept_obs), env_state

    def step(self, env_state, action):
        pong_action = self.action_map[action]
        # obs is the full stack (StackSize, ...)
        obs, next_state, _, done, info = self.env.step(env_state, pong_action)
        
        # --- Do NOT slice obs here. We need the history. ---
        # We only look at the LATEST frame [-1] for Reward calculation
        current_frame = jax.tree.map(lambda x: x[-1], obs)

        paddle_x = current_frame.player.x
        paddle_y = current_frame.player.y
        ball_x = current_frame.ball.x
        ball_y = current_frame.ball.y
        ball_radius = current_frame.ball.width / 2

        # Basic collision check (JAX-compatible)
        x_collision = (paddle_x < ball_x) & (ball_x < paddle_x + current_frame.player.width)
        y_collision = (paddle_y < ball_y + ball_radius) & (ball_y + ball_radius < paddle_y + current_frame.player.height)
        collision = x_collision & y_collision
        reward = jnp.where(collision, 1.0, 0.0)

        # 3. Process the FULL STACK
        final_obs = self._process_obs(obs)
        
        return final_obs, next_state, reward, done, info
    
    def _process_obs(self, obs_stack):
        """
        obs_stack: Struct where fields are arrays of shape (StackSize, ...)
        """
        # 1. Get Player Position History
        p_x_seq = obs_stack.player.x / self.SCREEN_WIDTH
        p_y_seq = obs_stack.player.y / self.SCREEN_HEIGHT

        # 2. Get Ball Position History
        obj_x_seq = obs_stack.ball.x / self.SCREEN_WIDTH
        obj_y_seq = obs_stack.ball.y / self.SCREEN_HEIGHT

        # --- FIX: Extract Width and Height to match SeaquestAdapter ---
        # Pong ball is constant size usually, but we can extract if available 
        # or use dummy values if constant (approx 2px or 4px in standard Pong)
        obj_w_seq = obs_stack.ball.width / self.SCREEN_WIDTH
        obj_h_seq = obs_stack.ball.height / self.SCREEN_HEIGHT

        # 3. Calculate relative sequence
        rel_x_seq = obj_x_seq - p_x_seq
        rel_y_seq = obj_y_seq - p_y_seq

        # Cheat a bit, mask x so the agent only has to think about up and down
        rel_x_seq = jnp.full_like(rel_x_seq, 0)
        p_x_seq = jnp.full_like(p_x_seq, 0)

        # 4. Valid Mask (Always 1.0 for Pong ball)
        valid_seq = jnp.ones((self.stack_size,), dtype=jnp.float32)

        # 6. Flatten History
        # FIX: Stack 5 features: [RelX, RelY, W, H, Val] to match SeaquestAdapter
        stack_feats = jnp.stack([rel_x_seq, rel_y_seq, obj_w_seq, obj_h_seq, valid_seq], axis=-1).flatten()

        player_history = jnp.stack([p_x_seq, p_y_seq], axis=-1).flatten()

        # 7. Concatenate: [Px, Py] + [Stack...]
        return jnp.concatenate([player_history, stack_feats])

    # Don't forget to update the shape property!
    def observation_space(self):
        # 2*Stack + 5*Stack = 7 * 4 = 28
        class Space: shape = (28,) 
        return Space()

    def action_space(self):
        class Space: n = 18 
        return Space()

