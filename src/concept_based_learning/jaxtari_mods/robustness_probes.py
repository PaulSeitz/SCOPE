"""Focused robustness probes: modest layout shifts and task simplifications.

Each registry key is a separate intervention. Simplifications keep the native
reward function; they do not establish that every frozen policy must improve.
Internal hooks act before collisions, rather than undoing deaths after a step.
"""
from functools import partial

import jax
import jax.numpy as jnp

from jaxtari.modification import JaxtariInternalModPlugin, JaxtariPostStepModPlugin


class NTGShiftColumnsMod(JaxtariInternalModPlugin):
    """Translate all eight tentacle bases right by 8px; retain all rules."""
    constants_overrides = {
        "tentacle_base_x": jnp.arange(24, 144, 16, dtype=jnp.int32),
    }


class NTGInfiniteOxygenMod(JaxtariInternalModPlugin):
    """Remove oxygen pressure; retain shark/tentacle tasks and their scores."""
    @partial(jax.jit, static_argnums=(0,))
    def _update_oxygen(self, state):
        full = jnp.asarray(self._env.consts.hud_bar_initial_px, jnp.int32)
        # Preserve the ordinary contact counter and any other oxygen effects.
        out = type(self._env)._update_oxygen(self._env, state.replace(oxy_bar_px=full))
        return out.replace(oxy_bar_px=full, oxygen_frames_remaining=full)


class TimePilotShiftAnchorMod(JaxtariInternalModPlugin):
    """Move the fixed player anchor 8px right, including shots and collisions."""
    constants_overrides = {"PLAYER_X": 84}


class TimePilotNoEnemyFireMod(JaxtariInternalModPlugin):
    """Remove enemy missiles, retaining aircraft, scoring and movement."""
    @partial(jax.jit, static_argnums=(0,))
    def enemy_missile_step(self, missile_states, enemy_states, enemy_attack, rng_key):
        out, key = type(self._env).enemy_missile_step(
            self._env, missile_states, enemy_states, enemy_attack, rng_key
        )
        # Consume the normal random draw, preserving the aircraft RNG stream.
        return out.at[:, 3].set(0), key


class SpaceInvadersShiftShieldRowMod(JaxtariInternalModPlugin):
    """Move the unchanged three bunkers up 6px."""
    constants_overrides = {
        "BARRICADE_POS": (jnp.array([41, 73, 105], dtype=jnp.int32), 151),
    }


class SpaceInvadersNoEnemyFireMod(JaxtariInternalModPlugin):
    """Remove incoming bullets without removing aliens, bunkers or the UFO."""
    @partial(jax.jit, static_argnums=(0,))
    def _update_enemy_bullets(self, state, key):
        active, x, y, cooldown = type(self._env)._update_enemy_bullets(self._env, state, key)
        return jnp.zeros_like(active), x, y, cooldown


class BeamriderShiftPlayerRowMod(JaxtariInternalModPlugin):
    """Move the player's fixed Y up 4px, retaining the seven-lane track."""
    constants_overrides = {"PLAYER_POS_Y": 160}


class BeamriderNoEnemyShotsMod(JaxtariInternalModPlugin):
    """Remove UFO shots only; preserve UFO patterns and all scoring objects."""
    @partial(jax.jit, static_argnums=(0,))
    def _enemy_shot_step(self, state, white_ufo_pos, white_ufo_pattern_id, white_ufo_pattern_timer):
        return (
            self._env.bullet_offscreen_shots,
            jnp.zeros_like(state.level.enemy_shot_vel),
            jnp.zeros_like(state.level.enemy_shot_timer),
            jnp.int32(0),
        )


class AsteroidsShiftInitialRocksMod(JaxtariPostStepModPlugin):
    """Translate initial active rocks 8px down; velocities and later spawns stay native."""
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        rocks = state.asteroid_states
        active = rocks[:, 3] != self._env.consts.INACTIVE
        rocks = rocks.at[:, 1].add(jnp.where(active, 8, 0))
        state = state.replace(asteroid_states=rocks)
        return self._env._get_observation(state), state


class AsteroidsNoShipCollisionsMod(JaxtariInternalModPlugin):
    """Keep moving, shootable and splitting rocks, but remove ship collision damage."""
    @partial(jax.jit, static_argnums=(0,))
    def player_hit(self, player_x, player_y, player_active, asteroid_states):
        return jnp.int32(-1)


class PongShiftPlayerPlaneMod(JaxtariInternalModPlugin):
    """Move the player's interception plane inward by 8px (140 -> 132)."""
    constants_overrides = {"PLAYER_X": 132}


class PongSlowerEnemyMod(JaxtariInternalModPlugin):
    """Halve opponent step size while retaining its native tracking policy."""
    constants_overrides = {"ENEMY_STEP_SIZE": 1}


class BreakoutShiftPaddleRowMod(JaxtariInternalModPlugin):
    """Move the paddle up 8px, preserving its width and ball motion rules."""
    constants_overrides = {"PLAYER_START_Y": 181}


class BreakoutWiderPaddleMod(JaxtariInternalModPlugin):
    """Add 8px to both native paddle widths; preserve shrink and ball rules."""
    constants_overrides = {"PLAYER_SIZE": (24, 4), "PLAYER_SIZE_SMALL": (20, 4)}


class FreewayShiftCarRowsMod(JaxtariInternalModPlugin):
    """Move each car row down 3px within its lane (offset 1 -> 4)."""
    constants_overrides = {"car_y_offset": 4}


class FreewayNoCarsMod(JaxtariInternalModPlugin):
    """Remove traffic from physics, rendering and OC activity; keep crossing score."""
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None):
        _, state = type(self._env).reset(self._env, key)
        state = state.replace(cars=state.cars.at[:, 1].set(-1000))
        return self._env._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # Exclude traffic BEFORE collision processing, including externally supplied states.
        state = state.replace(cars=state.cars.at[:, 1].set(-1000))
        return type(self._env).step(self._env, state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state):
        obs = type(self._env)._get_observation(self._env, state)
        return obs.replace(car=obs.car.replace(active=jnp.zeros_like(obs.car.active)))


class PhoenixShiftPlayerRowMod(JaxtariInternalModPlugin):
    """Move the fixed player row up 4px; retain X, width and movement speed."""
    constants_overrides = {"PLAYER_POSITION": (79, 169)}


class PhoenixNoEnemyFireMod(JaxtariInternalModPlugin):
    """Suppress new shots in bird, bat and boss waves; retain actor movement."""
    @partial(jax.jit, static_argnums=(0,))
    def _phoenix_small_bird_enemy_fire(self, state, pre_step_phoenix_do_attack, step_counter):
        out = type(self._env)._phoenix_small_bird_enemy_fire(
            self._env, state, pre_step_phoenix_do_attack, step_counter
        )
        return out._replace(
            salvo_fire_mask=jnp.zeros_like(out.salvo_fire_mask),
            salvo_x=jnp.full_like(out.salvo_x, -1),
            salvo_y=jnp.full_like(out.salvo_y, -1),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _bat_wave_enemy_fire(self, key, state, not_attacking):
        out = type(self._env)._bat_wave_enemy_fire(self._env, key, state, not_attacking)
        return out._replace(enemy_fire_mask=jnp.zeros_like(out.enemy_fire_mask))

    @partial(jax.jit, static_argnums=(0,))
    def _boss_enemy_fire(self, key, state, not_attacking):
        out = type(self._env)._boss_enemy_fire(self._env, key, state, not_attacking)
        return out._replace(enemy_fire_mask=jnp.zeros_like(out.enemy_fire_mask))
