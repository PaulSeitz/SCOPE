import jax
import jax.numpy as jnp
from functools import partial
from jaxtari.games.jax_spaceinvaders import SpaceInvadersState
from jaxtari.modification import JaxtariInternalModPlugin, JaxtariPostStepModPlugin

# --- Shield Modifications ---

class DisableShieldLeftMod(JaxtariPostStepModPlugin):
    """
    Erases the left bunker from the screen by clearing its specific memory data.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            barricade_health=new_state.barricade_health.at[0].set(0)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = state.replace(
            barricade_health=state.barricade_health.at[0].set(0)
        )
        return self._env._get_observation(state), state

class DisableShieldMiddleMod(JaxtariPostStepModPlugin):
    """
    Erases the middle bunker from the screen by clearing its specific memory data.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            barricade_health=new_state.barricade_health.at[1].set(0)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = state.replace(
            barricade_health=state.barricade_health.at[1].set(0)
        )
        return self._env._get_observation(state), state

class DisableShieldRightMod(JaxtariPostStepModPlugin):
    """
    Erases the right bunker from the screen by clearing its specific memory data.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            barricade_health=new_state.barricade_health.at[2].set(0)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = state.replace(
            barricade_health=state.barricade_health.at[2].set(0)
        )
        return self._env._get_observation(state), state

class ShiftShieldsMod(JaxtariInternalModPlugin):
    """
    Teleports all bunkers to new horizontal positions.
    """
    constants_overrides = {
        # Shifting all bunkers 10 pixels to the right from [41, 73, 105] to [51, 83, 115]
        "BARRICADE_POS": (jnp.array([51, 83, 115], dtype=jnp.int32), 157) # 210 - 53 = 157
    }

# --- Weapon & Gameplay Modifications ---

class ControllableMissileMod(JaxtariPostStepModPlugin):
    """
    Forces the fired missile's horizontal position to match the player's tank,
    allowing you to "steer" shots after firing.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            bullet_x=jnp.where(
                new_state.bullet_active,
                new_state.player_x - (self._env.consts.PLAYER_SIZE[0] // 2),
                new_state.bullet_x
            )
        )

class NoDangerMod(JaxtariPostStepModPlugin):
    """
    Removes all player shields and neutralizes incoming enemy projectiles.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Remove all shields
        new_barricade_health = new_state.barricade_health.at[:].set(0)
        # Neutralize enemy projectiles (deactivate them)
        new_enemy_bullets_active = jnp.zeros_like(new_state.enemy_bullets_active, dtype=jnp.bool_)
        
        return new_state.replace(
            barricade_health=new_barricade_health,
            enemy_bullets_active=new_enemy_bullets_active
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = state.replace(
            barricade_health=state.barricade_health.at[:].set(0),
            enemy_bullets_active=jnp.zeros_like(state.enemy_bullets_active, dtype=jnp.bool_)
        )
        return self._env._get_observation(state), state


class ShiftFormationMod(JaxtariInternalModPlugin):
    """S: shift alien formation origin X/Y."""

    name = "shift_formation"
    constants_overrides = {
        "OPPONENT_LIMIT_X": (32, 146),
        "OPPONENT_LIMIT_Y": (45, None),
    }


class SlowTankMod(JaxtariPostStepModPlugin):
    """D_p: slower player tank (undo X motion on odd frames)."""

    name = "slow_tank"

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        odd = (new_state.step_counter % 2) == 1
        px = jnp.where(odd, prev_state.player_x, new_state.player_x)
        ps = jnp.where(odd, jnp.zeros_like(new_state.player_speed), new_state.player_speed)
        return new_state.replace(player_x=px, player_speed=ps)


class FasterAliensMod(JaxtariInternalModPlugin):
    """D_t: aliens march on a faster cadence."""

    name = "faster_aliens"
    constants_overrides = {
        "MOVEMENT_RATES": jnp.array([8, 4, 2, 1, 1, 1], dtype=jnp.int32),
    }


class SaucerOnlyMod(JaxtariPostStepModPlugin):
    """A: hide alien grid; keep a looping mid-screen saucer for farm/alignment.

    Important: do NOT set every ``destroyed`` slot to 29 — that triggers
    ``wave_cleared`` → ``_next_wave`` every frame (player/ufo freeze loop).
    Leave one dummy alien "alive" but park the formation off-screen.

    Base SI awards UFO score on hit but rebuilds ``ufo_state`` from the
    pre-collision value, so the saucer never explodes. Force death here.
    """

    name = "saucer_only"
    # Stock UFO_Y is 11 (top strip) — too hard for an alignment farm.
    constants_overrides = {
        "UFO_Y": 120,  # lower mid-screen, easy to hit
    }
    RESPAWN_DELAY = 45

    def _hide_grid(self, state):
        # Slot 0 stays destroyed==0 so wave never clears; rest fully gone.
        destroyed = jnp.full_like(state.destroyed, 29)
        destroyed = destroyed.at[0].set(0)
        return state.replace(
            destroyed=destroyed,
            opponent_current_y=jnp.int32(250),  # off-screen
            enemy_bullets_active=jnp.zeros_like(state.enemy_bullets_active),
        )

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = self._hide_grid(state)
        state = state.replace(
            ufo_state=jnp.int32(1),
            ufo_x=jnp.int32(20),
            ufo_dir=jnp.int32(1),
            enemy_fire_cooldown=jnp.int32(0),
        )
        return self._env._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        new_state = self._hide_grid(new_state)
        ufo_score = jnp.int32(self._env.consts.UFO_SCORE)
        # Hit scored but ufo_state stayed 1 (base SI bug).
        hit = (
            (prev_state.ufo_state == 1)
            & (new_state.ufo_state == 1)
            & (new_state.player_score >= prev_state.player_score + ufo_score)
        )
        ufo_state = jnp.where(hit, jnp.int32(2), new_state.ufo_state)

        # After explosion finishes (state→0), wait then respawn.
        # Timer is ours (stored in enemy_fire_cooldown) — do not add onto the
        # game's cooldown value or we skip the delay and look immortal.
        dead = ufo_state == 0
        just_died = dead & (prev_state.ufo_state != 0)
        cool = jnp.where(
            just_died,
            jnp.int32(1),
            jnp.where(dead, prev_state.enemy_fire_cooldown + jnp.int32(1), jnp.int32(0)),
        )
        need = dead & (cool >= jnp.int32(self.RESPAWN_DELAY))
        ufo_state = jnp.where(need, jnp.int32(1), ufo_state)
        ufo_x = jnp.where(need, jnp.int32(20), new_state.ufo_x)
        ufo_dir = jnp.where(need, jnp.int32(1), new_state.ufo_dir)
        cool = jnp.where(need, jnp.int32(0), cool)
        return new_state.replace(
            ufo_state=ufo_state,
            ufo_x=ufo_x,
            ufo_dir=ufo_dir,
            enemy_fire_cooldown=cool,
        )
