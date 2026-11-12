from functools import partial
import jax
from jaxtari.modification import JaxtariInternalModPlugin, JaxtariPostStepModPlugin
import jax.numpy as jnp
from jaxtari.games.jax_phoenix import PhoenixState


def _reshuffled_formation_x() -> jnp.ndarray:
    """Scatter live slots across the playfield X (unused stay -1)."""
    # Deterministic chaos — same live/unused mask as stock formations.
    return jnp.array(
        [
            [18, 142, 48, 112, 72, 96, 32, 128],
            [24, 136, 56, 104, 80, 40, 120, 64],
            [12, 148, 88, 36, 124, 68, 100, -1],
            [28, 132, 60, 116, 44, 92, 76, -1],
            [80, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=jnp.float32,
    )


def _reshuffled_formation_y() -> jnp.ndarray:
    """Scatter live slots across playfield Y (230 = inactive)."""
    return jnp.array(
        [
            [40, 130, 70, 110, 55, 145, 90, 120],
            [48, 138, 78, 118, 62, 150, 98, 35],
            [42, 125, 85, 155, 58, 105, 140, 230],
            [50, 135, 75, 115, 95, 160, 38, 230],
            [100, 230, 230, 230, 230, 230, 230, 230],
        ],
        dtype=jnp.float32,
    )


class BossLateMissilesMod(JaxtariInternalModPlugin):
    """
    Make boss missiles appear a few pixels after spawn so they are visible
    later (e.g., after leaving dense boss-block area).
    """

    constants_overrides = {
        "BOSS_PROJECTILE_RENDER_DELAY_PX": 8,
    }


class InfiniteLivesMod(JaxtariInternalModPlugin):
    """
    Set player lives to 99.
    """
    constants_overrides = {
        "PLAYER_LIVES": 99,
    }


class FastPlayerMod(JaxtariInternalModPlugin):
    """
    Increases player movement speed.
    """
    constants_overrides = {
        "PLAYER_STEP_SIZE": 2,
    }


class InvinciblePlayerMod(JaxtariPostStepModPlugin):
    """
    Player is always invincible.
    """
    def run(self, prev_state: PhoenixState, new_state: PhoenixState) -> PhoenixState:
        return new_state.replace(invincibility=jnp.array(True))


class FastEnemyBulletsMod(JaxtariInternalModPlugin):
    """
    Increases speed of enemy projectiles.
    """
    constants_overrides = {
        "ENEMY_PROJECTILE_SPEED": 4,
    }


class NoAbilityCooldownMod(JaxtariInternalModPlugin):
    """
    Removes cooldown for the special ability (shield).
    """
    constants_overrides = {
        "ABILITY_COOLDOWN": 0,
    }

class NightMod(JaxtariInternalModPlugin):
    """Dims the entire screen by 50% for a night mode experience."""
    name = "night_mode"
    constants_overrides = {
        'SCORE_COLOR': (105, 105, 32),
        'PLAYER_COLOR': (106, 65, 37),
        'BOSS_BLUE_COLOR': (42, 46, 107),
        'BOSS_RED_COLOR': (100, 36, 36),
        'BOSS_GREEN_COLOR': (42, 80, 30),
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_FLOOR': (73, 35, 96),
        'RGB_PHOENIX_MAIN': (62, 24, 86),
        'RGB_BATS_BLUE': (66, 72, 126),
        'RGB_BATS_RED': (83, 13, 13),
    }

class GrayscaleMod(JaxtariInternalModPlugin):
    """Turns the entire game into grayscale."""
    name = "grayscale"
    constants_overrides = {
        'SCORE_COLOR': (170, 170, 170),
        'PLAYER_COLOR': (120, 120, 120),
        'BOSS_BLUE_COLOR': (90, 90, 90),
        'BOSS_RED_COLOR': (100, 100, 100),
        'BOSS_GREEN_COLOR': (110, 110, 110),
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_FLOOR': (100, 100, 100),
        'RGB_PHOENIX_MAIN': (80, 80, 80),
        'RGB_BATS_BLUE': (120, 120, 120),
        'RGB_BATS_RED': (70, 70, 70),
    }

class InvertedColorsMod(JaxtariInternalModPlugin):
    """Inverts all colors in the game."""
    name = "inverted_colors"
    constants_overrides = {
        'SCORE_COLOR': (45, 45, 191),
        'PLAYER_COLOR': (42, 125, 181),
        'BOSS_BLUE_COLOR': (171, 163, 41),
        'BOSS_RED_COLOR': (55, 183, 183),
        'BOSS_GREEN_COLOR': (171, 95, 195),
        'RGB_BACKGROUND': (255, 255, 255),
        'RGB_FLOOR': (109, 185, 63),
        'RGB_PHOENIX_MAIN': (130, 207, 82),
        'RGB_BATS_BLUE': (123, 111, 3),
        'RGB_BATS_RED': (88, 229, 229),
    }

class MatrixMod(JaxtariInternalModPlugin):
    """A Matrix-themed mod: black background, green elements."""
    name = "matrix_theme"
    constants_overrides = {
        'SCORE_COLOR': (0, 255, 0),
        'PLAYER_COLOR': (255, 255, 255),
        'BOSS_BLUE_COLOR': (0, 150, 0),
        'BOSS_RED_COLOR': (0, 200, 0),
        'BOSS_GREEN_COLOR': (50, 255, 50),
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_FLOOR': (0, 180, 0),
        'RGB_PHOENIX_MAIN': (0, 255, 100),
        'RGB_BATS_BLUE': (0, 255, 0),
        'RGB_BATS_RED': (50, 255, 50),
    }

class BloodMoonMod(JaxtariInternalModPlugin):
    """A dark red themed mod."""
    name = "blood_moon"
    constants_overrides = {
        'SCORE_COLOR': (255, 100, 100),
        'PLAYER_COLOR': (255, 255, 255),
        'BOSS_BLUE_COLOR': (150, 0, 0),
        'BOSS_RED_COLOR': (200, 50, 50),
        'BOSS_GREEN_COLOR': (180, 0, 0),
        'RGB_BACKGROUND': (40, 0, 0),
        'RGB_FLOOR': (200, 0, 0),
        'RGB_PHOENIX_MAIN': (255, 50, 50),
        'RGB_BATS_BLUE': (150, 0, 0),
        'RGB_BATS_RED': (255, 100, 100),
    }


class FormationReshuffleMod(JaxtariInternalModPlugin):
    """Bucket B: scatter formation slots all over the playfield (X and Y)."""

    name = "formation_reshuffle"
    constants_overrides = {
        "ENEMY_POSITIONS_X": _reshuffled_formation_x(),
        "ENEMY_POSITIONS_Y": _reshuffled_formation_y(),
    }


class PlayerDriftMod(JaxtariPostStepModPlugin):
    """D_p: slow rightward player drift every 4 frames."""

    name = "player_drift"

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: PhoenixState, new_state: PhoenixState) -> PhoenixState:
        del prev_state
        drift = jnp.where(new_state.step_counter % 4 == 0, 1, 0)
        # Match FastPlayerMod right bound usage in game (~width-ish)
        new_x = jnp.minimum(new_state.player_x + drift, jnp.int32(155))
        return new_state.replace(player_x=new_x.astype(jnp.int32))


class StaticScoreBaitMod(JaxtariPostStepModPlugin):
    """A: one static non-shooting enemy; teleports to a new XY after each hit."""

    name = "static_score_bait"
    BAIT_X = 77
    BAIT_Y = 90

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PhoenixState):
        ex = jnp.full_like(state.enemies_x, -1.0)
        ey = jnp.full_like(state.enemies_y, 230.0)
        ex = ex.at[0].set(jnp.float32(self.BAIT_X))
        ey = ey.at[0].set(jnp.float32(self.BAIT_Y))
        dying = jnp.zeros_like(state.phoenix_dying)
        timers = jnp.zeros_like(state.phoenix_death_timer)
        state = state.replace(
            enemies_x=ex,
            enemies_y=ey,
            phoenix_dying=dying,
            phoenix_death_timer=timers,
            enemy_projectile_x=jnp.full_like(state.enemy_projectile_x, -1),
            enemy_projectile_y=jnp.full_like(state.enemy_projectile_y, -1),
        )
        return self._env._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: PhoenixState, new_state: PhoenixState) -> PhoenixState:
        # Clear other enemies / projectiles; keep only bait slot 0.
        ex = jnp.full_like(new_state.enemies_x, -1.0)
        ey = jnp.full_like(new_state.enemies_y, 230.0)
        dying = jnp.zeros_like(new_state.phoenix_dying)
        timers = jnp.zeros_like(new_state.phoenix_death_timer)

        prev_alive = (
            (prev_state.enemies_x[0] > -1)
            & (prev_state.enemies_y[0] < 200)
            & (~prev_state.phoenix_dying[0])
        )
        now_dying = new_state.phoenix_dying[0] | (
            (new_state.enemies_x[0] <= -1) | (new_state.enemies_y[0] >= 200)
        )
        just_hit = prev_alive & now_dying

        # Sample a new bait location from step_counter (no RNG field on PhoenixState).
        rng = jax.random.PRNGKey(new_state.step_counter.astype(jnp.uint32))
        rng, kx, ky = jax.random.split(rng, 3)
        rand_x = jax.random.randint(kx, (), 20, 140).astype(jnp.float32)
        rand_y = jax.random.randint(ky, (), 40, 150).astype(jnp.float32)

        cur_x = jnp.where(
            just_hit,
            rand_x,
            jnp.where(prev_alive, prev_state.enemies_x[0], jnp.float32(self.BAIT_X)),
        )
        cur_y = jnp.where(
            just_hit,
            rand_y,
            jnp.where(prev_alive, prev_state.enemies_y[0], jnp.float32(self.BAIT_Y)),
        )
        # If somehow missing, ensure a bait exists.
        missing = (new_state.enemies_x[0] <= -1) & (~prev_alive)
        cur_x = jnp.where(missing, rand_x, cur_x)
        cur_y = jnp.where(missing, rand_y, cur_y)

        ex = ex.at[0].set(cur_x)
        ey = ey.at[0].set(cur_y)
        return new_state.replace(
            enemies_x=ex,
            enemies_y=ey,
            phoenix_dying=dying,
            phoenix_death_timer=timers,
            enemy_projectile_x=jnp.full_like(new_state.enemy_projectile_x, -1),
            enemy_projectile_y=jnp.full_like(new_state.enemy_projectile_y, -1),
        )
