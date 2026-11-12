
"""Paper-eval robustness mods for Name This Game."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxtari.modification import JaxtariInternalModPlugin, JaxtariPostStepModPlugin
from jaxtari.games.jax_namethisgame import NameThisGameState
from jaxtari.environment import ObjectObservation


def _rgba_swatch(rgb: tuple[int, int, int]) -> np.ndarray:
    """1×1 procedural RGBA pixel so ``rgb`` is present in the renderer palette."""
    return np.array([[[rgb[0], rgb[1], rgb[2], 255]]], dtype=np.uint8)


class ChangeBackgroundColorMod(JaxtariInternalModPlugin):
    """Magenta sky/water/seabed. Sprites and the orange/green HUD stay.

    Name This Game bakes those three bands into ``renderer.BACKGROUND`` at
    init and ignores a later ``background`` asset override. The swatch only
    seeds the palette; ``NameThisGameEnvMod`` remaps the baked ids afterwards.
    """

    name = "change_background_color"
    _NEW_BG = (180, 0, 180)
    # Bands from ``_create_static_procedural_sprites``.
    _BANDS = (
        (236, 212, 108),  # sky
        (24, 26, 167),  # water
        (100, 184, 220),  # seabed
    )

    asset_overrides = {
        "change_bg_swatch": {
            "name": "change_bg_swatch",
            "type": "procedural",
            "data": _rgba_swatch(_NEW_BG),
        }
    }


class ShiftTentacleSpawnXMod(JaxtariInternalModPlugin):
    """S: shift tentacle column spawn X by +16px."""

    name = "shift_tentacle_spawn_x"
    constants_overrides = {
        "tentacle_base_x": jnp.array([32, 48, 64, 80, 96, 112, 128, 144], dtype=jnp.int32),
    }


class SlowerPlayerMod(JaxtariPostStepModPlugin):
    """D_p: undo diver X motion on odd HUD frames (~half speed)."""

    name = "slower_player"

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: NameThisGameState, new_state: NameThisGameState) -> NameThisGameState:
        odd = (new_state.bar_frame_counter % 2) == 1
        x = jnp.where(odd, prev_state.diver_x, new_state.diver_x)
        return new_state.replace(diver_x=x)


class FasterSharkMod(JaxtariPostStepModPlugin):
    """D_t: extra shark horizontal step each frame."""

    name = "faster_shark"

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: NameThisGameState, new_state: NameThisGameState) -> NameThisGameState:
        del prev_state
        alive = new_state.shark_alive
        dx = jnp.sign(new_state.shark_dx).astype(jnp.int32)
        dx = jnp.where(dx == 0, jnp.int32(1), dx)
        new_x = new_state.shark_x + jnp.where(alive, dx, 0)
        return new_state.replace(shark_x=new_x)


class _NoSharkParkMod(JaxtariPostStepModPlugin):
    """Park shark offscreen; static length-1 tentacle stumps (no grow/slide).

    Stumps can be killed (len→0) and respawn after RESPAWN_FRAMES.
    ``shark_alive`` stays True — False is the NTG death signal.
    """

    name = "_no_shark_park"
    RESPAWN_FRAMES = 90

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: NameThisGameState):
        T = state.tentacle_len.shape[0]
        L = state.tentacle_cols.shape[1]
        # Length-1 stump: tip in middle column, no body.
        cols = jnp.zeros((T, L), dtype=jnp.int32).at[:, 0].set(1)
        state = state.replace(
            shark_alive=jnp.array(True, jnp.bool_),
            shark_x=jnp.int32(-80),
            tentacle_len=jnp.ones((T,), jnp.int32),
            tentacle_cols=cols,
            tentacle_dir=jnp.zeros((T,), jnp.int32),
            tentacle_edge_wait=jnp.zeros((T,), jnp.int32),
            tentacle_active=jnp.ones((T,), jnp.bool_),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: NameThisGameState, new_state: NameThisGameState) -> NameThisGameState:
        # Cap length at 1 (no growth); allow 0 after a spear hit.
        lens = jnp.minimum(new_state.tentacle_len, jnp.int32(1))

        # Freeze lateral sliding: keep previous columns while stump is alive.
        still = (prev_state.tentacle_len > 0) & (lens > 0)
        cols = jnp.where(still[:, None], prev_state.tentacle_cols, new_state.tentacle_cols)

        # Respawn timer while dead (reuse edge_wait).
        wait = jnp.where(lens == 0, prev_state.tentacle_edge_wait + 1, jnp.int32(0))
        respawn = (lens == 0) & (wait >= jnp.int32(self.RESPAWN_FRAMES))
        lens = jnp.where(respawn, jnp.int32(1), lens)
        wait = jnp.where(respawn, jnp.int32(0), wait)
        # Fresh stump tip on respawn.
        tip = jnp.zeros_like(cols).at[:, 0].set(1)
        cols = jnp.where(respawn[:, None], tip, cols)

        # Same REST exit as _NoSharkOnlyMod, so an agent that stops firing cannot stall the game.
        rest = new_state.resting
        return new_state.replace(
            resting=jnp.array(False, jnp.bool_),
            wave_bar_px=jnp.where(
                rest, jnp.array(self._env.consts.hud_bar_initial_px, jnp.int32), new_state.wave_bar_px
            ),
            bar_frame_counter=jnp.where(rest, jnp.array(0, jnp.int32), new_state.bar_frame_counter),
            shark_alive=jnp.array(True, jnp.bool_),
            shark_x=jnp.int32(-80),
            tentacle_len=lens,
            tentacle_cols=cols,
            tentacle_dir=jnp.zeros_like(new_state.tentacle_dir),
            tentacle_edge_wait=wait,
            tentacle_active=(lens > 0),
        )


class _NoSharkOnlyMod(JaxtariPostStepModPlugin):
    """Park the shark offscreen. Tentacles keep the normal grow and slide.

    ``shark_alive`` stays True — False is the Name This Game death signal.
    """

    name = "_no_shark_only"

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: NameThisGameState):
        state = state.replace(
            shark_alive=jnp.array(True, jnp.bool_),
            shark_x=jnp.int32(-80),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: NameThisGameState, new_state: NameThisGameState) -> NameThisGameState:
        del prev_state
        # REST ends on a fresh fire press, but without a shark no skill is selected to press
        # it. Ending REST here keeps the game going instead of idling to the time limit.
        rest = new_state.resting
        return new_state.replace(
            shark_alive=jnp.array(True, jnp.bool_),
            shark_x=jnp.int32(-80),
            resting=jnp.array(False, jnp.bool_),
            wave_bar_px=jnp.where(
                rest, jnp.array(self._env.consts.hud_bar_initial_px, jnp.int32), new_state.wave_bar_px
            ),
            bar_frame_counter=jnp.where(rest, jnp.array(0, jnp.int32), new_state.bar_frame_counter),
        )


class _NoSharkObsMod(JaxtariInternalModPlugin):
    """Force shark inactive in OC obs so heuristics never select it."""

    name = "_no_shark_obs"

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: NameThisGameState):
        from jaxtari.games.jax_namethisgame import JaxNameThisGame
        obs = JaxNameThisGame._get_observation(self._env, state)
        shark = ObjectObservation.create(
            x=obs.shark.x,
            y=obs.shark.y,
            width=obs.shark.width,
            height=obs.shark.height,
            active=jnp.array(0, jnp.int32),
            orientation=obs.shark.orientation,
        )
        return obs.replace(shark=shark)
