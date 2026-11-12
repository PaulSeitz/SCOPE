
import jax.numpy as jnp

from jaxtari.modification import JaxtariModController
from concept_based_learning.jaxtari_mods.namethisgame.namethisgame_mod_plugins import (
    ChangeBackgroundColorMod,
    ShiftTentacleSpawnXMod,
    SlowerPlayerMod,
    FasterSharkMod,
    _NoSharkOnlyMod,
    _NoSharkParkMod,
    _NoSharkObsMod,
)


from concept_based_learning.jaxtari_mods.robustness_probes import (
    NTGShiftColumnsMod,
    NTGInfiniteOxygenMod,
)

class NameThisGameEnvMod(JaxtariModController):
    """Game-specific mod controller for Name This Game."""

    REGISTRY = {
        "shift_tentacle_columns": NTGShiftColumnsMod,
        "infinite_oxygen": NTGInfiniteOxygenMod,

        "shift_tentacle_spawn_x": ShiftTentacleSpawnXMod,
        "slower_player": SlowerPlayerMod,
        "faster_shark": FasterSharkMod,
        "change_background_color": ChangeBackgroundColorMod,
        "_no_shark_only": _NoSharkOnlyMod,
        "_no_shark_park": _NoSharkParkMod,
        "_no_shark_obs": _NoSharkObsMod,
        # shark_alive=False is death in NTG — park offscreen + obs inactive instead.
        "no_shark": ["_no_shark_only", "_no_shark_obs"],
        "no_shark_short_tentacles": ["_no_shark_park", "_no_shark_obs"],
    }

    def __init__(self, env, mods_config: list = [], allow_conflicts: bool = False):
        self._has_change_background = "change_background_color" in list(mods_config or [])
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
        if self._has_change_background:
            self._apply_change_background_color()

    def _reapply_renderer_asset_mods(self) -> None:
        """Re-run after a renderer hot-swap. A fresh bake restores the bands."""
        if self._has_change_background:
            self._apply_change_background_color()

    def _apply_change_background_color(self) -> None:
        """Recolor baked sky/water/seabed ids. Leave every other id alone."""
        import numpy as np

        renderer = self._env.renderer
        new_rgb = tuple(ChangeBackgroundColorMod._NEW_BG)
        if new_rgb not in renderer.COLOR_TO_ID:
            raise KeyError(
                f"change_background_color: palette missing swatch {new_rgb}; "
                "ChangeBackgroundColorMod.asset_overrides must seed it."
            )
        new_id = int(renderer.COLOR_TO_ID[new_rgb])
        band_ids = []
        for rgb in ChangeBackgroundColorMod._BANDS:
            if rgb not in renderer.COLOR_TO_ID:
                raise KeyError(
                    f"change_background_color: palette missing background band {rgb}."
                )
            band_ids.append(int(renderer.COLOR_TO_ID[rgb]))
        background = np.asarray(renderer.BACKGROUND)
        recolored = background.copy()
        for band_id in band_ids:
            recolored[background == band_id] = new_id
        if np.array_equal(recolored, background):
            raise RuntimeError(
                "change_background_color: baked background contained none of the "
                "sky/water/seabed ids, so the mod would have been a no-op."
            )
        renderer.BACKGROUND = jnp.asarray(recolored)
