from jaxtari.modification import JaxtariModController
from concept_based_learning.jaxtari_mods.beamrider.beamrider_mod_plugins import (
    DoubleEnemySpeedMod,
    FogOfWarMod,
    HardcoreMod,
    MothershipLaserMod,
    SameEnemiesMod,
    TeleportUFOsMod,
    ThreeLanesMod,
    ToasterMod,
    SlowStrafeMod,
    TopIdleNoShootMod,
)


from concept_based_learning.jaxtari_mods.robustness_probes import (
    BeamriderShiftPlayerRowMod,
    BeamriderNoEnemyShotsMod,
)

class BeamRiderEnvMod(JaxtariModController):
    """
    Game-specific Mod Controller for Beamrider.
    """

    REGISTRY = {
        "shift_player_row": BeamriderShiftPlayerRowMod,
        "no_enemy_shots": BeamriderNoEnemyShotsMod,

        "double_enemy_speed": DoubleEnemySpeedMod,
        "fog_of_war": FogOfWarMod,
        "hardcore": HardcoreMod,
        "mothership_laser": MothershipLaserMod,
        "same_enemies": SameEnemiesMod,
        "teleport_ufos": TeleportUFOsMod,
        "three_lanes": ThreeLanesMod,
        "toaster": ToasterMod,
        "slow_strafe": SlowStrafeMod,
        "idle_at_range": TopIdleNoShootMod,
        "top_idle_no_shoot": TopIdleNoShootMod,  # alias
    }

    def __init__(
        self,
        env,
        mods_config: list = [],
        allow_conflicts: bool = False,
    ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY,
        )
