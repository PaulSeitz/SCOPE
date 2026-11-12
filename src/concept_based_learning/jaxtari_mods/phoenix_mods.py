from jaxtari.modification import JaxtariModController
from concept_based_learning.jaxtari_mods.phoenix.phoenix_mod_plugins import (
    BossLateMissilesMod,
    InfiniteLivesMod,
    FastPlayerMod,
    InvinciblePlayerMod,
    FastEnemyBulletsMod,
    NoAbilityCooldownMod,
    NightMod,
    GrayscaleMod,
    InvertedColorsMod,
    MatrixMod,
    BloodMoonMod,
    FormationReshuffleMod,
    PlayerDriftMod,
    StaticScoreBaitMod,
)


from concept_based_learning.jaxtari_mods.robustness_probes import (
    PhoenixShiftPlayerRowMod,
    PhoenixNoEnemyFireMod,
)

class PhoenixEnvMod(JaxtariModController):
    """
    Game-specific mod controller for Phoenix.
    """

    REGISTRY = {
        "shift_player_row": PhoenixShiftPlayerRowMod,
        "no_enemy_fire": PhoenixNoEnemyFireMod,

        "boss_late_missiles": BossLateMissilesMod,
        "infinite_lives": InfiniteLivesMod,
        "fast_player": FastPlayerMod,
        "invincible_player": InvinciblePlayerMod,
        "fast_enemy_bullets": FastEnemyBulletsMod,
        "no_ability_cooldown": NoAbilityCooldownMod,
        "night_mode": NightMod,
        "grayscale": GrayscaleMod,
        "inverted_colors": InvertedColorsMod,
        "matrix_theme": MatrixMod,
        "blood_moon": BloodMoonMod,
        "formation_reshuffle": FormationReshuffleMod,
        "player_drift": PlayerDriftMod,
        "static_score_bait": StaticScoreBaitMod,
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

