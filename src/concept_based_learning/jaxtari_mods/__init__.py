"""Register SCOPE's evaluated controllers with the pinned upstream JAXtari."""

_CONTROLLERS = {'asteroids': 'AsteroidsEnvMod', 'beamrider': 'BeamRiderEnvMod', 'freeway': 'FreewayEnvMod', 'namethisgame': 'NameThisGameEnvMod', 'phoenix': 'PhoenixEnvMod', 'pong': 'PongEnvMod', 'seaquest': 'SeaquestEnvMod', 'spaceinvaders': 'SpaceInvadersEnvMod'}


def register_modifications() -> None:
    """Use SCOPE's mod registry entries in this process only."""
    from jaxtari.core import MOD_MODULES

    MOD_MODULES.update({
        game: f"{__name__}.{game}_mods.{controller}"
        for game, controller in _CONTROLLERS.items()
    })
