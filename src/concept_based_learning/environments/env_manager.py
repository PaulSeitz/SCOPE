import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union

import jax.numpy as jnp
from jaxatari import make as make_atari_env
from jaxatari.core import MOD_MODULES as JAXATARI_MOD_MODULES
from jaxatari.wrappers import AtariWrapper
try:
    from jaxatari.wrappers import PixelAndObjectObsWrapper
except ImportError:  # pragma: no cover - compatibility with older jaxatari forks
    PixelAndObjectObsWrapper = None
from jaxatari.games.jax_seaquest import SeaquestRenderer
from jaxatari.games.jax_choppercommand import ChopperCommandRenderer
from jaxatari.games.jax_spaceinvaders import SpaceInvadersRenderer
from jaxatari.games.jax_phoenix import PhoenixRenderer
from jaxatari.games.jax_freeway import FreewayRenderer
from jaxatari.games.jax_pong import PongRenderer
from jaxatari.games.jax_breakout import BreakoutRenderer
from jaxatari.games.jax_beamrider import BeamriderRenderer
from jaxatari.games.jax_kangaroo import KangarooRenderer
from jaxatari.games.jax_skiing import SkiingRenderer
from jaxatari.games.jax_tennis import TennisRenderer
from jaxatari.games.jax_asterix import AsterixRenderer
from jaxatari.games.jax_montezumarevenge import MontezumaRevengeRenderer
from jaxatari.games.jax_venture import VentureRenderer
from jaxatari.games.jax_asteroids import AsteroidsRenderer
from jaxatari.games.jax_namethisgame import Renderer_NameThisGame
from jaxatari.games.jax_timepilot import TimePilotRenderer
# Kaboom is not in upstream jaxatari yet — keep label/policy code, do not register.
# from jaxatari.games.jax_kaboom import KaboomRenderer

from concept_based_learning.environments.labeling_wrapper import (
    JaxatariSemanticBaseWrapper,
)
from concept_based_learning.environments.seaquest.label import (
    SeaquestSemanticWrapper,
    seaquest_action_mapper,
    seaquest_inverse_action_mapper,
    extract_all_skills_rewards,
    set_seaquest_action_mapper,
    set_seaquest_ascent_block,
    set_seaquest_surface_label,
    set_seaquest_surfacing_death_guard,
)
from concept_based_learning.environments.seaquest.heuristic_policy import (
    get_heuristic_target,
    get_heuristic_target_with_state as seaquest_get_heuristic_with_state,
    NO_TARGET_IDX,
    apply_meta_mask_overrides as seaquest_meta_mask_hook,
    get_forced_target_override as seaquest_forced_target_hook,
)

from concept_based_learning.environments.choppercommand.choppercommand_label import (
    ChopperCommandSemanticWrapper,
    choppercommand_action_mapper,
    choppercommand_inverse_action_mapper,
    extract_all_skills_rewards as choppercommand_extract_all_skills_rewards,
)
from concept_based_learning.environments.choppercommand.choppercommand_policy import (
    get_heuristic_target as choppercommand_get_heuristic_target,
    NO_TARGET_IDX as choppercommand_NO_TARGET_IDX,
    apply_meta_mask_overrides as choppercommand_meta_mask_hook,
    get_forced_target_override as choppercommand_forced_target_hook,
)

from concept_based_learning.environments.spaceinvaders.spaceinvaders_label import (
    SpaceInvadersSemanticWrapper,
    spaceinvaders_action_mapper,
    spaceinvaders_inverse_action_mapper,
    extract_all_skills_rewards as spaceinvaders_extract_all_skills_rewards,
)
from concept_based_learning.environments.spaceinvaders.spaceinvaders_policy import (
    get_heuristic_target as spaceinvaders_get_heuristic_target,
    NO_TARGET_IDX as spaceinvaders_NO_TARGET_IDX,
)

from concept_based_learning.environments.phoenix.phoenix_label import (
    PhoenixSemanticWrapper,
    phoenix_action_mapper,
    phoenix_inverse_action_mapper,
    extract_all_skills_rewards as phoenix_extract_all_skills_rewards,
)
from concept_based_learning.environments.phoenix.phoenix_policy import (
    get_heuristic_target as phoenix_get_heuristic_target,
    NO_TARGET_IDX as phoenix_NO_TARGET_IDX,
)
from concept_based_learning.environments.meta_target_mask import phoenix_meta_mask_hook

from concept_based_learning.environments.freeway.freeway_label import (
    FreewaySemanticWrapper,
    freeway_action_mapper,
    freeway_inverse_action_mapper,
    extract_all_skills_rewards as freeway_extract_all_skills_rewards,
)
from concept_based_learning.environments.freeway.freeway_policy import (
    get_heuristic_target as freeway_get_heuristic_target,
    NO_TARGET_IDX as freeway_NO_TARGET_IDX,
    set_freeway_heuristic_params,
)
from concept_based_learning.environments.pong.pong_label import (
    PongSemanticWrapper,
    pong_action_mapper,
    pong_inverse_action_mapper,
    extract_all_skills_rewards as pong_extract_all_skills_rewards,
)
from concept_based_learning.environments.breakout.breakout_label import (
    BreakoutSemanticWrapper,
    breakout_action_mapper,
    breakout_inverse_action_mapper,
    extract_all_skills_rewards as breakout_extract_all_skills_rewards,
)
from concept_based_learning.environments.beamrider.beamrider_label import (
    BeamriderSemanticWrapper,
    beamrider_action_mapper,
    beamrider_inverse_action_mapper,
    extract_all_skills_rewards as beamrider_extract_all_skills_rewards,
)
from concept_based_learning.environments.beamrider.beamrider_policy import (
    get_heuristic_target as beamrider_get_heuristic_target,
    NO_TARGET_IDX as beamrider_NO_TARGET_IDX,
)
from concept_based_learning.environments.kangaroo.kangaroo_label import (
    KangarooSemanticWrapper,
    kangaroo_action_mapper,
    kangaroo_inverse_action_mapper,
    extract_all_skills_rewards as kangaroo_extract_all_skills_rewards,
)
from concept_based_learning.environments.kangaroo.kangaroo_policy import (
    get_heuristic_target as kangaroo_get_heuristic_target,
    get_heuristic_target_with_state as kangaroo_get_heuristic_with_state,
    NO_TARGET_IDX as kangaroo_NO_TARGET_IDX,
)
from concept_based_learning.environments.skiing.skiing_label import (
    SkiingSemanticWrapper,
    skiing_action_mapper,
    skiing_inverse_action_mapper,
    extract_all_skills_rewards as skiing_extract_all_skills_rewards,
)
from concept_based_learning.environments.tennis.tennis_label import (
    TennisSemanticWrapper,
    tennis_action_mapper,
    tennis_inverse_action_mapper,
    extract_all_skills_rewards as tennis_extract_all_skills_rewards,
)
from concept_based_learning.environments.asterix.asterix_label import (
    AsterixSemanticWrapper,
    asterix_action_mapper,
    asterix_inverse_action_mapper,
    extract_all_skills_rewards as asterix_extract_all_skills_rewards,
)
from concept_based_learning.environments.asterix.asterix_policy import (
    get_heuristic_target as asterix_get_heuristic_target,
    NO_TARGET_IDX as asterix_NO_TARGET_IDX,
)
from concept_based_learning.environments.montezumarevenge.montezuma_label import (
    MontezumaRevengeSemanticWrapper,
    montezuma_action_mapper,
    montezuma_inverse_action_mapper,
    extract_all_skills_rewards as montezuma_extract_all_skills_rewards,
)
from concept_based_learning.environments.montezumarevenge.montezuma_policy import (
    get_heuristic_target as montezuma_get_heuristic_target,
    get_heuristic_target_with_state as montezuma_get_heuristic_with_state,
    NO_TARGET_IDX as montezuma_NO_TARGET_IDX,
)
from concept_based_learning.environments.venture.venture_label import (
    VentureSemanticWrapper,
    venture_action_mapper,
    venture_inverse_action_mapper,
    extract_all_skills_rewards as venture_extract_all_skills_rewards,
)
from concept_based_learning.environments.venture.venture_policy import (
    get_heuristic_target as venture_get_heuristic_target,
    NO_TARGET_IDX as venture_NO_TARGET_IDX,
)
from concept_based_learning.environments.asteroids.asteroids_label import (
    AsteroidsSemanticWrapper,
    asteroids_action_mapper,
    asteroids_inverse_action_mapper,
    extract_all_skills_rewards as asteroids_extract_all_skills_rewards,
)
from concept_based_learning.environments.asteroids.asteroids_policy import (
    get_heuristic_target as asteroids_get_heuristic_target,
    get_heuristic_target_with_state as asteroids_get_heuristic_with_state,
    NO_TARGET_IDX as asteroids_NO_TARGET_IDX,
)
from concept_based_learning.environments.namethisgame.namethisgame_label import (
    NameThisGameSemanticWrapper,
    namethisgame_action_mapper,
    namethisgame_inverse_action_mapper,
    extract_all_skills_rewards as namethisgame_extract_all_skills_rewards,
)
from concept_based_learning.environments.namethisgame.namethisgame_policy import (
    get_heuristic_target as namethisgame_get_heuristic_target,
    NO_TARGET_IDX as namethisgame_NO_TARGET_IDX,
)
from concept_based_learning.environments.timepilot.timepilot_label import (
    TimePilotSemanticWrapper,
    timepilot_action_mapper,
    timepilot_inverse_action_mapper,
    extract_all_skills_rewards as timepilot_extract_all_skills_rewards,
)
from concept_based_learning.environments.timepilot.timepilot_policy import (
    get_heuristic_target as timepilot_get_heuristic_target,
    get_heuristic_target_with_state as timepilot_get_heuristic_with_state,
    NO_TARGET_IDX as timepilot_NO_TARGET_IDX,
)
# from concept_based_learning.environments.kaboom.kaboom_label import (
#     KaboomSemanticWrapper,
#     kaboom_action_mapper,
#     kaboom_inverse_action_mapper,
#     extract_all_skills_rewards as kaboom_extract_all_skills_rewards,
# )
# from concept_based_learning.environments.kaboom.kaboom_policy import (
#     get_heuristic_target as kaboom_get_heuristic_target,
#     NO_TARGET_IDX as kaboom_NO_TARGET_IDX,
# )
from concept_based_learning.environments.generic_heuristic import (
    get_heuristic_target as generic_get_heuristic_target,
    NO_TARGET_IDX as generic_NO_TARGET_IDX,
)


def _normalize_game_mods(raw: Union[None, str, List[Any]]) -> Optional[List[str]]:
    """
    Coerce Hydra / YAML GAME_MODS into a list of mod name strings.

    JAXAtari expects string keys matching the game mod controller REGISTRY
    (e.g. ``destroy``, ``avoid``, ``collect`` — not Seaquest-style ``destroy_curriculum``).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    out = [str(x) for x in list(raw)]
    return out if out else None


def _atari_wrapper_supports(param_name: str) -> bool:
    """Check AtariWrapper kwargs at runtime for jaxatari API compatibility."""
    return param_name in inspect.signature(AtariWrapper.__init__).parameters


def _default_meta_mask_hook(raw_obs, active_masks, config, objects=None):
    del raw_obs, config, objects
    return active_masks


def _default_forced_target_hook(raw_obs, current_target, config, rng=None):
    return current_target


def _default_fixed_target_idx_from_config(config: Dict[str, Any]) -> Optional[int]:
    return None


def _default_resolve_curriculum_mods(
    env_key: str, phase_name: str, curriculum_mods: Dict[str, Any]
) -> Optional[List[str]]:
    raw = (curriculum_mods or {}).get(phase_name)
    mods_list: List[str] = [] if raw is None else [str(x) for x in list(raw)]
    return mods_list if mods_list else None


def _lives_from_env_state(env_state: Any) -> jnp.ndarray:
    """Read remaining lives from a JAXAtari env_state (field name varies by game)."""
    if hasattr(env_state, "lives"):
        return env_state.lives
    if hasattr(env_state, "player_lives"):
        return env_state.player_lives
    if hasattr(env_state, "lives_remaining"):
        return env_state.lives_remaining
    raise AttributeError(
        "env_state has no `lives`, `player_lives`, or `lives_remaining` field; "
        "register a custom extract_life_lost for this game."
    )


def _unwrap_atari_env_state(wrapper_state: Any) -> Any:
    """Recursively unwrap nested wrapper states until an env_state object is reached."""
    cur = wrapper_state
    for _ in range(8):
        if hasattr(cur, "env_state"):
            return cur.env_state
        if hasattr(cur, "atari_state"):
            cur = cur.atari_state
            continue
        break
    raise AttributeError(
        "Could not unwrap env_state from wrapper state; "
        f"received type {type(wrapper_state)!r}."
    )


def _default_extract_life_lost(prev_env_state: Any, next_env_state: Any) -> jnp.ndarray:
    prev_es = _unwrap_atari_env_state(prev_env_state)
    next_es = _unwrap_atari_env_state(next_env_state)
    prev_lives = _lives_from_env_state(prev_es)
    next_lives = _lives_from_env_state(next_es)
    return (prev_lives - next_lives).astype(jnp.int32)


def _freeway_extract_life_lost(prev_env_state: Any, next_env_state: Any) -> jnp.ndarray:
    """
    Freeway has no lives; treat the start of a collision cooldown as a 'life lost'
    event so the AVOID skill receives the appropriate failure signal.
    """
    prev_es = _unwrap_atari_env_state(prev_env_state)
    next_es = _unwrap_atari_env_state(next_env_state)
    collision = jnp.logical_and(prev_es.cooldown == 0, next_es.cooldown > 0)
    return collision.astype(jnp.int32)


def _pong_extract_life_lost(prev_env_state: Any, next_env_state: Any) -> jnp.ndarray:
    """Pong has no lives; enemy scoring a point is the 'life lost' signal."""
    prev_es = _unwrap_atari_env_state(prev_env_state)
    next_es = _unwrap_atari_env_state(next_env_state)
    return (next_es.enemy_score - prev_es.enemy_score).astype(jnp.int32)


def _tennis_extract_life_lost(prev_env_state: Any, next_env_state: Any) -> jnp.ndarray:
    """Tennis has no lives; enemy scoring a point is the 'life lost' signal."""
    prev_es = _unwrap_atari_env_state(prev_env_state)
    next_es = _unwrap_atari_env_state(next_env_state)
    prev_pts = prev_es.game_state.enemy_score
    next_pts = next_es.game_state.enemy_score
    return (next_pts - prev_pts).astype(jnp.int32)


def _skiing_extract_life_lost(prev_env_state: Any, next_env_state: Any) -> jnp.ndarray:
    """Skiing has no lives; treat start of collision cooldown as a 'life lost' event."""
    prev_es = _unwrap_atari_env_state(prev_env_state)
    next_es = _unwrap_atari_env_state(next_env_state)
    collision = jnp.logical_and(prev_es.collision_cooldown == 0, next_es.collision_cooldown > 0)
    return collision.astype(jnp.int32)


# def _kaboom_extract_life_lost(prev_env_state: Any, next_env_state: Any) -> jnp.ndarray:
#     """Kaboom has no lives counter; losing a bucket is the life-lost signal."""
#     prev_es = _unwrap_atari_env_state(prev_env_state)
#     next_es = _unwrap_atari_env_state(next_env_state)
#     prev_n = jnp.sum(prev_es.buckets_pos[:, 2] > 0)
#     next_n = jnp.sum(next_es.buckets_pos[:, 2] > 0)
#     return jnp.maximum(prev_n - next_n, 0).astype(jnp.int32)


def _wrap_reward_extractor_5arg(reward_extractor_5arg: Callable[..., Any]):
    def _wrapped(
        prev_raw_obs: Any,
        next_raw_obs: Any,
        prev_target: Any,
        next_target: Any,
        life_lost: jnp.ndarray,
        info: Optional[Dict[str, Any]],
        config: Dict[str, Any],
    ):
        try:
            return reward_extractor_5arg(
                prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost,
                info, config,
            )
        except TypeError:
            return reward_extractor_5arg(
                prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost
            )

    return _wrapped


def semantic_max_objects_from_env(env) -> int:
    """Return ``max_objects`` from the outermost ``JaxatariSemanticBaseWrapper``."""
    cur = env
    for _ in range(16):
        if isinstance(cur, JaxatariSemanticBaseWrapper):
            return int(cur.max_objects)
        nxt = getattr(cur, "env", None)
        if nxt is None:
            nxt = getattr(cur, "_env", None)
        if nxt is None:
            break
        cur = nxt
    raise ValueError(
        f"Could not find JaxatariSemanticBaseWrapper in env chain "
        f"(started from {type(env).__name__})."
    )


def align_config_max_objects(
    config: Dict[str, Any], env, *, env_name: str = ""
) -> None:
    """Set ``config['MAX_OBJECTS']`` from the semantic wrapper when missing or wrong."""
    env_max = semantic_max_objects_from_env(env)
    cfg_max = config.get("MAX_OBJECTS")
    label = env_name or str(config.get("ENV_NAME", "env"))
    if cfg_max is None:
        config["MAX_OBJECTS"] = env_max
        print(f"MAX_OBJECTS not set; using {env_max} for {label}.")
        return
    cfg_max = int(cfg_max)
    if cfg_max != env_max:
        warnings.warn(
            f"MAX_OBJECTS={cfg_max} does not match semantic wrapper ({env_max}) "
            f"for ENV_NAME={label!r}; using env value {env_max}.",
            UserWarning,
            stacklevel=2,
        )
        config["MAX_OBJECTS"] = env_max


@dataclass(frozen=True)
class EnvComponents:
    """
    Environment-specific hooks used by the training pipeline and visualizers.

    All environments that plug into the concept-based skills architecture
    should provide:
    - A semantic wrapper subclassing `JaxatariSemanticBaseWrapper`
    - A renderer for visualization
    - A heuristic meta-policy over objects (`get_heuristic_target`)
    - A sentinel value for "no target on screen" (`no_target_idx`)
    - An egocentric → game action mapper (`action_mapper`)
    - The underlying JaxAtari game id (`jax_game_id`)
    """

    name: str
    jax_game_id: str
    semantic_wrapper_cls: Type[JaxatariSemanticBaseWrapper]
    renderer_cls: Type[Any]
    get_heuristic_target: Callable[..., Any]
    no_target_idx: int
    action_mapper: Callable[..., Any]
    inverse_action_mapper: Callable[..., Any]
    # Uniform trainer hook:
    # reward_extractor(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info, config)
    reward_extractor: Callable[..., Any]
    fixed_target_idx_from_config: Callable[[Dict[str, Any]], Optional[int]]
    extract_life_lost: Callable[..., jnp.ndarray]
    resolve_curriculum_mods: Callable[..., Optional[List[str]]]
    meta_mask_hook: Callable[..., Any]
    forced_target_hook: Callable[..., Any]
    # Optional stateful heuristic: (raw, objects, prev_target, PolicyState, rng)
    # -> (target, PolicyState).  Used for Sequence / ClimbLadders / SI sweep.
    get_heuristic_with_state: Optional[Callable[..., Any]] = None
    # If True, clear PolicyState (plan_idx / mission) on life loss as well as
    # episode reset.  Needed for Sequence plans that should restart on respawn
    # (e.g. Kangaroo); leave False when stickiness across lives is fine.
    reset_policy_on_life_lost: bool = False
    # Game-action ids used by action_mapper for NOOP / stationary FIRE (legal-mask).
    ego_mapper_noop_action: int = 0
    ego_mapper_fire_action: int = 1
    # Game action when the heuristic returns no_target_idx (Beamrider: FIRE to
    # advance sectors when the screen is empty). Config ``NO_TARGET_GAME_ACTION``
    # overrides this when set.
    no_target_game_action: int = 0


def resolve_no_target_game_action(
    config: Dict[str, Any] | None,
    components: "EnvComponents | None" = None,
) -> int:
    """Idle game action when no object is targeted.

    Prefer explicit ``NO_TARGET_GAME_ACTION`` in config; otherwise the env
    component default (Beamrider → FIRE=1 so empty screens still advance).
    """
    if config is not None:
        raw = config.get("NO_TARGET_GAME_ACTION", None)
        if raw is not None and raw != "":
            return int(raw)
    if components is not None:
        return int(components.no_target_game_action)
    return 0


class EnvManager:
    """
    Registry/manager that maps high-level environment names (e.g. "Seaquest")
    to their environment-specific policy and labeling hooks.

    This is the single place where new games should be registered so that both
    the training pipeline and analysis tools (e.g. visualizers) can look up
    the correct semantic wrapper, heuristic meta-policy, and action mapping.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, EnvComponents] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        # Seaquest registration
        self._registry["seaquest"] = EnvComponents(
            name="Seaquest",
            jax_game_id="seaquest",
            semantic_wrapper_cls=SeaquestSemanticWrapper,
            renderer_cls=SeaquestRenderer,
            get_heuristic_target=get_heuristic_target,
            no_target_idx=NO_TARGET_IDX,
            action_mapper=seaquest_action_mapper,
            inverse_action_mapper=seaquest_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=seaquest_meta_mask_hook,
            forced_target_hook=seaquest_forced_target_hook,
            get_heuristic_with_state=seaquest_get_heuristic_with_state,
        )

        # ChopperCommand registration
        self._registry["choppercommand"] = EnvComponents(
            name="ChopperCommand",
            jax_game_id="choppercommand",
            semantic_wrapper_cls=ChopperCommandSemanticWrapper,
            renderer_cls=ChopperCommandRenderer,
            get_heuristic_target=choppercommand_get_heuristic_target,
            no_target_idx=choppercommand_NO_TARGET_IDX,
            action_mapper=choppercommand_action_mapper,
            inverse_action_mapper=choppercommand_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(
                choppercommand_extract_all_skills_rewards
            ),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=choppercommand_meta_mask_hook,
            forced_target_hook=choppercommand_forced_target_hook,
        )

        self._registry["spaceinvaders"] = EnvComponents(
            name="SpaceInvaders",
            jax_game_id="spaceinvaders",
            semantic_wrapper_cls=SpaceInvadersSemanticWrapper,
            renderer_cls=SpaceInvadersRenderer,
            get_heuristic_target=spaceinvaders_get_heuristic_target,
            no_target_idx=spaceinvaders_NO_TARGET_IDX,
            action_mapper=spaceinvaders_action_mapper,
            inverse_action_mapper=spaceinvaders_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(
                spaceinvaders_extract_all_skills_rewards
            ),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )

        self._registry["phoenix"] = EnvComponents(
            name="Phoenix",
            jax_game_id="phoenix",
            semantic_wrapper_cls=PhoenixSemanticWrapper,
            renderer_cls=PhoenixRenderer,
            get_heuristic_target=phoenix_get_heuristic_target,
            no_target_idx=phoenix_NO_TARGET_IDX,
            action_mapper=phoenix_action_mapper,
            inverse_action_mapper=phoenix_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(phoenix_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=phoenix_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )

        self._registry["freeway"] = EnvComponents(
            name="Freeway",
            jax_game_id="freeway",
            semantic_wrapper_cls=FreewaySemanticWrapper,
            renderer_cls=FreewayRenderer,
            get_heuristic_target=freeway_get_heuristic_target,
            no_target_idx=freeway_NO_TARGET_IDX,
            action_mapper=freeway_action_mapper,
            inverse_action_mapper=freeway_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(freeway_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_freeway_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
            ego_mapper_fire_action=0,  # no distinct FIRE
        )

        self._registry["pong"] = EnvComponents(
            name="Pong",
            jax_game_id="pong",
            semantic_wrapper_cls=PongSemanticWrapper,
            renderer_cls=PongRenderer,
            get_heuristic_target=generic_get_heuristic_target,
            no_target_idx=generic_NO_TARGET_IDX,
            action_mapper=pong_action_mapper,
            inverse_action_mapper=pong_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(pong_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_pong_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )

        self._registry["breakout"] = EnvComponents(
            name="Breakout",
            jax_game_id="breakout",
            semantic_wrapper_cls=BreakoutSemanticWrapper,
            renderer_cls=BreakoutRenderer,
            get_heuristic_target=generic_get_heuristic_target,
            no_target_idx=generic_NO_TARGET_IDX,
            action_mapper=breakout_action_mapper,
            inverse_action_mapper=breakout_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(breakout_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )

        self._registry["beamrider"] = EnvComponents(
            name="Beamrider",
            jax_game_id="beamrider",
            semantic_wrapper_cls=BeamriderSemanticWrapper,
            renderer_cls=BeamriderRenderer,
            get_heuristic_target=beamrider_get_heuristic_target,
            no_target_idx=beamrider_NO_TARGET_IDX,
            action_mapper=beamrider_action_mapper,
            inverse_action_mapper=beamrider_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(beamrider_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
            # Empty-screen sector advance requires FIRE; without a target the
            # skill policy never runs, so idle must not stay on NOOP.
            no_target_game_action=1,
        )

        self._registry["kangaroo"] = EnvComponents(
            name="Kangaroo",
            jax_game_id="kangaroo",
            semantic_wrapper_cls=KangarooSemanticWrapper,
            renderer_cls=KangarooRenderer,
            get_heuristic_target=kangaroo_get_heuristic_target,
            no_target_idx=kangaroo_NO_TARGET_IDX,
            action_mapper=kangaroo_action_mapper,
            inverse_action_mapper=kangaroo_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(kangaroo_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
            get_heuristic_with_state=kangaroo_get_heuristic_with_state,
            reset_policy_on_life_lost=True,
        )

        self._registry["skiing"] = EnvComponents(
            name="Skiing",
            jax_game_id="skiing",
            semantic_wrapper_cls=SkiingSemanticWrapper,
            renderer_cls=SkiingRenderer,
            get_heuristic_target=generic_get_heuristic_target,
            no_target_idx=generic_NO_TARGET_IDX,
            action_mapper=skiing_action_mapper,
            inverse_action_mapper=skiing_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(skiing_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_skiing_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
            ego_mapper_fire_action=0,  # no distinct FIRE
        )

        self._registry["tennis"] = EnvComponents(
            name="Tennis",
            jax_game_id="tennis",
            semantic_wrapper_cls=TennisSemanticWrapper,
            renderer_cls=TennisRenderer,
            get_heuristic_target=generic_get_heuristic_target,
            no_target_idx=generic_NO_TARGET_IDX,
            action_mapper=tennis_action_mapper,
            inverse_action_mapper=tennis_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(tennis_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_tennis_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )

        self._registry["asterix"] = EnvComponents(
            name="Asterix",
            jax_game_id="asterix",
            semantic_wrapper_cls=AsterixSemanticWrapper,
            renderer_cls=AsterixRenderer,
            get_heuristic_target=asterix_get_heuristic_target,
            no_target_idx=asterix_NO_TARGET_IDX,
            action_mapper=asterix_action_mapper,
            inverse_action_mapper=asterix_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(asterix_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )

        self._registry["montezumarevenge"] = EnvComponents(
            name="MontezumaRevenge",
            jax_game_id="montezumarevenge",
            semantic_wrapper_cls=MontezumaRevengeSemanticWrapper,
            renderer_cls=MontezumaRevengeRenderer,
            get_heuristic_target=montezuma_get_heuristic_target,
            no_target_idx=montezuma_NO_TARGET_IDX,
            action_mapper=montezuma_action_mapper,
            inverse_action_mapper=montezuma_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(
                montezuma_extract_all_skills_rewards
            ),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
            get_heuristic_with_state=montezuma_get_heuristic_with_state,
            reset_policy_on_life_lost=True,
        )
        # Alias accepted by jaxatari.make("montezuma_revenge").
        self._registry["montezuma_revenge"] = self._registry["montezumarevenge"]

        self._registry["venture"] = EnvComponents(
            name="Venture",
            jax_game_id="venture",
            semantic_wrapper_cls=VentureSemanticWrapper,
            renderer_cls=VentureRenderer,
            get_heuristic_target=venture_get_heuristic_target,
            no_target_idx=venture_NO_TARGET_IDX,
            action_mapper=venture_action_mapper,
            inverse_action_mapper=venture_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(
                venture_extract_all_skills_rewards
            ),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )

        self._registry["asteroids"] = EnvComponents(
            name="Asteroids",
            jax_game_id="asteroids",
            semantic_wrapper_cls=AsteroidsSemanticWrapper,
            renderer_cls=AsteroidsRenderer,
            get_heuristic_target=asteroids_get_heuristic_target,
            no_target_idx=asteroids_NO_TARGET_IDX,
            action_mapper=asteroids_action_mapper,
            inverse_action_mapper=asteroids_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(
                asteroids_extract_all_skills_rewards
            ),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
            get_heuristic_with_state=asteroids_get_heuristic_with_state,
        )

        self._registry["namethisgame"] = EnvComponents(
            name="NameThisGame",
            jax_game_id="namethisgame",
            semantic_wrapper_cls=NameThisGameSemanticWrapper,
            renderer_cls=Renderer_NameThisGame,
            get_heuristic_target=namethisgame_get_heuristic_target,
            no_target_idx=namethisgame_NO_TARGET_IDX,
            action_mapper=namethisgame_action_mapper,
            inverse_action_mapper=namethisgame_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(
                namethisgame_extract_all_skills_rewards
            ),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )
        self._registry["name_this_game"] = self._registry["namethisgame"]

        self._registry["timepilot"] = EnvComponents(
            name="TimePilot",
            jax_game_id="timepilot",
            semantic_wrapper_cls=TimePilotSemanticWrapper,
            renderer_cls=TimePilotRenderer,
            get_heuristic_target=timepilot_get_heuristic_target,
            no_target_idx=timepilot_NO_TARGET_IDX,
            action_mapper=timepilot_action_mapper,
            inverse_action_mapper=timepilot_inverse_action_mapper,
            reward_extractor=_wrap_reward_extractor_5arg(
                timepilot_extract_all_skills_rewards
            ),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
            get_heuristic_with_state=timepilot_get_heuristic_with_state,
        )
        self._registry["time_pilot"] = self._registry["timepilot"]

        # Kaboom: not in upstream jaxatari yet. Uncomment with the imports above
        # once jax_kaboom lands.
        # self._registry["kaboom"] = EnvComponents(
        #     name="Kaboom",
        #     jax_game_id="kaboom",
        #     semantic_wrapper_cls=KaboomSemanticWrapper,
        #     renderer_cls=KaboomRenderer,
        #     get_heuristic_target=kaboom_get_heuristic_target,
        #     no_target_idx=kaboom_NO_TARGET_IDX,
        #     action_mapper=kaboom_action_mapper,
        #     inverse_action_mapper=kaboom_inverse_action_mapper,
        #     reward_extractor=_wrap_reward_extractor_5arg(
        #         kaboom_extract_all_skills_rewards
        #     ),
        #     fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
        #     extract_life_lost=_kaboom_extract_life_lost,
        #     resolve_curriculum_mods=_default_resolve_curriculum_mods,
        #     meta_mask_hook=_default_meta_mask_hook,
        #     forced_target_hook=_default_forced_target_hook,
        # )

    def available_envs(self):
        return tuple(self._registry.keys())

    def get_components(self, env_name: str) -> EnvComponents:
        """
        Return the registered components for a given environment name.
        """
        env_key = str(env_name).lower()
        try:
            return self._registry[env_key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown environment '{env_name}'. "
                f"Registered environments: {sorted(self._registry.keys())}"
            ) from exc

    def make_env(self, env_name: str, config: Dict[str, Any], train: bool):
        """
        Construct a wrapped JaxAtari environment and matching renderer.

        The wrapper and renderer are chosen based on `env_name`, while
        generic flags such as sticky actions, episodic life, FIRST_FIRE, and GAME_MODS
        are read from the provided config dict.
        """
        components = self.get_components(env_name)
        if components.jax_game_id == "seaquest":
            set_seaquest_action_mapper(
                bool(config.get("USE_LEGACY_SEAQUEST_ACTION_MAPPER", False))
            )
            set_seaquest_ascent_block(
                bool(config.get("USE_LEGACY_SEAQUEST_ASCENT_BLOCK", True))
            )
            set_seaquest_surfacing_death_guard(
                bool(config.get("USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD", False))
            )
            # Always apply (default collect) so multi-job workers reset between stems.
            set_seaquest_surface_label(config.get("SEAQUEST_SURFACE_LABEL", "collect"))
        if components.jax_game_id == "freeway":
            set_freeway_heuristic_params(
                threat_dist_thresh=config.get("FREEWAY_THREAT_DIST_THRESH", None),
            )

        game_mods = _normalize_game_mods(config.get("GAME_MODS", None))
        # AtariWrapper expects a float probability in [0, 1], not a bool.
        # ``True`` becomes ``float(True) == 1.0`` (always repeat last action) and
        # collapses training to zero reward — match flat_pqn_train's 0.25 default.
        sticky_raw = config.get("STICKY_ACTIONS", False)
        if isinstance(sticky_raw, bool):
            sticky_actions = 0.25 if sticky_raw else 0.0
        else:
            sticky_actions = float(sticky_raw)
        episodic_life = config.get("EPISODIC_LIFE", False)
        first_fire = bool(config.get("FIRST_FIRE", True))
        # Must match frozen skill checkpoints: ego MLP length = EGO_OBS_CHANNELS_PER_FRAME * frame_stack_size.
        frame_stack_size = int(config.get("FRAME_STACK_SIZE", 4))
        frame_skip = int(config.get("FRAME_SKIP", 4))
        clip_reward = bool(config.get("CLIP_REWARD", False))
        meta_input_mode = str(config.get("META_INPUT_MODE", "flat")).lower().strip()
        skill_input_mode = str(config.get("SKILL_INPUT_MODE", "ego")).lower().strip()

        jax_id = components.jax_game_id
        if game_mods:
            if jax_id not in JAXATARI_MOD_MODULES:
                raise ValueError(
                    f"GAME_MODS={game_mods!r} was set for jax_game_id={jax_id!r}, but this "
                    f"JAXAtari build has no MOD_MODULES entry for that game. Training would "
                    f"silently fall back to an unmodified environment. Install a JAXAtari "
                    f"version that registers MOD_MODULES['{jax_id}'] (known games: "
                    f"{sorted(JAXATARI_MOD_MODULES)})."
                )
            env = make_atari_env(jax_id, mods=game_mods)
        else:
            env = make_atari_env(jax_id)

        wrapper_kwargs: Dict[str, Any] = dict(
            sticky_actions=sticky_actions,
            episodic_life=episodic_life,
            first_fire=first_fire,
        )
        if _atari_wrapper_supports("frame_stack_size"):
            wrapper_kwargs["frame_stack_size"] = frame_stack_size
        # jaxatari renamed this argument in newer releases.
        if train:
            if _atari_wrapper_supports("max_episode_length"):
                wrapper_kwargs["max_episode_length"] = 27000
            elif _atari_wrapper_supports("max_frames_per_episode"):
                # Old API uses frame count instead of Atari steps.
                wrapper_kwargs["max_frames_per_episode"] = 108000

        if train:
            # Training-time wrapper (optionally sticky + episodic life, with long horizon).
            env = AtariWrapper(env, **wrapper_kwargs)
        else:
            # Evaluation-time wrapper: clean rollouts without sticky actions or episodic life.
            wrapper_kwargs["sticky_actions"] = 0.0
            wrapper_kwargs["episodic_life"] = False
            env = AtariWrapper(env, **wrapper_kwargs)

        # Pixel-based meta encoders need both observation modalities from Jaxtari.
        if (meta_input_mode == "pixel_embed") or (
            skill_input_mode in ("pixel_embed", "pixel_embed_cond")
        ):
            if PixelAndObjectObsWrapper is None:
                raise ImportError(
                    "META_INPUT_MODE='pixel_embed' requires PixelAndObjectObsWrapper "
                    "from jaxtari/jaxatari.wrappers, but it is unavailable in this install."
                )
            resize_shape = config.get("META_PIXEL_RESIZE_SHAPE", (84, 84))
            if isinstance(resize_shape, list):
                resize_shape = tuple(resize_shape)
            env = PixelAndObjectObsWrapper(
                env,
                do_pixel_resize=bool(config.get("META_PIXEL_DO_RESIZE", False)),
                pixel_resize_shape=tuple(resize_shape),
                grayscale=bool(config.get("META_PIXEL_GRAYSCALE", False)),
                use_native_downscaling=bool(
                    config.get("META_PIXEL_USE_NATIVE_DOWNSCALING", False)
                ),
                smooth_image=bool(config.get("META_PIXEL_SMOOTH_IMAGE", False)),
                frame_stack_size=frame_stack_size,
                frame_skip=frame_skip,
                max_pooling=bool(config.get("META_PIXEL_MAX_POOLING", True)),
                clip_reward=clip_reward,
            )

        # Compatibility hints used by semantic wrappers when frame-stack handling
        # moved out of AtariWrapper in newer jaxatari versions.
        setattr(env, "_cbl_frame_stack_size", frame_stack_size)
        setattr(env, "_cbl_frame_skip", frame_skip)
        setattr(env, "_cbl_clip_reward", clip_reward)

        env = components.semantic_wrapper_cls(env)
        renderer = components.renderer_cls()
        return env, renderer

