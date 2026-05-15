import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union

import jax.numpy as jnp
from jaxatari import make as make_atari_env
from jaxatari.core import MOD_MODULES as JAXATARI_MOD_MODULES
from jaxatari.wrappers import AtariWrapper
from jaxatari.games.jax_seaquest import SeaquestRenderer
from jaxatari.games.jax_choppercommand import ChopperCommandRenderer
from jaxatari.games.jax_spaceinvaders import SpaceInvadersRenderer
from jaxatari.games.jax_phoenix import PhoenixRenderer

from concept_based_learning.environments.labeling_wrapper import (
    JaxatariSemanticBaseWrapper,
)
from concept_based_learning.environments.datatypes import (
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
)
from concept_based_learning.environments.seaquest.label import (
    SeaquestSemanticWrapper,
    seaquest_action_mapper,
    extract_all_skills_rewards,
    set_seaquest_action_mapper,
)
from concept_based_learning.environments.seaquest.heuristic_policy import (
    get_heuristic_target,
    NO_TARGET_IDX,
    apply_meta_mask_overrides as seaquest_meta_mask_hook,
    get_forced_target_override as seaquest_forced_target_hook,
)

from concept_based_learning.environments.choppercommand.choppercommand_label import (
    ChopperCommandSemanticWrapper,
    choppercommand_action_mapper,
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
    extract_all_skills_rewards as spaceinvaders_extract_all_skills_rewards,
)
from concept_based_learning.environments.spaceinvaders.spaceinvaders_policy import (
    get_heuristic_target as spaceinvaders_get_heuristic_target,
    NO_TARGET_IDX as spaceinvaders_NO_TARGET_IDX,
)

from concept_based_learning.environments.phoenix.phoenix_label import (
    PhoenixSemanticWrapper,
    phoenix_action_mapper,
    extract_all_skills_rewards as phoenix_extract_all_skills_rewards,
)
from concept_based_learning.environments.phoenix.phoenix_policy import (
    get_heuristic_target as phoenix_get_heuristic_target,
    NO_TARGET_IDX as phoenix_NO_TARGET_IDX,
)

from concept_based_learning.environments.bootcamp.bootcamp_label import (
    BootcampSemanticWrapper,
)
from concept_based_learning.environments.bootcamp.bootcamp_action_mapper import (
    bootcamp_action_mapper,
)
from concept_based_learning.environments.bootcamp.bootcamp_policy import (
    get_heuristic_target as bootcamp_get_heuristic_target,
    NO_TARGET_IDX as bootcamp_NO_TARGET_IDX,
    bootcamp_fixed_target_index_from_game_mods,
)


def _normalize_game_mods(raw: Union[None, str, List[Any]]) -> Optional[List[str]]:
    """
    Coerce Hydra / YAML GAME_MODS into a list of mod name strings.

    JAXAtari expects string keys matching the game mod controller REGISTRY (e.g. skillbootcamp:
    ``destroy``, ``avoid``, ``collect`` — not Seaquest-style ``destroy_curriculum``).
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


def _default_meta_mask_hook(raw_obs, active_masks, config):
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


def _default_extract_life_lost(prev_env_state: Any, next_env_state: Any) -> jnp.ndarray:
    # Default assumes JAXAtari-like state tree and returns (prev_lives - next_lives).
    prev_lives = prev_env_state.atari_state.env_state.lives
    next_lives = next_env_state.atari_state.env_state.lives
    return (prev_lives - next_lives).astype(jnp.int32)


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
        del info, config
        return reward_extractor_5arg(
            prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost
        )

    return _wrapped


def _bootcamp_fixed_target_idx_from_config(config: Dict[str, Any]) -> Optional[int]:
    return bootcamp_fixed_target_index_from_game_mods(
        _normalize_game_mods(config.get("GAME_MODS", None))
    )


def _bootcamp_resolve_curriculum_mods(
    env_key: str, phase_name: str, curriculum_mods: Dict[str, Any]
) -> Optional[List[str]]:
    del env_key
    raw = (curriculum_mods or {}).get(phase_name)
    mods_list: List[str] = [] if raw is None else [str(x) for x in list(raw)]
    if not mods_list and phase_name != "full" and phase_name in ("destroy", "collect", "avoid"):
        return [phase_name]
    return mods_list if mods_list else None


def _bootcamp_reward_extractor(
    prev_raw_obs: Any,
    next_raw_obs: Any,
    prev_target: Any,
    next_target: Any,
    life_lost: jnp.ndarray,
    info: Optional[Dict[str, Any]],
    config: Dict[str, Any],
):
    del config, prev_raw_obs, next_raw_obs, next_target
    life_lost_arr = jnp.atleast_1d(life_lost)

    # Bootcamp now runs continuously with immediate respawns. In that setting a
    # target can be consumed and replaced in one step, hiding active 1->0
    # transitions. Use env-reported per-step events from info for robust rewards.
    def _info_get(key: str):
        z = jnp.zeros_like(life_lost_arr, dtype=jnp.int32)
        if info is None:
            return z
        if isinstance(info, dict):
            return info.get(key, z)
        if hasattr(info, key):
            return getattr(info, key)
        return z

    def _last_frame(e):
        return jnp.ravel(e)[-1] if jnp.ndim(e) > 0 else e

    prev_label = _last_frame(prev_target.semantic_label)
    swd = jnp.atleast_1d(_info_get("step_win_destroy")) != 0
    swc = jnp.atleast_1d(_info_get("step_win_collect")) != 0
    sfa = jnp.atleast_1d(_info_get("step_fail_avoid")) != 0
    sfd = jnp.atleast_1d(_info_get("step_fail_destroy_touch")) != 0
    swa = jnp.atleast_1d(_info_get("step_win_avoid")) != 0

    is_destroy = jnp.full_like(swd, prev_label == LABEL_DESTROY, dtype=jnp.bool_)
    is_collect = jnp.full_like(swd, prev_label == LABEL_COLLECT, dtype=jnp.bool_)
    is_avoid = jnp.full_like(swd, prev_label == LABEL_AVOID, dtype=jnp.bool_)

    destroy_reward = jnp.where(
        is_destroy,
        jnp.where(swd, 1.0, jnp.where(sfd, -1.0, 0.0)),
        0.0,
    )
    destroy_done = jnp.logical_and(is_destroy, jnp.logical_or(swd, sfd))

    collect_reward = jnp.where(is_collect, jnp.where(swc, 1.0, 0.0), 0.0)
    collect_done = jnp.logical_and(is_collect, swc)

    avoid_reward = jnp.where(
        is_avoid,
        jnp.where(swa, 1.0, jnp.where(sfa, -10.0, 0.0)),
        0.0,
    )
    avoid_done = jnp.logical_and(is_avoid, jnp.logical_or(swa, sfa))

    rewards = jnp.stack([destroy_reward, collect_reward, avoid_reward]).astype(jnp.float32)
    skill_done = jnp.stack([destroy_done, collect_done, avoid_done]).astype(jnp.bool_)

    # Preserve scalar output shape when called outside vmapped training loops.
    if jnp.ndim(life_lost) == 0:
        rewards = rewards[..., 0]
        skill_done = skill_done[..., 0]
    return rewards, skill_done

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
    # Uniform trainer hook:
    # reward_extractor(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info, config)
    reward_extractor: Callable[..., Any]
    fixed_target_idx_from_config: Callable[[Dict[str, Any]], Optional[int]]
    extract_life_lost: Callable[..., jnp.ndarray]
    resolve_curriculum_mods: Callable[..., Optional[List[str]]]
    meta_mask_hook: Callable[..., Any]
    forced_target_hook: Callable[..., Any]


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
            reward_extractor=_wrap_reward_extractor_5arg(extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=seaquest_meta_mask_hook,
            forced_target_hook=seaquest_forced_target_hook,
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
            reward_extractor=_wrap_reward_extractor_5arg(phoenix_extract_all_skills_rewards),
            fixed_target_idx_from_config=_default_fixed_target_idx_from_config,
            extract_life_lost=_default_extract_life_lost,
            resolve_curriculum_mods=_default_resolve_curriculum_mods,
            meta_mask_hook=_default_meta_mask_hook,
            forced_target_hook=_default_forced_target_hook,
        )

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

        game_mods = _normalize_game_mods(config.get("GAME_MODS", None))
        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)
        first_fire = bool(config.get("FIRST_FIRE", True))
        # Must match frozen skill checkpoints: ego MLP length = EGO_OBS_CHANNELS_PER_FRAME * frame_stack_size.
        frame_stack_size = int(config.get("FRAME_STACK_SIZE", 4))
        frame_skip = int(config.get("FRAME_SKIP", 4))
        clip_reward = bool(config.get("CLIP_REWARD", False))

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
            wrapper_kwargs["sticky_actions"] = False
            wrapper_kwargs["episodic_life"] = False
            env = AtariWrapper(env, **wrapper_kwargs)

        # Compatibility hints used by semantic wrappers when frame-stack handling
        # moved out of AtariWrapper in newer jaxatari versions.
        setattr(env, "_cbl_frame_stack_size", frame_stack_size)
        setattr(env, "_cbl_frame_skip", frame_skip)
        setattr(env, "_cbl_clip_reward", clip_reward)

        env = components.semantic_wrapper_cls(env)
        renderer = components.renderer_cls()
        return env, renderer

