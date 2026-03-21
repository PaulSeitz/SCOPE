from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

from jaxatari import make as make_atari_env
from jaxatari.wrappers import AtariWrapper
from jaxatari.games.jax_seaquest import SeaquestRenderer
from jaxatari.games.jax_choppercommand import ChopperCommandRenderer
from jaxatari.games.jax_spaceinvaders import SpaceInvadersRenderer
from jaxatari.games.jax_phoenix import PhoenixRenderer

from concept_based_learning.environments.labeling_wrapper import (
    JaxatariSemanticBaseWrapper,
)
from concept_based_learning.environments.seaquest.seaquest_label import (
    SeaquestSemanticWrapper,
    seaquest_action_mapper,
    extract_all_skills_rewards,
)
from concept_based_learning.environments.seaquest.seaquest_policy import (
    get_heuristic_target,
    NO_TARGET_IDX,
)

from concept_based_learning.environments.choppercommand.choppercommand_label import (
    ChopperCommandSemanticWrapper,
    choppercommand_action_mapper,
    extract_all_skills_rewards as choppercommand_extract_all_skills_rewards,
)
from concept_based_learning.environments.choppercommand.choppercommand_policy import (
    get_heuristic_target as choppercommand_get_heuristic_target,
    NO_TARGET_IDX as choppercommand_NO_TARGET_IDX,
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
    reward_extractor: Callable[..., Any]


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
            reward_extractor=extract_all_skills_rewards,
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
            reward_extractor=choppercommand_extract_all_skills_rewards,
        )

        self._registry["spaceinvaders"] = EnvComponents(
            name="SpaceInvaders",
            jax_game_id="spaceinvaders",
            semantic_wrapper_cls=SpaceInvadersSemanticWrapper,
            renderer_cls=SpaceInvadersRenderer,
            get_heuristic_target=spaceinvaders_get_heuristic_target,
            no_target_idx=spaceinvaders_NO_TARGET_IDX,
            action_mapper=spaceinvaders_action_mapper,
            reward_extractor=spaceinvaders_extract_all_skills_rewards,
        )

        self._registry["phoenix"] = EnvComponents(
            name="Phoenix",
            jax_game_id="phoenix",
            semantic_wrapper_cls=PhoenixSemanticWrapper,
            renderer_cls=PhoenixRenderer,
            get_heuristic_target=phoenix_get_heuristic_target,
            no_target_idx=phoenix_NO_TARGET_IDX,
            action_mapper=phoenix_action_mapper,
            reward_extractor=phoenix_extract_all_skills_rewards,
        )


    def available_envs(self):
        return tuple(self._registry.keys())

    def get_components(self, env_name: str) -> EnvComponents:
        """
        Return the registered components for a given environment name.
        """
        try:
            return self._registry[env_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown environment '{env_name}'. "
                f"Registered environments: {sorted(self._registry.keys())}"
            ) from exc

    def make_env(self, env_name: str, config: Dict[str, Any], train: bool):
        """
        Construct a wrapped JaxAtari environment and matching renderer.

        The wrapper and renderer are chosen based on `env_name`, while
        generic flags such as sticky actions, episodic life, and GAME_MODS
        are read from the provided config dict.
        """
        components = self.get_components(env_name)

        game_mods = config.get("GAME_MODS", None)
        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)

        if game_mods:
            env = make_atari_env(components.jax_game_id, mods=set(game_mods))
        else:
            env = make_atari_env(components.jax_game_id)

        if train:
            # Training-time wrapper (optionally sticky + episodic life, with long horizon).
            env = AtariWrapper(
                env,
                sticky_actions=sticky_actions,
                episodic_life=episodic_life,
                max_episode_length=27000,
                first_fire=True,
            )
        else:
            # Evaluation-time wrapper: clean rollouts without sticky actions or episodic life.
            env = AtariWrapper(env, sticky_actions=False, episodic_life=False)

        env = components.semantic_wrapper_cls(env)
        renderer = components.renderer_cls()
        return env, renderer

