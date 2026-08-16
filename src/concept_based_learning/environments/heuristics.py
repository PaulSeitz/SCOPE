import functools

import jax.numpy as jnp

from concept_based_learning.environments.manual_policy.state import PolicyState
from concept_based_learning.environments.seaquest.heuristic_policy import (
    SURFACE_DANGER_CLOSE_RANGE as SEAQUEST_SURFACE_DANGER_CLOSE_RANGE,
)
from concept_based_learning.environments.spaceinvaders.spaceinvaders_sweep_policy import (
    get_heuristic_target as spaceinvaders_sweep_heuristic_target,
    get_heuristic_target_with_mission as spaceinvaders_sweep_heuristic_with_mission,
)
from concept_based_learning.observations.egocentric import player_object_aabb_gaps


def get_nearest_object_target(raw_obs, objects, last_target, rng, *, no_target_idx=-1):
    """Select the closest active object by AABB surface gap. Env-agnostic."""
    del last_target, rng
    distances, _, _, _, _ = player_object_aabb_gaps(raw_obs, objects)
    active = objects.active[-1] > 0.5
    dist_masked = jnp.where(active, distances, jnp.inf)

    nearest_idx = jnp.argmin(dist_masked)
    return jnp.where(jnp.any(active), nearest_idx, jnp.int32(no_target_idx))


def get_avoid_only_target(raw_obs, objects, last_target, rng, *, no_target_idx=-1):
    """Nearest active AVOID-labeled object (AVOID skill probe / oversampling)."""
    del last_target, rng
    from concept_based_learning.environments.datatypes import LABEL_AVOID

    distances, _, _, _, _ = player_object_aabb_gaps(raw_obs, objects)
    active = objects.active[-1] > 0.5
    is_avoid = objects.semantic_label[-1] == LABEL_AVOID
    mask = jnp.logical_and(active, is_avoid)
    dist_masked = jnp.where(mask, distances, jnp.inf)
    nearest_idx = jnp.argmin(dist_masked)
    return jnp.where(jnp.any(mask), nearest_idx, jnp.int32(no_target_idx))


def _seaquest_danger_close_range(config) -> float:
    """AABB surface-gap px for surface danger override; ``<=0`` disables."""
    return float(
        config.get("SURFACE_DANGER_CLOSE_RANGE", SEAQUEST_SURFACE_DANGER_CLOSE_RANGE)
    )


def resolve_heuristic_target_fn(config, env_name: str, components):
    """Pick heuristic meta-policy from HEURISTIC_MODE and environment."""
    heuristic_mode = str(config.get("HEURISTIC_MODE", "env_specific")).lower().strip()
    if heuristic_mode == "nearest":
        return functools.partial(
            get_nearest_object_target, no_target_idx=components.no_target_idx
        )
    if heuristic_mode in {"avoid_only", "avoid"}:
        return functools.partial(
            get_avoid_only_target, no_target_idx=components.no_target_idx
        )
    if heuristic_mode == "sweep" and str(env_name).lower() == "spaceinvaders":
        return spaceinvaders_sweep_heuristic_target
    fn = components.get_heuristic_target
    if str(env_name).lower() == "seaquest":
        fn = functools.partial(
            fn, danger_close_range=_seaquest_danger_close_range(config)
        )
    return fn


def uses_sweep_mission_state(config, env_name: str) -> bool:
    return (
        str(config.get("HEURISTIC_MODE", "")).lower() == "sweep"
        and str(env_name).lower() == "spaceinvaders"
    )


def resolve_heuristic_with_mission_fn(config, env_name: str):
    """Legacy SI-sweep helper; prefer ``resolve_heuristic_with_state_fn``."""
    if uses_sweep_mission_state(config, env_name):
        return spaceinvaders_sweep_heuristic_with_mission
    return None


def _sweep_as_policy_state(raw_obs, objects, prev_target, state, rng):
    target, mission = spaceinvaders_sweep_heuristic_with_mission(
        raw_obs, objects, prev_target, state.mission_target, rng
    )
    return target, state.replace(mission_target=mission.astype(jnp.int32))


def resolve_heuristic_with_state_fn(config, env_name: str, components):
    """Return ``(raw, objects, prev, PolicyState, rng) -> (target, PolicyState)`` or None."""
    if uses_sweep_mission_state(config, env_name):
        return _sweep_as_policy_state
    fn = getattr(components, "get_heuristic_with_state", None)
    if fn is not None and str(env_name).lower() == "seaquest":
        return functools.partial(
            fn, danger_close_range=_seaquest_danger_close_range(config)
        )
    return fn


def uses_heuristic_policy_state(config, env_name: str, components) -> bool:
    return resolve_heuristic_with_state_fn(config, env_name, components) is not None


def initial_policy_state(no_target_idx: int = -1) -> PolicyState:
    return PolicyState.zero(no_target_idx=no_target_idx)
