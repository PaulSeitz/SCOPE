import functools

import jax.numpy as jnp

from concept_based_learning.environments.spaceinvaders.spaceinvaders_sweep_policy import (
    get_heuristic_target as spaceinvaders_sweep_heuristic_target,
    get_heuristic_target_with_mission as spaceinvaders_sweep_heuristic_with_mission,
)


def get_nearest_object_target(raw_obs, objects, last_target, rng, *, no_target_idx=-1):
    """Select the closest active object to the player centre. Env-agnostic."""
    p_cx = raw_obs.player.x[-1] + raw_obs.player.width[-1] / 2.0
    p_cy = raw_obs.player.y[-1] + raw_obs.player.height[-1] / 2.0

    obj_cx = objects.x[-1] + objects.width[-1] / 2.0
    obj_cy = objects.y[-1] + objects.height[-1] / 2.0

    dist = jnp.sqrt((obj_cx - p_cx) ** 2 + (obj_cy - p_cy) ** 2)
    active = objects.active[-1] > 0.5
    dist_masked = jnp.where(active, dist, jnp.inf)

    nearest_idx = jnp.argmin(dist_masked)
    return jnp.where(jnp.any(active), nearest_idx, jnp.int32(no_target_idx))


def resolve_heuristic_target_fn(config, env_name: str, components):
    """Pick heuristic meta-policy from HEURISTIC_MODE and environment."""
    heuristic_mode = config.get("HEURISTIC_MODE", "env_specific")
    if heuristic_mode == "nearest":
        return functools.partial(
            get_nearest_object_target, no_target_idx=components.no_target_idx
        )
    if heuristic_mode == "sweep" and str(env_name).lower() == "spaceinvaders":
        return spaceinvaders_sweep_heuristic_target
    return components.get_heuristic_target


def uses_sweep_mission_state(config, env_name: str) -> bool:
    return (
        str(config.get("HEURISTIC_MODE", "")).lower() == "sweep"
        and str(env_name).lower() == "spaceinvaders"
    )


def resolve_heuristic_with_mission_fn(config, env_name: str):
    if uses_sweep_mission_state(config, env_name):
        return spaceinvaders_sweep_heuristic_with_mission
    return None
