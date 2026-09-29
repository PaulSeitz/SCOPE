import functools

import jax.numpy as jnp

from concept_based_learning.environments.manual_policy.state import PolicyState
from concept_based_learning.environments.seaquest.heuristic_policy import (
    AVOID_INTERRUPT_RANGE as SEAQUEST_AVOID_INTERRUPT_RANGE,
    AVOID_INTERRUPT_RANGE_Y as SEAQUEST_AVOID_INTERRUPT_RANGE_Y,
    COLLECT_EDGE_HARD as SEAQUEST_COLLECT_EDGE_HARD,
    COLLECT_EDGE_MARGIN_PX as SEAQUEST_COLLECT_EDGE_MARGIN_PX,
    COLLECT_REQUIRE_FULLY_ON_SCREEN as SEAQUEST_COLLECT_REQUIRE_FULLY_ON_SCREEN,
    COLLECT_SKIP_WHEN_FULL as SEAQUEST_COLLECT_SKIP_WHEN_FULL,
    COLLECT_NEAREST_IN_RANGE_ONLY as SEAQUEST_COLLECT_NEAREST_IN_RANGE_ONLY,
    COLLECT_NEAR_RANGE as SEAQUEST_COLLECT_NEAR_RANGE,
    DESTROY_INTERRUPT_RANGE as SEAQUEST_DESTROY_INTERRUPT_RANGE,
    DESTROY_NEAREST_IN_DANGER_ONLY as SEAQUEST_DESTROY_NEAREST_IN_DANGER_ONLY,
    DIVER_LANE_Y_PX as SEAQUEST_DIVER_LANE_Y_PX,
    DIVER_LANE_BLOCK_RANGE as SEAQUEST_DIVER_LANE_BLOCK_RANGE,
    ENEMY_REQUIRE_ON_SCREEN as SEAQUEST_ENEMY_REQUIRE_ON_SCREEN,
    SURFACE_DANGER_CLOSE_RANGE as SEAQUEST_SURFACE_DANGER_CLOSE_RANGE,
    SURFACE_SUB_AVOID_RANGE_X as SEAQUEST_SURFACE_SUB_AVOID_RANGE_X,
    SURFACE_SUB_AVOID_RANGE_Y as SEAQUEST_SURFACE_SUB_AVOID_RANGE_Y,
    SURFACE_SUB_NEAR_RANGE as SEAQUEST_SURFACE_SUB_NEAR_RANGE,
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
    """AABB surface-gap px for surface AVOID override; ``<=0`` disables."""
    return float(
        config.get("SURFACE_DANGER_CLOSE_RANGE", SEAQUEST_SURFACE_DANGER_CLOSE_RANGE)
    )


def _seaquest_destroy_interrupt_range(config) -> float:
    """AABB surface-gap px for DESTROY interrupt; ``<=0`` disables."""
    return float(
        config.get("DESTROY_INTERRUPT_RANGE", SEAQUEST_DESTROY_INTERRUPT_RANGE)
    )


def _seaquest_avoid_interrupt_range(config) -> float:
    """Horizontal AABB edge-gap px for AVOID (torpedo) ellipse; ``<=0`` disables."""
    return float(config.get("AVOID_INTERRUPT_RANGE", SEAQUEST_AVOID_INTERRUPT_RANGE))


def _seaquest_avoid_interrupt_range_y(config) -> float:
    """Vertical AABB edge-gap px for AVOID (torpedo) ellipse; ``<=0`` disables."""
    return float(
        config.get("AVOID_INTERRUPT_RANGE_Y", SEAQUEST_AVOID_INTERRUPT_RANGE_Y)
    )


def _seaquest_surface_sub_avoid_range_x(config) -> float:
    """Horizontal ellipse for anytime surface-sub AVOID; ``<=0`` disables."""
    return float(
        config.get("SURFACE_SUB_AVOID_RANGE_X", SEAQUEST_SURFACE_SUB_AVOID_RANGE_X)
    )


def _seaquest_surface_sub_avoid_range_y(config) -> float:
    """Vertical ellipse for anytime surface-sub AVOID; ``<=0`` disables."""
    return float(
        config.get("SURFACE_SUB_AVOID_RANGE_Y", SEAQUEST_SURFACE_SUB_AVOID_RANGE_Y)
    )


def _seaquest_surface_sub_near_range(config) -> float:
    """Nearest-target surface-sub gate (surface band only); ``<=0`` disables."""
    return float(config.get("SURFACE_SUB_NEAR_RANGE", SEAQUEST_SURFACE_SUB_NEAR_RANGE))


def _seaquest_diver_lane_y_px(config) -> float:
    """Centre-Δy band that blocks COLLECT; ``<=0`` disables lane gating."""
    return float(config.get("DIVER_LANE_Y_PX", SEAQUEST_DIVER_LANE_Y_PX))


def _seaquest_diver_lane_block_range(config) -> float:
    """AABB gap for same-lane DESTROY to block COLLECT; ``<=0`` = any distance."""
    return float(
        config.get("DIVER_LANE_BLOCK_RANGE", SEAQUEST_DIVER_LANE_BLOCK_RANGE)
    )


def _seaquest_collect_requires_no_destroy(config) -> bool:
    """If True, any live DESTROY beats COLLECT (COLLECT never fights)."""
    return bool(config.get("COLLECT_REQUIRES_NO_DESTROY", False))


def _seaquest_collect_edge_margin_px(config) -> float:
    """Midfield COLLECT margin; ``<=0`` disables (allow edge divers)."""
    return float(config.get("COLLECT_EDGE_MARGIN_PX", SEAQUEST_COLLECT_EDGE_MARGIN_PX))


def _seaquest_collect_edge_hard(config) -> bool:
    """If True, edge divers are never COLLECT targets (no soft fall-back)."""
    return bool(config.get("COLLECT_EDGE_HARD", SEAQUEST_COLLECT_EDGE_HARD))


def _seaquest_collect_require_fully_on_screen(config) -> bool:
    """Hard-drop divers whose AABB is not fully inside the screen horizontally."""
    return bool(
        config.get(
            "COLLECT_REQUIRE_FULLY_ON_SCREEN",
            SEAQUEST_COLLECT_REQUIRE_FULLY_ON_SCREEN,
        )
    )


def _seaquest_collect_skip_when_full(config) -> bool:
    """Never COLLECT a diver when ``collected_divers >= 6``."""
    return bool(config.get("COLLECT_SKIP_WHEN_FULL", SEAQUEST_COLLECT_SKIP_WHEN_FULL))


def _seaquest_collect_nearest_in_range_only(config) -> bool:
    """COLLECT nearest only inside COLLECT_NEAR_RANGE (DESTROY otherwise)."""
    return bool(
        config.get(
            "COLLECT_NEAREST_IN_RANGE_ONLY",
            SEAQUEST_COLLECT_NEAREST_IN_RANGE_ONLY,
        )
    )


def _seaquest_collect_near_range(config) -> float:
    """AABB gap for COLLECT nearest eligibility; ``<=0`` disables the gate."""
    return float(config.get("COLLECT_NEAR_RANGE", SEAQUEST_COLLECT_NEAR_RANGE))


def _seaquest_enemy_require_on_screen(config) -> bool:
    """DESTROY nearest/interrupt only once AABB is fully on-screen."""
    return bool(
        config.get("ENEMY_REQUIRE_ON_SCREEN", SEAQUEST_ENEMY_REQUIRE_ON_SCREEN)
    )


def _seaquest_destroy_nearest_in_danger_only(config) -> bool:
    """DESTROY nearest scoring only inside DESTROY_INTERRUPT_RANGE."""
    return bool(
        config.get(
            "DESTROY_NEAREST_IN_DANGER_ONLY",
            SEAQUEST_DESTROY_NEAREST_IN_DANGER_ONLY,
        )
    )


def _seaquest_heuristic_kwargs(config) -> dict:
    return {
        "danger_close_range": _seaquest_danger_close_range(config),
        "destroy_interrupt_range": _seaquest_destroy_interrupt_range(config),
        "avoid_interrupt_range": _seaquest_avoid_interrupt_range(config),
        "avoid_interrupt_range_y": _seaquest_avoid_interrupt_range_y(config),
        "surface_sub_avoid_range_x": _seaquest_surface_sub_avoid_range_x(config),
        "surface_sub_avoid_range_y": _seaquest_surface_sub_avoid_range_y(config),
        "surface_sub_near_range": _seaquest_surface_sub_near_range(config),
        "diver_lane_y_px": _seaquest_diver_lane_y_px(config),
        "diver_lane_block_range": _seaquest_diver_lane_block_range(config),
        "collect_requires_no_destroy": _seaquest_collect_requires_no_destroy(config),
        "collect_edge_margin_px": _seaquest_collect_edge_margin_px(config),
        "collect_edge_hard": _seaquest_collect_edge_hard(config),
        "collect_require_fully_on_screen": _seaquest_collect_require_fully_on_screen(
            config
        ),
        "collect_skip_when_full": _seaquest_collect_skip_when_full(config),
        "collect_nearest_in_range_only": _seaquest_collect_nearest_in_range_only(
            config
        ),
        "collect_near_range": _seaquest_collect_near_range(config),
        "enemy_require_on_screen": _seaquest_enemy_require_on_screen(config),
        "destroy_nearest_in_danger_only": _seaquest_destroy_nearest_in_danger_only(
            config
        ),
    }


def _spaceinvaders_heuristic_mode(config) -> str:
    return str(config.get("HEURISTIC_MODE", "env_specific")).lower().strip()


def uses_spaceinvaders_sweep(config, env_name: str) -> bool:
    """Bottom-row L→R sweep is the default SI heuristic (env_specific / sweep)."""
    if str(env_name).lower() != "spaceinvaders":
        return False
    mode = _spaceinvaders_heuristic_mode(config)
    # Legacy uniform-random bottom-row: HEURISTIC_MODE=random_bottom
    return mode not in {"nearest", "avoid_only", "avoid", "random_bottom"}


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
    if uses_spaceinvaders_sweep(config, env_name):
        return spaceinvaders_sweep_heuristic_target
    fn = components.get_heuristic_target
    if str(env_name).lower() == "seaquest":
        fn = functools.partial(fn, **_seaquest_heuristic_kwargs(config))
    return fn


def uses_sweep_mission_state(config, env_name: str) -> bool:
    return uses_spaceinvaders_sweep(config, env_name)


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
        return functools.partial(fn, **_seaquest_heuristic_kwargs(config))
    return fn


def uses_heuristic_policy_state(config, env_name: str, components) -> bool:
    return resolve_heuristic_with_state_fn(config, env_name, components) is not None


def initial_policy_state(no_target_idx: int = -1) -> PolicyState:
    return PolicyState.zero(no_target_idx=no_target_idx)
