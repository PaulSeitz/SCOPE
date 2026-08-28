"""POV-relative action mapping and synchronized ego observations.

When ``USE_POV_RELATIVE_MAPPER`` is enabled:

* **View direction** always points at the selected target (θ_view = bearing to target).
  The target is never "behind" the POV — the 180° field is centered on the target.
* **Player facing** (hull orientation) is independent of view direction.
* **Ego actions** are body-relative to player facing (FORWARD = along facing).
* **Mapper** translates ``(facing_bin, ego_action) → game_action`` with no target geometry.
* **Ego angle** encodes ψ = facing − view via ``sin(ψ), cos(ψ)``.
* **Optional velocity** (``USE_EGO_VELOCITY_FEATURES``): one signed view-radial
  approach channel (+ = target closing on player, − = receding).
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from concept_based_learning.observations.action_space import (
    EGO_MOVE_DELTAS_DEG,
    _WORLD_DIR_ANGLES_DEG,
    _default_get_player,
    _nearest_world_dir,
)

_USE_POV_RELATIVE: bool = False


def using_pov_relative() -> bool:
    return _USE_POV_RELATIVE


def configure_pov_relative_from_config(config: dict | None) -> None:
    global _USE_POV_RELATIVE
    _USE_POV_RELATIVE = bool((config or {}).get("USE_POV_RELATIVE_MAPPER", False))


# Per-env move specs: (noop, fire, move_table[8], move_fire_table[8])
_MOVE_SPECS: dict[str, tuple[int, int, list[int], list[int]]] = {
    "seaquest": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "pong": (0, 1, [0, 3, 3, 3, 0, 2, 2, 2], [5, 3, 3, 3, 5, 2, 2, 2]),
    "freeway": (0, 0, [0, 2, 2, 2, 0, 1, 1, 1], [0, 2, 2, 2, 0, 1, 1, 1]),
    "breakout": (0, 1, [2, 2, 0, 3, 3, 3, 0, 2], [2, 2, 1, 3, 3, 3, 1, 2]),
    "beamrider": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "asterix": (0, 0, [2, 7, 4, 8, 3, 6, 1, 5], [2, 7, 4, 8, 3, 6, 1, 5]),
    "tennis": (0, 1, [0, 3, 3, 3, 0, 2, 2, 2], [5, 3, 3, 3, 5, 2, 2, 2]),
    "phoenix": (0, 1, [2, 2, 4, 3, 3, 3, 0, 2], [5, 5, 7, 6, 6, 6, 1, 5]),
    "skiing": (0, 0, [0, 2, 2, 2, 0, 1, 1, 1], [0, 2, 2, 2, 0, 1, 1, 1]),
    "kangaroo": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "choppercommand": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "spaceinvaders": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "montezumarevenge": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "venture": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
}

# Facing reference angles (degrees, screen-Y-down) for mapper table bin 0 / bin 1.
# Seaquest: left/right hull. Pong/Freeway/Breakout: vertical control, pinned facing.
# ``bilateral_axis`` selects virtual sector geometry for USE_POV_FIXED_BILATERAL.
# Vertical games use bin1 = +90° (down) for the second sector table when fixed bilateral
# is on; legacy POV still reads sprite bin 0 only for those envs.
_POV_PROFILES: dict[str, dict[str, Any]] = {
    "seaquest": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "pong": {
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": 90.0,
        "bilateral_axis": "vertical",
        "synthetic_orientation_deg": -90.0,
    },
    "freeway": {
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": 90.0,
        "bilateral_axis": "vertical",
        "synthetic_orientation_deg": -90.0,
        "player_getter": lambda raw: raw.chicken,
    },
    "breakout": {
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": 90.0,
        "bilateral_axis": "vertical",
    },
    "beamrider": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "asterix": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "tennis": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "phoenix": {
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": 90.0,
        "bilateral_axis": "vertical",
        "synthetic_orientation_deg": 90.0,
    },
    "skiing": {
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": 90.0,
        "bilateral_axis": "vertical",
        "synthetic_orientation_deg": -90.0,
        "player_getter": lambda raw: raw.skier,
    },
    "kangaroo": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "choppercommand": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "spaceinvaders": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "montezumarevenge": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "venture": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
}


def _player_orientation_deg(raw_obs, profile: dict[str, Any], player_getter) -> jnp.ndarray:
    player = player_getter(raw_obs)
    ori = player.orientation[-1]
    synthetic = profile.get("synthetic_orientation_deg")
    if synthetic is not None:
        return jnp.asarray(synthetic, dtype=jnp.float32)
    return ori


def _facing_bin_idx(raw_obs, profile: dict[str, Any], player_getter) -> jnp.ndarray:
    ori = _player_orientation_deg(raw_obs, profile, player_getter)
    return (ori >= jnp.float32(180.0)).astype(jnp.int32)


def _target_delta_xy(raw_obs, objects, target_idx, player_getter):
    """Player→target centre offset for the latest stacked frame."""
    from concept_based_learning.observations.egocentric import _center_y, _uses_bottom_edge_y

    player = player_getter(raw_obs)
    y_is_bottom_edge = _uses_bottom_edge_y(raw_obs)
    p_cx = player.x[-1] + player.width[-1] / jnp.float32(2.0)
    p_cy = _center_y(
        player.y[-1], player.height[-1], y_is_bottom_edge=y_is_bottom_edge
    )
    t_cx = objects.x[-1, target_idx] + objects.width[-1, target_idx] / jnp.float32(2.0)
    t_cy = _center_y(
        objects.y[-1, target_idx],
        objects.height[-1, target_idx],
        y_is_bottom_edge=y_is_bottom_edge,
    )
    return t_cx - p_cx, t_cy - p_cy


def _mapper_sector_idx(raw_obs, objects, target_idx, profile, player_getter) -> jnp.ndarray:
    from concept_based_learning.observations.pov_fixed_bilateral import (
        bilateral_axis_from_profile,
        target_sector_idx,
        using_pov_fixed_bilateral,
    )

    if using_pov_fixed_bilateral():
        dx, dy = _target_delta_xy(raw_obs, objects, target_idx, player_getter)
        axis = bilateral_axis_from_profile(profile)
        return target_sector_idx(dx, dy, axis=axis)
    return _facing_bin_idx(raw_obs, profile, player_getter)


def build_facing_action_table(
    *,
    noop_action: int,
    fire_action: int,
    move_table,
    move_fire_table,
    facing_deg: float,
) -> jnp.ndarray:
    """Map each ego action to a game action for a fixed body-facing angle."""
    _noop = jnp.int32(noop_action)
    _fire = jnp.int32(fire_action)
    _move = jnp.array(move_table, dtype=jnp.int32)
    _mfir = jnp.array(move_fire_table, dtype=jnp.int32)
    facing = jnp.float32(facing_deg)

    def _one(ego_action: jnp.ndarray) -> jnp.ndarray:
        delta = EGO_MOVE_DELTAS_DEG[ego_action]
        desired_deg = facing + delta
        dir_idx = _nearest_world_dir(desired_deg)
        has_fire = ego_action >= 10
        is_fire_only = ego_action == 1
        is_noop = ego_action == 0
        game_action = jnp.where(has_fire, _mfir[dir_idx], _move[dir_idx])
        game_action = jnp.where(is_fire_only, _fire, game_action)
        game_action = jnp.where(is_noop, _noop, game_action)
        return game_action.astype(jnp.int32)

    return jax.vmap(_one)(jnp.arange(18, dtype=jnp.int32))


def make_facing_relative_mappers(
    jax_game_id: str,
    *,
    player_getter: Callable | None = None,
) -> tuple[Callable, Callable]:
    """Return (forward_mapper, inverse_mapper) for a game id."""
    game_key = str(jax_game_id).lower()
    if game_key not in _MOVE_SPECS:
        raise ValueError(
            f"No POV-relative move spec for jax_game_id={jax_game_id!r}. "
            f"Known: {sorted(_MOVE_SPECS)}"
        )
    profile = _POV_PROFILES.get(game_key, {"facing_deg_bin0": 0.0, "facing_deg_bin1": 180.0})
    noop, fire, move, mfire = _MOVE_SPECS[game_key]
    from concept_based_learning.observations.pov_fixed_bilateral import (
        active_bilateral_axis,
        bilateral_facing_degrees,
        using_pov_fixed_bilateral,
    )

    if using_pov_fixed_bilateral():
        facing0, facing1 = bilateral_facing_degrees(profile, axis=active_bilateral_axis())
    else:
        facing0 = float(profile["facing_deg_bin0"])
        facing1 = float(profile["facing_deg_bin1"])
    table0 = build_facing_action_table(
        noop_action=noop,
        fire_action=fire,
        move_table=move,
        move_fire_table=mfire,
        facing_deg=facing0,
    )
    table1 = build_facing_action_table(
        noop_action=noop,
        fire_action=fire,
        move_table=move,
        move_fire_table=mfire,
        facing_deg=facing1,
    )
    _getter = player_getter or profile.get("player_getter") or _default_get_player
    _noop_i = jnp.int32(noop)
    _fire_i = jnp.int32(fire)

    def mapper(ego_action, raw_obs, target_idx, objects):
        bin_idx = _mapper_sector_idx(raw_obs, objects, target_idx, profile, _getter)
        table = jnp.where(bin_idx == 0, table0, table1)
        return table[ego_action]

    def inverse_mapper(game_action, raw_obs, target_idx, objects):
        bin_idx = _mapper_sector_idx(raw_obs, objects, target_idx, profile, _getter)
        table = jnp.where(bin_idx == 0, table0, table1)
        game_action = jnp.asarray(game_action, dtype=jnp.int32)
        matches = table == game_action
        has_match = jnp.any(matches)
        ego_idx = jnp.argmax(matches).astype(jnp.int32)
        ego_idx = jnp.where(game_action == _noop_i, jnp.int32(0), ego_idx)
        ego_idx = jnp.where(
            jnp.logical_and(game_action == _fire_i, jnp.logical_not(has_match)),
            jnp.int32(1),
            ego_idx,
        )
        ego_idx = jnp.where(
            jnp.logical_and(jnp.logical_not(has_match), game_action != _noop_i),
            jnp.int32(0),
            ego_idx,
        )
        return ego_idx.astype(jnp.int32)

    return mapper, inverse_mapper


def resolve_action_mappers(components, config: dict) -> tuple[Callable, Callable]:
    """Pick POV-relative or registered env mappers from config."""
    if not bool(config.get("USE_POV_RELATIVE_MAPPER", False)):
        return components.action_mapper, components.inverse_action_mapper

    mapper, inverse = make_facing_relative_mappers(components.jax_game_id)
    if str(components.jax_game_id).lower() == "seaquest":
        mapper = _wrap_seaquest_ascent_block(mapper)
    return mapper, inverse


def apply_seaquest_ascent_block(mapped_action, raw_obs, target_idx):
    """Legacy guard: drop ascent actions near the surface unless the target is it."""
    from concept_based_learning.environments.seaquest.label import (
        SURFACE_DUMMY_IDX,
        SURFACE_Y_THRESHOLD,
        using_legacy_seaquest_ascent_block,
    )
    from jaxatari.environment import JAXAtariAction as Atari

    if not using_legacy_seaquest_ascent_block():
        return mapped_action

    near_surface = raw_obs.player.y[-1] <= SURFACE_Y_THRESHOLD
    surface_target_exception = target_idx == SURFACE_DUMMY_IDX
    should_block_ascent = jnp.logical_and(
        near_surface,
        jnp.logical_not(surface_target_exception),
    )

    ascent_actions = jnp.array(
        [
            Atari.UP,
            Atari.UPLEFT,
            Atari.UPRIGHT,
            Atari.UPFIRE,
            Atari.UPLEFTFIRE,
            Atari.UPRIGHTFIRE,
        ],
        dtype=jnp.int32,
    )
    ascent_fire_actions = jnp.array(
        [Atari.UPFIRE, Atari.UPLEFTFIRE, Atari.UPRIGHTFIRE],
        dtype=jnp.int32,
    )

    is_ascent_action = jnp.any(mapped_action == ascent_actions)
    is_ascent_fire_action = jnp.any(mapped_action == ascent_fire_actions)
    blocked_replacement = jnp.where(is_ascent_fire_action, Atari.FIRE, Atari.NOOP)

    return jnp.where(
        jnp.logical_and(should_block_ascent, is_ascent_action),
        blocked_replacement,
        mapped_action,
    )


def _wrap_seaquest_ascent_block(base_mapper):
    def mapper(ego_action, raw_obs, target_idx, objects):
        return apply_seaquest_ascent_block(
            base_mapper(ego_action, raw_obs, target_idx, objects),
            raw_obs,
            target_idx,
        )

    return mapper


def pov_facing_psi_rad(
    player_orientation_deg: jnp.ndarray,
    theta_target_rad: jnp.ndarray,
) -> jnp.ndarray:
    """ψ = player facing minus view direction (view = toward target)."""
    facing_rad = player_orientation_deg * (jnp.pi / 180.0)
    psi = facing_rad - theta_target_rad
    return jnp.arctan2(jnp.sin(psi), jnp.cos(psi))
