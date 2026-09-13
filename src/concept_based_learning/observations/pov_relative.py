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
    "pong": (0, 1, [0, 3, 3, 3, 0, 2, 2, 2], [1, 5, 5, 5, 1, 4, 4, 4]),
    "freeway": (0, 0, [0, 2, 2, 2, 0, 1, 1, 1], [0, 2, 2, 2, 0, 1, 1, 1]),
    "breakout": (0, 1, [2, 2, 0, 3, 3, 3, 0, 2], [2, 2, 1, 3, 3, 3, 1, 2]),
    "beamrider": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "asterix": (0, 0, [2, 7, 4, 8, 3, 6, 1, 5], [2, 7, 4, 8, 3, 6, 1, 5]),
    # Full 18-action ALE set (matches tennis_label.py). Measured 2026-08-31:
    # all 8 compass dirs displace toward the probe under facing-relative mapping.
    "tennis": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "phoenix": (0, 1, [2, 2, 4, 3, 3, 3, 0, 2], [5, 5, 7, 6, 6, 6, 1, 5]),
    "skiing": (0, 0, [0, 2, 2, 2, 0, 1, 1, 1], [0, 2, 2, 2, 0, 1, 1, 1]),
    # TODO(POV_BACKLOG): skiing — only LEFT/RIGHT in ACTION_SET; table wrong. See backlog.
    "kangaroo": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    # TODO(POV_BACKLOG): choppercommand — ALE RIGHT moves the heli left; see backlog.
    "choppercommand": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    # Compact ACTION_SET: 0=NOOP,1=FIRE,2=RIGHT,3=LEFT,4=RIGHTFIRE,5=LEFTFIRE
    # (same layout as spaceinvaders_label.py / breakout). Do NOT use Seaquest ALE ids —
    # env.step indexes ACTION_SET, so ALE RIGHT=3 would become LEFT.
    "spaceinvaders": (0, 1, [2, 2, 0, 3, 3, 3, 0, 2], [4, 4, 1, 5, 5, 5, 1, 4]),
    "montezumarevenge": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    "venture": (0, 1, [3, 8, 5, 9, 4, 7, 2, 6], [11, 16, 13, 17, 12, 15, 10, 14]),
    # Asteroids ACTION_SET indices. DOWN/DOWNFIRE are hyperspace — map D→UP.
    "asteroids": (0, 1, [3, 6, 2, 7, 4, 7, 2, 6], [9, 12, 8, 13, 10, 13, 8, 12]),
    # Name This Game: same compact horizontal set as SpaceInvaders.
    "namethisgame": (0, 1, [2, 2, 0, 3, 3, 3, 0, 2], [4, 4, 1, 5, 5, 5, 1, 4]),
    # Time Pilot: cardinal-only ACTION_SET; diagonals collapse.
    "timepilot": (0, 1, [3, 3, 5, 4, 4, 4, 2, 3], [7, 7, 9, 8, 8, 8, 6, 7]),
    # Kaboom deferred (not in upstream jaxatari yet).
    # "kaboom": (0, 1, [2, 2, 0, 3, 3, 3, 0, 2], [2, 2, 1, 3, 3, 3, 1, 2]),
}

# Per-env view axes: the body facings (screen degrees, y down) the player can
# actually point along. ``facing_deg_bin0`` / ``facing_deg_bin1`` are both the
# mapper table angles *and* the angles the observation frame is built in.
#
# Two-axis envs turn left/right and pick their axis from the sprite orientation.
# Single-axis envs cannot turn, so they pin ``single_view_axis`` and look into the
# field (Pong's paddle faces the court, bottom-of-screen shooters face up).
#
# ``bilateral_axis`` is unrelated: it only selects virtual sector geometry for
# USE_POV_FIXED_BILATERAL, which ignores sprite orientation entirely.
_POV_PROFILES: dict[str, dict[str, Any]] = {
    "seaquest": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "pong": {
        # Player paddle at x≈140 (right). Face into the field (−x) so FPS view /
        # FORWARD point at the opponent; horizontal motion is NOOP, LEFT/RIGHT
        # strafe the paddle (fairest Seaquest→Pong transfer).
        "facing_deg_bin0": 180.0,
        "facing_deg_bin1": 180.0,
        "single_view_axis": True,
        "bilateral_axis": "vertical",
    },
    "freeway": {
        # Chicken crosses upward; no turning.
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": -90.0,
        "single_view_axis": True,
        "bilateral_axis": "vertical",
        "player_getter": lambda raw: raw.chicken,
    },
    "breakout": {
        # Paddle at the bottom, ball and bricks above.
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": -90.0,
        "single_view_axis": True,
        "bilateral_axis": "vertical",
    },
    "beamrider": {
        # Ship at the bottom strafing left/right, enemies come down the beams.
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": -90.0,
        "single_view_axis": True,
        "bilateral_axis": "vertical",
    },
    "asterix": {
        "facing_deg_bin0": 0.0,
        "facing_deg_bin1": 180.0,
        "bilateral_axis": "horizontal",
    },
    "tennis": {
        # Top half of court faces down (+y), bottom half faces up (−y). Sprite
        # orientation is only L/R walk direction — not court bearing.
        "facing_deg_bin0": 90.0,
        "facing_deg_bin1": -90.0,
        "bilateral_axis": "vertical",
        "court_side_vertical_facing": True,
        "court_midline_y": 108.0,
    },
    "phoenix": {
        # Ship at the bottom, birds above.
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": -90.0,
        "single_view_axis": True,
        "bilateral_axis": "vertical",
    },
    "skiing": {
        # Skier descends; obstacles approach from below the viewport.
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": -90.0,
        "single_view_axis": True,
        "bilateral_axis": "vertical",
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
        # Cannon at the bottom strafing left/right, invaders above.
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": -90.0,
        "single_view_axis": True,
        "bilateral_axis": "vertical",
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
    "asteroids": {
        # 16-way heading (22.5°). Screen degrees = sprite + (−90°). Honest motor
        # is USE_POV_TURN_PRIMITIVES; these bins keep facing-frame obs/mapper
        # aligned when turn primitives are off.
        "facing_deg_bins": tuple(
            ((-90.0 + i * 22.5 + 180.0) % 360.0) - 180.0 for i in range(16)
        ),
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": 90.0,
        "bilateral_axis": "horizontal",
    },
    "namethisgame": {
        # Diver at the bottom; spears / threats are above (same as Space Invaders).
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": -90.0,
        "single_view_axis": True,
        "bilateral_axis": "vertical",
    },
    "timepilot": {
        # 8-way absolute-aim + stepwise rotate. Buttons request a compass heading;
        # the game rotates one tick toward it (shortest path). 8 bins make ego
        # FORWARD = current nose direction rather than a coarse L/R strafe.
        "facing_deg_bins": (0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0),
        "facing_deg_bin0": -90.0,
        "facing_deg_bin1": 90.0,
        "bilateral_axis": "horizontal",
    },
    # Kaboom deferred (not in upstream jaxatari yet).
    # "kaboom": {
    #     "facing_deg_bin0": -90.0,
    #     "facing_deg_bin1": -90.0,
    #     "single_view_axis": True,
    #     "bilateral_axis": "vertical",
    # },
}


# JAXAtari's ``ObjectObservation.orientation`` measures 0° = up, 90° = right,
# 180° = down, 270° = left. Every body-frame computation here uses screen angles
# (0° = right, 90° = down, 180° = left, −90° = up), so sprite values need a shift.
_SPRITE_ORIENTATION_OFFSET_DEG: float = -90.0

# JAXAtari also reports orientation 0.0 for "inactive / no direction yet", which
# must not be read as a real facing (it would alias to −90° after the shift).
_SPRITE_ORIENTATION_UNSET_DEG: float = 0.0


def _wrap180_deg(deg: jnp.ndarray) -> jnp.ndarray:
    return (deg + jnp.float32(180.0)) % jnp.float32(360.0) - jnp.float32(180.0)


def view_axes_deg(profile: dict[str, Any]) -> tuple[float, float]:
    """Declared body-facing angles (screen degrees) for view axis 0 and 1."""
    return (
        float(profile.get("facing_deg_bin0", 0.0)),
        float(profile.get("facing_deg_bin1", 180.0)),
    )


def facing_bins_deg(profile: dict[str, Any]) -> tuple[float, ...]:
    """All declared body-facing snap angles (screen degrees).

    Prefer ``facing_deg_bins`` (N-way) when present; otherwise the legacy
    two-axis ``facing_deg_bin0/1`` pair.
    """
    bins = profile.get("facing_deg_bins")
    if bins is not None:
        return tuple(float(x) for x in bins)
    return view_axes_deg(profile)


def has_single_view_axis(profile: dict[str, Any]) -> bool:
    """True when the player body cannot turn, so only view axis 0 is reachable."""
    if bool(profile.get("single_view_axis", False)):
        return True
    bins = facing_bins_deg(profile)
    if len(bins) <= 1:
        return True
    if len(bins) == 2:
        return abs(bins[0] - bins[1]) < 1e-6
    return False


def _facing_bin_idx(raw_obs, profile: dict[str, Any], player_getter) -> jnp.ndarray:
    """Which declared view axis the player body currently points along."""
    if has_single_view_axis(profile):
        return jnp.int32(0)
    if bool(profile.get("court_side_vertical_facing", False)):
        player = player_getter(raw_obs)
        y = jnp.asarray(player.y[-1], dtype=jnp.float32)
        mid = jnp.float32(profile.get("court_midline_y", 108.0))
        # Top court (smaller y) → bin 0 (face down); bottom → bin 1 (face up).
        return jnp.where(y < mid, jnp.int32(0), jnp.int32(1)).astype(jnp.int32)
    bins = jnp.asarray(facing_bins_deg(profile), dtype=jnp.float32)
    ori = jnp.asarray(player_getter(raw_obs).orientation[-1], dtype=jnp.float32)
    screen_deg = ori + jnp.float32(_SPRITE_ORIENTATION_OFFSET_DEG)
    diffs = jnp.abs(_wrap180_deg(screen_deg - bins))
    nearest = jnp.argmin(diffs).astype(jnp.int32)
    # Legacy 2-bin hulls use orientation 0.0 as "unset". Multi-bin facers
    # (Time Pilot 8-way, Asteroids 16-way) legitimately face up at 0° sprite.
    if int(bins.shape[0]) <= 2:
        unset = (
            jnp.abs(ori - jnp.float32(_SPRITE_ORIENTATION_UNSET_DEG)) < jnp.float32(1e-3)
        )
        return jnp.where(unset, jnp.int32(0), nearest)
    return nearest


def _player_orientation_deg(raw_obs, profile: dict[str, Any], player_getter) -> jnp.ndarray:
    """Body facing in screen degrees, snapped to the env's declared view axis.

    This deliberately never returns a raw sprite angle: the observation frame and
    the action mapper both resolve their facing through this function, so keying
    it on the same per-env axis table is what stops the two frames drifting apart.
    """
    bins = jnp.asarray(facing_bins_deg(profile), dtype=jnp.float32)
    bin_idx = _facing_bin_idx(raw_obs, profile, player_getter)
    return bins[bin_idx]


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
        facing_list = (facing0, facing1)
    else:
        facing_list = facing_bins_deg(profile)

    tables = jnp.stack(
        [
            build_facing_action_table(
                noop_action=noop,
                fire_action=fire,
                move_table=move,
                move_fire_table=mfire,
                facing_deg=facing,
            )
            for facing in facing_list
        ],
        axis=0,
    )
    _getter = player_getter or profile.get("player_getter") or _default_get_player
    _noop_i = jnp.int32(noop)
    _fire_i = jnp.int32(fire)
    _n_bins = jnp.int32(len(facing_list))

    def mapper(ego_action, raw_obs, target_idx, objects):
        bin_idx = _mapper_sector_idx(raw_obs, objects, target_idx, profile, _getter)
        bin_idx = jnp.clip(bin_idx, jnp.int32(0), _n_bins - jnp.int32(1))
        return tables[bin_idx, ego_action]

    def inverse_mapper(game_action, raw_obs, target_idx, objects):
        bin_idx = _mapper_sector_idx(raw_obs, objects, target_idx, profile, _getter)
        bin_idx = jnp.clip(bin_idx, jnp.int32(0), _n_bins - jnp.int32(1))
        table = tables[bin_idx]
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
    if str(components.jax_game_id).lower() in {"seaquest", "old_seaquest"}:
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
