"""Persistent-heading turn primitives on top of the POV-relative mapper.

Enabled with ``USE_POV_TURN_PRIMITIVES`` (requires ``USE_POV_RELATIVE_MAPPER``).

Instead of picking one of 18 instantaneous body-relative compass intentions per
step, each environment carries an internal heading θ_h that integrates the chosen
turns over time.  The ego action space shrinks to 6 motion primitives::

    0 NOOP          θ_h unchanged, game NOOP
    1 FIRE          θ_h unchanged, stationary fire
    2 FORWARD       move along θ_h
    3 TURN_LEFT     θ_h -= step, then move along the new θ_h
    4 TURN_RIGHT    θ_h += step, then move along the new θ_h
    5 FORWARD_FIRE  move along θ_h while firing

Turns are not free frames: the agent commits to a rotation and keeps moving, so
every primitive makes progress (the classic differential-drive formulation).

The mapper only ever sees ``(θ_h, ego_action)`` — no target geometry — so the
action space stays as target-agnostic as the plain POV mapper, preserving its
transfer property.  What changes is the learning problem: instead of 18-way angle
classification the network only has to servo ψ = θ_h − θ_view toward 0, which is
closed-loop proportional control.

θ_h is *not* the sprite orientation.  When turn primitives are active the POV ego
observation derives ψ from θ_h and ignores ``player.orientation`` entirely, which
is what closes the loop: the network sees the heading it has been commanding.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from concept_based_learning.observations.action_space import (
    _nearest_world_dir,
    _default_get_player,
)
from concept_based_learning.observations.pov_relative import (
    _MOVE_SPECS,
    _POV_PROFILES,
)

# Ego action indices for the turn-primitive space.
TURN_NOOP = 0
TURN_FIRE = 1
TURN_FORWARD = 2
TURN_LEFT = 3
TURN_RIGHT = 4
TURN_FORWARD_FIRE = 5

TURN_EGO_ACTION_DIM = 6
TURN_EGO_ACTION_NAMES = (
    "NOOP",
    "FIRE",
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "FORWARD_FIRE",
)

# Fire-capable primitives (mirrors action_space.fire_ego_action_indices).
TURN_FIRE_EGO_ACTION_INDICES = jnp.array([TURN_FIRE, TURN_FORWARD_FIRE], dtype=jnp.int32)

DEFAULT_TURN_STEP_DEG = 45.0

_USE_POV_TURN: bool = False
_TURN_STEP_DEG: float = DEFAULT_TURN_STEP_DEG
_INITIAL_HEADING_DEG: float = 0.0


def using_pov_turn_primitives() -> bool:
    return _USE_POV_TURN


def turn_step_deg() -> float:
    return _TURN_STEP_DEG


def initial_heading_deg() -> float:
    """Env-specific heading the integrator starts from (and resets to)."""
    return _INITIAL_HEADING_DEG


def default_initial_heading_deg(env_name: str | None) -> float:
    """Bin-0 facing of the POV profile: the env's natural "forward"."""
    profile = _POV_PROFILES.get(str(env_name or "").lower())
    if profile is None:
        return 0.0
    return float(profile.get("facing_deg_bin0", 0.0))


def configure_pov_turn_from_config(config: dict | None) -> None:
    """Latch turn-primitive knobs from a run config (mirrors the other ego setters)."""
    global _USE_POV_TURN, _TURN_STEP_DEG, _INITIAL_HEADING_DEG

    cfg = config or {}
    enabled = bool(cfg.get("USE_POV_TURN_PRIMITIVES", False))
    if enabled and not bool(cfg.get("USE_POV_RELATIVE_MAPPER", False)):
        raise ValueError(
            "USE_POV_TURN_PRIMITIVES requires USE_POV_RELATIVE_MAPPER: the turn "
            "integrator replaces the POV facing bin, it is not a standalone mapper."
        )
    if enabled and bool(cfg.get("RELABEL_ALL_OBJECTS", False)):
        raise ValueError(
            "USE_POV_TURN_PRIMITIVES is incompatible with RELABEL_ALL_OBJECTS: the "
            "inverse mapper cannot recover which turn produced a game action."
        )
    _USE_POV_TURN = enabled

    step = cfg.get("POV_TURN_STEP_DEG")
    _TURN_STEP_DEG = DEFAULT_TURN_STEP_DEG if step is None else float(step)
    if enabled and not (0.0 < _TURN_STEP_DEG <= 180.0):
        raise ValueError(f"POV_TURN_STEP_DEG must be in (0, 180], got {_TURN_STEP_DEG}")

    override = cfg.get("POV_TURN_INITIAL_HEADING_DEG")
    _INITIAL_HEADING_DEG = (
        default_initial_heading_deg(cfg.get("ENV_NAME"))
        if override is None
        else float(override)
    )


def wrap_deg(angle_deg: jnp.ndarray) -> jnp.ndarray:
    """Wrap degrees to [-180, 180)."""
    return (jnp.asarray(angle_deg, dtype=jnp.float32) + 180.0) % 360.0 - 180.0


def heading_after(
    heading_deg: jnp.ndarray,
    ego_action: jnp.ndarray,
    *,
    step_deg: float | None = None,
) -> jnp.ndarray:
    """Integrate one turn primitive into the persistent heading.

    Screen-Y-down convention: TURN_RIGHT is clockwise (increasing angle).
    """
    step = jnp.float32(_TURN_STEP_DEG if step_deg is None else step_deg)
    ego = jnp.asarray(ego_action, dtype=jnp.int32)
    delta = jnp.where(
        ego == TURN_LEFT,
        -step,
        jnp.where(ego == TURN_RIGHT, step, jnp.float32(0.0)),
    )
    return wrap_deg(jnp.asarray(heading_deg, dtype=jnp.float32) + delta)


def build_turn_action_fn(
    *,
    noop_action: int,
    fire_action: int,
    move_table,
    move_fire_table,
) -> Callable:
    """Return ``(ego_action, heading_deg) -> game_action`` for one env's move spec."""
    _noop = jnp.int32(noop_action)
    _fire = jnp.int32(fire_action)
    _move = jnp.asarray(move_table, dtype=jnp.int32)
    _mfir = jnp.asarray(move_fire_table, dtype=jnp.int32)

    def turn_action(ego_action: jnp.ndarray, heading_deg: jnp.ndarray) -> jnp.ndarray:
        ego = jnp.asarray(ego_action, dtype=jnp.int32)
        dir_idx = _nearest_world_dir(jnp.asarray(heading_deg, dtype=jnp.float32))
        moving = jnp.logical_and(ego >= TURN_FORWARD, ego <= TURN_FORWARD_FIRE)
        with_fire = ego == TURN_FORWARD_FIRE
        game_action = jnp.where(with_fire, _mfir[dir_idx], _move[dir_idx])
        stationary = jnp.where(ego == TURN_FIRE, _fire, _noop)
        return jnp.where(moving, game_action, stationary).astype(jnp.int32)

    return turn_action


def make_turn_primitive_mappers(
    jax_game_id: str,
    *,
    player_getter: Callable | None = None,
) -> tuple[Callable, Callable]:
    """Return ``(mapper, inverse_mapper)`` for the turn-primitive action space.

    ``mapper(ego_action, heading_deg, raw_obs, target_idx, objects)`` takes the
    heading *after* the turn was integrated, so callers must apply
    :func:`heading_after` first.

    The inverse mapper exists only to satisfy call sites that expect the pair; a
    game action does not determine which turn produced it, so it degrades to NOOP.
    Guarded against silent misuse by rejecting ``RELABEL_ALL_OBJECTS`` in
    :func:`configure_pov_turn_from_config`.
    """
    game_key = str(jax_game_id).lower()
    if game_key not in _MOVE_SPECS:
        raise ValueError(
            f"No POV move spec for jax_game_id={jax_game_id!r}. "
            f"Known: {sorted(_MOVE_SPECS)}"
        )
    noop, fire, move, mfire = _MOVE_SPECS[game_key]
    turn_action = build_turn_action_fn(
        noop_action=noop,
        fire_action=fire,
        move_table=move,
        move_fire_table=mfire,
    )
    del player_getter

    def mapper(ego_action, heading_deg, raw_obs, target_idx, objects):
        del raw_obs, target_idx, objects
        return turn_action(ego_action, heading_deg)

    def inverse_mapper(game_action, raw_obs, target_idx, objects):
        del raw_obs, target_idx, objects
        return jnp.zeros_like(jnp.asarray(game_action, dtype=jnp.int32))

    return mapper, inverse_mapper


def resolve_turn_mapper(components, config: dict) -> Callable | None:
    """Return the 5-arg turn mapper for this env, or ``None`` when the flag is off."""
    if not bool((config or {}).get("USE_POV_TURN_PRIMITIVES", False)):
        return None

    base_mapper, _inverse = make_turn_primitive_mappers(components.jax_game_id)
    if str(components.jax_game_id).lower() != "seaquest":
        return base_mapper

    from concept_based_learning.observations.pov_relative import (
        apply_seaquest_ascent_block,
    )

    def mapper(ego_action, heading_deg, raw_obs, target_idx, objects):
        return apply_seaquest_ascent_block(
            base_mapper(ego_action, heading_deg, raw_obs, target_idx, objects),
            raw_obs,
            target_idx,
        )

    return mapper


def make_turn_legal_mask_fn(
    turn_mapper,
    *,
    noop_action: int = 0,
    fire_action: int = 1,
) -> Callable:
    """Mask over turn primitives whose movement the env can actually realize.

    ``legal_mask(heading_deg)`` — NOOP / FIRE are always legal; a moving primitive
    is legal iff its mapped game action is not the env's NOOP (e.g. a horizontal
    heading in Pong maps to NOOP and stalls the paddle).
    """
    _noop = jnp.int32(noop_action)
    _fire = jnp.int32(fire_action)
    _distinct_fire = bool(int(fire_action) != int(noop_action))

    def legal_mask(heading_deg):
        egos = jnp.arange(TURN_EGO_ACTION_DIM, dtype=jnp.int32)

        def _one(ego):
            heading = heading_after(heading_deg, ego)
            g = turn_mapper(ego, heading, None, None, None)
            stationary = jnp.logical_or(ego == TURN_NOOP, ego == TURN_FIRE)
            with_fire = ego == TURN_FORWARD_FIRE
            ok_move = g != _noop
            ok_move_fire = jnp.where(
                _distinct_fire,
                jnp.logical_and(g != _noop, g != _fire),
                g != _noop,
            )
            return jnp.where(
                stationary, True, jnp.where(with_fire, ok_move_fire, ok_move)
            )

        return jax.vmap(_one)(egos)

    return legal_mask
