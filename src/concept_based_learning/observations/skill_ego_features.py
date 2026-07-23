"""Optional weapon-state channel appended to egocentric skill observations.

When ``USE_EGO_WEAPON_FEATURES`` is enabled, each frame gains exactly one
extra channel — ternary *weapon_state*:

  +1  weapon ready (can fire)
   0  shot in flight, aimed at the target when fired
  -1  shot in flight, aimed away when fired

Aim quality is latched on the can_fire 1→0 edge (within the frame stack) so
bobbing / mid-flight geometry changes cannot flicker the value.

Occupancy histograms list this channel in ``DISCRETE_SKILL_EGO_FLAGS``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

_USE_EGO_WEAPON_FEATURES = False
_ENV_NAME = ""

# Weapon-state channel values.
WEAPON_READY = 1.0
WEAPON_IN_FLIGHT_AIMED = 0.0
WEAPON_IN_FLIGHT_WASTED = -1.0

# Single channel when weapon features are enabled.
WEAPON_CHANNELS_PER_FRAME = 1


class DiscreteSkillEgoFlag(NamedTuple):
    """A discrete channel in the weapon extras block (offset within extras).

    ``n_levels`` is the occupancy alphabet size. The ternary weapon_state uses
    3 with values encoded as ``round(value + 1)`` so {-1, 0, +1} → {0, 1, 2}.
    """

    offset: int
    name: str
    n_levels: int = 2
    level_offset: float = 0.0


DISCRETE_SKILL_EGO_FLAGS: tuple[DiscreteSkillEgoFlag, ...] = (
    DiscreteSkillEgoFlag(
        offset=0,
        name="weapon_state",
        n_levels=3,
        level_offset=1.0,  # {-1,0,+1} → {0,1,2}
    ),
)

# Envs where shot direction follows left/right player facing (Seaquest-style).
# Others default to always-aimed (in-flight → 0), since fire direction is fixed.
_FACING_LR_AIM_ENVS = frozenset({"seaquest", "choppercommand"})


def configure_ego_weapon_features(
    *,
    enabled: bool,
    env_name: str,
    num_skills: int | None = None,
    extra_dim_per_skill: int | None = None,
) -> None:
    """Enable/disable the shared weapon_state ego channel.

    ``num_skills`` / ``extra_dim_per_skill`` are ignored (kept for call-site
    compatibility with older configs that passed a padded per-skill dim).
    """
    del num_skills, extra_dim_per_skill
    global _USE_EGO_WEAPON_FEATURES, _ENV_NAME
    _USE_EGO_WEAPON_FEATURES = bool(enabled)
    _ENV_NAME = str(env_name).lower()


def configure_skill_ego_features(
    *,
    enabled: bool,
    num_skills: int = 3,
    extra_dim_per_skill: int = 1,
    env_name: str = "",
) -> None:
    """Backward-compatible alias for ``configure_ego_weapon_features``."""
    configure_ego_weapon_features(
        enabled=enabled,
        env_name=env_name,
        num_skills=num_skills,
        extra_dim_per_skill=extra_dim_per_skill,
    )


def using_ego_weapon_features() -> bool:
    return _USE_EGO_WEAPON_FEATURES


def using_skill_ego_features() -> bool:
    """Backward-compatible alias for ``using_ego_weapon_features``."""
    return using_ego_weapon_features()


def skill_ego_extra_channels_per_frame() -> int:
    """Extra ego channels/frame from weapon features (0 or 1)."""
    return WEAPON_CHANNELS_PER_FRAME if _USE_EGO_WEAPON_FEATURES else 0


def skill_ego_extra_dim_per_skill() -> int:
    """Backward-compatible alias: weapon extras are a single shared channel."""
    return WEAPON_CHANNELS_PER_FRAME if _USE_EGO_WEAPON_FEATURES else 0


def _stack_len(raw_observation) -> int:
    for name in ("player", "chicken", "agent", "skier"):
        if hasattr(raw_observation, name):
            x = getattr(raw_observation, name).x
            return int(x.shape[0])
    return 1


def _get_player(raw_observation):
    for name in ("player", "chicken", "agent", "skier"):
        if hasattr(raw_observation, name):
            return getattr(raw_observation, name)
    raise AttributeError(
        f"Cannot find a player entity in {type(raw_observation).__name__}."
    )


def _player_projectile_active_per_frame(
    active: jnp.ndarray, projectile_slot: int | None
) -> jnp.ndarray:
    """Reduce projectile.active to one in-flight scalar per stacked frame."""
    active = active.astype(jnp.float32)
    if active.ndim == 1:
        return active
    if projectile_slot is not None:
        return active[:, projectile_slot]
    return jnp.max(active, axis=-1)


def _facing_toward_target_lr(
    player_ori: jnp.ndarray,
    player_x: jnp.ndarray,
    player_w: jnp.ndarray,
    target_x: jnp.ndarray,
    target_w: jnp.ndarray,
) -> jnp.ndarray:
    """
    True when left/right facing matches the target's horizontal side.

    Seaquest convention (also used by the Seaquest action mapper):
    ``orientation < 180`` → facing right.
    """
    facing_right = player_ori < 180.0
    p_cx = player_x.astype(jnp.float32) + player_w.astype(jnp.float32) / 2.0
    t_cx = target_x.astype(jnp.float32) + target_w.astype(jnp.float32) / 2.0
    target_right = t_cx >= p_cx
    return jnp.logical_not(jnp.logical_xor(facing_right, target_right))


def _aim_ok_per_frame(
    raw_observation,
    objects: Any | None,
    target_idx: jnp.ndarray | int | None,
    stack_len: int,
) -> jnp.ndarray:
    """Per-frame bool: facing aimed at the target (or always True)."""
    if _ENV_NAME not in _FACING_LR_AIM_ENVS or objects is None or target_idx is None:
        return jnp.ones((stack_len,), dtype=jnp.bool_)

    player = _get_player(raw_observation)
    if not hasattr(player, "orientation"):
        return jnp.ones((stack_len,), dtype=jnp.bool_)

    idx = jnp.asarray(target_idx, dtype=jnp.int32)
    return _facing_toward_target_lr(
        player.orientation.astype(jnp.float32),
        player.x,
        player.width,
        objects.x[:, idx],
        objects.width[:, idx],
    )


def latch_weapon_state(
    can_fire: jnp.ndarray, aim_ok: jnp.ndarray
) -> jnp.ndarray:
    """
    Ternary weapon_state over a frame stack with fire-time aim latching.

    On each can_fire 1→0 edge, latch 0 (aimed) or -1 (wasted) from ``aim_ok``
    at that frame and hold until the weapon is ready again. Scan init treats a
    stack that begins already in-flight as a fire edge on frame 0, so the
    oldest in-flight frame supplies the latch when the true fire left the window.
    """
    can_fire = can_fire.astype(jnp.float32)
    aim_ok = aim_ok.astype(jnp.bool_)

    def _step(carry, xs):
        prev_can_fire, latch = carry
        cf, aimed = xs
        just_fired = jnp.logical_and(prev_can_fire > 0.5, cf <= 0.5)
        fired_latch = jnp.where(aimed, WEAPON_IN_FLIGHT_AIMED, WEAPON_IN_FLIGHT_WASTED)
        latch = jnp.where(just_fired, fired_latch, latch)
        latch = jnp.where(cf > 0.5, WEAPON_IN_FLIGHT_AIMED, latch)
        out = jnp.where(cf > 0.5, WEAPON_READY, latch)
        return (cf, latch), out

    # prev_can_fire=1 makes an already-in-flight frame 0 count as just_fired.
    init = (jnp.float32(1.0), jnp.float32(WEAPON_IN_FLIGHT_AIMED))
    _, weapon_state = jax.lax.scan(_step, init, (can_fire, aim_ok))
    return weapon_state.astype(jnp.float32)


def _weapon_state_channel_stacked(
    raw_observation,
    objects: Any | None,
    target_idx: jnp.ndarray | int | None,
    *,
    projectile_attr: str,
    projectile_slot: int | None = None,
) -> jnp.ndarray:
    """Per-frame weapon_state, shape [stack_len, 1]."""
    stack_len = _stack_len(raw_observation)
    if not hasattr(raw_observation, projectile_attr):
        return jnp.full((stack_len, 1), WEAPON_READY, dtype=jnp.float32)

    projectiles = getattr(raw_observation, projectile_attr)
    player_proj_active = _player_projectile_active_per_frame(
        projectiles.active, projectile_slot
    )
    can_fire = 1.0 - player_proj_active
    aim_ok = _aim_ok_per_frame(raw_observation, objects, target_idx, stack_len)
    weapon_state = latch_weapon_state(can_fire, aim_ok)
    return weapon_state[:, None]


def build_skill_ego_extras_stacked(
    raw_observation,
    objects: Any | None = None,
    target_idx: jnp.ndarray | int | None = None,
) -> jnp.ndarray:
    """
    Per-frame weapon extras. Shape ``[stack_len, 1]`` when enabled, else
    ``[stack_len, 0]``.

    ``objects`` / ``target_idx`` are required for fire-time aim latching on
    facing-dependent envs (Seaquest, Chopper Command); other envs ignore them.
    """
    stack_len = _stack_len(raw_observation)
    if not _USE_EGO_WEAPON_FEATURES:
        return jnp.zeros((stack_len, 0), dtype=jnp.float32)

    common = dict(objects=objects, target_idx=target_idx)

    if _ENV_NAME == "seaquest":
        return _weapon_state_channel_stacked(
            raw_observation,
            projectile_attr="projectiles",
            projectile_slot=0,
            **common,
        )
    if _ENV_NAME == "phoenix":
        return _weapon_state_channel_stacked(
            raw_observation,
            projectile_attr="player_projectile",
            **common,
        )
    if _ENV_NAME == "spaceinvaders":
        return _weapon_state_channel_stacked(
            raw_observation,
            projectile_attr="player_bullet",
            **common,
        )
    if _ENV_NAME == "choppercommand":
        return _weapon_state_channel_stacked(
            raw_observation,
            projectile_attr="player_missiles",
            projectile_slot=0,
            **common,
        )
    if _ENV_NAME == "beamrider":
        return _weapon_state_channel_stacked(
            raw_observation,
            projectile_attr="player_shots",
            projectile_slot=0,
            **common,
        )

    return jnp.full((stack_len, 1), WEAPON_READY, dtype=jnp.float32)


def _latest_can_fire(
    raw_observation, projectile_attr: str, projectile_slot: int | None = None
) -> jnp.ndarray:
    if not hasattr(raw_observation, projectile_attr):
        return jnp.array(1.0, dtype=jnp.float32)
    active = getattr(raw_observation, projectile_attr).active[-1]
    in_flight = _player_projectile_active_per_frame(
        jnp.expand_dims(active, axis=0), projectile_slot
    )[0]
    return 1.0 - in_flight


def seaquest_can_fire(raw_observation) -> jnp.ndarray:
    """Latest-frame can_fire helper for action masking (Seaquest)."""
    return _latest_can_fire(raw_observation, "projectiles", projectile_slot=0)


def phoenix_can_fire(raw_observation) -> jnp.ndarray:
    """Latest-frame can_fire helper (Phoenix player projectile)."""
    return _latest_can_fire(raw_observation, "player_projectile")


def spaceinvaders_can_fire(raw_observation) -> jnp.ndarray:
    """Latest-frame can_fire helper (Space Invaders player bullet)."""
    return _latest_can_fire(raw_observation, "player_bullet")


def choppercommand_can_fire(raw_observation) -> jnp.ndarray:
    """Latest-frame can_fire helper (Chopper Command player missile)."""
    return _latest_can_fire(raw_observation, "player_missiles", projectile_slot=0)


def beamrider_can_fire(raw_observation) -> jnp.ndarray:
    """Latest-frame can_fire helper (Beamrider player shot)."""
    return _latest_can_fire(raw_observation, "player_shots", projectile_slot=0)
