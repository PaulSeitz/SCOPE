"""Optional weapon ego channels appended to skill observations.

When ``USE_EGO_WEAPON_FEATURES`` is enabled, each frame gains two channels:

  can_fire       +1 weapon ready / -1 shot in flight
  facing_enemy   +1 facing the target's L/R side / -1 not
                 (always +1 on envs whose shot direction is not L/R-gated)

``facing_enemy`` is pure geometry (player ori + player/target centres). It is
always visible — not latched on fire — so skills can turn before shooting.
Occupancy histograms list both channels in ``DISCRETE_SKILL_EGO_FLAGS``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

_USE_EGO_WEAPON_FEATURES = False
_ENV_NAME = ""

# Channel values (±1 binaries).
CAN_FIRE_READY = 1.0
CAN_FIRE_IN_FLIGHT = -1.0
FACING_ENEMY_YES = 1.0
FACING_ENEMY_NO = -1.0

# Backward-compat aliases for older tests / imports.
WEAPON_READY = CAN_FIRE_READY
WEAPON_IN_FLIGHT_AIMED = 0.0  # legacy ternary mid value
WEAPON_IN_FLIGHT_WASTED = CAN_FIRE_IN_FLIGHT

# Two channels when weapon features are enabled.
WEAPON_CHANNELS_PER_FRAME = 2


class DiscreteSkillEgoFlag(NamedTuple):
    """A discrete channel in the weapon extras block (offset within extras).

    ``n_levels`` is the occupancy alphabet size. ±1 binaries use
    ``level_offset=1`` so ``round(value + 1)`` maps {-1,+1} → {0,2}, then
    clamped to {0,1}.
    """

    offset: int
    name: str
    n_levels: int = 2
    level_offset: float = 0.0


DISCRETE_SKILL_EGO_FLAGS: tuple[DiscreteSkillEgoFlag, ...] = (
    DiscreteSkillEgoFlag(
        offset=0,
        name="can_fire",
        n_levels=2,
        level_offset=1.0,  # {-1,+1} → {0,1} after clamp
    ),
    DiscreteSkillEgoFlag(
        offset=1,
        name="facing_enemy",
        n_levels=2,
        level_offset=1.0,
    ),
)

# Envs where shot direction follows left/right player facing (Seaquest-style).
_FACING_LR_AIM_ENVS = frozenset({"seaquest", "choppercommand"})


def configure_ego_weapon_features(
    *,
    enabled: bool,
    env_name: str,
    num_skills: int | None = None,
    extra_dim_per_skill: int | None = None,
) -> None:
    """Enable/disable the shared can_fire + facing_enemy ego channels.

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
    """Extra ego channels/frame from weapon features (0 or 2)."""
    return WEAPON_CHANNELS_PER_FRAME if _USE_EGO_WEAPON_FEATURES else 0


def skill_ego_extra_dim_per_skill() -> int:
    """Backward-compatible alias: weapon extras are shared channels."""
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

    Raw-geometry only. Works for jaxatari L/R oris (90=right, 270=left) and
    for 0/180: ``orientation < 180`` → facing right.
    """
    facing_right = player_ori < 180.0
    p_cx = player_x.astype(jnp.float32) + player_w.astype(jnp.float32) / 2.0
    t_cx = target_x.astype(jnp.float32) + target_w.astype(jnp.float32) / 2.0
    target_right = t_cx >= p_cx
    return jnp.logical_not(jnp.logical_xor(facing_right, target_right))


def _facing_enemy_per_frame(
    raw_observation,
    objects: Any | None,
    target_idx: jnp.ndarray | int | None,
    stack_len: int,
) -> jnp.ndarray:
    """Per-frame ±1 facing_enemy (always +1 when L/R aim does not apply)."""
    if _ENV_NAME not in _FACING_LR_AIM_ENVS or objects is None or target_idx is None:
        return jnp.full((stack_len,), FACING_ENEMY_YES, dtype=jnp.float32)

    player = _get_player(raw_observation)
    if not hasattr(player, "orientation"):
        return jnp.full((stack_len,), FACING_ENEMY_YES, dtype=jnp.float32)

    idx = jnp.asarray(target_idx, dtype=jnp.int32)
    aimed = _facing_toward_target_lr(
        player.orientation.astype(jnp.float32),
        player.x,
        player.width,
        objects.x[:, idx],
        objects.width[:, idx],
    )
    return jnp.where(aimed, FACING_ENEMY_YES, FACING_ENEMY_NO).astype(jnp.float32)


def latch_weapon_state(
    can_fire: jnp.ndarray, aim_ok: jnp.ndarray
) -> jnp.ndarray:
    """Legacy ternary latch kept for unit tests; not used in ego extras."""
    import jax

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

    init = (jnp.float32(1.0), jnp.float32(WEAPON_IN_FLIGHT_AIMED))
    _, weapon_state = jax.lax.scan(_step, init, (can_fire, aim_ok))
    return weapon_state.astype(jnp.float32)


def _weapon_channels_stacked(
    raw_observation,
    objects: Any | None,
    target_idx: jnp.ndarray | int | None,
    *,
    projectile_attr: str,
    projectile_slot: int | None = None,
) -> jnp.ndarray:
    """Per-frame [can_fire, facing_enemy], shape [stack_len, 2]."""
    stack_len = _stack_len(raw_observation)
    if not hasattr(raw_observation, projectile_attr):
        can_fire = jnp.full((stack_len,), CAN_FIRE_READY, dtype=jnp.float32)
    else:
        projectiles = getattr(raw_observation, projectile_attr)
        player_proj_active = _player_projectile_active_per_frame(
            projectiles.active, projectile_slot
        )
        can_fire = jnp.where(
            player_proj_active > 0.5,
            CAN_FIRE_IN_FLIGHT,
            CAN_FIRE_READY,
        ).astype(jnp.float32)

    facing = _facing_enemy_per_frame(
        raw_observation, objects, target_idx, stack_len
    )
    return jnp.stack([can_fire, facing], axis=-1)


def build_skill_ego_extras_stacked(
    raw_observation,
    objects: Any | None = None,
    target_idx: jnp.ndarray | int | None = None,
) -> jnp.ndarray:
    """
    Per-frame weapon extras. Shape ``[stack_len, 2]`` when enabled, else
    ``[stack_len, 0]``.

    ``objects`` / ``target_idx`` drive ``facing_enemy`` on L/R envs
    (Seaquest, Chopper Command); other envs keep ``facing_enemy=+1``.
    """
    stack_len = _stack_len(raw_observation)
    if not _USE_EGO_WEAPON_FEATURES:
        return jnp.zeros((stack_len, 0), dtype=jnp.float32)

    common = dict(objects=objects, target_idx=target_idx)

    if _ENV_NAME == "seaquest":
        return _weapon_channels_stacked(
            raw_observation,
            projectile_attr="projectiles",
            projectile_slot=0,
            **common,
        )
    if _ENV_NAME == "phoenix":
        return _weapon_channels_stacked(
            raw_observation,
            projectile_attr="player_projectile",
            **common,
        )
    if _ENV_NAME == "spaceinvaders":
        return _weapon_channels_stacked(
            raw_observation,
            projectile_attr="player_bullet",
            **common,
        )
    if _ENV_NAME == "choppercommand":
        return _weapon_channels_stacked(
            raw_observation,
            projectile_attr="player_missiles",
            projectile_slot=0,
            **common,
        )
    if _ENV_NAME == "beamrider":
        return _weapon_channels_stacked(
            raw_observation,
            projectile_attr="player_shots",
            projectile_slot=0,
            **common,
        )

    return jnp.stack(
        [
            jnp.full((stack_len,), CAN_FIRE_READY, dtype=jnp.float32),
            jnp.full((stack_len,), FACING_ENEMY_YES, dtype=jnp.float32),
        ],
        axis=-1,
    )


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
