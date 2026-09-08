"""Optional weapon ego channels appended to skill observations.

When ``USE_EGO_WEAPON_FEATURES`` is enabled, each frame gains:

  can_fire       +1 weapon ready / -1 shot in flight
                 (Phoenix: +1 shield active / -1 shield off — visible ability
                 sprite via ``player.state``, not the hidden cooldown timer)
  facing_enemy   +1 weapon can plausibly hit (L/R facing matches *and*
                 centres share a vertical band) / -1 not
                 (always +1 on envs whose shot direction is not L/R-gated)
                 Optional: set ``USE_EGO_FACING_ENEMY_FEATURES: false`` to
                 keep ``can_fire`` alone (channel-deletion ablation).

``facing_enemy`` is pure geometry (player ori + player/target centres). It is
always visible — not latched on fire — so skills can turn before shooting.
Nearly identical x (``|Δx| < 2px``) skips the L/R check so co-located targets
do not invent a rightward bias, but a vertical band (``|Δy| < 28px``) is
still required so overhead threats (falling coco, surface WP) stay un-aimed.
Occupancy histograms list both channels in ``DISCRETE_SKILL_EGO_FLAGS``.

``USE_EGO_SHOT_BETWEEN`` (requires weapon features) appends one *binary*
channel: the player projectile currently sits on the open player→target
segment (same occupancy idea as ``los_blocked`` / blockers). +1 when the
missile centre projects onto that open segment within a missile-width
corridor; 0 when there is no bullet, the shot is off the corridor (including
the wrong way), or the target is not DESTROY. Zeros on games without
bullets (Pong / Freeway). Current geometry only — not a fire-time latch.

``USE_EGO_SHOT_FEATURES`` (requires weapon features) appends two *continuous*
channels that locate the player projectile relative to the skill target:

  shot_dx, shot_dy   (target_centre - missile_centre) / screen, clipped [-1, 1]
                     zeros when there is no projectile attr, the slot is
                     inactive, or the env has no bullets (Pong / Freeway).

Channel count is therefore identical across games: missing bullets are a
zero vector, not a missing dim, so transfer is not OOD in shape. Looking
direction is already in ``facing_enemy`` + ``sin/cos(ψ)``; shot geometry is
the missing “where is my missile going” cue.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_DESTROY
from concept_based_learning.observations.line_of_sight import missile_on_open_segment

_USE_EGO_WEAPON_FEATURES = False
_USE_EGO_FACING_ENEMY_FEATURES = True  # only applies when weapon extras are on
_USE_EGO_SHOT_FEATURES = False
_USE_EGO_SHOT_BETWEEN = False
_FORCE_CAN_FIRE: float | None = None  # diagnostic override (±1); None = live
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

# Base weapon extras: can_fire always; facing_enemy optional (see flag).
# Shot geometry is optional on top.
WEAPON_CAN_FIRE_CHANNELS_PER_FRAME = 1
WEAPON_FACING_ENEMY_CHANNELS_PER_FRAME = 1
WEAPON_BASE_CHANNELS_PER_FRAME = 2  # can_fire + facing_enemy (legacy full set)
SHOT_BETWEEN_CHANNELS_PER_FRAME = 1
SHOT_CHANNELS_PER_FRAME = 2
# Backward-compat alias: base extras only (shot flag off).
WEAPON_CHANNELS_PER_FRAME = WEAPON_BASE_CHANNELS_PER_FRAME

# Screen scales for shot Δx/Δy (Atari native). Avoid importing egocentric
# (circular: egocentric → this module).
_SHOT_DX_SCALE = 160.0
_SHOT_DY_SCALE = 210.0


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


def _weapon_base_channel_count() -> int:
    """Active base weapon channels (can_fire ± optional facing_enemy)."""
    if not _USE_EGO_WEAPON_FEATURES:
        return 0
    n = WEAPON_CAN_FIRE_CHANNELS_PER_FRAME
    if _USE_EGO_FACING_ENEMY_FEATURES:
        n += WEAPON_FACING_ENEMY_CHANNELS_PER_FRAME
    return n


def discrete_skill_ego_flags() -> tuple[DiscreteSkillEgoFlag, ...]:
    """Active discrete weapon extras (can_fire / shield_active, facing, optional shot_between)."""
    if not _USE_EGO_WEAPON_FEATURES:
        return ()
    can_fire_name = "shield_active" if _ENV_NAME == "phoenix" else "can_fire"
    flags: tuple[DiscreteSkillEgoFlag, ...] = (
        DiscreteSkillEgoFlag(
            offset=0,
            name=can_fire_name,
            n_levels=2,
            level_offset=1.0,  # {-1,+1} → {0,1} after clamp
        ),
    )
    if _USE_EGO_FACING_ENEMY_FEATURES:
        flags = flags + (
            DiscreteSkillEgoFlag(
                offset=1,
                name="facing_enemy",
                n_levels=2,
                level_offset=1.0,
            ),
        )
    if _USE_EGO_SHOT_BETWEEN:
        flags = flags + (
            DiscreteSkillEgoFlag(
                offset=_weapon_base_channel_count(),
                name="shot_between",
                n_levels=2,
                level_offset=0.0,  # {0,1}
            ),
        )
    return flags


# Keep the module-level registry for docs / older imports; runtime uses
# ``discrete_skill_ego_flags()`` so Phoenix can rename the first channel.
DISCRETE_SKILL_EGO_FLAGS: tuple[DiscreteSkillEgoFlag, ...] = (
    DiscreteSkillEgoFlag(
        offset=0,
        name="can_fire",
        n_levels=2,
        level_offset=1.0,
    ),
    DiscreteSkillEgoFlag(
        offset=1,
        name="facing_enemy",
        n_levels=2,
        level_offset=1.0,
    ),
)
_SHOT_BETWEEN_FLAG = DiscreteSkillEgoFlag(
    offset=WEAPON_BASE_CHANNELS_PER_FRAME,
    name="shot_between",
    n_levels=2,
    level_offset=0.0,  # {0,1}
)
# Envs where the weapon / punch aims along left/right player facing.
_FACING_LR_AIM_ENVS = frozenset({"seaquest", "choppercommand", "kangaroo"})


def configure_ego_weapon_features(
    *,
    enabled: bool,
    env_name: str,
    num_skills: int | None = None,
    extra_dim_per_skill: int | None = None,
    shot_features: bool = False,
    shot_between: bool = False,
    facing_enemy: bool = True,
    force_can_fire: float | None = None,
) -> None:
    """Enable/disable shared weapon ego channels.

    When ``enabled``, always emits ``can_fire`` (or Phoenix ``shield_active``).
    ``facing_enemy`` (default True) appends the L/R aim geometry bit; set False
    to keep ``can_fire`` alone (channel-deletion ablation).

    ``force_can_fire`` (diagnostic): if set to ±1, overrides the live can_fire /
    shield channel every frame (used to probe transfer sensitivity to the
    constant +1 Pong sees). ``None`` keeps the live signal.

    ``shot_features`` adds shot-to-target Δx/Δy (zeros when no bullet).
    ``shot_between`` adds a DESTROY-only 0/1 occupancy bit (missile on the
    open player→target segment).
    ``num_skills`` / ``extra_dim_per_skill`` are ignored (kept for call-site
    compatibility with older configs that passed a padded per-skill dim).
    Different skill in-dims use ``USE_DESTROY_ONLY_EGO_EXTRAS`` (pose for
    COLLECT/AVOID/NAVIGATE; extras concatenated only on DESTROY).
    """
    del num_skills, extra_dim_per_skill
    global _USE_EGO_WEAPON_FEATURES
    global _USE_EGO_FACING_ENEMY_FEATURES
    global _USE_EGO_SHOT_FEATURES
    global _USE_EGO_SHOT_BETWEEN
    global _FORCE_CAN_FIRE
    global _ENV_NAME
    _USE_EGO_WEAPON_FEATURES = bool(enabled)
    _USE_EGO_FACING_ENEMY_FEATURES = bool(enabled) and bool(facing_enemy)
    _USE_EGO_SHOT_FEATURES = bool(enabled) and bool(shot_features)
    _USE_EGO_SHOT_BETWEEN = bool(enabled) and bool(shot_between)
    if force_can_fire is None or force_can_fire is False or force_can_fire == "":
        _FORCE_CAN_FIRE = None
    else:
        v = float(force_can_fire)
        if abs(abs(v) - 1.0) > 1e-5:
            raise ValueError(
                f"EGO_FORCE_CAN_FIRE must be ±1 or null, got {force_can_fire!r}"
            )
        _FORCE_CAN_FIRE = 1.0 if v > 0 else -1.0
    _ENV_NAME = str(env_name).lower()


def configure_skill_ego_features(
    *,
    enabled: bool,
    num_skills: int = 3,
    extra_dim_per_skill: int = 1,
    env_name: str = "",
    shot_features: bool = False,
    shot_between: bool = False,
    facing_enemy: bool = True,
    force_can_fire: float | None = None,
) -> None:
    """Backward-compatible alias for ``configure_ego_weapon_features``."""
    configure_ego_weapon_features(
        enabled=enabled,
        env_name=env_name,
        num_skills=num_skills,
        extra_dim_per_skill=extra_dim_per_skill,
        shot_features=shot_features,
        shot_between=shot_between,
        facing_enemy=facing_enemy,
        force_can_fire=force_can_fire,
    )


def using_ego_weapon_features() -> bool:
    return _USE_EGO_WEAPON_FEATURES


def using_ego_facing_enemy_features() -> bool:
    return _USE_EGO_FACING_ENEMY_FEATURES


def using_ego_shot_features() -> bool:
    return _USE_EGO_SHOT_FEATURES


def using_ego_shot_between() -> bool:
    return _USE_EGO_SHOT_BETWEEN


def using_skill_ego_features() -> bool:
    """Backward-compatible alias for ``using_ego_weapon_features``."""
    return using_ego_weapon_features()


def skill_ego_extra_channels_per_frame() -> int:
    """Extra ego channels/frame from weapon features (0–6)."""
    if not _USE_EGO_WEAPON_FEATURES:
        return 0
    extra = _weapon_base_channel_count()
    if _USE_EGO_SHOT_BETWEEN:
        extra += SHOT_BETWEEN_CHANNELS_PER_FRAME
    if _USE_EGO_SHOT_FEATURES:
        extra += SHOT_CHANNELS_PER_FRAME
    return extra


def skill_ego_extra_dim_per_skill() -> int:
    """Backward-compatible alias: weapon extras are shared channels."""
    return skill_ego_extra_channels_per_frame()


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


def _leaf_per_frame(arr, projectile_slot: int | None, stack_len: int) -> jnp.ndarray:
    """Take one projectile-slot scalar per stacked frame from x/y/w/h."""
    arr = jnp.asarray(arr, dtype=jnp.float32)
    if arr.ndim == 0:
        return jnp.full((stack_len,), arr)
    if arr.ndim == 1:
        if arr.shape[0] == stack_len:
            return arr
        if projectile_slot is not None:
            return jnp.full((stack_len,), arr[projectile_slot])
        return jnp.broadcast_to(arr[0], (stack_len,))
    if projectile_slot is not None:
        return arr[:, projectile_slot]
    return arr[:, 0]


def _shot_to_target_xy(
    raw_observation,
    objects: Any | None,
    target_idx: jnp.ndarray | int | None,
    *,
    projectile_attr: str,
    projectile_slot: int | None,
    stack_len: int,
    in_flight: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-frame (Δx, Δy) from missile centre to target centre; 0 if no shot."""
    zeros = jnp.zeros((stack_len,), dtype=jnp.float32)
    if not _USE_EGO_SHOT_FEATURES:
        return zeros, zeros
    if objects is None or target_idx is None:
        return zeros, zeros
    if not hasattr(raw_observation, projectile_attr):
        return zeros, zeros
    projectiles = getattr(raw_observation, projectile_attr)
    if not hasattr(projectiles, "x") or not hasattr(projectiles, "y"):
        return zeros, zeros

    mx = _leaf_per_frame(projectiles.x, projectile_slot, stack_len)
    my = _leaf_per_frame(projectiles.y, projectile_slot, stack_len)
    mw = _leaf_per_frame(getattr(projectiles, "width", 0.0), projectile_slot, stack_len)
    mh = _leaf_per_frame(getattr(projectiles, "height", 0.0), projectile_slot, stack_len)
    m_cx = mx + mw / 2.0
    m_cy = my + mh / 2.0

    idx = jnp.asarray(target_idx, dtype=jnp.int32)
    ox = objects.x.astype(jnp.float32)
    oy = objects.y.astype(jnp.float32)
    ow = objects.width.astype(jnp.float32)
    oh = objects.height.astype(jnp.float32)
    t_cx = ox[:, idx] + ow[:, idx] / 2.0
    t_cy = oy[:, idx] + oh[:, idx] / 2.0

    dx = jnp.clip((t_cx - m_cx) / jnp.float32(_SHOT_DX_SCALE), -1.0, 1.0)
    dy = jnp.clip((t_cy - m_cy) / jnp.float32(_SHOT_DY_SCALE), -1.0, 1.0)
    flying = in_flight > 0.5
    return jnp.where(flying, dx, zeros), jnp.where(flying, dy, zeros)


def _shot_between_per_frame(
    raw_observation,
    objects: Any | None,
    target_idx: jnp.ndarray | int | None,
    *,
    projectile_attr: str,
    projectile_slot: int | None,
    stack_len: int,
    in_flight: jnp.ndarray,
) -> jnp.ndarray:
    """Per-frame 0/1: missile centre on the open player→target segment."""
    zeros = jnp.zeros((stack_len,), dtype=jnp.float32)
    if not _USE_EGO_SHOT_BETWEEN:
        return zeros
    if objects is None or target_idx is None:
        return zeros
    if not hasattr(raw_observation, projectile_attr):
        return zeros
    projectiles = getattr(raw_observation, projectile_attr)
    if not hasattr(projectiles, "x") or not hasattr(projectiles, "y"):
        return zeros
    player = _get_player(raw_observation)
    if not hasattr(player, "y") or not hasattr(player, "height"):
        return zeros

    mx = _leaf_per_frame(projectiles.x, projectile_slot, stack_len)
    my = _leaf_per_frame(projectiles.y, projectile_slot, stack_len)
    mw = _leaf_per_frame(getattr(projectiles, "width", 0.0), projectile_slot, stack_len)
    mh = _leaf_per_frame(getattr(projectiles, "height", 0.0), projectile_slot, stack_len)
    m_cx = mx + mw / 2.0
    m_cy = my + mh / 2.0

    p_cx = player.x.astype(jnp.float32) + player.width.astype(jnp.float32) / 2.0
    p_cy = player.y.astype(jnp.float32) + player.height.astype(jnp.float32) / 2.0

    idx = jnp.asarray(target_idx, dtype=jnp.int32)
    t_cx = objects.x.astype(jnp.float32)[:, idx] + objects.width.astype(jnp.float32)[:, idx] / 2.0
    t_cy = objects.y.astype(jnp.float32)[:, idx] + objects.height.astype(jnp.float32)[:, idx] / 2.0

    on_seg = missile_on_open_segment(
        p_cx, p_cy, t_cx, t_cy, m_cx, m_cy, mw, mh
    )
    flying = in_flight > 0.5
    bit = jnp.where(jnp.logical_and(flying, on_seg), jnp.float32(1.0), zeros)
    if hasattr(objects, "semantic_label"):
        tgt_labels = objects.semantic_label[
            jnp.arange(objects.semantic_label.shape[0]), idx
        ]
        bit = jnp.where(tgt_labels == LABEL_DESTROY, bit, zeros)
    return bit


# When |Δx| is below this, L/R aim is undefined (pure vertical / co-located).
# Seaquest's surface NAVIGATE waypoint shares the player x every frame; without
# a deadzone, ``t_cx >= p_cx`` falsely marks it "to the right" and biases
# facing_enemy toward rightward facing.
_FACING_LR_DEADZONE_PX = 2.0
# Horizontal punch / torpedo only hits near the player's band. Targets far
# above/below (falling coco, surface WP) must not count as aimed.
_FACING_LR_Y_ALIGN_PX = 28.0


def _facing_toward_target_lr(
    player_ori: jnp.ndarray,
    player_x: jnp.ndarray,
    player_w: jnp.ndarray,
    player_y: jnp.ndarray,
    player_h: jnp.ndarray,
    target_x: jnp.ndarray,
    target_w: jnp.ndarray,
    target_y: jnp.ndarray,
    target_h: jnp.ndarray,
) -> jnp.ndarray:
    """
    True when the weapon can plausibly hit: L/R facing matches the target's
    horizontal side *and* centres share a vertical band.

    Raw-geometry only. Works for jaxatari L/R oris (90=right, 270=left) and
    for 0/180: ``orientation < 180`` → facing right.

    Horizontally overlapping centres (``|Δx| < deadzone``) skip the L/R check
    so a co-located target does not invent a rightward bias, but still require
    the vertical band (overhead threats stay un-aimed).
    """
    facing_right = player_ori < 180.0
    p_cx = player_x.astype(jnp.float32) + player_w.astype(jnp.float32) / 2.0
    t_cx = target_x.astype(jnp.float32) + target_w.astype(jnp.float32) / 2.0
    p_cy = player_y.astype(jnp.float32) + player_h.astype(jnp.float32) / 2.0
    t_cy = target_y.astype(jnp.float32) + target_h.astype(jnp.float32) / 2.0
    dx = t_cx - p_cx
    dy = t_cy - p_cy
    no_lr = jnp.abs(dx) < jnp.float32(_FACING_LR_DEADZONE_PX)
    y_ok = jnp.abs(dy) < jnp.float32(_FACING_LR_Y_ALIGN_PX)
    target_right = dx > 0.0
    side_ok = jnp.logical_not(jnp.logical_xor(facing_right, target_right))
    lr_ok = jnp.where(no_lr, True, side_ok)
    return jnp.logical_and(lr_ok, y_ok)


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
        player.y,
        player.height,
        objects.x[:, idx],
        objects.width[:, idx],
        objects.y[:, idx],
        objects.height[:, idx],
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
    can_fire_from_player_state: bool = False,
) -> jnp.ndarray:
    """Per-frame weapon extras: [can_fire] (+ facing_enemy) (+ shot_between) (+ shot_dx, shot_dy).

    When ``can_fire_from_player_state`` is True (Phoenix), the first channel is
    shield-active from ``player.state`` (+1 on / -1 off) instead of projectile ready.
    """
    stack_len = _stack_len(raw_observation)
    if can_fire_from_player_state:
        player = _get_player(raw_observation)
        st = jnp.asarray(getattr(player, "state", 0), dtype=jnp.float32)
        if st.ndim == 0:
            st = jnp.full((stack_len,), st)
        elif st.shape[0] != stack_len:
            st = jnp.broadcast_to(st.reshape(-1)[0], (stack_len,))
        can_fire = jnp.where(
            st > 0.5,
            jnp.float32(CAN_FIRE_READY),
            jnp.float32(CAN_FIRE_IN_FLIGHT),
        )
        if hasattr(raw_observation, projectile_attr):
            projectiles = getattr(raw_observation, projectile_attr)
            in_flight = _player_projectile_active_per_frame(
                projectiles.active, projectile_slot
            )
        else:
            in_flight = jnp.zeros((stack_len,), dtype=jnp.float32)
    elif not hasattr(raw_observation, projectile_attr):
        can_fire = jnp.full((stack_len,), CAN_FIRE_READY, dtype=jnp.float32)
        in_flight = jnp.zeros((stack_len,), dtype=jnp.float32)
    else:
        projectiles = getattr(raw_observation, projectile_attr)
        player_proj_active = _player_projectile_active_per_frame(
            projectiles.active, projectile_slot
        )
        in_flight = player_proj_active
        can_fire = jnp.where(
            player_proj_active > 0.5,
            CAN_FIRE_IN_FLIGHT,
            CAN_FIRE_READY,
        ).astype(jnp.float32)

    if _FORCE_CAN_FIRE is not None:
        can_fire = jnp.full((stack_len,), jnp.float32(_FORCE_CAN_FIRE))

    facing = _facing_enemy_per_frame(
        raw_observation, objects, target_idx, stack_len
    )
    channels = [can_fire]
    if _USE_EGO_FACING_ENEMY_FEATURES:
        channels.append(facing)
    if _USE_EGO_SHOT_BETWEEN:
        channels.append(
            _shot_between_per_frame(
                raw_observation,
                objects,
                target_idx,
                projectile_attr=projectile_attr,
                projectile_slot=projectile_slot,
                stack_len=stack_len,
                in_flight=in_flight,
            )
        )
    if _USE_EGO_SHOT_FEATURES:
        shot_dx, shot_dy = _shot_to_target_xy(
            raw_observation,
            objects,
            target_idx,
            projectile_attr=projectile_attr,
            projectile_slot=projectile_slot,
            stack_len=stack_len,
            in_flight=in_flight,
        )
        channels.extend([shot_dx, shot_dy])
    return jnp.stack(channels, axis=-1)


def build_skill_ego_extras_stacked(
    raw_observation,
    objects: Any | None = None,
    target_idx: jnp.ndarray | int | None = None,
) -> jnp.ndarray:
    """
    Per-frame weapon extras. Shape ``[stack_len, 1|2]`` when enabled
    (``can_fire`` ± ``facing_enemy``), plus ``shot_between`` and/or
    ``shot_dx/dy`` when those flags are on, else ``[stack_len, 0]``.

    ``objects`` / ``target_idx`` drive ``facing_enemy`` on L/R-aim envs
    (Seaquest, Chopper Command, Kangaroo) and shot-to-target Δx/Δy when
    the shot flag is on; other envs keep ``facing_enemy=+1`` and shot Δ=0
    if no projectile is present.
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
            can_fire_from_player_state=True,
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

    # Default: always-ready weapon (no projectile attr) + geometry facing_enemy
    # for L/R-aim envs (e.g. Kangaroo punch). Non-L/R envs still get +1.
    ready = jnp.full((stack_len,), CAN_FIRE_READY, dtype=jnp.float32)
    if _FORCE_CAN_FIRE is not None:
        ready = jnp.full((stack_len,), jnp.float32(_FORCE_CAN_FIRE))
    channels = [ready]
    if _USE_EGO_FACING_ENEMY_FEATURES:
        facing = _facing_enemy_per_frame(
            raw_observation, objects, target_idx, stack_len
        )
        channels.append(facing)
    if _USE_EGO_SHOT_BETWEEN:
        channels.append(jnp.zeros((stack_len,), dtype=jnp.float32))
    if _USE_EGO_SHOT_FEATURES:
        zeros = jnp.zeros((stack_len,), dtype=jnp.float32)
        channels.extend([zeros, zeros])
    return jnp.stack(channels, axis=-1)


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
    """Latest-frame player-projectile ready helper (legal-fire masking).

    Distinct from the Phoenix ego ``can_fire`` channel, which encodes shield-active.
    """
    return _latest_can_fire(raw_observation, "player_projectile")


def phoenix_shield_active(raw_observation) -> jnp.ndarray:
    """Latest-frame shield-active (+1 / -1) from stamped ``player.state``."""
    player = _get_player(raw_observation)
    st = jnp.asarray(getattr(player, "state", 0), dtype=jnp.float32)
    latest = st[-1] if st.ndim > 0 else st
    return jnp.where(
        latest > 0.5,
        jnp.float32(CAN_FIRE_READY),
        jnp.float32(CAN_FIRE_IN_FLIGHT),
    )


def spaceinvaders_can_fire(raw_observation) -> jnp.ndarray:
    """Latest-frame can_fire helper (Space Invaders player bullet)."""
    return _latest_can_fire(raw_observation, "player_bullet")


def choppercommand_can_fire(raw_observation) -> jnp.ndarray:
    """Latest-frame can_fire helper (Chopper Command player missile)."""
    return _latest_can_fire(raw_observation, "player_missiles", projectile_slot=0)


def beamrider_can_fire(raw_observation) -> jnp.ndarray:
    """Latest-frame can_fire helper (Beamrider player shot)."""
    return _latest_can_fire(raw_observation, "player_shots", projectile_slot=0)
