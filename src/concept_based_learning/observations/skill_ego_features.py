"""Per-skill extra channels appended to egocentric skill observations.

Layout (per frame, before stack flatten):
  [skill_0 extras | skill_1 extras | ... | skill_{N-1} extras]
Each skill block has ``SKILL_EGO_EXTRA_DIM`` channels (zero-padded when unused).

Env-specific builders fill only the blocks they care about; reserved slots stay
available for future skills or features without changing the QNetwork input width.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_USE_SKILL_EGO_FEATURES = False
_NUM_SKILLS = 3
_SKILL_EGO_EXTRA_DIM = 4
_ENV_NAME = ""


def configure_skill_ego_features(
    *,
    enabled: bool,
    num_skills: int,
    extra_dim_per_skill: int,
    env_name: str,
) -> None:
    global _USE_SKILL_EGO_FEATURES, _NUM_SKILLS, _SKILL_EGO_EXTRA_DIM, _ENV_NAME
    _USE_SKILL_EGO_FEATURES = bool(enabled)
    _NUM_SKILLS = max(1, int(num_skills))
    _SKILL_EGO_EXTRA_DIM = max(1, int(extra_dim_per_skill))
    _ENV_NAME = str(env_name).lower()


def using_skill_ego_features() -> bool:
    return _USE_SKILL_EGO_FEATURES


def skill_ego_extra_channels_per_frame() -> int:
    if not _USE_SKILL_EGO_FEATURES:
        return 0
    return _NUM_SKILLS * _SKILL_EGO_EXTRA_DIM


def skill_ego_extra_dim_per_skill() -> int:
    return _SKILL_EGO_EXTRA_DIM


def _stack_len(raw_observation) -> int:
    for name in ("player", "chicken", "agent", "skier"):
        if hasattr(raw_observation, name):
            x = getattr(raw_observation, name).x
            return int(x.shape[0])
    return 1


def _zeros_block(extra_dim: int) -> jnp.ndarray:
    return jnp.zeros((extra_dim,), dtype=jnp.float32)


def _destroy_can_fire_extras_one_frame(
    can_fire: jnp.ndarray, extra_dim: int, num_skills: int
) -> jnp.ndarray:
    """Skill 0 (DESTROY): slot 0 = can_fire (1.0 when a new shot may be launched)."""
    blocks = []
    destroy = _zeros_block(extra_dim)
    destroy = destroy.at[0].set(can_fire)
    blocks.append(destroy)
    for _ in range(1, num_skills):
        blocks.append(_zeros_block(extra_dim))
    return jnp.concatenate(blocks, axis=0)


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


def _can_fire_extras_stacked(
    raw_observation,
    *,
    extra_dim: int,
    num_skills: int,
    projectile_attr: str,
    projectile_slot: int | None = None,
) -> jnp.ndarray:
    """Build per-frame DESTROY can_fire extras from a player projectile field."""
    stack_len = _stack_len(raw_observation)
    width = num_skills * extra_dim
    if not hasattr(raw_observation, projectile_attr):
        return jnp.zeros((stack_len, width), dtype=jnp.float32)

    projectiles = getattr(raw_observation, projectile_attr)
    player_proj_active = _player_projectile_active_per_frame(
        projectiles.active, projectile_slot
    )
    can_fire = 1.0 - player_proj_active
    return jax.vmap(
        lambda cf: _destroy_can_fire_extras_one_frame(cf, extra_dim, num_skills)
    )(can_fire)


def _seaquest_extras_stacked(raw_observation, extra_dim: int, num_skills: int) -> jnp.ndarray:
    # JAXAtari Seaquest: projectiles.active[:, 0] is the player torpedo.
    return _can_fire_extras_stacked(
        raw_observation,
        extra_dim=extra_dim,
        num_skills=num_skills,
        projectile_attr="projectiles",
        projectile_slot=0,
    )


def build_skill_ego_extras_stacked(raw_observation) -> jnp.ndarray:
    """
    Per-frame skill extra channels. Shape [stack_len, num_skills * extra_dim].
    """
    if not _USE_SKILL_EGO_FEATURES:
        stack_len = _stack_len(raw_observation)
        return jnp.zeros((stack_len, 0), dtype=jnp.float32)

    extra_dim = _SKILL_EGO_EXTRA_DIM
    num_skills = _NUM_SKILLS
    stack_len = _stack_len(raw_observation)
    width = num_skills * extra_dim

    if _ENV_NAME == "seaquest":
        return _seaquest_extras_stacked(raw_observation, extra_dim, num_skills)
    if _ENV_NAME == "phoenix":
        return _can_fire_extras_stacked(
            raw_observation,
            extra_dim=extra_dim,
            num_skills=num_skills,
            projectile_attr="player_projectile",
        )
    if _ENV_NAME == "spaceinvaders":
        return _can_fire_extras_stacked(
            raw_observation,
            extra_dim=extra_dim,
            num_skills=num_skills,
            projectile_attr="player_bullet",
        )
    if _ENV_NAME == "choppercommand":
        return _can_fire_extras_stacked(
            raw_observation,
            extra_dim=extra_dim,
            num_skills=num_skills,
            projectile_attr="player_missiles",
            projectile_slot=0,
        )
    if _ENV_NAME == "beamrider":
        return _can_fire_extras_stacked(
            raw_observation,
            extra_dim=extra_dim,
            num_skills=num_skills,
            projectile_attr="player_shots",
            projectile_slot=0,
        )

    return jnp.zeros((stack_len, width), dtype=jnp.float32)


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
