"""
Heuristic meta-targets for Kinematic Bootcamp (three labeled entities).
"""

from typing import Optional

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)

NO_TARGET_IDX = -1


def bootcamp_fixed_target_index_from_game_mods(game_mods) -> Optional[int]:
    """
    Single-mod Bootcamp curriculum phases: fixed object slot for `get_ego_obs` / rewards.

    Layout matches ``BootcampSemanticWrapper``:
    index 0 = destroy enemy, 1 = avoid enemy, 2 = diver (collect).

    Returns None for full Bootcamp (no mods, multiple mods, or unknown mod) so training
    falls back to ``get_heuristic_target``.
    """
    if game_mods is None:
        return None
    if isinstance(game_mods, str):
        mods_list = [game_mods]
    else:
        mods_list = list(game_mods)
    if len(mods_list) != 1:
        return None
    key = str(mods_list[0]).lower()
    return {"destroy": 0, "avoid": 1, "collect": 2}.get(key)


def get_heuristic_target(raw_observation, objects, prev_target_idx, rng):
    del prev_target_idx, rng

    px = raw_observation.player.x[-1]
    py = raw_observation.player.y[-1]
    pw = raw_observation.player.width[-1]
    ph = raw_observation.player.height[-1]

    ox = objects.x[-1]
    oy = objects.y[-1]
    ow = objects.width[-1]
    oh = objects.height[-1]

    active = objects.active[-1] > 0
    labels = objects.semantic_label[-1]

    skill_object = jnp.logical_and(
        active,
        jnp.logical_or(
            labels == LABEL_DESTROY,
            jnp.logical_or(labels == LABEL_AVOID, labels == LABEL_COLLECT),
        ),
    )

    px_c = px.astype(jnp.float32) + pw.astype(jnp.float32) * 0.5
    py_c = py.astype(jnp.float32) + ph.astype(jnp.float32) * 0.5
    ox_c = ox.astype(jnp.float32) + ow.astype(jnp.float32) * 0.5
    oy_c = oy.astype(jnp.float32) + oh.astype(jnp.float32) * 0.5

    dx = px_c - ox_c
    dy = py_c - oy_c
    distances = jnp.sqrt(dx * dx + dy * dy)

    large = jnp.max(jnp.where(skill_object, distances, 0.0)) + 1.0
    masked = jnp.where(skill_object, distances, large)
    best = jnp.argmin(masked).astype(jnp.int32)
    any_skill = jnp.any(skill_object)

    return jnp.where(
        any_skill,
        best,
        jnp.array(NO_TARGET_IDX, dtype=jnp.int32),
    ).astype(jnp.int32)
