"""Slim / relational / role-pooled meta OC features (``META_INPUT_MODE: oc_simple*``).

Modes
-----
``oc_simple``
    Global: ``[player_xywh, oxygen, collected_divers]``
    Per slot: ``[x, y, w, h, active] + one_hot(label)``

``oc_simple_rel``
    Same global; per slot uses player-relative geometry
    ``[Δx, Δy, dist, same_lane, lane_delta, active] + one_hot(label)``
    (no absolute object coords / no visual_id).

``oc_simple_role``
    ``oc_simple`` object rows + role-pool summary appended to global
    (active counts + min player-distance for COLLECT/DESTROY/AVOID/NAVIGATE).

``oc_simple_rel_role``
    ``oc_simple_rel`` object rows + same role-pool global extras.

``oc_simple_inv``
    ``oc_simple`` plus ``active_collect_count`` and ``lives`` on the global stream.
"""

from __future__ import annotations

from typing import Any

import jax.nn as jnn
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NAVIGATE,
    NUM_LABELS,
)
from concept_based_learning.observations.egocentric import (
    MAX_DIST,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    _get_player,
)

# Match Seaquest heuristic diver-lane band (centre Δy).
_SAME_LANE_Y_PX = 14.0

_ROLE_LABELS = (LABEL_COLLECT, LABEL_DESTROY, LABEL_AVOID, LABEL_NAVIGATE)

_OC_SIMPLE_ALIASES = {
    "oc_simple": "oc_simple",
    "simple_oc": "oc_simple",
    "oc_slim": "oc_simple",
    "oc_simple_rel": "oc_simple_rel",
    "oc_rel": "oc_simple_rel",
    "oc_simple_role": "oc_simple_role",
    "oc_role": "oc_simple_role",
    "oc_simple_rel_role": "oc_simple_rel_role",
    "oc_rel_role": "oc_simple_rel_role",
    "oc_simple_inv": "oc_simple_inv",
    "oc_inv": "oc_simple_inv",
}


def normalize_meta_oc_simple_mode(mode: str) -> str | None:
    """Return canonical oc_simple* mode or ``None`` if not in this family."""
    key = str(mode or "").lower().strip()
    return _OC_SIMPLE_ALIASES.get(key)


def _latest_scalar(value, *, default: float = 0.0) -> jnp.ndarray:
    if value is None:
        return jnp.asarray(default, dtype=jnp.float32)
    arr = jnp.asarray(value, dtype=jnp.float32)
    if arr.ndim == 0:
        return arr
    return arr.reshape(-1)[-1]


def _latest_object_leaf(objects: Any, name: str) -> jnp.ndarray:
    arr = jnp.asarray(getattr(objects, name))
    if arr.ndim == 0:
        return arr
    return arr[-1] if arr.ndim >= 2 else arr


def _latest_player_box(raw_obs: Any) -> jnp.ndarray:
    player = _get_player(raw_obs)
    return jnp.stack(
        [
            _latest_scalar(player.x),
            _latest_scalar(player.y),
            _latest_scalar(player.width),
            _latest_scalar(player.height),
        ],
        axis=-1,
    ).astype(jnp.float32)


def _optional_raw_scalar(raw_obs: Any, name: str, *, default: float = 0.0) -> jnp.ndarray:
    if raw_obs is None or not hasattr(raw_obs, name):
        return jnp.asarray(default, dtype=jnp.float32)
    return _latest_scalar(getattr(raw_obs, name), default=default)


def _player_center_wh(player_box: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    px = player_box[0] + player_box[2] * 0.5
    py = player_box[1] + player_box[3] * 0.5
    return px, py


def _pad_or_trim_rows(mat: jnp.ndarray, max_objects: int) -> jnp.ndarray:
    m = int(mat.shape[0])
    if m == max_objects:
        return mat
    if m > max_objects:
        return mat[:max_objects]
    pad = jnp.zeros((max_objects - m, mat.shape[-1]), dtype=jnp.float32)
    return jnp.concatenate([mat, pad], axis=0)


def _label_one_hot(labels: jnp.ndarray) -> jnp.ndarray:
    return jnn.one_hot(
        jnp.clip(labels.astype(jnp.int32), 0, NUM_LABELS - 1), NUM_LABELS
    ).astype(jnp.float32)


def _object_geometry(objects: Any) -> tuple[jnp.ndarray, ...]:
    x = _latest_object_leaf(objects, "x").astype(jnp.float32)
    y = _latest_object_leaf(objects, "y").astype(jnp.float32)
    w = _latest_object_leaf(objects, "width").astype(jnp.float32)
    h = _latest_object_leaf(objects, "height").astype(jnp.float32)
    active = _latest_object_leaf(objects, "active").astype(jnp.float32)
    labels = _latest_object_leaf(objects, "semantic_label").astype(jnp.int32)
    return x, y, w, h, active, labels


def build_meta_oc_simple_global(
    raw_obs: Any,
    *,
    include_inventory_extras: bool = False,
    objects: Any | None = None,
    max_objects: int | None = None,
) -> jnp.ndarray:
    """Player bbox + oxygen/collected_divers (+ optional lives / active collect count)."""
    player_box = _latest_player_box(raw_obs)
    oxygen = _optional_raw_scalar(raw_obs, "oxygen_level")
    collected = _optional_raw_scalar(raw_obs, "collected_divers")
    parts = [player_box, jnp.atleast_1d(oxygen), jnp.atleast_1d(collected)]
    if include_inventory_extras:
        lives = _optional_raw_scalar(raw_obs, "lives")
        if objects is None or max_objects is None:
            active_collect = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            active_collect = count_active_collect_objects(objects, max_objects=max_objects)
        parts.extend([jnp.atleast_1d(lives), jnp.atleast_1d(active_collect)])
    return jnp.concatenate(parts).astype(jnp.float32)


def build_meta_oc_simple_object_matrix(
    objects: Any,
    *,
    max_objects: int,
) -> jnp.ndarray:
    """Absolute ``[x,y,w,h,active] + label OH``."""
    if objects is None:
        raise ValueError("objects is required for META_INPUT_MODE=oc_simple.")
    x, y, w, h, active, labels = _object_geometry(objects)
    box = jnp.stack([x, y, w, h, active], axis=-1)
    mat = jnp.concatenate([box, _label_one_hot(labels)], axis=-1)
    return _pad_or_trim_rows(mat, max_objects)


def build_meta_oc_simple_rel_object_matrix(
    raw_obs: Any,
    objects: Any,
    *,
    max_objects: int,
) -> jnp.ndarray:
    """Player-relative ``[Δx,Δy,dist,same_lane,lane_delta,active] + label OH``."""
    if objects is None:
        raise ValueError("objects is required for META_INPUT_MODE=oc_simple_rel.")
    player_box = _latest_player_box(raw_obs)
    px, py = _player_center_wh(player_box)
    x, y, w, h, active, labels = _object_geometry(objects)
    cx = x + w * 0.5
    cy = y + h * 0.5
    dx = (cx - px) / jnp.asarray(SCREEN_WIDTH, dtype=jnp.float32)
    dy = (cy - py) / jnp.asarray(SCREEN_HEIGHT, dtype=jnp.float32)
    dist = jnp.sqrt(jnp.maximum(dx * dx + dy * dy, 0.0))
    # Keep a pixel-space lane band, then a normalized lane_delta.
    dy_px = cy - py
    same_lane = (jnp.abs(dy_px) <= _SAME_LANE_Y_PX).astype(jnp.float32)
    lane_delta = dy_px / jnp.asarray(SCREEN_HEIGHT, dtype=jnp.float32)
    # Inactive / empty rows: zero geometry so padding does not look like a target.
    geom = jnp.stack([dx, dy, dist, same_lane, lane_delta, active], axis=-1)
    geom = geom * active[:, None]
    mat = jnp.concatenate([geom, _label_one_hot(labels)], axis=-1)
    return _pad_or_trim_rows(mat, max_objects)


def build_meta_role_pool_features(
    raw_obs: Any,
    objects: Any,
    *,
    max_objects: int,
) -> jnp.ndarray:
    """Per-role active counts + min normalised distance to player (4 roles × 2)."""
    if objects is None:
        raise ValueError("objects is required for role-pool meta features.")
    player_box = _latest_player_box(raw_obs)
    px, py = _player_center_wh(player_box)
    x, y, w, h, active, labels = _object_geometry(objects)
    m = min(int(x.shape[0]), int(max_objects))
    x, y, w, h = x[:m], y[:m], w[:m], h[:m]
    active = active[:m]
    labels = labels[:m]
    cx = x + w * 0.5
    cy = y + h * 0.5
    dist = jnp.sqrt((cx - px) ** 2 + (cy - py) ** 2) / jnp.asarray(MAX_DIST, dtype=jnp.float32)
    # Large sentinel so inactive / wrong-role slots lose the min.
    big = jnp.asarray(1.0, dtype=jnp.float32)
    feats = []
    for role in _ROLE_LABELS:
        mask = jnp.logical_and(active > 0.5, labels == role)
        count = jnp.sum(mask.astype(jnp.float32))
        masked_dist = jnp.where(mask, dist, big)
        min_dist = jnp.min(masked_dist)
        min_dist = jnp.where(count > 0.0, min_dist, big)
        feats.extend([count, min_dist])
    return jnp.stack(feats, axis=-1).astype(jnp.float32)


def count_active_collect_objects(objects: Any, *, max_objects: int) -> jnp.ndarray:
    """On-screen active COLLECT count."""
    labels = _latest_object_leaf(objects, "semantic_label").astype(jnp.int32)
    active = _latest_object_leaf(objects, "active").astype(jnp.float32)
    m = min(int(labels.shape[0]), int(max_objects))
    is_collect = labels[:m] == LABEL_COLLECT
    return jnp.sum(jnp.where(is_collect, active[:m], 0.0)).astype(jnp.float32)


def build_meta_oc_simple_features(
    raw_obs: Any,
    objects: Any,
    *,
    max_objects: int,
    mode: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(global_features, object_matrix)`` for an oc_simple* mode."""
    canon = normalize_meta_oc_simple_mode(mode)
    if canon is None:
        raise ValueError(f"Not an oc_simple* mode: {mode!r}")

    if canon == "oc_simple":
        global_f = build_meta_oc_simple_global(raw_obs)
        obj_m = build_meta_oc_simple_object_matrix(objects, max_objects=max_objects)
    elif canon == "oc_simple_rel":
        global_f = build_meta_oc_simple_global(raw_obs)
        obj_m = build_meta_oc_simple_rel_object_matrix(
            raw_obs, objects, max_objects=max_objects
        )
    elif canon == "oc_simple_role":
        global_f = jnp.concatenate(
            [
                build_meta_oc_simple_global(raw_obs),
                build_meta_role_pool_features(
                    raw_obs, objects, max_objects=max_objects
                ),
            ]
        )
        obj_m = build_meta_oc_simple_object_matrix(objects, max_objects=max_objects)
    elif canon == "oc_simple_rel_role":
        global_f = jnp.concatenate(
            [
                build_meta_oc_simple_global(raw_obs),
                build_meta_role_pool_features(
                    raw_obs, objects, max_objects=max_objects
                ),
            ]
        )
        obj_m = build_meta_oc_simple_rel_object_matrix(
            raw_obs, objects, max_objects=max_objects
        )
    elif canon == "oc_simple_inv":
        global_f = build_meta_oc_simple_global(
            raw_obs,
            include_inventory_extras=True,
            objects=objects,
            max_objects=max_objects,
        )
        obj_m = build_meta_oc_simple_object_matrix(objects, max_objects=max_objects)
    else:
        raise ValueError(f"Unhandled oc_simple mode: {canon!r}")
    return global_f.astype(jnp.float32), obj_m.astype(jnp.float32)


def build_skill_oc_flat_rel_obs(
    raw_obs: Any,
    objects: Any,
    target_idx: Any,
    *,
    max_objects: int,
) -> jnp.ndarray:
    """Flattened relative object list + globals for skill ``oc_flat_rel``.

    Per-object row (slot order = labeled observation order, pad/trim to
    ``max_objects``)::

        [Δx, Δy, dist, same_lane, lane_delta, active] + one_hot(label) + is_target

    Then ``flatten(rows) || global`` where global is
    ``[player_x, player_y, player_w, player_h, oxygen, collected_divers]``.

    Latest frame only (no frame-stack), matching meta ``oc_simple_rel``.
    """
    obj_m = build_meta_oc_simple_rel_object_matrix(
        raw_obs, objects, max_objects=max_objects
    )
    # obj_m: [M, 6 + NUM_LABELS]; col 5 is active.
    m = int(obj_m.shape[0])
    idx = jnp.asarray(target_idx, dtype=jnp.int32)
    is_target = jnn.one_hot(jnp.clip(idx, 0, m - 1), m, dtype=jnp.float32)
    rows = jnp.concatenate([obj_m, is_target[:, None]], axis=-1)
    active = obj_m[:, 5]
    rows = rows * active[:, None]
    flat = rows.reshape((-1,))
    global_f = build_meta_oc_simple_global(raw_obs)
    return jnp.concatenate([flat, global_f]).astype(jnp.float32)


def skill_oc_flat_rel_dim(max_objects: int) -> int:
    """Input width for ``oc_flat_rel`` (latest frame)."""
    per_obj = 6 + NUM_LABELS + 1  # geom + label OH + is_target
    global_dim = 6  # player xywh + oxygen + collected_divers
    return int(max_objects) * per_obj + global_dim
