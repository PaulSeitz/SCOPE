"""Context ablations of the reference ego skill input.

The additive modes keep the reference ``get_ego_obs`` vector as an unchanged prefix,
so the heuristic selector, skill rewards, ego action mapping, MLP head and
optimizer stay identical to the reference arm.

``ego_abs_target`` replaces the prefix entirely with per-frame
``[target_cx, target_cy, target_w, target_h, active, player_cx, player_cy,
player_w, player_h]`` normalized by screen dimensions. Inactive target
frames are zero. It exposes no relative features or other objects.

``ego_scene_rel`` (broader context, relative only)
    ``ref || flatten(rows)``. One row per object slot (pad/trim to
    ``max_objects``), built with the *same* per-object encoding as the
    reference target (AABB-gap distance + facing-relative ψ bins, four-frame
    history), masked per frame by that slot's ``active`` flag::

        row_j = [ego(j)_f0, active_f0, …, ego(j)_f3, active_f3, is_target_j]

    No absolute coordinates, no player box, no labels, no oxygen. Slot order
    is fixed per game, so slot position identifies the entity type. Note that
    static or lane-bound objects (Seaquest surface waypoint, Name This Game
    tentacles) make absolute position *recoverable* from relative offsets;
    this mode withholds explicit absolute location, not all of its correlates.

``ego_abs_pos`` (restricted input + absolute location)
    ``ref || [px_f0, py_f0, …, px_f3, py_f3]``: the player's bbox centre in
    normalized screen coordinates (x / 160, y / 210) for every frame of the
    same frame stack as the reference features.

``ego_scene_rows`` (``ego_scene_rel`` without the reference prefix)
    ``flatten(rows)`` only. The target stays identified by ``is_target``, so the
    same information is present but not handed over as a direct input. Separates
    "the direct target-relative input protects" from "relative inputs protect".

``ego_oc_native`` (restricted input + full native OC input)
    ``ref || native_vector || one_hot(target)``: the complete ``oc_native`` skill
    input, including absolute object coordinates, sizes and global fields,
    behind the reference prefix. Separates "absolute inputs harm" from "the
    direct target-relative input protects" in the native-OC comparison. Needs
    ``OC_NATIVE_STACK: true`` and, unlike the context modes above, the aligned
    native vector, so it is dispatched separately from them.

``oc_native_target_front`` (native OC input + absolute target in fixed slots)
    ``target_row || native_vector || one_hot(target)`` with
    ``target_row = [x_f0, y_f0, w_f0, h_f0, active_f0, ...]`` in normalized screen
    coordinates. Adds fixed input slots for the target without any target-relative
    information, the counterpart to ``ego_scene_rows``. Same native-vector
    requirement as ``ego_oc_native``.
"""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp

from concept_based_learning.observations.egocentric import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    _center_y,
    _get_player,
    _uses_bottom_edge_y,
    get_ego_obs,
)

EGO_SCENE_REL = "ego_scene_rel"
EGO_ABS_POS = "ego_abs_pos"
EGO_SCENE_ROWS = "ego_scene_rows"
EGO_ABS_TARGET = "ego_abs_target"
EGO_CONTEXT_SKILL_INPUT_MODES = (EGO_SCENE_REL, EGO_ABS_POS, EGO_SCENE_ROWS, EGO_ABS_TARGET)
EGO_OC_NATIVE = "ego_oc_native"
OC_NATIVE_TARGET_FRONT = "oc_native_target_front"


def _stack_size(objects: Any) -> int:
    return int(jnp.asarray(objects.x).shape[0])


def player_abs_position_features(raw_obs: Any) -> jnp.ndarray:
    """Normalized player bbox centre per frame, flattened ``[x0, y0, x1, y1, …]``."""
    player = _get_player(raw_obs)
    px = jnp.asarray(player.x, dtype=jnp.float32)
    py = jnp.asarray(player.y, dtype=jnp.float32)
    pw = jnp.asarray(player.width, dtype=jnp.float32)
    ph = jnp.asarray(player.height, dtype=jnp.float32)
    cx = (px + pw * 0.5) / jnp.float32(SCREEN_WIDTH)
    cy = _center_y(py, ph, y_is_bottom_edge=_uses_bottom_edge_y(raw_obs)) / jnp.float32(
        SCREEN_HEIGHT
    )
    return jnp.stack([cx, cy], axis=-1).reshape((-1,)).astype(jnp.float32)


def apply_abs_pos_intervention(tail: jnp.ndarray, spec: str | None) -> jnp.ndarray:
    """Eval-only test of whether a policy uses the abs-pos tail ``[x0, y0, x1, y1, ...]``.

    ``spec``: ``None``/empty (unchanged), ``frozen:X,Y`` (constant normalized
    position in every frame), or ``mirror_x`` (x -> 1 - x, y unchanged).
    """
    if not spec:
        return tail
    if spec == "mirror_x":
        pairs = tail.reshape((-1, 2))
        return jnp.stack([1.0 - pairs[:, 0], pairs[:, 1]], axis=-1).reshape((-1,))
    if spec.startswith("frozen:"):
        x, y = (float(v) for v in spec.split(":", 1)[1].split(","))
        fixed = jnp.array([x, y], dtype=jnp.float32)
        return jnp.tile(fixed, tail.shape[0] // 2)
    raise ValueError(f"Unknown CBL_ABS_POS_INTERVENTION {spec!r}")


def build_ego_abs_pos_obs(
    raw_obs: Any, objects: Any, target_idx: Any, distance_clip_px=None
) -> jnp.ndarray:
    ref = get_ego_obs(raw_obs, objects, target_idx, distance_clip_px=distance_clip_px)
    # Environment variable so parallel mod-eval workers inherit it. Never set in training.
    tail = apply_abs_pos_intervention(
        player_abs_position_features(raw_obs),
        os.environ.get("CBL_ABS_POS_INTERVENTION"),
    )
    return jnp.concatenate([ref, tail]).astype(jnp.float32)


def build_ego_abs_target_obs(raw_obs: Any, objects: Any, target_idx: Any) -> jnp.ndarray:
    """Full absolute target/player input, frame-major; no reference ego prefix."""
    player = _get_player(raw_obs)
    idx = jnp.asarray(target_idx, dtype=jnp.int32)
    bottom = _uses_bottom_edge_y(raw_obs)
    width, height = jnp.float32(SCREEN_WIDTH), jnp.float32(SCREEN_HEIGHT)
    tx, ty, tw, th = (
        jnp.asarray(getattr(objects, field), dtype=jnp.float32)[:, idx]
        for field in ("x", "y", "width", "height")
    )
    px, py, pw, ph = (
        jnp.asarray(getattr(player, field), dtype=jnp.float32)
        for field in ("x", "y", "width", "height")
    )
    active = jnp.asarray(objects.active)[:, idx] > 0
    frames = jnp.stack([
        (tx + tw * 0.5) / width,
        _center_y(ty, th, y_is_bottom_edge=bottom) / height,
        tw / width, th / height, active.astype(jnp.float32),
        (px + pw * 0.5) / width,
        _center_y(py, ph, y_is_bottom_edge=bottom) / height,
        pw / width, ph / height,
    ], axis=-1)
    return jnp.where(active[:, None], frames, 0.0).reshape((-1,)).astype(jnp.float32)


def build_ego_scene_rows(
    raw_obs: Any, objects: Any, target_idx: Any, *, max_objects: int, distance_clip_px=None
) -> jnp.ndarray:
    """``[max_objects, stack * (C + 1) + 1]`` relative scene rows (see module doc)."""
    stack = _stack_size(objects)
    m_env = int(jnp.asarray(objects.x).shape[-1])
    slot_ids = jnp.arange(m_env, dtype=jnp.int32)
    ego_all = jax.vmap(
        lambda j: get_ego_obs(raw_obs, objects, j, distance_clip_px=distance_clip_px)
    )(slot_ids)  # [M, stack * C]
    ego_all = ego_all.reshape((m_env, stack, -1))
    active = (jnp.asarray(objects.active) > 0).astype(jnp.float32).T  # [M, stack]
    per_frame = jnp.concatenate(
        [ego_all * active[..., None], active[..., None]], axis=-1
    )  # [M, stack, C + 1]
    idx = jnp.asarray(target_idx, dtype=jnp.int32)
    is_target = (slot_ids == idx).astype(jnp.float32)[:, None]
    rows = jnp.concatenate([per_frame.reshape((m_env, -1)), is_target], axis=-1)
    if m_env > max_objects:
        rows = rows[:max_objects]
    elif m_env < max_objects:
        rows = jnp.concatenate(
            [rows, jnp.zeros((max_objects - m_env, rows.shape[-1]), jnp.float32)]
        )
    return rows.astype(jnp.float32)


def build_ego_scene_rel_obs(
    raw_obs: Any, objects: Any, target_idx: Any, *, max_objects: int, distance_clip_px=None
) -> jnp.ndarray:
    ref = get_ego_obs(raw_obs, objects, target_idx, distance_clip_px=distance_clip_px)
    rows = build_ego_scene_rows(
        raw_obs,
        objects,
        target_idx,
        max_objects=max_objects,
        distance_clip_px=distance_clip_px,
    )
    return jnp.concatenate([ref, rows.reshape((-1,))]).astype(jnp.float32)


def build_ego_oc_native_obs(
    raw_obs: Any, objects: Any, target_idx: Any, native_vector: Any, *, max_objects: int
) -> jnp.ndarray:
    """``ref || oc_native_skill_features(native_vector, target_idx, max_objects)``."""
    from concept_based_learning.observations.native_oc_stack import (
        oc_native_skill_features,
    )

    if native_vector is None:
        raise ValueError("ego_oc_native skills require the aligned native OC vector.")
    ref = get_ego_obs(raw_obs, objects, target_idx)
    native = oc_native_skill_features(native_vector, target_idx, max_objects)
    return jnp.concatenate([ref, native]).astype(jnp.float32)


def build_oc_native_target_front_obs(
    objects: Any, target_idx: Any, native_vector: Any, *, max_objects: int
) -> jnp.ndarray:
    """``absolute target row || oc_native_skill_features(native_vector, target_idx, max_objects)``."""
    from concept_based_learning.observations.native_oc_stack import (
        oc_native_skill_features,
    )

    if native_vector is None:
        raise ValueError("oc_native_target_front skills require the aligned native OC vector.")
    idx = jnp.asarray(target_idx, dtype=jnp.int32)
    width = jnp.float32(SCREEN_WIDTH)
    height = jnp.float32(SCREEN_HEIGHT)
    row = jnp.stack(
        [
            jnp.asarray(objects.x, dtype=jnp.float32)[..., idx] / width,
            jnp.asarray(objects.y, dtype=jnp.float32)[..., idx] / height,
            jnp.asarray(objects.width, dtype=jnp.float32)[..., idx] / width,
            jnp.asarray(objects.height, dtype=jnp.float32)[..., idx] / height,
            (jnp.asarray(objects.active)[..., idx] > 0).astype(jnp.float32),
        ],
        axis=-1,
    )
    native = oc_native_skill_features(native_vector, target_idx, max_objects)
    return jnp.concatenate([jnp.clip(row, 0.0, 1.0).reshape((-1,)), native]).astype(jnp.float32)


def build_ego_context_skill_obs(
    raw_obs: Any,
    objects: Any,
    target_idx: Any,
    *,
    mode: str,
    max_objects: int,
    distance_clip_px=None,
) -> jnp.ndarray:
    canon = str(mode).lower().strip()
    if canon == EGO_ABS_TARGET:
        return build_ego_abs_target_obs(raw_obs, objects, target_idx)
    if canon == EGO_ABS_POS:
        return build_ego_abs_pos_obs(
            raw_obs, objects, target_idx, distance_clip_px=distance_clip_px
        )
    if canon == EGO_SCENE_ROWS:
        return build_ego_scene_rows(
            raw_obs,
            objects,
            target_idx,
            max_objects=max_objects,
            distance_clip_px=distance_clip_px,
        ).reshape((-1,))
    if canon == EGO_SCENE_REL:
        return build_ego_scene_rel_obs(
            raw_obs,
            objects,
            target_idx,
            max_objects=max_objects,
            distance_clip_px=distance_clip_px,
        )
    raise ValueError(
        f"Unknown ego-context skill mode {mode!r}; expected one of "
        f"{EGO_CONTEXT_SKILL_INPUT_MODES}."
    )


def ego_context_extra_dim(
    mode: str, *, ego_channels_per_frame: int, stack_size: int, max_objects: int
) -> int:
    """Width appended after the reference ego vector (additive modes only).

    Replacement modes build their full input directly; network initialization
    infers its width from that vector rather than adding this helper.
    """
    canon = str(mode).lower().strip()
    if canon == EGO_ABS_POS:
        return 2 * int(stack_size)
    if canon == EGO_SCENE_REL:
        return int(max_objects) * (int(stack_size) * (int(ego_channels_per_frame) + 1) + 1)
    raise ValueError(f"Not an ego-context mode: {mode!r}")
