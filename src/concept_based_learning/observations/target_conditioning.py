"""Index-invariant target cues for skill pixel / OC inputs.

``target_idx`` is only a gather key. The network never receives a bare slot
one-hot glued onto a global embedding:

* **pixel_embed** — paint a red (or bright grayscale) outline on the target bbox
* **oc_embed** — full labeled ``objects`` matrix (same leaves as meta);
  shared per-object ``Dense``, then gather the target row (transfer-friendly)
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from concept_based_learning.observations.egocentric import SCREEN_HEIGHT, SCREEN_WIDTH

# Outline thickness in pixel-observation coordinates (after any resize).
DEFAULT_OUTLINE_THICKNESS = 2


def _last_frame_objects(objects):
    """Take the last frame-stack entry: leaves ``[..., stack, max_objects]`` → ``[..., max_objects]``."""
    return jax.tree_util.tree_map(
        lambda x: x[..., -1, :] if jnp.ndim(x) >= 2 else x,
        objects,
    )


def scale_bbox_to_frame(x, y, w, h, frame_h: int, frame_w: int):
    """Map native Atari bbox coords onto a (possibly resized) frame."""
    sx = jnp.asarray(frame_w, dtype=jnp.float32) / jnp.asarray(
        SCREEN_WIDTH, dtype=jnp.float32
    )
    sy = jnp.asarray(frame_h, dtype=jnp.float32) / jnp.asarray(
        SCREEN_HEIGHT, dtype=jnp.float32
    )
    x = jnp.asarray(x, dtype=jnp.float32) * sx
    y = jnp.asarray(y, dtype=jnp.float32) * sy
    w = jnp.asarray(w, dtype=jnp.float32) * sx
    h = jnp.asarray(h, dtype=jnp.float32) * sy
    return x, y, w, h


def outline_mask(frame_h: int, frame_w: int, x, y, w, h, thickness: int):
    """Boolean [H, W] border mask for a single axis-aligned box."""
    yy = jnp.arange(frame_h, dtype=jnp.float32)[:, None]
    xx = jnp.arange(frame_w, dtype=jnp.float32)[None, :]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    t = jnp.asarray(thickness, dtype=jnp.float32)
    inset = (xx >= x1 + t) & (xx < x2 - t) & (yy >= y1 + t) & (yy < y2 - t)
    too_small = ((x2 - x1) <= (2.0 * t)) | ((y2 - y1) <= (2.0 * t))
    return jnp.where(too_small, inside, inside & jnp.logical_not(inset))


def _outline_color(num_channels: int) -> jnp.ndarray:
    """RGB red when C>=3; bright value for grayscale."""
    if num_channels >= 3:
        color = jnp.zeros((num_channels,), dtype=jnp.float32)
        color = color.at[0].set(1.0)
        return color
    return jnp.ones((num_channels,), dtype=jnp.float32)


def paint_outline_on_hwc(frame_hwc: jnp.ndarray, x, y, w, h, *, thickness: int, active):
    """Paint outline onto a single [H, W, C] float frame in [0, 1]."""
    frame = frame_hwc.astype(jnp.float32)
    h_img, w_img, c = frame.shape
    color = _outline_color(int(c))
    mask = outline_mask(h_img, w_img, x, y, w, h, thickness)
    mask = jnp.logical_and(mask, jnp.asarray(active, dtype=jnp.bool_))
    mask = mask.astype(jnp.float32)[..., None]
    return frame * (1.0 - mask) + color * mask


def _gather_target_bbox(objects, target_idx, max_objects: int):
    """Gather bbox fields for ``target_idx`` without ``jnp.take`` batch pitfalls.

    ``jnp.take(arr, idx, axis=-1)`` with ``arr=[B,M]`` and ``idx=[B]`` yields
    ``[B,B]`` (indices applied independently of the batch axis). Use one-hot
    gather so shapes stay ``[B]`` / ``[B,K]`` aligned with ``target_idx``.
    """
    objs = _last_frame_objects(objects)
    idx = jnp.clip(jnp.asarray(target_idx, dtype=jnp.int32), 0, max_objects - 1)
    ref = jnp.asarray(objs.x)
    m = ref.shape[-1]
    obj_lead = ref.shape[:-1]
    extra = len(idx.shape) - len(obj_lead)
    if extra < 0:
        raise ValueError(
            f"target_idx rank {len(idx.shape)} must be >= object lead rank {len(obj_lead)}."
        )
    is_target = jax.nn.one_hot(idx, m, dtype=jnp.float32)

    def _sel(arr):
        arr_f = jnp.asarray(arr, dtype=jnp.float32)
        if extra == 0:
            # is_target [..., M] ; arr [..., M] → [...]
            return jnp.sum(arr_f * is_target, axis=-1)
        # is_target [..., K, M] ; arr [..., M] → [..., K]
        return jnp.einsum("...m,...km->...k", arr_f, is_target)

    x = _sel(objs.x)
    y = _sel(objs.y)
    w = _sel(objs.width)
    h = _sel(objs.height)
    active = _sel(objs.active) > 0.5
    valid = jnp.logical_and(active, jnp.logical_and(w > 0, h > 0))
    return x, y, w, h, valid


def normalize_pixel_obs(pixel_obs: jnp.ndarray) -> jnp.ndarray:
    """Cast ``pixel_obs`` to float32 in ``[0, 1]`` (accepts ``[0, 255]``)."""
    if pixel_obs is None:
        raise ValueError("pixel_obs is required.")
    img = pixel_obs.astype(jnp.float32)
    return jnp.where(jnp.max(img) > 1.5, img / 255.0, img)


def paint_target_outline_pixels(
    pixel_obs: jnp.ndarray,
    objects,
    target_idx,
    *,
    max_objects: int,
    thickness: int = DEFAULT_OUTLINE_THICKNESS,
) -> jnp.ndarray:
    """
    Draw a target outline on every frame of ``pixel_obs``.

    ``pixel_obs`` layout: ``[..., T, H, W, C]`` (float or uint8).
    When ``target_idx`` has extra trailing object axes beyond the pixel batch
    axes, the image is broadcast per target so each slot gets its own outline
    (needed for relabel-all-objects).
    """
    if pixel_obs is None:
        raise ValueError("pixel_obs is required to paint a target outline.")
    if objects is None:
        raise ValueError("objects is required to paint a target outline.")

    img = normalize_pixel_obs(pixel_obs)

    if img.ndim < 4:
        raise ValueError(
            f"pixel_obs must have rank >=4 with trailing [T,H,W,C], got rank {img.ndim}."
        )

    pixel_lead = img.shape[:-4]
    target_idx = jnp.asarray(target_idx, dtype=jnp.int32)
    target_lead = target_idx.shape
    extra = len(target_lead) - len(pixel_lead)
    if extra < 0:
        raise ValueError(
            f"target_idx rank {len(target_lead)} must be >= pixel lead rank {len(pixel_lead)}."
        )

    t, h_img, w_img, c = img.shape[-4:]
    lead = target_lead if extra > 0 else pixel_lead
    if extra > 0:
        for _ in range(extra):
            img = jnp.expand_dims(img, axis=len(pixel_lead))
        img = jnp.broadcast_to(img, lead + (t, h_img, w_img, c))

    x, y, w, h, valid = _gather_target_bbox(objects, target_idx, max_objects)
    x, y, w, h = scale_bbox_to_frame(x, y, w, h, h_img, w_img)

    n = math.prod(lead) if lead else 1
    img_flat = jnp.reshape(img, (n, t, h_img, w_img, c))
    x = jnp.reshape(jnp.broadcast_to(x, lead), (n,))
    y = jnp.reshape(jnp.broadcast_to(y, lead), (n,))
    w = jnp.reshape(jnp.broadcast_to(w, lead), (n,))
    h = jnp.reshape(jnp.broadcast_to(h, lead), (n,))
    valid = jnp.reshape(jnp.broadcast_to(valid, lead), (n,))

    def _paint_one(frame_t, xi, yi, wi, hi, vivi):
        def _paint_frame(fr):
            return paint_outline_on_hwc(
                fr, xi, yi, wi, hi, thickness=thickness, active=vivi
            )

        return jax.vmap(_paint_frame)(frame_t)

    painted = jax.vmap(_paint_one)(img_flat, x, y, w, h, valid)
    if lead:
        return jnp.reshape(painted, lead + (t, h_img, w_img, c))
    return painted[0]


def _fit_object_axis(arr: jnp.ndarray, max_objects: int) -> jnp.ndarray:
    """Pad/truncate the trailing object axis to ``max_objects``."""
    m = arr.shape[-1]
    if m == max_objects:
        return arr
    if m > max_objects:
        return arr[..., :max_objects]
    pad = [(0, 0)] * (arr.ndim - 1) + [(0, max_objects - m)]
    return jnp.pad(arr, pad)


def build_oc_object_matrix(objects, *, max_objects: int) -> jnp.ndarray:
    """
    Full object-observation matrix ``[..., max_objects, F]`` (no is_target).

    Flattens **all** numeric leaves of the labeled ``objects`` pytree (same
    source meta uses for its per-object features): bbox, active, visual_id,
    state, orientation, semantic_label, flags, …
    """
    if objects is None:
        raise ValueError("objects is required for oc_embed features.")
    objs = _last_frame_objects(objects)
    ref = jnp.asarray(objs.x, dtype=jnp.float32)
    ref = _fit_object_axis(ref, max_objects)
    lead_shape = ref.shape[:-1]

    leaves, _ = jax.tree_util.tree_flatten(objs)
    cols = []
    for leaf in leaves:
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        if arr.ndim == 0:
            arr = jnp.broadcast_to(arr, ref.shape)
        else:
            # Align leading dims to ref, then pad/truncate the object axis.
            if arr.shape[:-1] != lead_shape and arr.ndim >= 1:
                # Object-axis length may still differ from ref before fitting.
                try:
                    arr = jnp.broadcast_to(arr, lead_shape + (arr.shape[-1],))
                except ValueError:
                    flat = arr.reshape(-1)
                    arr = jnp.broadcast_to(flat, lead_shape + flat.shape)
                    if arr.shape[-1] != ref.shape[-1] and arr.shape != ref.shape:
                        # Non-object-shaped exotic leaf — skip.
                        continue
            arr = _fit_object_axis(arr, max_objects)
        cols.append(arr[..., None])
    if not cols:
        raise ValueError("objects pytree produced no numeric leaves for oc_embed.")
    return jnp.concatenate(cols, axis=-1)


def gather_object_rows(
    features: jnp.ndarray,
    target_idx,
    *,
    max_objects: int,
) -> jnp.ndarray:
    """
    Select object row(s) from ``[..., M, D]`` using ``target_idx``.

    ``target_idx`` shape is either the feature lead dims (single target →
    ``[..., D]``) or lead + ``(K,)`` (relabel → ``[..., K, D]``).
    """
    features = jnp.asarray(features, dtype=jnp.float32)
    target_idx = jnp.asarray(target_idx, dtype=jnp.int32)
    m = features.shape[-2]
    obj_lead = features.shape[:-2]
    extra = len(target_idx.shape) - len(obj_lead)
    if extra < 0:
        raise ValueError(
            f"target_idx rank {len(target_idx.shape)} must be >= feature lead "
            f"rank {len(obj_lead)}."
        )
    is_target = jax.nn.one_hot(
        jnp.clip(target_idx, 0, max_objects - 1), m
    ).astype(jnp.float32)
    if extra == 0:
        # is_target: [..., M] ; features: [..., M, D]
        return jnp.sum(features * is_target[..., None], axis=-2)
    # is_target: [..., K, M] ; features: [..., M, D]
    return jnp.einsum("...md,...km->...kd", features, is_target)


def oc_per_object_feature_dim(objects, max_objects: int) -> int:
    """Probe per-object channel count ``F`` (Dense input width for skill OC)."""
    return int(build_oc_object_matrix(objects, max_objects=max_objects).shape[-1])
