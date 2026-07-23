"""
Env-agnostic line-of-sight test between player and target.

Computes whether any occluding object lies on the straight segment from the
player center to the target center.  Occluders are simply object slots whose
semantic label is in BLOCKING_LABELS (enemies, hazards, jump obstacles, and
LABEL_BLOCKER virtual objects emitted by env label files for structural
geometry like walls or shields).  Because the blocker geometry arrives through
the ordinary object list, this module needs no per-env branches — env-specific
knowledge lives entirely in the label files.

Semantics: a *thin* ray (line of fire / direct sight), not a body-width sweep.
The target slot itself never blocks; an occluder overlapping the player center
counts as blocking (you are in contact with something).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import BLOCKING_LABELS

# Segment direction components smaller than this are treated as axis-parallel.
_PARALLEL_EPS = 1e-6


def _axis_t_range(p, d, lo, hi):
    """
    Entry/exit parameters of segment p + t*d against slab [lo, hi] on one axis.

    Parallel segments yield (-inf, +inf) when inside the slab and an empty
    range when outside, per the standard robust slab test.
    """
    parallel = jnp.abs(d) < _PARALLEL_EPS
    safe_d = jnp.where(parallel, 1.0, d)
    t1 = (lo - p) / safe_d
    t2 = (hi - p) / safe_d
    t_min = jnp.minimum(t1, t2)
    t_max = jnp.maximum(t1, t2)
    inside = jnp.logical_and(p >= lo, p <= hi)
    t_min = jnp.where(parallel, jnp.where(inside, -jnp.inf, jnp.inf), t_min)
    t_max = jnp.where(parallel, jnp.where(inside, jnp.inf, -jnp.inf), t_max)
    return t_min, t_max


def segment_intersects_box(p_x, p_y, q_x, q_y, box_x_min, box_y_min, box_x_max, box_y_max):
    """True when segment (p → q) intersects the AABB (slab test, t ∈ [0, 1])."""
    d_x = q_x - p_x
    d_y = q_y - p_y
    tx_min, tx_max = _axis_t_range(p_x, d_x, box_x_min, box_x_max)
    ty_min, ty_max = _axis_t_range(p_y, d_y, box_y_min, box_y_max)
    t_enter = jnp.maximum(tx_min, ty_min)
    t_exit = jnp.minimum(tx_max, ty_max)
    return jnp.logical_and(
        t_enter <= t_exit,
        jnp.logical_and(t_exit >= 0.0, t_enter <= 1.0),
    )


def line_of_sight_blocked(
    p_cx,
    p_cy,
    t_cx,
    t_cy,
    target_idx,
    obj_x,
    obj_y,
    obj_w,
    obj_h,
    obj_active,
    obj_labels,
    *,
    y_is_bottom_edge: bool,
):
    """
    Single-frame line-of-sight test: is the player→target segment occluded?

    Args:
        p_cx, p_cy   : player center (scalars, float)
        t_cx, t_cy   : target center (scalars, float)
        target_idx   : int32 scalar — target slot (excluded from occluders)
        obj_*        : object arrays of shape [max_objects]
        y_is_bottom_edge : env y convention (True e.g. for Freeway)

    Returns:
        bool scalar — True when any active object with a label in
        BLOCKING_LABELS (other than the target) intersects the segment.
    """
    x_min = obj_x.astype(jnp.float32)
    x_max = x_min + obj_w.astype(jnp.float32)
    if y_is_bottom_edge:
        y_max = obj_y.astype(jnp.float32)
        y_min = y_max - obj_h.astype(jnp.float32)
    else:
        y_min = obj_y.astype(jnp.float32)
        y_max = y_min + obj_h.astype(jnp.float32)

    hits = jax.vmap(
        lambda bx_min, by_min, bx_max, by_max: segment_intersects_box(
            p_cx, p_cy, t_cx, t_cy, bx_min, by_min, bx_max, by_max
        )
    )(x_min, y_min, x_max, y_max)

    is_blocking_label = jnp.zeros_like(obj_labels, dtype=jnp.bool_)
    for label in BLOCKING_LABELS:
        is_blocking_label = jnp.logical_or(is_blocking_label, obj_labels == label)

    occluder = jnp.logical_and(
        obj_active > 0,
        jnp.logical_and(
            is_blocking_label,
            jnp.arange(obj_x.shape[0], dtype=jnp.int32) != target_idx,
        ),
    )
    return jnp.any(jnp.logical_and(hits, occluder))
