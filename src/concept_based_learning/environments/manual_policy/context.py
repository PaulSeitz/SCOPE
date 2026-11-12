"""Per-step geometric context shared by plan / interrupt evaluation."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from concept_based_learning.observations.egocentric import player_object_aabb_gaps


class FrameContext(NamedTuple):
    """Cached last-frame geometry for one env step.

    ``distances`` are AABB surface gaps (same metric as skill ego distance).
    ``px``/``py``/``ox``/``oy`` are centres; ``dx``/``dy`` are absolute centre
    offsets used by elliptical interrupt ranges.
    ``feet_y`` is the player bbox bottom (screen-y down).
    """

    px: jnp.ndarray
    py: jnp.ndarray
    ox: jnp.ndarray
    oy: jnp.ndarray
    active: jnp.ndarray
    labels: jnp.ndarray
    distances: jnp.ndarray
    dx: jnp.ndarray
    dy: jnp.ndarray
    feet_y: jnp.ndarray
    n: int


def build_frame_context(raw_observation: Any, objects: Any) -> FrameContext:
    distances, px, py, ox, oy = player_object_aabb_gaps(raw_observation, objects)
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]
    dx = jnp.abs(px - ox)
    dy = jnp.abs(py - oy)
    player = raw_observation.player
    feet_y = player.y[-1] + player.height[-1]
    return FrameContext(
        px=px,
        py=py,
        ox=ox,
        oy=oy,
        active=active,
        labels=labels,
        distances=distances,
        dx=dx,
        dy=dy,
        feet_y=feet_y,
        n=int(active.shape[0]),
    )
