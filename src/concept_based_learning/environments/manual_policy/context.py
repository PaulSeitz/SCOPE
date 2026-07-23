"""Per-step geometric context shared by plan / interrupt evaluation."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from concept_based_learning.observations.egocentric import _get_player


class FrameContext(NamedTuple):
    """Cached last-frame geometry for one env step.

    All arrays are unbatched (single env).  Centres use the same convention as
    ``observations.egocentric`` (x + w/2, y + h/2).
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
    n: int


def build_frame_context(raw_observation: Any, objects: Any) -> FrameContext:
    player = _get_player(raw_observation)
    px = player.x[-1] + player.width[-1] / 2.0
    py = player.y[-1] + player.height[-1] / 2.0

    ox = objects.x[-1] + objects.width[-1] / 2.0
    oy = objects.y[-1] + objects.height[-1] / 2.0
    active = objects.active[-1] > 0.5
    labels = objects.semantic_label[-1]
    dx = jnp.abs(px - ox)
    dy = jnp.abs(py - oy)
    distances = jnp.sqrt(dx ** 2 + dy ** 2)
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
        n=int(active.shape[0]),
    )
