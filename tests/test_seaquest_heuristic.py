"""Unit tests for Seaquest heuristic target selection."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NAVIGATE,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX
from concept_based_learning.environments.seaquest.heuristic_policy import (
    CLOSE_DANGER_RANGE,
    SURFACE_DUMMY_IDX,
    SURFACE_SUB_IDX,
    SURFACE_SUB_NEAR_RANGE,
    get_heuristic_target,
)


def _stack(value, n=1):
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return jnp.full((n,), arr)
    return jnp.broadcast_to(arr, (n,) + arr.shape)


def _make_raw(
    *,
    px=80.0,
    py=100.0,
    pw=16.0,
    ph=11.0,
    oxygen=64.0,
    divers=0,
    surface_sub_x=20.0,
    surface_sub_active=0,
):
    n = 1
    enemies_x = jnp.zeros(25, dtype=jnp.float32)
    enemies_w = jnp.full(25, 8.0, dtype=jnp.float32)
    enemies_active = jnp.zeros(25, dtype=jnp.float32)
    enemies_x = enemies_x.at[24].set(surface_sub_x)
    enemies_active = enemies_active.at[24].set(float(surface_sub_active))
    return SimpleNamespace(
        player=SimpleNamespace(
            x=_stack(px, n),
            y=_stack(py, n),
            width=_stack(pw, n),
            height=_stack(ph, n),
        ),
        oxygen_level=_stack(oxygen, n),
        collected_divers=_stack(divers, n),
        enemies=SimpleNamespace(
            x=enemies_x,
            width=enemies_w,
            active=enemies_active,
        ),
    )


def _make_objects(
    *,
    diver_xy=None,
    shark_xy=None,
    surface_sub_xy=None,
    surface_sub_active=False,
    n_slots=35,
):
    """Build stacked objects with meta-NOOP at 0 (caller uses game slots 1..)."""
    x = jnp.zeros((1, n_slots), dtype=jnp.float32)
    y = jnp.zeros((1, n_slots), dtype=jnp.float32)
    w = jnp.full((1, n_slots), 8.0, dtype=jnp.float32)
    h = jnp.full((1, n_slots), 8.0, dtype=jnp.float32)
    active = jnp.zeros((1, n_slots), dtype=jnp.float32)
    labels = jnp.full((1, n_slots), -1, dtype=jnp.int32)

    # Slot layout after meta NOOP: divers 1-4, sharks 5-16, ..., surface sub 29, WP 34
    labels = labels.at[0, 1:5].set(LABEL_COLLECT)
    labels = labels.at[0, 5:17].set(LABEL_DESTROY)
    labels = labels.at[0, 17:29].set(LABEL_DESTROY)
    labels = labels.at[0, SURFACE_SUB_IDX].set(LABEL_AVOID)
    labels = labels.at[0, 30:34].set(LABEL_AVOID)
    labels = labels.at[0, SURFACE_DUMMY_IDX].set(LABEL_NAVIGATE)
    active = active.at[0, SURFACE_DUMMY_IDX].set(1.0)
    # Surface WP tracks player-ish; exact pos unused when excluded from scoring.
    y = y.at[0, SURFACE_DUMMY_IDX].set(44.0)

    if diver_xy is not None:
        dx, dy = diver_xy
        x = x.at[0, 1].set(dx)
        y = y.at[0, 1].set(dy)
        active = active.at[0, 1].set(1.0)

    if shark_xy is not None:
        sx, sy = shark_xy
        x = x.at[0, 5].set(sx)
        y = y.at[0, 5].set(sy)
        active = active.at[0, 5].set(1.0)

    if surface_sub_xy is not None:
        ssx, ssy = surface_sub_xy
        x = x.at[0, SURFACE_SUB_IDX].set(ssx)
        y = y.at[0, SURFACE_SUB_IDX].set(ssy)
        w = w.at[0, SURFACE_SUB_IDX].set(8.0)
        h = h.at[0, SURFACE_SUB_IDX].set(11.0)
        if surface_sub_active:
            active = active.at[0, SURFACE_SUB_IDX].set(1.0)

    return SimpleNamespace(
        x=x,
        y=y,
        width=w,
        height=h,
        active=active,
        semantic_label=labels,
    )


def test_close_danger_range_matches_plan():
    assert CLOSE_DANGER_RANGE == 30.0
    assert SURFACE_SUB_NEAR_RANGE == 15.0


def test_prefers_diver_when_threat_is_far():
    # Player at (80,100); diver nearby; shark far away (>30px).
    raw = _make_raw(px=80.0, py=100.0)
    objs = _make_objects(diver_xy=(100.0, 100.0), shark_xy=(80.0, 160.0))
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 1  # diver slot


def test_close_threat_overrides_diver():
    raw = _make_raw(px=80.0, py=100.0)
    # Shark within 30px, diver also present.
    objs = _make_objects(diver_xy=(120.0, 100.0), shark_xy=(90.0, 105.0))
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 5  # shark


def test_surface_sub_ignored_when_deep():
    raw = _make_raw(px=80.0, py=120.0)  # deep underwater
    objs = _make_objects(
        diver_xy=(100.0, 120.0),
        surface_sub_xy=(85.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 1  # diver, not surface sub


def test_surface_sub_selectable_near_surface_and_close():
    raw = _make_raw(px=80.0, py=50.0)  # near surface
    objs = _make_objects(
        diver_xy=(140.0, 100.0),
        surface_sub_xy=(85.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == SURFACE_SUB_IDX


def test_surface_wp_when_full_divers_and_sub_left():
    raw = _make_raw(
        px=80.0,
        py=100.0,
        divers=6,
        oxygen=64.0,
        surface_sub_x=20.0,
        surface_sub_active=1,
    )
    objs = _make_objects(
        diver_xy=(100.0, 100.0),
        surface_sub_xy=(20.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == SURFACE_DUMMY_IDX


def test_surface_wp_blocked_when_sub_on_right_unless_critical_o2():
    raw = _make_raw(
        px=80.0,
        py=100.0,
        divers=6,
        oxygen=64.0,
        surface_sub_x=120.0,
        surface_sub_active=1,
    )
    objs = _make_objects(
        diver_xy=(100.0, 100.0),
        surface_sub_xy=(120.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 1  # stay on diver; wait for clear approach

    raw_crit = _make_raw(
        px=80.0,
        py=100.0,
        divers=6,
        oxygen=4.0,
        surface_sub_x=120.0,
        surface_sub_active=1,
    )
    target_crit = get_heuristic_target(raw_crit, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target_crit) == SURFACE_DUMMY_IDX
