"""Unit tests for Seaquest heuristic target selection."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX
from concept_based_learning.environments.seaquest.heuristic_policy import (
    AVOID_INTERRUPT_RANGE,
    AVOID_INTERRUPT_RANGE_Y,
    DESTROY_INTERRUPT_RANGE,
    DIVER_LANE_Y_PX,
    O2_CRITICAL_ENEMY_INTERRUPT_RANGE,
    O2_LOW_ENEMY_INTERRUPT_RANGE,
    SURFACE_DANGER_CLOSE_RANGE,
    SURFACE_DUMMY_IDX,
    SURFACE_SUB_AVOID_RANGE_X,
    SURFACE_SUB_AVOID_RANGE_Y,
    SURFACE_SUB_IDX,
    SURFACE_SUB_NEAR_RANGE,
    apply_meta_mask_overrides,
    get_heuristic_target,
)
from concept_based_learning.environments.seaquest.label import (
    SEAQUEST_SURFACE_LABEL,
    SURFACE_WP_SLOT_CX,
    SURFACE_WP_SUB_CLEAR_PX,
    select_surface_waypoint_x,
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
        oxygen_level=_stack(oxygen, n) if jnp.asarray(oxygen).ndim == 0 else jnp.asarray(oxygen, dtype=jnp.float32),
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
    shark2_xy=None,
    sub_xy=None,
    missile_xy=None,
    surface_sub_xy=None,
    surface_sub_active=False,
    stack_len=1,
    n_slots=35,
):
    """Build stacked objects: game slots 0..33, trailing meta-NOOP at 34."""
    from concept_based_learning.environments.datatypes import LABEL_NOOP

    x = jnp.zeros((stack_len, n_slots), dtype=jnp.float32)
    y = jnp.zeros((stack_len, n_slots), dtype=jnp.float32)
    w = jnp.full((stack_len, n_slots), 8.0, dtype=jnp.float32)
    h = jnp.full((stack_len, n_slots), 8.0, dtype=jnp.float32)
    active = jnp.zeros((stack_len, n_slots), dtype=jnp.float32)
    labels = jnp.full((stack_len, n_slots), -1, dtype=jnp.int32)

    # Append-NOOP layout (matches SeaquestSemanticWrapper + append_meta_reserved_slots).
    labels = labels.at[:, 0:4].set(LABEL_COLLECT)       # divers
    labels = labels.at[:, 4:16].set(LABEL_DESTROY)      # sharks
    labels = labels.at[:, 16:28].set(LABEL_DESTROY)     # subs
    labels = labels.at[:, SURFACE_SUB_IDX].set(LABEL_AVOID)
    labels = labels.at[:, 29:33].set(LABEL_AVOID)       # missiles
    labels = labels.at[:, SURFACE_DUMMY_IDX].set(SEAQUEST_SURFACE_LABEL)
    labels = labels.at[:, n_slots - 1].set(LABEL_NOOP)
    active = active.at[:, SURFACE_DUMMY_IDX].set(1.0)
    y = y.at[:, SURFACE_DUMMY_IDX].set(44.0)

    if diver_xy is not None:
        dx, dy = diver_xy
        x = x.at[:, 0].set(dx)
        y = y.at[:, 0].set(dy)
        active = active.at[:, 0].set(1.0)

    if shark_xy is not None:
        sx, sy = shark_xy
        x = x.at[:, 4].set(sx)
        y = y.at[:, 4].set(sy)
        active = active.at[:, 4].set(1.0)

    if shark2_xy is not None:
        sx, sy = shark2_xy
        x = x.at[:, 5].set(sx)
        y = y.at[:, 5].set(sy)
        active = active.at[:, 5].set(1.0)

    if sub_xy is not None:
        sx, sy = sub_xy
        x = x.at[:, 16].set(sx)
        y = y.at[:, 16].set(sy)
        active = active.at[:, 16].set(1.0)

    if missile_xy is not None:
        mx, my = missile_xy
        x = x.at[:, 29].set(mx)
        y = y.at[:, 29].set(my)
        w = w.at[:, 29].set(4.0)
        h = h.at[:, 29].set(2.0)
        active = active.at[:, 29].set(1.0)

    if surface_sub_xy is not None:
        ssx, ssy = surface_sub_xy
        x = x.at[:, SURFACE_SUB_IDX].set(ssx)
        y = y.at[:, SURFACE_SUB_IDX].set(ssy)
        w = w.at[:, SURFACE_SUB_IDX].set(8.0)
        h = h.at[:, SURFACE_SUB_IDX].set(11.0)
        if surface_sub_active:
            active = active.at[:, SURFACE_SUB_IDX].set(1.0)

    return SimpleNamespace(
        x=x,
        y=y,
        width=w,
        height=h,
        active=active,
        semantic_label=labels,
    )


def test_surface_waypoint_uses_collect_skill():
    import concept_based_learning.environments.seaquest.label as sq_label

    assert sq_label.SEAQUEST_SURFACE_LABEL == LABEL_COLLECT


def test_set_seaquest_surface_label_navigate_and_reset():
    import concept_based_learning.environments.seaquest.label as sq_label
    from concept_based_learning.environments.datatypes import LABEL_NAVIGATE

    sq_label.set_seaquest_surface_label("navigate")
    assert sq_label.SEAQUEST_SURFACE_LABEL == LABEL_NAVIGATE
    sq_label.set_seaquest_surface_label("collect")
    assert sq_label.SEAQUEST_SURFACE_LABEL == LABEL_COLLECT


def test_surface_waypoint_picks_nearest_slot_without_sub():
    # Player width 16 → left edges 20 / 72 / 124 for centres 28 / 80 / 132.
    mid = float(
        select_surface_waypoint_x(
            80.0, 16.0, sub_active=False, sub_x=0.0, sub_w=8.0
        )
    )
    left = float(
        select_surface_waypoint_x(
            20.0, 16.0, sub_active=False, sub_x=0.0, sub_w=8.0
        )
    )
    right = float(
        select_surface_waypoint_x(
            140.0, 16.0, sub_active=False, sub_x=0.0, sub_w=8.0
        )
    )
    assert mid == pytest.approx(SURFACE_WP_SLOT_CX[1] - 8.0)
    assert left == pytest.approx(SURFACE_WP_SLOT_CX[0] - 8.0)
    assert right == pytest.approx(SURFACE_WP_SLOT_CX[2] - 8.0)


def test_surface_waypoint_avoids_sub_on_the_right():
    # Player at mid, sub approaching from the right → left slot, not 10px behind.
    x = float(
        select_surface_waypoint_x(
            80.0, 16.0, sub_active=True, sub_x=120.0, sub_w=8.0
        )
    )
    sub_cx = 120.0 + 4.0
    wp_cx = x + 8.0
    assert wp_cx == pytest.approx(SURFACE_WP_SLOT_CX[0])
    assert abs(wp_cx - sub_cx) >= SURFACE_WP_SUB_CLEAR_PX


def test_surface_danger_close_range():
    assert SURFACE_DANGER_CLOSE_RANGE == 25.0
    assert DESTROY_INTERRUPT_RANGE == 40.0
    assert AVOID_INTERRUPT_RANGE == 55.0
    assert AVOID_INTERRUPT_RANGE_Y == 18.0
    assert SURFACE_SUB_AVOID_RANGE_X == 5.0
    assert SURFACE_SUB_AVOID_RANGE_Y == 5.0
    assert DIVER_LANE_Y_PX == 14.0
    assert O2_LOW_ENEMY_INTERRUPT_RANGE == 12.0
    assert O2_CRITICAL_ENEMY_INTERRUPT_RANGE == 0.0
    assert SURFACE_SUB_NEAR_RANGE == 5.0


def test_always_nearest_object():
    raw = _make_raw(px=80.0, py=100.0)
    # Shark closer than diver → shark.
    objs = _make_objects(diver_xy=(140.0, 100.0), shark_xy=(100.0, 100.0))
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_destroy_interrupt_beats_nearer_diver():
    """Shark within DESTROY_INTERRUPT_RANGE wins even if a diver is closer."""
    raw = _make_raw(px=80.0, py=100.0, pw=16.0, ph=11.0)
    objs = _make_objects(diver_xy=(95.0, 100.0), shark_xy=(120.0, 100.0))
    from concept_based_learning.observations.egocentric import aabb_surface_gap_px

    gap_diver = float(aabb_surface_gap_px(80.0, 100.0, 16.0, 11.0, 95.0, 100.0, 8.0, 8.0))
    gap_shark = float(aabb_surface_gap_px(80.0, 100.0, 16.0, 11.0, 120.0, 100.0, 8.0, 8.0))
    assert gap_diver < gap_shark
    assert gap_shark <= DESTROY_INTERRUPT_RANGE

    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_avoid_interrupt_steals_destroy_when_torpedo_is_in_band():
    """A fired torpedo inside AVOID_INTERRUPT_RANGE beats a closer shark/sub."""
    raw = _make_raw(px=80.0, py=100.0, pw=16.0, ph=11.0)
    objs = _make_objects(shark_xy=(95.0, 100.0), missile_xy=(145.0, 100.0))
    from concept_based_learning.observations.egocentric import aabb_surface_gap_px

    gap_missile = float(
        aabb_surface_gap_px(80.0, 100.0, 16.0, 11.0, 145.0, 100.0, 4.0, 2.0)
    )
    assert DESTROY_INTERRUPT_RANGE < gap_missile <= AVOID_INTERRUPT_RANGE
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 29


def test_far_avoid_does_not_steal_nearest_destroy():
    """AVOID outside the interrupt band stays out of DESTROY/COLLECT fights."""
    raw = _make_raw(px=80.0, py=100.0, pw=16.0, ph=11.0)
    objs = _make_objects(shark_xy=(95.0, 100.0), missile_xy=(170.0, 100.0))
    from concept_based_learning.observations.egocentric import aabb_surface_gap_px

    gap_missile = float(
        aabb_surface_gap_px(80.0, 100.0, 16.0, 11.0, 170.0, 100.0, 4.0, 2.0)
    )
    assert gap_missile > AVOID_INTERRUPT_RANGE
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_other_lane_missile_does_not_steal_destroy():
    """Oval AVOID: a torpedo far in Y (other row) must not lock dodge."""
    raw = _make_raw(px=80.0, py=100.0, pw=16.0, ph=11.0)
    objs = _make_objects(shark_xy=(95.0, 100.0), missile_xy=(90.0, 50.0))
    from concept_based_learning.observations.egocentric import aabb_closest_delta_px

    gap_dx, gap_dy, _ = aabb_closest_delta_px(
        80.0, 100.0, 16.0, 11.0, 90.0, 50.0, 4.0, 2.0
    )
    radius = (float(gap_dx) / AVOID_INTERRUPT_RANGE) ** 2 + (
        float(gap_dy) / AVOID_INTERRUPT_RANGE_Y
    ) ** 2
    assert radius > 1.0
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_same_lane_far_shark_blocks_diver():
    """Same-row shark (beyond DESTROY_INTERRUPT_RANGE) still beats the diver."""
    raw = _make_raw(px=80.0, py=100.0)
    objs = _make_objects(diver_xy=(95.0, 100.0), shark_xy=(140.0, 100.0))
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_other_lane_shark_allows_diver():
    """A shark in a different row does not block a closer clear-lane diver."""
    raw = _make_raw(px=80.0, py=100.0)
    objs = _make_objects(diver_xy=(95.0, 100.0), shark_xy=(140.0, 160.0))
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 0


def test_collect_requires_no_destroy_steals_other_lane_diver():
    """Any live DESTROY beats COLLECT, including a far other-lane shark."""
    raw = _make_raw(px=80.0, py=100.0)
    objs = _make_objects(diver_xy=(95.0, 100.0), shark_xy=(140.0, 160.0))
    target = get_heuristic_target(
        raw,
        objs,
        jnp.int32(NO_TARGET_IDX),
        None,
        collect_requires_no_destroy=True,
    )
    assert int(target) == 4


def test_collect_requires_no_destroy_does_not_steal_surface_wp():
    """Surface WP still wins when O2/divers require surfacing, even with a shark."""
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
        shark_xy=(140.0, 160.0),
        surface_sub_xy=(20.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(
        raw,
        objs,
        jnp.int32(NO_TARGET_IDX),
        None,
        collect_requires_no_destroy=True,
    )
    assert int(target) == SURFACE_DUMMY_IDX


def test_nearest_among_multiple_sharks():
    raw = _make_raw(px=80.0, py=100.0, pw=16.0, ph=11.0)
    objs = _make_objects(
        diver_xy=(160.0, 100.0),
        shark_xy=(120.0, 100.0),
        shark2_xy=(100.0, 100.0),
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 5  # nearer shark2 slot (append layout)


def test_edge_gap_prefers_nose_shark_over_center_nearer_diver():
    """Centre→centre would pick diver below; surface-gap picks nose-on shark.

    Matches the failure mode where COLLECT is selected while a shark sits on
    the player's nose (edge gap ~0, centre distance still larger than diver).
    """
    raw = _make_raw(px=80.0, py=100.0, pw=16.0, ph=11.0)
    objs = _make_objects(
        diver_xy=(84.0, 114.0),
        shark_xy=(98.0, 102.0),
    )
    from concept_based_learning.observations.egocentric import aabb_surface_gap_px

    pw, ph = 16.0, 11.0
    p_cx, p_cy = 80.0 + pw / 2.0, 100.0 + ph / 2.0
    diver_cx, diver_cy = 84.0 + 4.0, 114.0 + 4.0
    shark_cx, shark_cy = 98.0 + 4.0, 102.0 + 4.0
    center_diver = ((diver_cx - p_cx) ** 2 + (diver_cy - p_cy) ** 2) ** 0.5
    center_shark = ((shark_cx - p_cx) ** 2 + (shark_cy - p_cy) ** 2) ** 0.5
    assert center_diver < center_shark

    gap_diver = float(
        aabb_surface_gap_px(80.0, 100.0, pw, ph, 84.0, 114.0, 8.0, 8.0)
    )
    gap_shark = float(
        aabb_surface_gap_px(80.0, 100.0, pw, ph, 98.0, 102.0, 8.0, 8.0)
    )
    assert gap_shark < gap_diver

    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_surface_sub_ignored_when_deep():
    raw = _make_raw(px=80.0, py=120.0)
    objs = _make_objects(
        diver_xy=(100.0, 120.0),
        surface_sub_xy=(85.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 0


def test_surface_sub_avoids_when_close_even_if_deep():
    """Ram-prevention: only at contact range (~5px), even below the surface band."""
    raw = _make_raw(px=80.0, py=60.0, pw=16.0, ph=11.0)
    objs = _make_objects(
        diver_xy=(120.0, 60.0),
        surface_sub_xy=(85.0, 46.0),
        surface_sub_active=True,
    )
    from concept_based_learning.observations.egocentric import aabb_closest_delta_px

    gap_dx, gap_dy, _ = aabb_closest_delta_px(
        80.0, 60.0, 16.0, 11.0, 85.0, 46.0, 8.0, 11.0
    )
    radius = (float(gap_dx) / SURFACE_SUB_AVOID_RANGE_X) ** 2 + (
        float(gap_dy) / SURFACE_SUB_AVOID_RANGE_Y
    ) ** 2
    assert radius <= 1.0
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == SURFACE_SUB_IDX


def test_surface_sub_does_not_steal_from_a_lane_away():
    """A shark in the upper lane must win over a still-distant surface sub."""
    raw = _make_raw(px=80.0, py=70.0, pw=16.0, ph=11.0)
    objs = _make_objects(
        shark_xy=(95.0, 70.0),
        surface_sub_xy=(85.0, 46.0),
        surface_sub_active=True,
    )
    from concept_based_learning.observations.egocentric import aabb_closest_delta_px

    gap_dx, gap_dy, _ = aabb_closest_delta_px(
        80.0, 70.0, 16.0, 11.0, 85.0, 46.0, 8.0, 11.0
    )
    radius = (float(gap_dx) / SURFACE_SUB_AVOID_RANGE_X) ** 2 + (
        float(gap_dy) / SURFACE_SUB_AVOID_RANGE_Y
    ) ** 2
    assert radius > 1.0
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_surface_sub_selectable_near_surface_and_close():
    raw = _make_raw(px=80.0, py=50.0)
    objs = _make_objects(
        diver_xy=(140.0, 100.0),
        surface_sub_xy=(85.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == SURFACE_SUB_IDX


def test_surface_refill_selects_nothing():
    """Rising oxygen means refill (actions ignored) — no object, any y."""
    raw = _make_raw(px=80.0, py=100.0, oxygen=(31.0, 32.0), divers=2)
    objs = _make_objects(
        diver_xy=(100.0, 100.0),
        shark_xy=(110.0, 100.0),
        missile_xy=(90.0, 90.0),
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 34


def test_breathing_band_with_low_oxygen_still_targets():
    """Frozen/draining O₂ is not refill — keep targeting in the breathing band."""
    raw = _make_raw(px=80.0, py=50.0, oxygen=(32.0, 32.0), divers=2)
    objs = _make_objects(diver_xy=(100.0, 100.0))
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 0


def test_oxygen_falling_still_targets():
    raw = _make_raw(px=80.0, py=46.0, oxygen=(40.0, 39.0), divers=2)
    objs = _make_objects(diver_xy=(100.0, 100.0))
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 0


def test_surface_full_oxygen_resumes_targeting():
    raw = _make_raw(px=80.0, py=46.0, oxygen=64.0, divers=2)
    objs = _make_objects(diver_xy=(100.0, 100.0))
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 0


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


def test_surface_wp_blocked_when_sub_on_right_unless_oxygen_low():
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
    assert int(target) == 0  # stay on nearest diver; wait for clear approach

    raw_low = _make_raw(
        px=80.0,
        py=100.0,
        divers=6,
        oxygen=10.0,
        surface_sub_x=120.0,
        surface_sub_active=1,
    )
    target_low = get_heuristic_target(raw_low, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target_low) == SURFACE_DUMMY_IDX

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


def test_surface_danger_close_overrides_waypoint():
    # Full divers + clear approach, but shark within 25px → interrupt surface.
    raw = _make_raw(
        px=80.0,
        py=100.0,
        divers=6,
        oxygen=64.0,
        surface_sub_x=20.0,
        surface_sub_active=1,
    )
    objs = _make_objects(
        diver_xy=(160.0, 100.0),
        shark_xy=(95.0, 100.0),
        surface_sub_xy=(20.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_surface_danger_close_range_knob_disables_override():
    raw = _make_raw(
        px=80.0,
        py=100.0,
        divers=6,
        oxygen=64.0,
        surface_sub_x=20.0,
        surface_sub_active=1,
    )
    objs = _make_objects(
        diver_xy=(160.0, 100.0),
        shark_xy=(95.0, 100.0),
        surface_sub_xy=(20.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(
        raw,
        objs,
        jnp.int32(NO_TARGET_IDX),
        None,
        danger_close_range=0.0,
        destroy_interrupt_range=0.0,
    )
    assert int(target) == SURFACE_DUMMY_IDX


def test_surface_no_danger_close_keeps_waypoint():
    # Shark farther than 25px while forced surface → stay on WP.
    raw = _make_raw(
        px=80.0,
        py=100.0,
        divers=6,
        oxygen=64.0,
        surface_sub_x=20.0,
        surface_sub_active=1,
    )
    objs = _make_objects(
        diver_xy=(160.0, 100.0),
        shark_xy=(140.0, 100.0),
        surface_sub_xy=(20.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == SURFACE_DUMMY_IDX


def test_low_oxygen_forces_waypoint_over_midrange_shark():
    """O₂ low shrinks DESTROY interrupt (40 → 12); a 24px shark must not block ascent."""
    raw = _make_raw(
        px=80.0,
        py=100.0,
        divers=0,
        oxygen=10.0,
        surface_sub_x=20.0,
        surface_sub_active=1,
    )
    objs = _make_objects(
        diver_xy=(160.0, 100.0),
        shark_xy=(120.0, 100.0),
        surface_sub_xy=(20.0, 46.0),
        surface_sub_active=True,
    )
    from concept_based_learning.observations.egocentric import aabb_surface_gap_px

    gap_shark = float(
        aabb_surface_gap_px(80.0, 100.0, 16.0, 11.0, 120.0, 100.0, 8.0, 8.0)
    )
    assert O2_LOW_ENEMY_INTERRUPT_RANGE < gap_shark <= DESTROY_INTERRUPT_RANGE
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == SURFACE_DUMMY_IDX


def test_low_oxygen_still_interrupts_nose_shark():
    """Low O₂ still dodges a shark on the nose (within the shrunk 12px band)."""
    raw = _make_raw(
        px=80.0,
        py=100.0,
        divers=0,
        oxygen=10.0,
        surface_sub_x=20.0,
        surface_sub_active=1,
    )
    objs = _make_objects(
        diver_xy=(160.0, 100.0),
        shark_xy=(95.0, 100.0),
        surface_sub_xy=(20.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == 4


def test_critical_oxygen_forces_waypoint_even_with_nose_shark():
    raw = _make_raw(
        px=80.0,
        py=100.0,
        divers=0,
        oxygen=4.0,
        surface_sub_x=20.0,
        surface_sub_active=1,
    )
    objs = _make_objects(
        diver_xy=(160.0, 100.0),
        shark_xy=(95.0, 100.0),
        surface_sub_xy=(20.0, 46.0),
        surface_sub_active=True,
    )
    target = get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None)
    assert int(target) == SURFACE_DUMMY_IDX


def test_meta_mask_overrides_batched_objects_surface_gate():
    """Batched objects are [B, stack, N]; must not index stack as object slots."""
    num_envs, stack_len, n_slots = 4, 2, 35
    px = jnp.full((num_envs, stack_len), 80.0, dtype=jnp.float32)
    py = jnp.full((num_envs, stack_len), 100.0, dtype=jnp.float32)
    pw = jnp.full((num_envs, stack_len), 16.0, dtype=jnp.float32)
    oxygen = jnp.full((num_envs, stack_len), 64.0, dtype=jnp.float32)
    divers = jnp.full((num_envs, stack_len), 6.0, dtype=jnp.float32)

    x = jnp.zeros((num_envs, stack_len, n_slots), dtype=jnp.float32)
    w = jnp.full((num_envs, stack_len, n_slots), 8.0, dtype=jnp.float32)
    active = jnp.zeros((num_envs, stack_len, n_slots), dtype=jnp.float32)
    x = x.at[0, -1, SURFACE_SUB_IDX].set(20.0)
    x = x.at[1, -1, SURFACE_SUB_IDX].set(120.0)
    active = active.at[:, -1, SURFACE_SUB_IDX].set(1.0)
    active = active.at[:, -1, SURFACE_DUMMY_IDX].set(1.0)

    raw = SimpleNamespace(
        player=SimpleNamespace(x=px, y=py, width=pw),
        oxygen_level=oxygen,
        collected_divers=divers,
    )
    objs = SimpleNamespace(x=x, width=w, active=active)
    masks = jnp.ones((num_envs, n_slots), dtype=bool)

    out = apply_meta_mask_overrides(raw, masks, {}, objects=objs)
    assert out.shape == (num_envs, n_slots)
    assert bool(out[0, SURFACE_DUMMY_IDX]) is True
    assert bool(out[1, SURFACE_DUMMY_IDX]) is False


def test_midfield_margin_prefers_far_mid_diver_over_near_edge():
    """With margin>0, a farther midfield diver beats a nearer edge diver."""
    # Player near left; edge diver just outside the sub; mid diver farther right.
    raw = _make_raw(px=40.0, py=100.0)
    objs = _make_objects(diver_xy=(2.0, 100.0))  # centre ~6, outside margin=24
    objs.x = objs.x.at[:, 2].set(110.0)  # centre ~114, midfield, farther
    objs.y = objs.y.at[:, 2].set(100.0)
    objs.active = objs.active.at[:, 2].set(1.0)

    without = int(get_heuristic_target(raw, objs, jnp.int32(NO_TARGET_IDX), None))
    with_mid = int(
        get_heuristic_target(
            raw,
            objs,
            jnp.int32(NO_TARGET_IDX),
            None,
            collect_edge_margin_px=24.0,
        )
    )
    assert without == 0  # nearer edge diver
    assert with_mid == 2  # midfield preference


def test_midfield_margin_falls_back_to_edge_when_no_mid_diver():
    raw = _make_raw(px=40.0, py=100.0)
    objs = _make_objects(diver_xy=(2.0, 100.0))
    target = int(
        get_heuristic_target(
            raw,
            objs,
            jnp.int32(NO_TARGET_IDX),
            None,
            collect_edge_margin_px=24.0,
        )
    )
    assert target == 0


def test_midfield_margin_zero_keeps_nearest_edge_diver():
    raw = _make_raw(px=40.0, py=100.0)
    objs = _make_objects(diver_xy=(2.0, 100.0))
    objs.x = objs.x.at[:, 2].set(110.0)
    objs.y = objs.y.at[:, 2].set(100.0)
    objs.active = objs.active.at[:, 2].set(1.0)
    target = int(
        get_heuristic_target(
            raw,
            objs,
            jnp.int32(NO_TARGET_IDX),
            None,
            collect_edge_margin_px=0.0,
        )
    )
    assert target == 0
