"""Declarative IR nodes for manual policies.

These frozen dataclasses are the Python helper surface today and the intended
AST for a future YAML/DSL frontend.  Prefer data (slots, labels, thresholds,
named predicates) over opaque callables so a DSL can round-trip later.

Opaque callables are allowed on ``ForceWhen.predicate`` / ``Interrupt.suppress``
for env-specific logic that has not been registered yet; pass a ``str`` name
instead once it lives in ``PREDICATE_REGISTRY``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_JUMP_OVER,
    LABEL_NAVIGATE,
)

# ---------------------------------------------------------------------------
# Named predicate registry (DSL hook)
# ---------------------------------------------------------------------------

PredicateFn = Callable[..., jnp.ndarray]
PREDICATE_REGISTRY: dict[str, PredicateFn] = {}


def register_predicate(name: str, fn: PredicateFn) -> PredicateFn:
    """Register a JAX-traceable predicate for DSL / ForceWhen lookup."""
    existing = PREDICATE_REGISTRY.get(name)
    if existing is not None and existing is not fn:
        raise ValueError(f"Predicate {name!r} already registered")
    PREDICATE_REGISTRY[name] = fn
    return fn


def resolve_predicate(pred: Union[str, PredicateFn]) -> PredicateFn:
    if isinstance(pred, str):
        if pred not in PREDICATE_REGISTRY:
            raise KeyError(
                f"Unknown predicate {pred!r}; known: {sorted(PREDICATE_REGISTRY)}"
            )
        return PREDICATE_REGISTRY[pred]
    return pred


# ---------------------------------------------------------------------------
# Plan nodes
# ---------------------------------------------------------------------------

# Default sticky mission labels: tasks the agent commits to until done.
DEFAULT_MISSION_LABELS: Tuple[int, ...] = (
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_NAVIGATE,
    LABEL_JUMP_OVER,
)


@dataclass(frozen=True)
class NavigateTo:
    """Target a fixed object slot until the stage completion rule fires.

    ``slot`` is an index into the full object array (including meta-reserved
    slots).      ``complete_when``:
      - ``"arrive"``: player centre within ``arrive_dist`` of the slot centre
        (ladder / surface waypoints).
      - ``"inactive"``: slot becomes inactive (fruit collected, enemy gone).
        Stays locked on the slot until then; ``arrive_dist`` is unused.
      - ``"either"``: arrive **or** inactive (skip already-cleared stages).
      - ``"crossed"``: player centre has passed the slot centre horizontally
        (``cross_dir`` +1 → ``px > ox + margin``, −1 → ``px < ox - margin``).
        Used for JUMP_OVER hazards; ``arrive_dist`` is unused.

    ``skip_if_inactive`` (arrive mode only): when True (default), a permanently
    inactive padded slot completes the stage immediately.  Set False when the
    slot may be gated off temporarily (e.g. kangaroo ladder under a falling
    coconut) so the plan waits instead of skipping.

    ``platform_reach_*`` (arrive mode): optional floor snap for raised top
    waypoints — stage also completes when the player stands on the platform
    near the slot even if centre-distance arrive never fires.  Prefer
    ``platform_reach_dy_min`` / ``platform_reach_dy_max`` (player centre below
    waypoint by dy = py - oy) over loose ``platform_reach_y_slack``.

    ``arrive_feet_oy_min`` / ``arrive_feet_oy_max`` (arrive mode): when set,
    AABB ``arrive_dist`` alone is not enough — also require
    ``arrive_feet_oy_min <= feet_y - oy <= arrive_feet_oy_max`` (omit a bound
    to leave it open).  ``min`` stops ladder-bottom stages completing mid-climb
    when the tall player box already overlaps a low waypoint; ``max`` stops
    ladder-top stages completing while still low in the shaft.

    ``arrive_x_slack`` (arrive mode): when set, also require
    ``|px - ox| <= arrive_x_slack`` so a vertically-overlapping but
    horizontally-far player (e.g. still in a ladder shaft beside a belt-lip
    COLLECT) cannot complete the stage.

    ``arrive_center_max`` (arrive mode): when set, stage completes when the
    player centre is within this many pixels of the slot centre
    (``hypot(px-ox, py-oy)``) — ignores AABB ``arrive_dist``.

    ``past_x`` (``complete_when="past_x"``): absolute clear line. With
    ``cross_dir`` +1, done when ``px > past_x``; with −1 when ``px < past_x``.

    ``skip_if_player_above_px``: when set, the stage completes (and stops
    targeting the slot) if the player's feet are higher on screen than the
    object centre by more than this many pixels.  Used for hanging fruits that
    must be grabbed from the floor below — once the agent has climbed past,
    do not dive back down.
    """

    slot: int
    arrive_dist: float = 8.0
    complete_when: str = "arrive"  # arrive|inactive|either|crossed|past_x
    skip_if_inactive: bool = True
    platform_reach_y_slack: Optional[float] = None
    platform_reach_x_slack: Optional[float] = None
    platform_reach_dy_min: Optional[float] = None
    platform_reach_dy_max: Optional[float] = None
    arrive_feet_oy_min: Optional[float] = None
    arrive_feet_oy_max: Optional[float] = None
    arrive_x_slack: Optional[float] = None
    arrive_center_max: Optional[float] = None
    skip_if_player_above_px: Optional[float] = None
    # crossed / past_x: +1 = clear to the right, -1 = clear to the left.
    cross_dir: Optional[int] = None
    cross_margin_px: float = 0.0
    past_x: Optional[float] = None
    kind: str = field(default="navigate_to", init=False)


@dataclass(frozen=True)
class DynamicPick:
    """Recompute a mission target from label priority + optional stickiness.

    ``prefer`` is highest-priority first (e.g. ``(DESTROY, COLLECT)``).
    ``exclude_slots`` are never chosen (e.g. SeaQuest surface dummy during
    normal play).  ``slot_mask`` if set is a boolean array factory
    ``(n,) -> mask`` evaluated each step for additional filtering.
    """

    prefer: Tuple[int, ...] = (LABEL_DESTROY, LABEL_COLLECT, LABEL_NAVIGATE)
    sticky: bool = True
    sticky_labels: Tuple[int, ...] = DEFAULT_MISSION_LABELS
    exclude_slots: Tuple[int, ...] = ()
    # Optional callable (n_objects,) -> bool mask; None = all slots eligible.
    # DSL later: named mask id in a registry.
    slot_mask: Optional[Callable[[int], jnp.ndarray]] = None
    kind: str = field(default="dynamic_pick", init=False)


@dataclass(frozen=True)
class ForceWhen:
    """Replace the plan target while ``predicate`` is true.

    ``predicate`` may be a callable ``(raw, objects, ctx) -> bool`` or a
    registered name string.  ``target_slot`` is the forced object index.
    """

    predicate: Union[str, PredicateFn]
    target_slot: int
    kind: str = field(default="force_when", init=False)


@dataclass(frozen=True)
class ClimbLadders:
    """Kangaroo-style paired bottom→top climb over a contiguous slot range.

    Bottom slots ``[bottom_start, bottom_start + n)`` pair with tops at
    ``top_start + i``.  While not climbing, optionally prefer ``between_prefer``
    labels (fruits, etc.) before picking the next ladder bottom.
    """

    bottom_start: int
    top_start: int
    n_ladders: int = 20
    arrive_dist: float = 10.0
    between_prefer: Tuple[int, ...] = (LABEL_COLLECT,)
    kind: str = field(default="climb_ladders", init=False)


PlanNode = Union[NavigateTo, DynamicPick, ClimbLadders]


@dataclass(frozen=True)
class Sequence:
    """Ordered plan stages; advance when the current stage reports done."""

    stages: Tuple[PlanNode, ...]
    kind: str = field(default="sequence", init=False)


@dataclass(frozen=True)
class Interrupt:
    """Reactive override that does not advance / clear the plan mission.

    Range can be isotropic (``within``) or axis-weighted (``within_x`` /
    ``within_y``).  With both axes set, a threat is in range when it lies
    inside the ellipse ``(dx/within_x)^2 + (dy/within_y)^2 <= 1`` — so a
    larger ``within_y`` reacts earlier to falling threats overhead while
    keeping side approaches tighter.

    If ``release`` is set, hysteresis keeps the interrupt latched until all
    matching threats leave the release range (same metric).  ``suppress``
    (optional predicate) disables the interrupt (e.g. while surfacing).

    ``threat_mask`` if set is ``(raw, objects, ctx, plan_target) -> bool[n]``
    further filtering which label-matched threats may fire the interrupt.
    """

    labels: Tuple[int, ...]
    within: float = 0.0
    within_x: Optional[float] = None
    within_y: Optional[float] = None
    release: Optional[float] = None
    release_x: Optional[float] = None
    release_y: Optional[float] = None
    suppress: Optional[Union[str, PredicateFn]] = None
    threat_mask: Optional[Callable[..., jnp.ndarray]] = None
    kind: str = field(default="interrupt", init=False)


@dataclass(frozen=True)
class ManualPolicySpec:
    """Top-level IR: a plan plus ordered interrupts (highest priority last).

    Interrupts are applied in list order; later entries override earlier ones
    when active (so put AVOID after DESTROY if dodge must win).
    ``force`` overlays sit above the plan but below interrupts unless
    ``force_beats_interrupt`` is True (SeaQuest surface commit).
    """

    plan: Union[PlanNode, Sequence]
    interrupts: Tuple[Interrupt, ...] = ()
    force: Tuple[ForceWhen, ...] = ()
    force_beats_interrupt: bool = False
    no_target_idx: int = -1
    kind: str = field(default="manual_policy", init=False)


def as_sequence(plan: Union[PlanNode, Sequence]) -> Sequence:
    if isinstance(plan, Sequence):
        return plan
    return Sequence(stages=(plan,))
