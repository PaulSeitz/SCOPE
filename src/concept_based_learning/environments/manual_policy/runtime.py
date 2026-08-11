"""Compile ``ManualPolicySpec`` IR into a JAX-traceable step function."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import jax.numpy as jnp

from concept_based_learning.environments.manual_policy.context import (
    FrameContext,
    build_frame_context,
)
from concept_based_learning.environments.manual_policy.ir import (
    ClimbLadders,
    DynamicPick,
    ForceWhen,
    Interrupt,
    ManualPolicySpec,
    NavigateTo,
    PlanNode,
    Sequence,
    as_sequence,
    resolve_predicate,
)
from concept_based_learning.environments.manual_policy.state import (
    NO_TARGET_IDX,
    PolicyState,
)


def _slot_eligible(ctx: FrameContext, exclude_slots: Tuple[int, ...]) -> jnp.ndarray:
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    eligible = ctx.active
    for slot in exclude_slots:
        eligible = jnp.logical_and(eligible, idxs != int(slot))
    return eligible


def _nearest_label(
    ctx: FrameContext,
    label: int,
    eligible: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mask = jnp.logical_and(eligible, ctx.labels == label)
    dist = jnp.where(mask, ctx.distances, jnp.inf)
    idx = jnp.argmin(dist).astype(jnp.int32)
    return idx, jnp.any(mask)


def _pick_by_prefer(
    ctx: FrameContext,
    prefer: Tuple[int, ...],
    eligible: jnp.ndarray,
    no_target_idx: int,
) -> jnp.ndarray:
    target = jnp.int32(no_target_idx)
    found = jnp.bool_(False)
    # Lower priority first so higher overwrites.
    for label in reversed(prefer):
        idx, any_ = _nearest_label(ctx, int(label), eligible)
        target = jnp.where(any_, idx, target)
        found = jnp.logical_or(found, any_)
    return target


def _mission_still_valid(
    ctx: FrameContext,
    mission: jnp.ndarray,
    sticky_labels: Tuple[int, ...],
    eligible: jnp.ndarray,
) -> jnp.ndarray:
    valid = jnp.logical_and(mission >= 0, mission < ctx.n)
    active = jnp.where(valid, eligible[mission], False)
    label = jnp.where(valid, ctx.labels[mission], -1)
    is_sticky = jnp.bool_(False)
    for lab in sticky_labels:
        is_sticky = jnp.logical_or(is_sticky, label == int(lab))
    return jnp.logical_and(active, is_sticky)


def _eval_dynamic_pick(
    ctx: FrameContext,
    node: DynamicPick,
    state: PolicyState,
    no_target_idx: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, PolicyState]:
    """Return (target, stage_done, new_state). DynamicPick never auto-advances."""
    eligible = _slot_eligible(ctx, node.exclude_slots)
    if node.slot_mask is not None:
        eligible = jnp.logical_and(eligible, node.slot_mask(ctx.n))

    fresh = _pick_by_prefer(ctx, node.prefer, eligible, no_target_idx)
    if node.sticky:
        keep = _mission_still_valid(ctx, state.mission_target, node.sticky_labels, eligible)
        # Only keep if previous mission is still in prefer set conceptually via sticky_labels.
        target = jnp.where(keep, state.mission_target, fresh)
    else:
        target = fresh

    new_state = state.replace(mission_target=target)
    # Never done — Sequence with a single DynamicPick stays forever (by design).
    return target, jnp.bool_(False), new_state


def _eval_navigate_to(
    ctx: FrameContext,
    node: NavigateTo,
    state: PolicyState,
    no_target_idx: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, PolicyState]:
    slot = jnp.int32(node.slot)
    active = jnp.where(
        jnp.logical_and(slot >= 0, slot < ctx.n),
        ctx.active[slot],
        False,
    )
    dist = jnp.where(active, ctx.distances[slot], jnp.inf)
    arrived = jnp.logical_and(active, dist <= float(node.arrive_dist))
    inactive = jnp.logical_not(active)

    mode = str(node.complete_when)
    if mode == "inactive":
        done = inactive
    elif mode == "either":
        done = jnp.logical_or(arrived, inactive)
    else:  # "arrive"
        done = arrived
        # Skip stages whose waypoint was never spawned (padded inactive ladders).
        done = jnp.logical_or(done, inactive)

    target = jnp.where(active, slot, jnp.int32(no_target_idx))
    # Keep mission on the slot even while inactive-wait isn't needed; if inactive
    # and complete_when=inactive, target is no_target but done is True so Sequence
    # advances immediately.
    new_state = state.replace(mission_target=jnp.where(active, slot, state.mission_target))
    return target, done, new_state


def _eval_climb_ladders(
    ctx: FrameContext,
    node: ClimbLadders,
    state: PolicyState,
    no_target_idx: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, PolicyState]:
    """Bottom→top climb; ``aux0`` holds paired top while ascending."""
    bot0 = int(node.bottom_start)
    top0 = int(node.top_start)
    n = int(node.n_ladders)
    arrive = float(node.arrive_dist)

    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    is_bottom = jnp.logical_and(idxs >= bot0, idxs < bot0 + n)
    is_top = jnp.logical_and(idxs >= top0, idxs < top0 + n)
    bottom_eligible = jnp.logical_and(ctx.active, is_bottom)
    top_eligible = jnp.logical_and(ctx.active, is_top)

    # Collectibles between climbs (fruits / bell / child).
    between = _pick_by_prefer(ctx, node.between_prefer, ctx.active, no_target_idx)
    any_between = between != no_target_idx

    paired_top = state.aux0
    climbing = jnp.logical_and(paired_top >= top0, paired_top < top0 + n)

    # If climbing: stick to paired top until arrived / inactive.
    top_active = jnp.where(
        jnp.logical_and(paired_top >= 0, paired_top < ctx.n),
        ctx.active[paired_top],
        False,
    )
    top_dist = jnp.where(top_active, ctx.distances[paired_top], jnp.inf)
    top_arrived = jnp.logical_and(top_active, top_dist <= arrive)
    climb_done = jnp.logical_or(top_arrived, jnp.logical_and(climbing, jnp.logical_not(top_active)))

    # Mission may be a bottom we're approaching.
    mission = state.mission_target
    mission_is_bottom = jnp.logical_and(
        jnp.logical_and(mission >= bot0, mission < bot0 + n),
        jnp.where(
            jnp.logical_and(mission >= 0, mission < ctx.n),
            ctx.active[mission],
            False,
        ),
    )
    bot_dist = jnp.where(mission_is_bottom, ctx.distances[mission], jnp.inf)
    bot_arrived = jnp.logical_and(mission_is_bottom, bot_dist <= arrive)

    # On bottom arrival → lock paired top.
    bot_offset = mission - bot0
    new_paired = jnp.where(bot_arrived, top0 + bot_offset, paired_top)
    new_paired = jnp.where(climb_done, jnp.int32(no_target_idx), new_paired)
    now_climbing = jnp.logical_or(
        jnp.logical_and(climbing, jnp.logical_not(climb_done)),
        bot_arrived,
    )

    # Nearest active ladder bottom (label-agnostic; waypoints are NAVIGATE).
    bot_dist_all = jnp.where(bottom_eligible, ctx.distances, jnp.inf)
    nearest_bottom = jnp.argmin(bot_dist_all).astype(jnp.int32)
    any_bottom = jnp.any(bottom_eligible)

    # Fresh pick when idle: between-collect if any, else nearest ladder bottom.
    idle_target = jnp.where(any_between, between, nearest_bottom)
    idle_target = jnp.where(
        jnp.logical_or(any_between, any_bottom),
        idle_target,
        jnp.int32(no_target_idx),
    )

    # Resolve current target.
    target = jnp.where(now_climbing, new_paired, idle_target)
    # If approaching a bottom (mission sticky), keep it until arrived.
    keep_bottom = jnp.logical_and(mission_is_bottom, jnp.logical_not(bot_arrived))
    keep_bottom = jnp.logical_and(keep_bottom, jnp.logical_not(now_climbing))
    target = jnp.where(keep_bottom, mission, target)
    # Sticky between-collect while active.
    keep_between = jnp.logical_and(
        jnp.logical_not(now_climbing),
        jnp.logical_not(keep_bottom),
    )
    if node.between_prefer:
        prev_ok = _mission_still_valid(
            ctx, mission, tuple(node.between_prefer), ctx.active
        )
        keep_between = jnp.logical_and(keep_between, prev_ok)
        target = jnp.where(keep_between, mission, target)

    # Same-step arrival: idle pick landed on a bottom already within arrive_dist.
    idle_is_bottom = jnp.logical_and(
        jnp.logical_and(target >= bot0, target < bot0 + n),
        jnp.logical_not(now_climbing),
    )
    idle_bot_dist = jnp.where(
        jnp.logical_and(idle_is_bottom, target >= 0),
        ctx.distances[jnp.clip(target, 0, ctx.n - 1)],
        jnp.inf,
    )
    idle_arrived = jnp.logical_and(idle_is_bottom, idle_bot_dist <= arrive)
    idle_offset = target - bot0
    new_paired = jnp.where(idle_arrived, top0 + idle_offset, new_paired)
    target = jnp.where(idle_arrived, new_paired, target)

    new_state = state.replace(mission_target=target, aux0=new_paired)
    # ClimbLadders never advances Sequence (it's a perpetual controller).
    return target, jnp.bool_(False), new_state


def _eval_plan_node(
    ctx: FrameContext,
    node: PlanNode,
    state: PolicyState,
    no_target_idx: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, PolicyState]:
    if isinstance(node, DynamicPick):
        return _eval_dynamic_pick(ctx, node, state, no_target_idx)
    if isinstance(node, NavigateTo):
        return _eval_navigate_to(ctx, node, state, no_target_idx)
    if isinstance(node, ClimbLadders):
        return _eval_climb_ladders(ctx, node, state, no_target_idx)
    raise TypeError(f"Unknown plan node: {type(node)!r}")


def _apply_forces(
    raw: Any,
    objects: Any,
    ctx: FrameContext,
    forces: Tuple[ForceWhen, ...],
    target: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (target, any_force_active). Later forces override earlier."""
    forced = target
    any_force = jnp.bool_(False)
    for force in forces:
        pred = resolve_predicate(force.predicate)
        active = pred(raw, objects, ctx)
        # Predicates may return 0-d or shaped arrays; squeeze to scalar bool.
        active = jnp.asarray(active).reshape(())
        forced = jnp.where(active, jnp.int32(force.target_slot), forced)
        any_force = jnp.logical_or(any_force, active)
    return forced, any_force


def _interrupt_metric(
    ctx: FrameContext,
    *,
    within: float,
    within_x: Optional[float],
    within_y: Optional[float],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (score, in_range) for interrupt selection.

    Isotropic: score = AABB surface gap, in_range = dist <= within.
    Axis-weighted: score = elliptical radius on centre offsets, in_range = ellipse <= 1.
    """
    if within_x is not None or within_y is not None:
        rx = float(within_x if within_x is not None else within)
        ry = float(within_y if within_y is not None else within)
        rx = max(rx, 1e-6)
        ry = max(ry, 1e-6)
        score = jnp.sqrt((ctx.dx / rx) ** 2 + (ctx.dy / ry) ** 2)
        return score, score <= 1.0
    score = ctx.distances
    return score, score <= float(within)


def _apply_interrupts(
    raw: Any,
    objects: Any,
    ctx: FrameContext,
    interrupts: Tuple[Interrupt, ...],
    state: PolicyState,
    plan_target: jnp.ndarray,
) -> Tuple[jnp.ndarray, PolicyState]:
    target = plan_target
    latched = state.interrupt_latched

    for interrupt in interrupts:
        suppress = jnp.bool_(False)
        if interrupt.suppress is not None:
            suppress = jnp.asarray(
                resolve_predicate(interrupt.suppress)(raw, objects, ctx)
            ).reshape(())

        label_mask = jnp.zeros((ctx.n,), dtype=bool)
        for lab in interrupt.labels:
            label_mask = jnp.logical_or(label_mask, ctx.labels == int(lab))
        threat = jnp.logical_and(ctx.active, label_mask)

        score, in_within = _interrupt_metric(
            ctx,
            within=interrupt.within,
            within_x=interrupt.within_x,
            within_y=interrupt.within_y,
        )
        in_within = jnp.logical_and(threat, in_within)
        have_within = jnp.any(in_within)

        if (
            interrupt.release is not None
            or interrupt.release_x is not None
            or interrupt.release_y is not None
        ):
            rel_within = (
                float(interrupt.release)
                if interrupt.release is not None
                else float(interrupt.within)
            )
            _, in_release = _interrupt_metric(
                ctx,
                within=rel_within,
                within_x=interrupt.release_x,
                within_y=interrupt.release_y,
            )
            in_release = jnp.logical_and(threat, in_release)
            latched = jnp.where(have_within, True, latched)
            latched = jnp.where(jnp.any(in_release), latched, False)
            active_interrupt = jnp.logical_and(latched, jnp.logical_not(suppress))
            dist = jnp.where(in_release, score, jnp.inf)
        else:
            active_interrupt = jnp.logical_and(have_within, jnp.logical_not(suppress))
            dist = jnp.where(in_within, score, jnp.inf)

        best = jnp.argmin(dist).astype(jnp.int32)
        target = jnp.where(active_interrupt, best, target)

    return target, state.replace(interrupt_latched=latched)


class ManualPolicy:
    """Compiled manual policy: IR → ``step`` / legacy heuristic wrappers."""

    def __init__(self, spec: ManualPolicySpec):
        self.spec = spec
        self.sequence = as_sequence(spec.plan)
        self.needs_state = True  # always; Sequence + interrupts use carry

    def initial_state(self) -> PolicyState:
        return PolicyState.zero(no_target_idx=self.spec.no_target_idx)

    def step(
        self,
        raw_observation: Any,
        objects: Any,
        state: PolicyState,
        rng: Any = None,
    ) -> Tuple[jnp.ndarray, PolicyState]:
        del rng
        ctx = build_frame_context(raw_observation, objects)
        no_target = int(self.spec.no_target_idx)
        stages = self.sequence.stages
        n_stages = len(stages)

        # Evaluate every stage (static Python loop); select by plan_idx.
        targets = []
        dones = []
        states = []
        for stage in stages:
            t, d, s = _eval_plan_node(ctx, stage, state, no_target)
            targets.append(t)
            dones.append(d)
            states.append(s)

        plan_idx = jnp.clip(state.plan_idx, 0, max(n_stages - 1, 0))
        # Gather selected stage outputs without lax.switch for small n.
        target = targets[0] if n_stages == 1 else targets[0]
        done = dones[0] if n_stages == 1 else dones[0]
        stage_state = states[0] if n_stages == 1 else states[0]
        for i in range(n_stages):
            take = plan_idx == i
            target = jnp.where(take, targets[i], target)
            done = jnp.where(take, dones[i], done)
            # Merge state fields from the active stage.
            stage_state = PolicyState(
                plan_idx=stage_state.plan_idx,
                mission_target=jnp.where(
                    take, states[i].mission_target, stage_state.mission_target
                ),
                interrupt_latched=stage_state.interrupt_latched,
                aux0=jnp.where(take, states[i].aux0, stage_state.aux0),
            )

        # Advance sequence when stage done.
        next_idx = jnp.where(
            done,
            jnp.minimum(plan_idx + 1, jnp.int32(max(n_stages - 1, 0))),
            plan_idx,
        )
        # On advance, adopt the next stage's target immediately (same step).
        advanced_target = targets[0]
        advanced_mission = states[0].mission_target
        advanced_aux = states[0].aux0
        for i in range(n_stages):
            take_next = next_idx == i
            advanced_target = jnp.where(take_next, targets[i], advanced_target)
            advanced_mission = jnp.where(
                take_next, states[i].mission_target, advanced_mission
            )
            advanced_aux = jnp.where(take_next, states[i].aux0, advanced_aux)

        target = jnp.where(done, advanced_target, target)
        mission = jnp.where(done, advanced_mission, stage_state.mission_target)
        aux0 = jnp.where(done, advanced_aux, stage_state.aux0)
        plan_state = stage_state.replace(
            plan_idx=next_idx, mission_target=mission, aux0=aux0
        )

        # Force overlays (e.g. surface).
        forced_target, any_force = _apply_forces(
            raw_observation, objects, ctx, self.spec.force, target
        )
        # When forcing, update mission to the forced slot so stickiness resumes cleanly.
        plan_state = plan_state.replace(
            mission_target=jnp.where(any_force, forced_target, plan_state.mission_target)
        )
        target = forced_target

        # Interrupts (reactive).
        final_target, plan_state = _apply_interrupts(
            raw_observation,
            objects,
            ctx,
            self.spec.interrupts,
            plan_state,
            target,
        )

        if self.spec.force_beats_interrupt:
            final_target = jnp.where(any_force, forced_target, final_target)

        # Empty screen → no target (unless a force is active).
        any_active = jnp.any(ctx.active)
        final_target = jnp.where(
            jnp.logical_or(any_active, any_force),
            final_target,
            jnp.int32(no_target),
        )
        return final_target.astype(jnp.int32), plan_state

    def as_legacy_heuristic(self):
        """``(raw, objects, prev_target_idx, rng) -> target`` using ephemeral state.

        Stickiness uses ``prev_target_idx`` as ``mission_target``; ``plan_idx`` /
        climb ``aux0`` reset every call — fine for flat DynamicPick policies,
        insufficient for Sequence / ClimbLadders (use ``step`` + carry).
        """

        def _fn(raw_observation, objects, prev_target_idx, rng):
            state = self.initial_state().replace(
                mission_target=jnp.asarray(prev_target_idx, dtype=jnp.int32)
            )
            target, _ = self.step(raw_observation, objects, state, rng)
            return target

        return _fn

    def as_stateful_heuristic(self):
        """``(raw, objects, prev_target_idx, state, rng) -> (target, state)``.

        ``prev_target_idx`` is accepted for call-site uniformity with the sweep
        helper but ignored — mission lives in ``state``.
        """

        def _fn(raw_observation, objects, prev_target_idx, state, rng):
            del prev_target_idx
            return self.step(raw_observation, objects, state, rng)

        return _fn


def compile_policy(spec: ManualPolicySpec) -> ManualPolicy:
    return ManualPolicy(spec)
