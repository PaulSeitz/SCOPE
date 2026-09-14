"""Montezuma's Revenge manual policy: room-1 (room_id 4) key path.

Edit the numbered ``WP*`` constants below to move trigger points.
Coordinates are **obs / render space** (same as ``player.x`` / ``player.y``
after the HUD offset).  ``y`` grows downward:

  * Each arrive WP is a ``(2·ARRIVE_R)²`` box centred on ``(x, y)``.
    Stage completes on **AABB overlap** (same contact as COLLECT).
  * climbing DOWN  — increase ``y`` by 2 → box (and trigger) 2px lower
  * climbing UP    — decrease ``y`` by 2 → box 2px higher
  * WP2 / WP11     — rope COLLECT (jump the chasm by grabbing the rope)

Hot-reload while experimenting (no process restart):
  * Press **W** in ``reward_dashboard`` / play logger after saving this file, OR
  * Edit ``tmp/montezuma_waypoints.json`` (auto-picked up); dump defaults with
    ``python -c "from concept_based_learning.environments.montezumarevenge.montezuma_policy import write_waypoints_json; write_waypoints_json()"``.

Interrupt: rolling skull AVOID only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_NOOP,
)
from concept_based_learning.environments.manual_policy import (
    Interrupt,
    ManualPolicySpec,
    NavigateTo,
    Sequence,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX, PolicyState
from concept_based_learning.environments.montezumarevenge.montezuma_label import (
    PLAYER_STATE_DYING,
)

# =============================================================================
# Tunable room-1 waypoints (OBS centres).  Index = plan order.
# =============================================================================
#
#   0  shaft bead          COLLECT   arrive at (x, y)
#   1  conveyor edge       COLLECT   arrive at (x, y)
#   2  rope grab           COLLECT   arrive (jump chasm via rope)
#   3  right ladder top    COLLECT   arrive
#   4  right ladder bot    COLLECT   arrive
#   5  left ladder bot     COLLECT   arrive
#   6  left ladder top     COLLECT   arrive
#   7  key                 COLLECT   until inactive (live item; xy ignored)
#   8  left bot (return)   COLLECT   arrive
#   9  right bot (return)  COLLECT   arrive
#  10  right top (return)  COLLECT   arrive
#  11  rope grab (return)  COLLECT   arrive (same rope, jump back)
#  12  center ladder bot   COLLECT   arrive
#  13  center ladder top   COLLECT   arrive
#  14  left door           COLLECT   arrive or inactive (live door; xy ignored)

ARRIVE_R = 6.0  # half-size of the trigger box around each WP centre (px)

WP0 = (80.0, 140.0)   # shaft bead (mid center ladder)
WP1 = (102.0, 140.0)  # conveyor-right lip
WP2 = (111.0, 102.0)  # rope grab (aligns with injected rope COLLECT)
WP3 = (136.0, 121.0)  # L1 top (right mid landing)
WP4 = (136.0, 185.0)  # L1 bot
WP5 = (24.0, 185.0)   # L2 bot
WP6 = (24.0, 121.0)   # L2 top (key ledge)
WP7 = (16.0, 103.0)   # key (informational; completion = item inactive)
WP8 = (24.0, 171.0)   # L2 bot return
WP9 = (136.0, 171.0)  # L1 bot return
WP10 = (136.0, 121.0) # L1 top return
WP11 = (111.0, 102.0) # rope grab return (same rope)
WP12 = (80.0, 129.0)  # L0 bot
WP13 = (80.0, 78.0)   # L0 top
WP14 = (18.0, 73.0)   # door (informational; completion = arrive/inactive)

# Object slots (meta-NOOP prepended). Keep in sync with montezuma_label layout.
_SLOT_SHAFT = 20
_SLOT_EDGE = 23
_SLOT_GAP_L = 21
_SLOT_GAP_R = 22
_SLOT_L0_BOT, _SLOT_L0_TOP = 4, 8
_SLOT_L1_BOT, _SLOT_L1_TOP = 5, 9
_SLOT_L2_BOT, _SLOT_L2_TOP = 6, 10
_SLOT_KEY = 12
_SLOT_DOOR = 15
_SLOT_ROPE = 17
_ENEMY_SLOT0 = 1
_ENEMY_SLOT1 = 3

# Trigger box stamped around each WP centre. Overlap == stage complete
# (same contact rule as COLLECT skill success).
_WP_W = float(ARRIVE_R) * 2.0
_WP_H = float(ARRIVE_R) * 2.0

# Unused gap slots (kept off-screen; was JUMP_OVER, now plan uses rope COLLECT).
GAP_R_BOX = (0.0, -40.0, 1.0, 1.0)
GAP_L_BOX = (0.0, -40.0, 1.0, 1.0)

# Skull AVOID disc + hysteresis.
SKULL_AVOID_RANGE = 28.0
SKULL_AVOID_RELEASE = 40.0
CLOSE_AVOID_RANGE = SKULL_AVOID_RANGE  # legacy alias for tests

# ---------------------------------------------------------------------------
# Slot aliases kept for tests / play logger.
# ---------------------------------------------------------------------------
_SHAFT_BEAD_SLOT = _SLOT_SHAFT
_EDGE_COLLECT_SLOT = _SLOT_EDGE
_RIGHT_GAP_JUMP_SLOT = _SLOT_GAP_R
_LEFT_GAP_JUMP_SLOT = _SLOT_GAP_L
_L0_BOT, _L0_TOP = _SLOT_L0_BOT, _SLOT_L0_TOP
_L1_BOT, _L1_TOP = _SLOT_L1_BOT, _SLOT_L1_TOP
_L2_BOT, _L2_TOP = _SLOT_L2_BOT, _SLOT_L2_TOP
_KEY_SLOT = _SLOT_KEY
_LEFT_DOOR_SLOT = _SLOT_DOOR
_ROPE_GRAB_SLOT = _SLOT_ROPE

_WP_NAMES = tuple(f"WP{i}" for i in range(15))
_JSON_MTIME: float = -1.0


def default_waypoints_json_path() -> Path:
    """``tmp/montezuma_waypoints.json`` under the repo root."""
    # .../src/concept_based_learning/environments/montezumarevenge/this.py
    repo = Path(__file__).resolve().parents[4]
    return repo / "tmp" / "montezuma_waypoints.json"


def waypoints_as_dict() -> dict:
    g = globals()
    return {
        "ARRIVE_R": float(g["ARRIVE_R"]),
        **{name: [float(g[name][0]), float(g[name][1])] for name in _WP_NAMES},
        "GAP_R_BOX": [float(x) for x in GAP_R_BOX],
        "GAP_L_BOX": [float(x) for x in GAP_L_BOX],
    }


def write_waypoints_json(path: Optional[Path] = None) -> Path:
    """Write current WP* / ARRIVE_R to JSON for live editing."""
    path = Path(path) if path is not None else default_waypoints_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(waypoints_as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def apply_waypoints_dict(data: dict) -> None:
    """Overwrite module-level WP* / ARRIVE_R / gap boxes from a dict."""
    global ARRIVE_R, _WP_W, _WP_H, GAP_R_BOX, GAP_L_BOX
    g = globals()
    if "ARRIVE_R" in data:
        ARRIVE_R = float(data["ARRIVE_R"])
        _WP_W = float(ARRIVE_R) * 2.0
        _WP_H = float(ARRIVE_R) * 2.0
        g["ARRIVE_R"] = ARRIVE_R
        g["_WP_W"] = _WP_W
        g["_WP_H"] = _WP_H
    for name in _WP_NAMES:
        if name in data:
            xy = data[name]
            g[name] = (float(xy[0]), float(xy[1]))
    if "GAP_R_BOX" in data:
        b = data["GAP_R_BOX"]
        GAP_R_BOX = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        g["GAP_R_BOX"] = GAP_R_BOX
    if "GAP_L_BOX" in data:
        b = data["GAP_L_BOX"]
        GAP_L_BOX = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        g["GAP_L_BOX"] = GAP_L_BOX


def _tl(xy: Tuple[float, float], w: float = None, h: float = None):
    """Centre (x, y) → top-left (x, y, w, h) in obs space."""
    ww = float(_WP_W if w is None else w)
    hh = float(_WP_H if h is None else h)
    return (xy[0] - ww * 0.5, xy[1] - hh * 0.5, ww, hh)


def room4_path_stamps() -> Dict[int, Tuple[float, float, float, float, int]]:
    """Obs-space stamps for the label wrapper: slot → (x, y, w, h, label)."""
    g = globals()
    stamps: Dict[int, Tuple[float, float, float, float, int]] = {}
    for slot, name in (
        (_SLOT_SHAFT, "WP0"),
        (_SLOT_EDGE, "WP1"),
        (_SLOT_ROPE, "WP2"),  # rope COLLECT (WP11 reuses same slot/geometry)
        (_SLOT_L1_TOP, "WP3"),
        (_SLOT_L1_BOT, "WP4"),
        (_SLOT_L2_BOT, "WP5"),
        (_SLOT_L2_TOP, "WP6"),
        (_SLOT_L0_BOT, "WP12"),
        (_SLOT_L0_TOP, "WP13"),
    ):
        x, y, w, h = _tl(g[name])
        stamps[slot] = (x, y, w, h, int(LABEL_COLLECT))
    # Park old gap slots (no longer JUMP_OVER targets).
    stamps[_SLOT_GAP_L] = (*GAP_L_BOX, int(LABEL_NOOP))
    stamps[_SLOT_GAP_R] = (*GAP_R_BOX, int(LABEL_NOOP))
    return stamps


def _arrive(slot: int, xy: Tuple[float, float], r: float = None) -> NavigateTo:
    """Complete on AABB overlap with the stamped WP box around ``xy``."""
    del xy, r
    return NavigateTo(
        slot=slot,
        arrive_dist=0.0,
        complete_when="arrive",
        skip_if_inactive=False,
    )


def _collect_item(slot: int) -> NavigateTo:
    return NavigateTo(
        slot=slot,
        arrive_dist=10.0,
        complete_when="inactive",
        skip_if_inactive=True,
    )


def _open_door(slot: int) -> NavigateTo:
    return NavigateTo(
        slot=slot,
        arrive_dist=12.0,
        complete_when="either",
        skip_if_inactive=True,
    )


def _skull_threat_mask(raw, objects, ctx, plan_target):
    del raw, objects, plan_target
    idxs = jnp.arange(ctx.n, dtype=jnp.int32)
    return jnp.logical_and(idxs >= _ENEMY_SLOT0, idxs <= _ENEMY_SLOT1)


def _player_last_scalar(arr) -> jnp.ndarray:
    return arr[-1] if jnp.ndim(arr) > 0 else arr


def _player_is_dying(raw, objects=None, ctx=None) -> jnp.ndarray:
    del objects, ctx
    player = getattr(raw, "player", None)
    if player is not None and hasattr(player, "state"):
        return _player_last_scalar(player.state) == jnp.int32(PLAYER_STATE_DYING)
    return jnp.bool_(False)


def _build_stages():
    g = globals()
    return (
        _arrive(_SLOT_SHAFT, g["WP0"]),
        _arrive(_SLOT_EDGE, g["WP1"]),
        _arrive(_SLOT_ROPE, g["WP2"]),       # rope COLLECT (outbound)
        _arrive(_SLOT_L1_TOP, g["WP3"]),
        _arrive(_SLOT_L1_BOT, g["WP4"]),
        _arrive(_SLOT_L2_BOT, g["WP5"]),
        _arrive(_SLOT_L2_TOP, g["WP6"]),
        _collect_item(_SLOT_KEY),
        _arrive(_SLOT_L2_BOT, g["WP8"]),
        _arrive(_SLOT_L1_BOT, g["WP9"]),
        _arrive(_SLOT_L1_TOP, g["WP10"]),
        _arrive(_SLOT_ROPE, g["WP11"]),      # rope COLLECT (return)
        _arrive(_SLOT_L0_BOT, g["WP12"]),
        _arrive(_SLOT_L0_TOP, g["WP13"]),
        _open_door(_SLOT_DOOR),
    )


class _DeathFrozenMontezumaPolicy:
    """Hold the Sequence at stage 0 for the whole death animation."""

    def __init__(self, core):
        self._core = core
        self.spec = core.spec
        self.needs_state = True

    def initial_state(self) -> PolicyState:
        return self._core.initial_state()

    def step(self, raw_observation, objects, state, rng):
        dying = _player_is_dying(raw_observation)
        zero = self._core.initial_state()
        state_in = jax.lax.cond(dying, lambda _: zero, lambda s: s, state)
        target, new_state = self._core.step(raw_observation, objects, state_in, rng)
        new_state = jax.lax.cond(dying, lambda _: zero, lambda s: s, new_state)
        target = jnp.where(dying, jnp.int32(NO_TARGET_IDX), target)
        return target.astype(jnp.int32), new_state

    def as_legacy_heuristic(self):
        return self._core.as_legacy_heuristic()

    def as_stateful_heuristic(self):
        def _fn(raw_observation, objects, prev_target_idx, state, rng):
            del prev_target_idx
            return self.step(raw_observation, objects, state, rng)

        return _fn


def rebuild_policy() -> None:
    """Recompile Sequence + heuristic callables from current WP* globals."""
    global MONTEZUMA_L1_STAGES, _MONTEZUMA_POLICY_CORE, MONTEZUMA_POLICY
    global get_heuristic_target, get_heuristic_target_with_state
    MONTEZUMA_L1_STAGES = _build_stages()
    _MONTEZUMA_POLICY_CORE = compile_policy(
        ManualPolicySpec(
            plan=Sequence(stages=MONTEZUMA_L1_STAGES),
            interrupts=(
                Interrupt(
                    labels=(LABEL_AVOID,),
                    within_x=SKULL_AVOID_RANGE,
                    within_y=SKULL_AVOID_RANGE,
                    release_x=SKULL_AVOID_RELEASE,
                    release_y=SKULL_AVOID_RELEASE,
                    threat_mask=_skull_threat_mask,
                ),
            ),
            no_target_idx=NO_TARGET_IDX,
        )
    )
    MONTEZUMA_POLICY = _DeathFrozenMontezumaPolicy(_MONTEZUMA_POLICY_CORE)
    get_heuristic_target = MONTEZUMA_POLICY.as_legacy_heuristic()
    get_heuristic_target_with_state = MONTEZUMA_POLICY.as_stateful_heuristic()


def load_waypoints_json(path: Optional[Path] = None) -> bool:
    """Load JSON overrides and rebuild the policy. Returns True if loaded."""
    global _JSON_MTIME
    path = Path(path) if path is not None else default_waypoints_json_path()
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    apply_waypoints_dict(data)
    rebuild_policy()
    _JSON_MTIME = path.stat().st_mtime
    return True


def maybe_reload_waypoints_json(path: Optional[Path] = None) -> bool:
    """If the JSON file changed on disk, reload it. Returns True when applied."""
    global _JSON_MTIME
    path = Path(path) if path is not None else default_waypoints_json_path()
    if not path.is_file():
        return False
    mtime = path.stat().st_mtime
    if mtime == _JSON_MTIME:
        return False
    return load_waypoints_json(path)


def hot_reload():
    """Reload this ``.py`` from disk (picks up WP* edits) and return the module.

    Callers that cached ``get_heuristic_target_with_state`` must rebind it from
    the returned module.  An existing waypoints JSON is *not* re-applied (so .py
    edits win); edit the JSON afterward (or touch it) to override again.
    """
    import importlib

    mod = importlib.reload(sys.modules[__name__])
    # Module bottom set _JSON_MTIME=-1; freeze current JSON mtime so the next
    # stamp pass does not immediately overwrite the freshly loaded WP* values.
    path = mod.default_waypoints_json_path()
    if path.is_file():
        mod._JSON_MTIME = float(path.stat().st_mtime)
    return mod


# Initial compile from the WP* defaults above.
MONTEZUMA_L1_STAGES = ()
_MONTEZUMA_POLICY_CORE = None
MONTEZUMA_POLICY = None
get_heuristic_target = None
get_heuristic_target_with_state = None
rebuild_policy()
# JSON overrides (if ``tmp/montezuma_waypoints.json`` exists) apply on the
# first stamp / ``maybe_reload_waypoints_json`` call, not here — so ``hot_reload``
# can keep freshly edited WP* from this file without being overwritten.
