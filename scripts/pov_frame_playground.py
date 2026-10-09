#!/usr/bin/env python3
"""POV / FPS frame playground: verify body-frame obs and action mapping per env.

Three checks, all run against the *real* environment:

1. **Facing consistency** - the angle used to build the observation
   (``resolve_facing_deg`` -> sprite/synthetic orientation) must equal the angle
   the action mapper keys its table on (``facing_deg_bin0/1``). If these differ,
   the skill sees the world rotated relative to the actions it can take.

2. **Probe ring** - static objects on a compass around a stationary player.
   For each we report world delta, body pose ``(fwd, lat, phi)``, the ego action
   a correct skill should pick, the emitted game action, the ALE console action
   the env resolves it to (via ``ACTION_SET``), and the player displacement
   *measured by stepping the env*. Catches ``ACTION_SET`` index remaps such as
   Pong (emitted 2 -> ALE RIGHT -> paddle moves *up*).

3. **Mirror symmetry** - two targets placed symmetrically about the player, each
   viewed with the facing that turns toward it. A correct FPS frame gives the two
   the same |phi| with opposite sign (mirrored situations, mirrored actions).

Usage
-----
    python scripts/pov_frame_playground.py --env seaquest --out tmp/pov_playground
    python scripts/pov_frame_playground.py --env pong --facing 180
    python scripts/pov_frame_playground.py --all --out tmp/pov_playground
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from concept_based_learning.observations.action_space import (
    EGO_MOVE_DELTAS_DEG,
    _default_get_player,
)
from concept_based_learning.observations.pov_facing_frame import (
    facing_frame_phi_rad,
    world_delta_to_facing_frame,
)
from concept_based_learning.observations.pov_relative import (
    _MOVE_SPECS,
    _POV_PROFILES,
    _player_orientation_deg,
    build_facing_action_table,
    has_single_view_axis,
    view_axes_deg,
)

ALE_NAMES = {
    0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
    6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT", 10: "UPFIRE",
    11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE", 14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE",
}

EGO_NAMES = {
    0: "NOOP", 1: "FIRE", 2: "FORWARD", 3: "RIGHT", 4: "LEFT", 5: "BACKWARD",
    6: "FWD_RIGHT", 7: "FWD_LEFT", 8: "BACK_RIGHT", 9: "BACK_LEFT",
    10: "FORWARD_F", 11: "RIGHT_F", 12: "LEFT_F", 13: "BACKWARD_F",
    14: "FWD_RIGHT_F", 15: "FWD_LEFT_F", 16: "BACK_RIGHT_F", 17: "BACK_LEFT_F",
}

# How many view axes each env physically has, i.e. which directions the player
# body can meaningfully point. Envs with a single axis should pin their facing
# (synthetic orientation) instead of reading a sprite orientation.
ENV_VIEW_AXES: dict[str, tuple[str, ...]] = {
    "seaquest": ("right", "left"),
    "tennis": ("down (top court)", "up (bottom court)"),
    "kangaroo": ("right", "left"),
    "montezumarevenge": ("right", "left"),
    "venture": ("right", "left"),
    "choppercommand": ("right", "left"),
    "pong": ("into field (left)",),
    "breakout": ("into field (up)",),
    "beamrider": ("into field (up)",),
    "freeway": ("into field (up)",),
    "phoenix": ("into field (up)",),
    "spaceinvaders": ("into field (up)",),
    # Unresolved: asterix moves in 4 directions with no meaningful hull facing,
    # and skiing's declared axis is kept at 'up' only to preserve the previous
    # behaviour while its move table is still wrong (see playground check 2).
    "asterix": ("right", "left"),
    "skiing": ("UNRESOLVED (physically down, declared up)",),
}

# JAXAtari ObjectObservation.orientation uses 0 deg = up, 90 = right, 180 = down,
# 270 = left. Body-frame maths here uses screen angles: 0 = right, 90 = down,
# 180 = left, -90 = up. Converting between them is a -90 deg shift.
SPRITE_TO_SCREEN_OFFSET_DEG = -90.0

PROBE_RADIUS_PX = 40.0
PROBES: tuple[tuple[str, float, float], ...] = (
    ("E  (right)", 1.0, 0.0),
    ("SE (below-right)", 0.7071, 0.7071),
    ("S  (below)", 0.0, 1.0),
    ("SW (below-left)", -0.7071, 0.7071),
    ("W  (left)", -1.0, 0.0),
    ("NW (above-left)", -0.7071, -0.7071),
    ("N  (above)", 0.0, -1.0),
    ("NE (above-right)", 0.7071, -0.7071),
)

DIR_PROBE_ALE = {"RIGHT": 3, "LEFT": 4, "UP": 2, "DOWN": 5}


def wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def body_pose(dx: float, dy: float, facing_deg: float) -> tuple[float, float, float]:
    fwd, lat = world_delta_to_facing_frame(
        jnp.float32(dx), jnp.float32(dy), jnp.float32(facing_deg)
    )
    return float(fwd), float(lat), math.degrees(float(facing_frame_phi_rad(fwd, lat)))


def expected_ego_move(phi_deg: float) -> int:
    deltas = np.asarray(EGO_MOVE_DELTAS_DEG[2:10], dtype=np.float64)
    return 2 + int(np.argmin([abs(wrap180(d - phi_deg)) for d in deltas]))


def make_raw_env(env: str):
    from jaxtari import make as make_atari_env
    from jaxtari.wrappers import AtariWrapper

    base = make_atari_env(env)
    if env == "asterix":
        from concept_based_learning.environments.asterix.asterix_label import (
            AsterixSemanticWrapper,
        )

        wrapped = AtariWrapper(
            base, sticky_actions=0.0, episodic_life=False, first_fire=True, noop_max=0
        )
        wrapped._cbl_frame_stack_size = 1
        wrapped._cbl_frame_skip = 1
        wrapped._cbl_clip_reward = False
        return AsterixSemanticWrapper(wrapped)
    return base


def _playground_obs(obs):
    """Unwrap SemanticObservation from env wrappers used in paper-eval paths."""
    return obs.raw_observation if hasattr(obs, "raw_observation") else obs


def action_set_of(raw_env):
    action_set = getattr(raw_env, "ACTION_SET", None)
    if action_set is None:
        action_set = getattr(getattr(raw_env, "_env", None), "ACTION_SET", None)
    return None if action_set is None else np.asarray(action_set)


def resolve_ale(action_set, emitted: int) -> int:
    if action_set is None:
        return int(emitted)
    idx = int(emitted)
    return -1 if idx < 0 or idx >= action_set.shape[0] else int(action_set[idx])


def player_of(obs, env: str):
    obs = _playground_obs(obs)
    getter = _POV_PROFILES.get(env, {}).get("player_getter") or _default_get_player
    return getter(obs)


def player_xy(obs, env: str) -> tuple[float, float]:
    p = player_of(obs, env)
    return float(np.ravel(p.x)[-1]), float(np.ravel(p.y)[-1])


def hold_action(raw_env, env: str, emitted: int, warmup: int, steps: int):
    """Fresh reset -> NOOP warmup -> hold ``emitted``. Returns (displacement, obs)."""
    obs, state = raw_env.reset(jax.random.PRNGKey(0))
    noop = jnp.int32(0)
    for _ in range(warmup):
        obs, state, *_ = raw_env.step(state, noop)
    x0, y0 = player_xy(obs, env)
    act = jnp.int32(emitted)
    for _ in range(steps):
        obs, state, *_ = raw_env.step(state, act)
    x1, y1 = player_xy(obs, env)
    return (x1 - x0, y1 - y0), obs


def find_live_warmup(raw_env, env: str, action_set, warmup: int, steps: int) -> int:
    """Grow the NOOP warmup until the player actually responds to input.

    Needed for envs that freeze the player after reset (Seaquest holds the sub at
    the surface until oxygen refills, so every action is forced to NOOP).
    """
    probe_emitted = []
    for ale in DIR_PROBE_ALE.values():
        if action_set is None:
            probe_emitted.append(ale)
        else:
            hits = np.nonzero(action_set == ale)[0]
            if hits.size:
                probe_emitted.append(int(hits[0]))
    w = warmup
    for _ in range(6):
        for emitted in probe_emitted:
            disp, _ = hold_action(raw_env, env, emitted, w, steps)
            if math.hypot(*disp) > 1e-3:
                return w
        w = max(w * 2, w + 25)
    return warmup


def observed_orientations(raw_env, env: str, warmup: int, steps: int) -> list[float]:
    """Sprite orientation values the env actually emits while moving around."""
    action_set = action_set_of(raw_env)
    found: set[float] = set()
    for ale in DIR_PROBE_ALE.values():
        emitted = ale
        if action_set is not None:
            hits = np.nonzero(action_set == ale)[0]
            if not hits.size:
                continue
            emitted = int(hits[0])
        _, obs = hold_action(raw_env, env, emitted, warmup, steps)
        ori = getattr(player_of(obs, env), "orientation", None)
        if ori is not None:
            found.add(round(float(np.ravel(ori)[-1]), 1))
    return sorted(found)


def mapper_facings(env: str, override: float | None) -> list[float]:
    if override is not None:
        return [float(override)]
    profile = _POV_PROFILES.get(env, {})
    axis0, axis1 = view_axes_deg(profile)
    return [axis0] if has_single_view_axis(profile) else [axis0, axis1]


def _stub_obs(env: str, sprite_ori: float):
    """Minimal raw obs carrying just a player orientation, for frame queries."""
    del env

    class _P:
        orientation = np.array([sprite_ori], dtype=np.float32)

    stub = type("_Stub", (), {})()
    for attr in ("player", "chicken", "skier"):
        setattr(stub, attr, _P())
    return stub


def obs_facing_for(env: str, sprite_ori: float) -> float:
    """Facing the observation pipeline would use for this sprite orientation."""
    profile = _POV_PROFILES.get(env, {})
    getter = profile.get("player_getter") or _default_get_player
    return float(_player_orientation_deg(_stub_obs(env, sprite_ori), profile, getter))


def toward_verdict(disp: tuple[float, float], dx: float, dy: float) -> str:
    if math.hypot(*disp) < 1e-3:
        return "no-move"
    dot = disp[0] * dx + disp[1] * dy
    return "TOWARD" if dot > 1e-3 else ("AWAY" if dot < -1e-3 else "orthogonal")


def action_table(env: str, facing_deg: float) -> np.ndarray:
    noop, fire, move, mfire = _MOVE_SPECS[env]
    return np.asarray(
        build_facing_action_table(
            noop_action=noop,
            fire_action=fire,
            move_table=move,
            move_fire_table=mfire,
            facing_deg=facing_deg,
        )
    )


def facing_word(deg: float) -> str:
    d = wrap180(deg)
    for ref, word in ((0.0, "right"), (90.0, "down"), (180.0, "left"), (-90.0, "up")):
        if abs(wrap180(d - ref)) < 45.0:
            return word
    return f"{d:.0f}deg"


def check_facing_consistency(env: str, raw_env, warmup: int, steps: int) -> None:
    print("\n-- 1. facing consistency (observation frame vs mapper frame) --")
    from concept_based_learning.observations.pov_relative import _facing_bin_idx

    axes = ENV_VIEW_AXES.get(env)
    print(f"   physical view axes (expected): {', '.join(axes) if axes else 'undeclared'}")
    profile = _POV_PROFILES.get(env, {})
    axis0, axis1 = view_axes_deg(profile)
    single = has_single_view_axis(profile)
    oris = observed_orientations(raw_env, env, warmup, steps)
    print(f"   sprite orientations emitted by env: {oris if oris else 'none'}")
    print(f"   declared axes: {axis0:.0f}deg ({facing_word(axis0)})"
          f"{'  [single axis, sprite ignored]' if single else f' / {axis1:.0f}deg ({facing_word(axis1)})'}")

    getter = profile.get("player_getter") or _default_get_player
    print(f"   {'sprite ori':>11} {'-> obs facing':>14} {'axis':>5} "
          f"{'mapper facing':>14}  {'obs points':<8} {'mapper points':<14} verdict")
    bad = False
    for ori in oris or [0.0]:
        obs_f = obs_facing_for(env, ori)
        bin_idx = int(_facing_bin_idx(_stub_obs(env, ori), profile, getter))
        map_f = (axis0, axis1)[bin_idx]
        delta = wrap180(obs_f - map_f)
        ok = abs(delta) < 1.0
        bad = bad or not ok
        print(f"   {ori:>11.1f} {obs_f:>14.1f} {bin_idx:>4} {map_f:>14.1f}  "
              f"{facing_word(obs_f):<8} {facing_word(map_f):<14} "
              f"{'OK' if ok else f'MISMATCH {delta:+.0f}deg'}")
    if bad:
        print("   !! observation frame is rotated relative to the action frame.")
        print("      A skill trained here learns a rotated obs->action association,")
        print("      which breaks when transferred to an env whose frames do agree.")


def check_probe_ring(
    env: str, raw_env, facing_deg: float, warmup: int, steps: int
) -> list[dict]:
    table = action_table(env, facing_deg)
    action_set = action_set_of(raw_env)
    cache: dict[int, tuple[float, float]] = {}

    def disp_for(emitted: int) -> tuple[float, float]:
        if emitted not in cache:
            cache[emitted], _ = hold_action(raw_env, env, emitted, warmup, steps)
        return cache[emitted]

    print(f"\n-- 2a. ego action -> emitted -> ALE -> measured displacement "
          f"(mapper facing {facing_deg:.0f}deg = {facing_word(facing_deg)}) --")
    print(f"   {'ego':<13} {'emitted':>7} {'ALE':<14} {'dx':>7} {'dy':>7}  moves")
    for ego in range(18):
        emitted = int(table[ego])
        ale = resolve_ale(action_set, emitted)
        dxm, dym = disp_for(emitted)
        words = []
        if dym < -0.05:
            words.append("up")
        if dym > 0.05:
            words.append("down")
        if dxm < -0.05:
            words.append("left")
        if dxm > 0.05:
            words.append("right")
        print(f"   {EGO_NAMES[ego]:<13} {emitted:>7} {ALE_NAMES.get(ale, str(ale)):<14} "
              f"{dxm:>7.2f} {dym:>7.2f}  {'+'.join(words) or 'none'}")

    print("\n-- 2b. probe ring (player stationary at centre) --")
    header = (f"   {'probe':<18} {'dx':>5} {'dy':>5} | {'fwd':>6} {'lat':>6} {'phi':>7} | "
              f"{'ego wanted':<11} {'emitted':>7} {'ALE':<11} {'measured':<15} verdict")
    print(header)
    print("   " + "-" * (len(header) - 3))
    rows = []
    for name, ux, uy in PROBES:
        dx, dy = ux * PROBE_RADIUS_PX, uy * PROBE_RADIUS_PX
        fwd, lat, phi = body_pose(dx, dy, facing_deg)
        ego = expected_ego_move(phi)
        emitted = int(table[ego])
        ale = resolve_ale(action_set, emitted)
        disp = disp_for(emitted)
        verdict = toward_verdict(disp, dx, dy)
        print(f"   {name:<18} {dx:>5.0f} {dy:>5.0f} | {fwd:>6.1f} {lat:>6.1f} {phi:>7.1f} | "
              f"{EGO_NAMES[ego]:<11} {emitted:>7} {ALE_NAMES.get(ale, str(ale)):<11} "
              f"{f'({disp[0]:.2f},{disp[1]:.2f})':<15} {verdict}")
        rows.append(dict(name=name, dx=dx, dy=dy, fwd=fwd, lat=lat, phi=phi, ego=ego,
                         emitted=emitted, ale=ale, disp=disp, verdict=verdict))

    away = [r for r in rows if r["verdict"] == "AWAY"]
    if away:
        print(f"   !! {len(away)} probe(s) move AWAY from their target:")
        for r in away:
            print(f"      {r['name']}: {EGO_NAMES[r['ego']]} -> "
                  f"{ALE_NAMES.get(r['ale'], r['ale'])} -> disp {r['disp']}")
    else:
        print("   OK: every probe on a controllable axis moves toward its target.")
    return rows


def check_mirror(env: str, offset_px: float, ahead_px: float) -> list[dict]:
    """Two-target test: mirrored situations must give mirrored phi.

    Two-axis envs turn the body toward each target (the Seaquest shark pair).
    Single-axis envs cannot turn, so instead we mirror the target across the
    body's forward axis, which must still produce mirrored phi.
    """
    profile = _POV_PROFILES.get(env, {})
    b0, b1 = view_axes_deg(profile)
    single_axis = has_single_view_axis(profile)

    if single_axis:
        print(f"\n-- 3. mirror symmetry (single view axis: target mirrored across "
              f"forward, {ahead_px:.0f}px ahead, +/-{offset_px:.0f}px to each side) --")
        frad = math.radians(b0)
        fx, fy = math.cos(frad), math.sin(frad)
        rx, ry = math.cos(frad + math.pi / 2), math.sin(frad + math.pi / 2)
        cases = [
            ("target ahead and to my RIGHT",
             fx * ahead_px + rx * offset_px, fy * ahead_px + ry * offset_px, b0, 0.0),
            ("target ahead and to my LEFT ",
             fx * ahead_px - rx * offset_px, fy * ahead_px - ry * offset_px, b0, 0.0),
        ]
    else:
        print(f"\n-- 3. mirror symmetry (two view axes: body turns toward each target, "
              f"{ahead_px:.0f}px above, +/-{offset_px:.0f}px sideways) --")
        cases = [
            ("target above-RIGHT, body turned right", offset_px, -ahead_px, b0, 90.0),
            ("target above-LEFT,  body turned left ", -offset_px, -ahead_px, b1, 270.0),
        ]
    print(f"   {'case':<40} {'facing':>7} {'fwd':>7} {'lat':>7} {'phi':>8}  ego")
    rows = []
    for label, dx, dy, map_f, sprite_ori in cases:
        obs_f = obs_facing_for(env, sprite_ori)
        for tag, facing in (("as-built (obs facing)", obs_f), ("mapper facing", map_f)):
            fwd, lat, phi = body_pose(dx, dy, facing)
            ego = expected_ego_move(phi)
            print(f"   {label + ' [' + tag + ']':<40} {facing:>7.0f} {fwd:>7.1f} "
                  f"{lat:>7.1f} {phi:>8.1f}  {EGO_NAMES[ego]}")
            rows.append(dict(label=label, tag=tag, dx=dx, dy=dy, facing=facing,
                             fwd=fwd, lat=lat, phi=phi, ego=ego))
    for tag in ("as-built (obs facing)", "mapper facing"):
        pair = [r for r in rows if r["tag"] == tag]
        if len(pair) != 2:
            continue
        p0, p1 = pair
        mirrored = (abs(abs(p0["phi"]) - abs(p1["phi"])) < 1.0
                    and p0["phi"] * p1["phi"] <= 0.0)
        print(f"   [{tag}] phi = {p0['phi']:+.1f} vs {p1['phi']:+.1f}  -> "
              f"{'mirrored (correct)' if mirrored else 'NOT mirrored (frame is wrong)'}")
    return rows


def draw_diagram(env: str, facing_deg: float, rows: list[dict], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    lim = PROBE_RADIUS_PX * 2.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)  # screen coords: y down
    ax.set_aspect("equal")
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)

    ax.plot([0], [0], marker="s", ms=14, color="#1f77b4", zorder=5)
    frad = math.radians(facing_deg)
    fx, fy = math.cos(frad), math.sin(frad)
    ax.annotate("", xy=(fx * PROBE_RADIUS_PX * 0.85, fy * PROBE_RADIUS_PX * 0.85),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#ff9900", lw=3.0), zorder=6)
    ax.text(fx * PROBE_RADIUS_PX * 0.95, fy * PROBE_RADIUS_PX * 0.95, " FORWARD",
            color="#cc7a00", fontsize=10, fontweight="bold", va="center")
    rx, ry = math.cos(frad + math.pi / 2), math.sin(frad + math.pi / 2)
    ax.annotate("", xy=(rx * PROBE_RADIUS_PX * 0.5, ry * PROBE_RADIUS_PX * 0.5),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#9467bd", lw=2.0, ls="dashed"),
                zorder=6)
    ax.text(rx * PROBE_RADIUS_PX * 0.58, ry * PROBE_RADIUS_PX * 0.58, "+lat (my right)",
            color="#6a3d9a", fontsize=9, va="center")

    colors = {"TOWARD": "#2ca02c", "AWAY": "#d62728", "no-move": "#888888",
              "orthogonal": "#ff7f0e"}
    for r in rows:
        dx, dy = r["dx"], r["dy"]
        col = colors.get(r["verdict"], "#888888")
        ax.plot([dx], [dy], marker="o", ms=11, color=col, zorder=5)
        ax.plot([0, dx], [0, dy], color="#bbbbbb", lw=0.9, ls="dotted", zorder=1)
        ax.annotate(
            f"{r['name'].split(' ')[0]}\nphi={r['phi']:.0f}  lat={r['lat']:.0f}\n"
            f"{EGO_NAMES[r['ego']]} -> {ALE_NAMES.get(r['ale'], r['ale'])}\n{r['verdict']}",
            xy=(dx, dy), xytext=(dx * 1.34, dy * 1.34), ha="center", va="center",
            fontsize=8.0, color=col,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=col, lw=0.9, alpha=0.92),
            zorder=7)
        mx, my = r["disp"]
        mag = math.hypot(mx, my)
        if mag > 1e-3:
            scale = PROBE_RADIUS_PX * 0.42 / mag
            ax.annotate("", xy=(mx * scale, my * scale), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, alpha=0.75),
                        zorder=4)

    ax.set_title(f"{env}  |  mapper facing {facing_deg:.0f}deg ({facing_word(facing_deg)})"
                 f"  |  screen coords, y down\n"
                 "orange = FORWARD, purple dashed = +lat, coloured arrow = measured movement",
                 fontsize=11)
    ax.set_xlabel("screen x (px, right +)")
    ax.set_ylabel("screen y (px, down +)")
    tag = f"{env}_facing{int(round(facing_deg))}".replace("-", "neg")
    path = out_dir / f"pov_frame_{tag}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def draw_mirror(env: str, rows: list[dict], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    tags = ["as-built (obs facing)", "mapper facing"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.0))
    for ax, tag in zip(axes, tags):
        pair = [r for r in rows if r["tag"] == tag]
        lim = 70.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(lim, -lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="#eeeeee", lw=0.8)
        ax.axvline(0, color="#eeeeee", lw=0.8)
        for r, col in zip(pair, ("#1f77b4", "#d62728")):
            frad = math.radians(r["facing"])
            ax.annotate("", xy=(math.cos(frad) * 26, math.sin(frad) * 26), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6))
            ax.plot([r["dx"]], [r["dy"]], marker="o", ms=12, color=col)
            ax.plot([0, r["dx"]], [0, r["dy"]], ls="dotted", color=col, lw=1.0)
            ax.annotate(f"phi={r['phi']:+.0f}\nlat={r['lat']:+.0f}\n{EGO_NAMES[r['ego']]}",
                        xy=(r["dx"], r["dy"]), xytext=(r["dx"] * 1.05, r["dy"] * 1.6),
                        ha="center", va="center", fontsize=9, color=col,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=0.9))
        ax.plot([0], [0], marker="s", ms=13, color="#333333")
        phis = [r["phi"] for r in pair]
        mirrored = (len(phis) == 2 and abs(abs(phis[0]) - abs(phis[1])) < 1.0
                    and phis[0] * phis[1] <= 0.0)
        ax.set_title(f"{tag}\nphi {phis[0]:+.0f} vs {phis[1]:+.0f} -> "
                     f"{'mirrored (correct)' if mirrored else 'NOT mirrored'}",
                     fontsize=11,
                     color="#2ca02c" if mirrored else "#d62728")
        ax.set_xlabel("screen x (px)")
        ax.set_ylabel("screen y (px, down +)")
    fig.suptitle(f"{env}: mirrored targets seen from mirrored facings", fontsize=13)
    path = out_dir / f"pov_mirror_{env}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def report_env(env: str, args, out_dir: Path | None) -> bool:
    raw_env = make_raw_env(env)
    action_set = action_set_of(raw_env)
    warmup = find_live_warmup(raw_env, env, action_set, args.warmup, args.steps)

    print("=" * 104)
    print(f"ENV = {env}    (NOOP warmup until player responds: {warmup} steps)")
    print("=" * 104)

    check_facing_consistency(env, raw_env, warmup, args.steps)

    inverted = False
    for facing in mapper_facings(env, args.facing):
        rows = check_probe_ring(env, raw_env, facing, warmup, args.steps)
        inverted = inverted or any(r["verdict"] == "AWAY" for r in rows)
        if out_dir is not None:
            print(f"   diagram -> {draw_diagram(env, facing, rows, out_dir)}")

    mirror_rows = check_mirror(env, args.mirror_offset, args.mirror_ahead)
    if out_dir is not None:
        print(f"   diagram -> {draw_mirror(env, mirror_rows, out_dir)}")
    return inverted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=None, help="Env name (e.g. seaquest, pong)")
    parser.add_argument("--all", action="store_true", help="Every env with a POV spec")
    parser.add_argument("--facing", type=float, default=None,
                        help="Override mapper facing in deg (default: profile bins)")
    parser.add_argument("--steps", type=int, default=12,
                        help="Env steps to hold each action when measuring")
    parser.add_argument("--warmup", type=int, default=6,
                        help="Initial NOOP warmup after reset (auto-grown if frozen)")
    parser.add_argument("--mirror-offset", type=float, default=20.0,
                        help="Sideways offset for the mirror test (px)")
    parser.add_argument("--mirror-ahead", type=float, default=40.0,
                        help="Distance above the player for the mirror test (px)")
    parser.add_argument("--out", default=None, help="Directory for PNG diagrams")
    args = parser.parse_args()

    if args.all:
        envs = [e for e in _MOVE_SPECS if e in _POV_PROFILES]
    elif args.env:
        envs = [args.env.lower()]
    else:
        parser.error("pass --env NAME or --all")

    out_dir = Path(args.out) if args.out else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    failures: list[str] = []
    for env in envs:
        if env not in _MOVE_SPECS:
            print(f"skip {env}: no POV move spec")
            continue
        try:
            if report_env(env, args, out_dir):
                failures.append(env)
        except Exception as exc:  # noqa: BLE001 - playground keeps going
            print(f"   ERROR {env}: {type(exc).__name__}: {exc}")
        print()

    if failures:
        print(f"INVERTED ACTION MAPPINGS: {', '.join(failures)}")
    else:
        print("No inverted action mappings detected.")


if __name__ == "__main__":
    main()
