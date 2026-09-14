#!/usr/bin/env python3
"""Play Montezuma room-1 while logging player / waypoints / target + screenshots.

Controls (same as reward_dashboard):
  Arrows + Space  move / jump (FIRE)
  R               reset
  F               freeze
  N               step one frame while frozen
  W               hot-reload WP* from montezuma_policy.py (no env restart)
  Q / Esc         quit (flushes logs)

Writes under ``--out-dir`` (default ``tmp/montezuma_play_log/<timestamp>/``):
  frames.jsonl    one record per env step
  summary.txt     human-readable running notes
  shots/fXXXX.png every ``--shot-every`` frames (also on reset / quit)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from omegaconf import OmegaConf

# Repo root on path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from concept_based_learning.environments.datatypes import (  # noqa: E402
    LABEL_AVOID,
    LABEL_BLOCKER,
    LABEL_COLLECT,
    LABEL_JUMP_OVER,
    LABEL_NAVIGATE,
    LABEL_NOOP,
)
from concept_based_learning.environments.env_manager import EnvManager  # noqa: E402
from concept_based_learning.environments.heuristics import (  # noqa: E402
    initial_policy_state,
    resolve_heuristic_target_fn,
    resolve_heuristic_with_state_fn,
)
from concept_based_learning.environments.montezumarevenge.montezuma_policy import (  # noqa: E402
    _EDGE_COLLECT_SLOT,
    _KEY_SLOT,
    _L0_BOT,
    _L0_TOP,
    _LEFT_DOOR_SLOT,
    _ROPE_GRAB_SLOT,
    _SHAFT_BEAD_SLOT,
    hot_reload,
    maybe_reload_waypoints_json,
)

# ALE action ids (dashboard remaps via ACTION_SET)
ACTION_NOOP = 0
ACTION_FIRE = 1
ACTION_UP = 2
ACTION_RIGHT = 3
ACTION_LEFT = 4
ACTION_DOWN = 5
ACTION_UPRIGHT = 6
ACTION_UPLEFT = 7
ACTION_DOWNRIGHT = 8
ACTION_DOWNLEFT = 9
ACTION_UPFIRE = 10
ACTION_RIGHTFIRE = 11
ACTION_LEFTFIRE = 12
ACTION_DOWNFIRE = 13
ACTION_UPRIGHTFIRE = 14
ACTION_UPLEFTFIRE = 15
ACTION_DOWNRIGHTFIRE = 16
ACTION_DOWNLEFTFIRE = 17

_LABEL_NAME = {
    int(LABEL_NOOP): "NOOP",
    int(LABEL_COLLECT): "COLLECT",
    int(LABEL_AVOID): "AVOID",
    int(LABEL_NAVIGATE): "NAVIGATE",
    int(LABEL_JUMP_OVER): "JUMP_OVER",
    int(LABEL_BLOCKER): "BLOCKER",
}

_SLOT_NAME = {
    _L0_BOT: "L0_BOT",
    _L0_TOP: "L0_TOP",
    _KEY_SLOT: "KEY",
    _LEFT_DOOR_SLOT: "LEFT_DOOR",
    _SHAFT_BEAD_SLOT: "SHAFT_BEAD",
    _EDGE_COLLECT_SLOT: "EDGE_COLLECT",
    _ROPE_GRAB_SLOT: "ROPE_GRAB",
    1: "SKULL",
}


def _unwrap_action_set(env):
    cur = env
    for _ in range(12):
        aset = getattr(cur, "ACTION_SET", None)
        if aset is not None:
            return [int(x) for x in list(aset)]
        nxt = getattr(cur, "env", None) or getattr(cur, "_env", None)
        if nxt is None:
            break
        cur = nxt
    return list(range(18))


def _ale_to_idx(action_set):
    return {int(a): i for i, a in enumerate(action_set)}


def _player_xywh(obs):
    p = obs.raw_observation.player
    x = float(np.asarray(p.x)[-1] if np.ndim(p.x) else p.x)
    y = float(np.asarray(p.y)[-1] if np.ndim(p.y) else p.y)
    w = float(np.asarray(p.width)[-1] if np.ndim(p.width) else p.width)
    h = float(np.asarray(p.height)[-1] if np.ndim(p.height) else p.height)
    return x, y, w, h


def _active_objects(obs):
    o = obs.objects
    out = []
    n = int(o.x.shape[1])
    for i in range(n):
        if int(o.active[-1, i]) <= 0:
            continue
        lab = int(o.semantic_label[-1, i])
        x, y = float(o.x[-1, i]), float(o.y[-1, i])
        w, h = float(o.width[-1, i]), float(o.height[-1, i])
        out.append(
            {
                "slot": i,
                "name": _SLOT_NAME.get(i, f"obj{i}"),
                "label": _LABEL_NAME.get(lab, str(lab)),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + w * 0.5,
                "cy": y + h * 0.5,
            }
        )
    return out


def _waypoints_only(objects_list):
    """Plan-relevant COLLECT scaffolding + ladders + key + door."""
    keep = set(range(4, 12)) | {
        _KEY_SLOT,
        _LEFT_DOOR_SLOT,
        _SHAFT_BEAD_SLOT,
        _EDGE_COLLECT_SLOT,
        _ROPE_GRAB_SLOT,
    }
    return [o for o in objects_list if o["slot"] in keep]


def _overlay_plan_objects(objects_list):
    """Objects drawn on the play overlay (scaffolding + ladder WPs + key/door)."""
    return [
        o
        for o in _waypoints_only(objects_list)
        if o["label"] in ("COLLECT", "NAVIGATE")
    ]


def _overlay_color(obj, current_target: int):
    if obj["slot"] == current_target:
        return (255, 60, 60)
    return (80, 200, 255)


class MontezumaPlayLogger:
    def __init__(self, env, renderer, components, cfg, out_dir: Path, shot_every: int):
        self.env = env
        self.renderer = renderer
        self.out_dir = out_dir
        self.shots_dir = out_dir / "shots"
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self.shot_every = max(1, int(shot_every))
        self.jsonl_path = out_dir / "frames.jsonl"
        self.summary_path = out_dir / "summary.txt"

        self.action_set = _unwrap_action_set(env)
        self.ale_to_idx = _ale_to_idx(self.action_set)
        self.no_target_idx = int(components.no_target_idx)
        self.get_heuristic_with_state = resolve_heuristic_with_state_fn(
            cfg, "montezumarevenge", components
        )
        self.get_heuristic_target = resolve_heuristic_target_fn(
            cfg, "montezumarevenge", components
        )
        self.extract_life_lost = components.extract_life_lost
        self.reset_policy_on_life_lost = bool(
            getattr(components, "reset_policy_on_life_lost", False)
        )

        pygame.init()
        self.game_scale = 3
        # 160x210 game → scaled + side panel
        self.screen = pygame.display.set_mode((160 * self.game_scale + 420, 210 * self.game_scale))
        pygame.display.set_caption("Montezuma waypoint debug — play room 1, Q to quit")
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_sm = pygame.font.SysFont("monospace", 12)

        self.rng_key = jax.random.PRNGKey(0)
        self.obs = self.state = None
        self.policy_state = initial_policy_state(self.no_target_idx)
        self.prev_target = self.no_target_idx
        self.current_target = self.no_target_idx
        self.step_count = 0
        self.frozen = False
        self._jsonl = open(self.jsonl_path, "w", encoding="utf-8")
        self._summary = open(self.summary_path, "w", encoding="utf-8")
        self._write_summary(
            f"Montezuma play log started {datetime.now().isoformat()}\n"
            f"out_dir={out_dir}\nshot_every={self.shot_every}\n"
            f"plan slots: SHAFT={_SHAFT_BEAD_SLOT} EDGE={_EDGE_COLLECT_SLOT} "
            f"ROPE={_ROPE_GRAB_SLOT} L0_BOT={_L0_BOT} KEY={_KEY_SLOT} "
            f"L0_TOP={_L0_TOP} LEFT_DOOR={_LEFT_DOOR_SLOT}\n"
            "Controls: arrows+space, R reset, F freeze, N step, W reload WP*, Q quit\n\n"
        )

    def _rebind_heuristic(self) -> None:
        import concept_based_learning.environments.montezumarevenge.montezuma_policy as mp

        self.get_heuristic_with_state = mp.get_heuristic_target_with_state
        self.get_heuristic_target = mp.get_heuristic_target

    def _reload_waypoints(self, *, from_py: bool) -> None:
        if from_py:
            mp = hot_reload()
            self._rebind_heuristic()
            self.policy_state = initial_policy_state(self.no_target_idx)
            self.prev_target = self.no_target_idx
            self._write_summary(
                f"W: reloaded WP* from .py  ARRIVE_R={mp.ARRIVE_R} WP0={mp.WP0}\n"
            )
            return
        if maybe_reload_waypoints_json():
            self._rebind_heuristic()
            self.policy_state = initial_policy_state(self.no_target_idx)
            self.prev_target = self.no_target_idx
            import concept_based_learning.environments.montezumarevenge.montezuma_policy as mp

            self._write_summary(
                f"JSON: reloaded waypoints  ARRIVE_R={mp.ARRIVE_R} WP0={mp.WP0}\n"
            )

    def _write_summary(self, text: str):
        self._summary.write(text)
        self._summary.flush()
        print(text, end="" if text.endswith("\n") else "\n")

    def _ale_to_env(self, ale: int) -> int:
        if ale in self.ale_to_idx:
            return self.ale_to_idx[ale]
        return self.ale_to_idx.get(ACTION_NOOP, 0)

    def _poll_input(self):
        keys = pygame.key.get_pressed()
        ale = ACTION_NOOP
        quit_ = reset_ = freeze_ = step_ = reload_wp_ = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_ = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    quit_ = True
                elif ev.key == pygame.K_r:
                    reset_ = True
                elif ev.key == pygame.K_f:
                    freeze_ = True
                elif ev.key == pygame.K_n:
                    step_ = True
                elif ev.key == pygame.K_w:
                    reload_wp_ = True
        u, d, l, r, sp = (
            keys[pygame.K_UP],
            keys[pygame.K_DOWN],
            keys[pygame.K_LEFT],
            keys[pygame.K_RIGHT],
            keys[pygame.K_SPACE],
        )
        if u and r:
            ale = ACTION_UPRIGHTFIRE if sp else ACTION_UPRIGHT
        elif u and l:
            ale = ACTION_UPLEFTFIRE if sp else ACTION_UPLEFT
        elif d and r:
            ale = ACTION_DOWNRIGHTFIRE if sp else ACTION_DOWNRIGHT
        elif d and l:
            ale = ACTION_DOWNLEFTFIRE if sp else ACTION_DOWNLEFT
        elif u:
            ale = ACTION_UPFIRE if sp else ACTION_UP
        elif d:
            ale = ACTION_DOWNFIRE if sp else ACTION_DOWN
        elif r:
            ale = ACTION_RIGHTFIRE if sp else ACTION_RIGHT
        elif l:
            ale = ACTION_LEFTFIRE if sp else ACTION_LEFT
        elif sp:
            ale = ACTION_FIRE
        return self._ale_to_env(ale), quit_, reset_, freeze_, step_, reload_wp_

    def reset(self):
        self.rng_key, sub = jax.random.split(self.rng_key)
        self.obs, self.state = self.env.reset(sub)
        self.policy_state = initial_policy_state(self.no_target_idx)
        self.prev_target = self.no_target_idx
        self.current_target = self.no_target_idx
        self.step_count = 0
        self._write_summary(f"\n=== RESET @ {datetime.now().isoformat()} ===\n")
        self._log_frame(action=0, note="reset")
        self._save_shot(force=True)

    def _pick_target(self):
        self.rng_key, key = jax.random.split(self.rng_key)
        raw = self.obs.raw_observation
        objs = self.obs.objects
        prev = jnp.int32(self.prev_target)
        if self.get_heuristic_with_state is not None:
            out = self.get_heuristic_with_state(
                raw, objs, prev, self.policy_state, key
            )
            tgt = int(jnp.ravel(out[0])[0])
            self.policy_state = out[1]
        else:
            tgt = int(
                jnp.ravel(self.get_heuristic_target(raw, objs, prev, key))[0]
            )
        self.prev_target = tgt
        self.current_target = tgt
        return tgt

    def _log_frame(self, action: int, note: str = ""):
        px, py, pw, ph = _player_xywh(self.obs)
        objs = _active_objects(self.obs)
        wps = _waypoints_only(objs)
        plan_idx = int(getattr(self.policy_state, "plan_idx", -1))
        mission = int(getattr(self.policy_state, "mission_target", self.no_target_idx))
        tgt = int(self.current_target)
        tgt_name = _SLOT_NAME.get(tgt, f"slot{tgt}" if tgt != self.no_target_idx else "NONE")

        # Distance player-centre → target centre if active
        pcx, pcy = px + pw * 0.5, py + ph * 0.5
        tgt_info = next((o for o in objs if o["slot"] == tgt), None)
        dist = None
        if tgt_info is not None:
            dist = float(
                np.hypot(pcx - tgt_info["cx"], pcy - tgt_info["cy"])
            )

        rec = {
            "step": self.step_count,
            "note": note,
            "action": int(action),
            "player": {
                "x": px,
                "y": py,
                "w": pw,
                "h": ph,
                "cx": pcx,
                "cy": pcy,
            },
            "plan_idx": plan_idx,
            "mission_target": mission,
            "selected_target": tgt,
            "selected_name": tgt_name,
            "dist_to_target": dist,
            "waypoints": wps,
            "avoid_active": [o for o in objs if o["label"] == "AVOID"],
            "nav_active": [o for o in objs if o["label"] == "NAVIGATE"],
        }
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()

        wp_str = " | ".join(
            f"{w['name']}@{w['cx']:.0f},{w['cy']:.0f}({w['w']:.0f}x{w['h']:.0f})"
            for w in wps
        )
        line = (
            f"t={self.step_count:5d}  player=({px:5.1f},{py:5.1f})  "
            f"plan={plan_idx}  target={tgt_name}({tgt})  "
            f"dist={dist if dist is not None else float('nan'):6.1f}  "
            f"WPs:[{wp_str}]\n"
        )
        self._write_summary(line)

    def _save_shot(self, force: bool = False):
        if not force and (self.step_count % self.shot_every) != 0:
            return
        # Render game frame
        try:
            # Unwrap env_state for renderer
            st = self.state
            for _ in range(8):
                if hasattr(st, "env_state"):
                    gs = st.env_state
                    break
                if hasattr(st, "atari_state"):
                    st = st.atari_state
                    continue
                gs = st
                break
            else:
                gs = self.state
            frame = np.asarray(jax.device_get(self.renderer.render(gs)))
        except Exception as exc:
            self._write_summary(f"[shot failed] {exc}\n")
            return

        # Draw overlays onto a pygame surface then save
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(
            surf, (160 * self.game_scale, 210 * self.game_scale)
        )

        # Overlay waypoints + player + target
        px, py, pw, ph = _player_xywh(self.obs)
        sc = self.game_scale

        def rect(x, y, w, h, color, width=2):
            pygame.draw.rect(
                surf,
                color,
                (int(x * sc), int(y * sc), max(1, int(w * sc)), max(1, int(h * sc))),
                width,
            )

        rect(px, py, pw, ph, (0, 255, 0), 2)
        for o in _overlay_plan_objects(_active_objects(self.obs)):
            rect(o["x"], o["y"], o["w"], o["h"], _overlay_color(o, self.current_target), 2)
        for o in _active_objects(self.obs):
            if o["label"] == "AVOID":
                rect(o["x"], o["y"], o["w"], o["h"], (255, 180, 40), 1)

        path = self.shots_dir / f"f{self.step_count:05d}.png"
        pygame.image.save(surf, str(path))

    def _draw_ui(self):
        self.screen.fill((10, 10, 14))
        try:
            st = self.state
            for _ in range(8):
                if hasattr(st, "env_state"):
                    gs = st.env_state
                    break
                if hasattr(st, "atari_state"):
                    st = st.atari_state
                    continue
                gs = st
                break
            else:
                gs = self.state
            frame = np.asarray(jax.device_get(self.renderer.render(gs)))
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surf = pygame.transform.scale(
                surf, (160 * self.game_scale, 210 * self.game_scale)
            )
            self.screen.blit(surf, (0, 0))
        except Exception:
            pass

        # Overlays on screen (game area)
        px, py, pw, ph = _player_xywh(self.obs)
        sc = self.game_scale

        def rect(x, y, w, h, color, width=2):
            pygame.draw.rect(
                self.screen,
                color,
                (int(x * sc), int(y * sc), max(1, int(w * sc)), max(1, int(h * sc))),
                width,
            )

        rect(px, py, pw, ph, (0, 255, 0), 2)
        for o in _overlay_plan_objects(_active_objects(self.obs)):
            color = _overlay_color(o, self.current_target)
            rect(o["x"], o["y"], o["w"], o["h"], color, 2)
            tag = self.font_sm.render(o["name"], True, color)
            self.screen.blit(tag, (int(o["x"] * sc), max(0, int(o["y"] * sc) - 14)))
        for o in _active_objects(self.obs):
            if o["label"] == "AVOID":
                rect(o["x"], o["y"], o["w"], o["h"], (255, 180, 40), 1)

        # Side panel
        x0 = 160 * sc + 12
        y = 12
        lines = [
            f"step {self.step_count}",
            f"player ({px:.1f},{py:.1f})",
            f"plan_idx {int(getattr(self.policy_state, 'plan_idx', -1))}",
            f"target {_SLOT_NAME.get(self.current_target, self.current_target)}",
            "",
            "active waypoints:",
        ]
        for o in _waypoints_only(_active_objects(self.obs)):
            mark = ">>" if o["slot"] == self.current_target else "  "
            lines.append(
                f"{mark}{o['name']:10s} ({o['cx']:.0f},{o['cy']:.0f}) "
                f"{o['w']:.0f}x{o['h']:.0f}"
            )
        lines += ["", "Q quit  R reset  F freeze  W reload WP*", f"shots every {self.shot_every}f"]
        if self.frozen:
            lines.append("*** FROZEN (N to step) ***")
        for line in lines:
            self.screen.blit(self.font.render(line, True, (220, 220, 220)), (x0, y))
            y += 18

    def step_env(self, action: int):
        tgt = self._pick_target()
        del tgt
        prev_state = self.state
        self.obs, self.state, reward, done, info = self.env.step(self.state, action)
        del reward, info
        if self.reset_policy_on_life_lost and prev_state is not None:
            try:
                life_lost = self.extract_life_lost(prev_state, self.state)
                if float(np.asarray(jax.device_get(life_lost)).reshape(())) > 0:
                    self.policy_state = initial_policy_state(self.no_target_idx)
                    self.prev_target = self.no_target_idx
                    self._write_summary(f"t={self.step_count} LIFE LOST — policy reset\n")
            except Exception:
                pass
        self.step_count += 1
        self._log_frame(action=action)
        self._save_shot()
        return bool(done)

    def run(self):
        self.reset()
        clock = pygame.time.Clock()
        running = True
        while running:
            action, quit_, reset_, freeze_, step_, reload_wp_ = self._poll_input()
            if quit_:
                break
            if reset_:
                self.reset()
                self.frozen = False
                continue
            if freeze_:
                self.frozen = not self.frozen
            if reload_wp_:
                self._reload_waypoints(from_py=True)
            else:
                self._reload_waypoints(from_py=False)
            if not self.frozen or step_:
                if self.step_env(action):
                    self._write_summary(f"t={self.step_count} EPISODE DONE — auto reset\n")
                    self.reset()
                    self.frozen = False
            self._draw_ui()
            pygame.display.flip()
            clock.tick(30)
        self._save_shot(force=True)
        self._write_summary(f"\n=== QUIT @ {datetime.now().isoformat()} steps={self.step_count} ===\n")
        self._jsonl.close()
        self._summary.close()
        pygame.quit()
        print(f"\nLogs written to {self.out_dir}")
        print(f"  {self.jsonl_path}")
        print(f"  {self.summary_path}")
        print(f"  {self.shots_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot-every", type=int, default=15)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Default: tmp/montezuma_play_log/<timestamp>/",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO / "tmp" / "montezuma_play_log" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = OmegaConf.load(_REPO / "configs" / "config.yaml")
    cfg = OmegaConf.to_container(base, resolve=True)
    cfg["FRAME_SKIP"] = 1
    cfg["STICKY_ACTIONS"] = False
    cfg["FIRST_FIRE"] = False
    cfg["FRAME_STACK_SIZE"] = 4
    cfg["ENV_NAME"] = "montezumarevenge"

    manager = EnvManager()
    env, renderer = manager.make_env("montezumarevenge", cfg, train=False)
    components = manager.get_components("montezumarevenge")

    print(f"Logging to {out_dir}")
    print("Play the first room; press Q when done.")
    MontezumaPlayLogger(
        env, renderer, components, cfg, out_dir, shot_every=args.shot_every
    ).run()


if __name__ == "__main__":
    main()
