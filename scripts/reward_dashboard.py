#!/usr/bin/env python3
"""
Interactive Reward Dashboard — manual play with live skill reward overlay.

Run with:
    python scripts/reward_dashboard.py                        # defaults to seaquest
    python scripts/reward_dashboard.py --experiment seaquest_tandem
    python scripts/reward_dashboard.py --env-name freeway

Controls:
    Arrow keys + Space : move / fire
    R                  : reset episode
    F                  : freeze / unfreeze
    N                  : single-step while frozen
    W                  : hot-reload Montezuma WP* from montezuma_policy.py
                         (also auto-reloads tmp/montezuma_waypoints.json)
    Q / ESC            : quit
"""
import sys
import os
import argparse
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from omegaconf import OmegaConf

from concept_based_learning.environments.datatypes import (
    LABEL_NOOP,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NAVIGATE,
    LABEL_JUMP_OVER,
)
from concept_based_learning.environments.env_manager import EnvManager
from concept_based_learning.environments.heuristics import (
    initial_policy_state,
    resolve_heuristic_target_fn,
    resolve_heuristic_with_mission_fn,
    resolve_heuristic_with_state_fn,
)
from concept_based_learning.environments.skill_rewards import object_reward_eligible

# ── colours ──────────────────────────────────────────────────────────────────
BLACK     = (0,   0,   0)
WHITE     = (255, 255, 255)
GREEN     = (0,   220, 0)
RED       = (220, 50,  50)
BLUE      = (80,  140, 255)
YELLOW    = (255, 220, 0)
CYAN      = (0,   220, 220)
MAGENTA   = (220, 0,   220)
GRAY      = (140, 140, 140)
DARK_GRAY = (50,  50,  50)
ORANGE    = (255, 160, 30)

TARGET_COLOR = (255, 60, 60)
NAVIGATE_COLOR = (80, 200, 255)
AVOID_COLOR = (255, 180, 40)
# Visual size of NAVIGATE waypoints in *game* pixels (objects themselves are 1x1).
NAVIGATE_MARKER_GAME_PX = 8
AVOID_MARKER_GAME_PX = 10

SKILL_NAMES  = ["Destroy", "Collect", "Avoid"]
# Colour per skill for reward display
SKILL_COLORS = [ORANGE, CYAN, MAGENTA]
# Dashboard shows the first three registry skills (destroy / collect / avoid).
DASHBOARD_SKILL_SLICE = slice(0, len(SKILL_NAMES))

LABEL_NAMES = {
    LABEL_NOOP:      "Noop",
    LABEL_DESTROY:   "Destroy",
    LABEL_COLLECT:   "Collect",
    LABEL_AVOID:     "Avoid",
    LABEL_NAVIGATE:  "Navigate",
    LABEL_JUMP_OVER: "JumpOver",
}

# Seaquest slot names — index 0 is the first game entity; meta-NOOP is last.
SEAQUEST_OBJECT_NAMES = (
    ["[NOOP]"]
    + [f"Diver {i}"   for i in range(4)]        # slots 1-4
    + [f"Shark {i}"   for i in range(12)]        # slots 5-16
    + [f"Sub {i}"     for i in range(12)]        # slots 17-28
    + ["Surface Sub"]                             # slot 29
    + [f"Missile {i}" for i in range(4)]         # slots 30-33
    + ["Surface WP"]                              # slot 34
)

FREEWAY_OBJECT_NAMES = (
    ["[NOOP]"]
    + [f"Car {i}" for i in range(10)]            # slots 1-10
    + ["Goal"]                                    # slot 11
)

# Fallback for unknown envs
def _default_names(n):
    return [f"Obj {i}" for i in range(n)]


# ── Atari action constants ────────────────────────────────────────────────────
ACTION_NOOP         = 0
ACTION_FIRE         = 1
ACTION_UP           = 2
ACTION_RIGHT        = 3
ACTION_LEFT         = 4
ACTION_DOWN         = 5
ACTION_UPRIGHT      = 6
ACTION_UPLEFT       = 7
ACTION_DOWNRIGHT    = 8
ACTION_DOWNLEFT     = 9
ACTION_UPFIRE       = 10
ACTION_RIGHTFIRE    = 11
ACTION_LEFTFIRE     = 12
ACTION_DOWNFIRE     = 13
ACTION_UPRIGHTFIRE  = 14
ACTION_UPLEFTFIRE   = 15
ACTION_DOWNRIGHTFIRE = 16
ACTION_DOWNLEFTFIRE = 17


def _unwrap_action_set(env):
    """Return the game ``ACTION_SET`` (ALE ids) from a possibly wrapped env."""
    cur = env
    for _ in range(12):
        action_set = getattr(cur, "ACTION_SET", None)
        if action_set is not None:
            return [int(x) for x in list(action_set)]
        nxt = getattr(cur, "env", None)
        if nxt is None:
            nxt = getattr(cur, "_env", None)
        if nxt is None:
            break
        cur = nxt
    return list(range(18))


def _ale_to_action_index_map(action_set):
    """Map ALE action id → Discrete index into ``ACTION_SET``."""
    return {int(ale): idx for idx, ale in enumerate(action_set)}


# When an ALE id is absent from a compact ACTION_SET, try these substitutes
# (first present wins). Keeps diagonals / fire combos playable on 4–10 action games.
_ALE_FALLBACKS: dict[int, tuple[int, ...]] = {
    ACTION_UPRIGHT: (ACTION_UP, ACTION_RIGHT),
    ACTION_UPLEFT: (ACTION_UP, ACTION_LEFT),
    ACTION_DOWNRIGHT: (ACTION_DOWN, ACTION_RIGHT),
    ACTION_DOWNLEFT: (ACTION_DOWN, ACTION_LEFT),
    ACTION_UPFIRE: (ACTION_FIRE, ACTION_UP),
    ACTION_RIGHTFIRE: (ACTION_FIRE, ACTION_RIGHT),
    ACTION_LEFTFIRE: (ACTION_FIRE, ACTION_LEFT),
    ACTION_DOWNFIRE: (ACTION_FIRE, ACTION_DOWN),
    ACTION_UPRIGHTFIRE: (
        ACTION_UPFIRE,
        ACTION_RIGHTFIRE,
        ACTION_UPRIGHT,
        ACTION_FIRE,
        ACTION_UP,
        ACTION_RIGHT,
    ),
    ACTION_UPLEFTFIRE: (
        ACTION_UPFIRE,
        ACTION_LEFTFIRE,
        ACTION_UPLEFT,
        ACTION_FIRE,
        ACTION_UP,
        ACTION_LEFT,
    ),
    ACTION_DOWNRIGHTFIRE: (
        ACTION_DOWNFIRE,
        ACTION_RIGHTFIRE,
        ACTION_DOWNRIGHT,
        ACTION_FIRE,
        ACTION_DOWN,
        ACTION_RIGHT,
    ),
    ACTION_DOWNLEFTFIRE: (
        ACTION_DOWNFIRE,
        ACTION_LEFTFIRE,
        ACTION_DOWNLEFT,
        ACTION_FIRE,
        ACTION_DOWN,
        ACTION_LEFT,
    ),
    ACTION_UP: (ACTION_NOOP,),
    ACTION_DOWN: (ACTION_NOOP,),
    ACTION_LEFT: (ACTION_NOOP,),
    ACTION_RIGHT: (ACTION_NOOP,),
    ACTION_FIRE: (ACTION_NOOP,),
}


class RewardDashboard:
    def __init__(
        self,
        env,
        renderer,
        *,
        env_name: str,
        get_heuristic_target,
        get_heuristic_with_mission=None,
        get_heuristic_with_state=None,
        reset_policy_on_life_lost=False,
        no_target_idx,
        reward_extractor,
        extract_life_lost,
        screen_width=1280,
        screen_height=630,
    ):
        self.env = env
        self.renderer = renderer
        self.env_name = env_name.lower()
        self.get_heuristic_target = get_heuristic_target
        self.get_heuristic_with_mission = get_heuristic_with_mission
        self.get_heuristic_with_state = get_heuristic_with_state
        self.reset_policy_on_life_lost = bool(reset_policy_on_life_lost)
        self.no_target_idx = no_target_idx
        self.reward_extractor = reward_extractor
        self.extract_life_lost = extract_life_lost

        # Keyboard emits Seaquest-style ALE ids; compact ACTION_SETs (Asteroids,
        # Time Pilot, …) need remapping so RIGHTFIRE≠DOWNFIRE index collision.
        self._action_set = _unwrap_action_set(env)
        self._ale_to_idx = _ale_to_action_index_map(self._action_set)

        # Object name table
        if "seaquest" in self.env_name:
            self._obj_names = SEAQUEST_OBJECT_NAMES
        elif "freeway" in self.env_name:
            self._obj_names = FREEWAY_OBJECT_NAMES
        else:
            self._obj_names = _default_names(128)

        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"{env_name.title()} — Skill Reward Dashboard")
        self.font       = pygame.font.SysFont("monospace", 13)
        self.bold_font  = pygame.font.SysFont("monospace", 15, bold=True)

        self.game_scale  = 2
        self.game_w      = 160 * self.game_scale   # 320
        self.game_h      = 210 * self.game_scale   # 420
        self.panel_x     = self.game_w + 20
        self.panel_w     = screen_width - self.panel_x - 10

        # State
        self.obs = self.state = self.prev_obs = self.prev_state = None
        self.step_count = 0
        self.total_reward = 0.0
        self.rng_key = jax.random.key(42)
        self.prev_target_idx = self.no_target_idx
        self.mission_target_idx = self.no_target_idx
        self.policy_state = initial_policy_state(self.no_target_idx)
        self.current_target_idx = self.no_target_idx
        # concept_rewards[skill, obj] — only the tracked target is computed each step
        self.concept_rewards: np.ndarray = np.zeros((3, 1), dtype=np.float32)
        # Recent event log: list of dicts
        self.event_log: list = []
        self.log_max_steps = 180  # keep entries for ~6 s at 30 fps

        self.frozen = False
        self.target_fps = 60
        self.fps_history: list = []
        self.actual_fps = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self):
        self.rng_key, sub = jax.random.split(self.rng_key)
        self.obs, self.state = self.env.reset(sub)
        self.prev_obs = self.prev_state = None
        self.step_count = 0
        self.total_reward = 0.0
        self.prev_target_idx = self.no_target_idx
        self.mission_target_idx = self.no_target_idx
        self.policy_state = initial_policy_state(self.no_target_idx)
        self.current_target_idx = self.no_target_idx
        self.concept_rewards = np.zeros((3, 1), dtype=np.float32)
        self.event_log = []

    def step(self, action) -> bool:
        self.prev_obs   = self.obs
        self.prev_state = self.state

        # Pick heuristic target from the pre-step observation.
        target_idx = self.no_target_idx
        mission_idx = self.mission_target_idx
        if self.prev_obs is not None:
            try:
                self.rng_key, key_pol = jax.random.split(self.rng_key)
                raw = self.prev_obs.raw_observation
                objs = self.prev_obs.objects
                prev_idx = jnp.array(self.prev_target_idx, dtype=jnp.int32)
                if self.get_heuristic_with_state is not None:
                    out = self.get_heuristic_with_state(
                        raw, objs, prev_idx, self.policy_state, key_pol
                    )
                    target_idx = int(jnp.ravel(out[0])[0])
                    self.policy_state = out[1]
                    mission_idx = int(jnp.ravel(self.policy_state.mission_target)[0])
                elif self.get_heuristic_with_mission is not None:
                    mission_arr = jnp.array(self.mission_target_idx, dtype=jnp.int32)
                    out = self.get_heuristic_with_mission(
                        raw, objs, prev_idx, mission_arr, key_pol
                    )
                    target_idx = int(jnp.ravel(out[0])[0])
                    mission_idx = int(jnp.ravel(out[1])[0])
                else:
                    target_idx = int(jnp.ravel(
                        self.get_heuristic_target(raw, objs, prev_idx, key_pol)
                    )[0])
                    mission_idx = target_idx
            except Exception as exc:
                print(f"[reward_dashboard] heuristic error: {type(exc).__name__}: {exc}")

        self.prev_target_idx = target_idx
        self.mission_target_idx = mission_idx
        self.current_target_idx = target_idx

        self.obs, self.state, env_reward, done, info = self.env.step(self.state, action)
        self.total_reward += float(env_reward)

        # Optional env-specific: restart Sequence plan on soft respawn.
        if self.reset_policy_on_life_lost and self.prev_state is not None:
            try:
                life_lost = self.extract_life_lost(self.prev_state, self.state)
                if float(np.asarray(jax.device_get(life_lost)).reshape(())) > 0:
                    self.policy_state = initial_policy_state(self.no_target_idx)
                    self.prev_target_idx = self.no_target_idx
                    self.mission_target_idx = self.no_target_idx
            except Exception:
                pass

        if self.prev_obs is not None:
            self._compute_rewards(target_idx, mission_idx, env_reward, info)

        self.step_count += 1
        # Purge stale log entries
        self.event_log = [e for e in self.event_log
                          if self.step_count - e["step"] <= self.log_max_steps]
        return bool(done)

    # ── reward computation ────────────────────────────────────────────────────

    def _slice_obj(self, obj_obs, idx: int):
        """Extract a single object's data across the frame stack."""
        def _leaf(x):
            if not hasattr(x, "ndim") or x.ndim == 0:
                return x
            if x.ndim == 1:
                return x[idx] if idx < x.shape[0] else x
            return x[..., idx] if idx < x.shape[-1] else x
        return jax.tree_util.tree_map(_leaf, obj_obs)

    def _compute_rewards(self, tracked_idx: int, mission_idx: int, env_reward: float, info):
        """Compute skill rewards for tracked (and mission) target objects."""
        objects      = self.obs.objects
        prev_objects = self.prev_obs.objects
        num_objects  = int(objects.x.shape[1])

        life_lost = self.extract_life_lost(self.prev_state, self.state)
        if self.concept_rewards.shape != (3, num_objects):
            self.concept_rewards = np.zeros((3, num_objects), dtype=np.float32)
        else:
            self.concept_rewards.fill(0.0)

        reward_idxs = [tracked_idx]
        if mission_idx != tracked_idx and mission_idx >= 0:
            reward_idxs.append(mission_idx)

        for obj_idx in reward_idxs:
            if obj_idx < 0 or obj_idx >= num_objects:
                continue
            self._compute_reward_for_object(
                obj_idx, life_lost, env_reward, info, tracked=(obj_idx == tracked_idx)
            )

    def _compute_reward_for_object(
        self, obj_idx: int, life_lost, env_reward: float, info, *, tracked: bool
    ):
        objects = self.obs.objects
        prev_objects = self.prev_obs.objects
        labels = np.array(objects.semantic_label[-1])

        prev_t = self._slice_obj(prev_objects, obj_idx)
        next_t = self._slice_obj(objects, obj_idx)
        if not bool(
            np.asarray(jax.device_get(object_reward_eligible(prev_t, next_t)))
        ):
            return

        try:
            out = self.reward_extractor(
                self.prev_obs.raw_observation,
                self.obs.raw_observation,
                prev_t, next_t, life_lost, info, {},
            )
            rvec = np.asarray(
                jax.device_get(out[0] if isinstance(out, tuple) else out),
                dtype=np.float32,
            )
            if rvec.shape[0] > len(SKILL_NAMES):
                rvec = rvec[DASHBOARD_SKILL_SLICE]
            elif rvec.shape[0] < len(SKILL_NAMES):
                padded = np.zeros(len(SKILL_NAMES), dtype=np.float32)
                padded[: rvec.shape[0]] = rvec
                rvec = padded
            self.concept_rewards[:, obj_idx] = rvec
        except Exception:
            return

        if not np.any(np.abs(rvec) > 1e-6):
            return

        tag = "TRACKED" if tracked else "MISSION"
        obj_name = self._obj_name(obj_idx)
        label_str = LABEL_NAMES.get(int(labels[obj_idx]), "?")
        entry = {
            "step":       self.step_count,
            "obj_idx":    obj_idx,
            "obj_name":   obj_name,
            "label":      label_str,
            "rvec":       rvec.tolist(),
            "env_reward": env_reward,
            "tracked":    tracked,
        }
        self.event_log.append(entry)
        print(
            f"step={self.step_count:5d} [{tag}] "
            f"{obj_name:<16s} [{label_str:<7s}] "
            f"D={rvec[0]:+.1f} C={rvec[1]:+.1f} A={rvec[2]:+.1f}  "
            f"env={env_reward:+.0f}"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _obj_name(self, idx: int) -> str:
        if idx < len(self._obj_names):
            return self._obj_names[idx]
        return f"Obj {idx}"

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_game(self):
        frame = None
        try:
            if self.state is not None and self.renderer is not None:
                gs = getattr(
                    getattr(self.state, "atari_state", self.state),
                    "env_state", self.state
                )
                frame = np.asarray(jax.device_get(self.renderer.render(gs)))
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) \
                        if frame.max() <= 1.0 else np.clip(frame, 0, 255).astype(np.uint8)
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[2] == 4:
                    frame = frame[:, :, :3]
        except Exception:
            frame = None

        if frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (self.game_w, self.game_h))
            self.screen.blit(surf, (10, 10))
        else:
            pygame.draw.rect(self.screen, DARK_GRAY, (10, 10, self.game_w, self.game_h))

        self._draw_navigate_checkpoints()
        self._draw_avoid_target_marker()
        self._draw_target_outline()

    def _object_screen_rect(self, x, y, w, h, *, min_game_px: int = 1):
        """Map an object AABB to screen pixels; enforce a minimum on-screen size."""
        gw = max(int(w), int(min_game_px))
        gh = max(int(h), int(min_game_px))
        # Centre the inflated rect on the object's geometric centre.
        cx = float(x) + float(w) / 2.0
        cy = float(y) + float(h) / 2.0
        sx = 10 + int(round((cx - gw / 2.0) * self.game_scale))
        sy = 10 + int(round((cy - gh / 2.0) * self.game_scale))
        return sx, sy, gw * self.game_scale, gh * self.game_scale

    def _draw_navigate_checkpoints(self):
        """Draw enlarged markers for small NAVIGATE / ladder waypoints (often 1–2 px)."""
        if self.obs is None:
            return
        try:
            xa = np.array(self.obs.objects.x[-1])
            ya = np.array(self.obs.objects.y[-1])
            wa = np.array(self.obs.objects.width[-1])
            ha = np.array(self.obs.objects.height[-1])
            act = np.array(self.obs.objects.active[-1])
            labels = np.array(self.obs.objects.semantic_label[-1])
        except Exception:
            return

        for i in range(int(act.shape[0])):
            if act[i] <= 0:
                continue
            lab = int(labels[i])
            # Compact NAVIGATE markers only — tall doors/ropes keep native AABBs
            # (drawing them as "checkpoints" was confusing in Montezuma).
            is_small_nav = (
                lab == LABEL_NAVIGATE and max(float(wa[i]), float(ha[i])) <= 12.0
            )
            # Kangaroo ladder waypoints are COLLECT slots 1..40.
            is_kangaroo_wp = (
                "kangaroo" in self.env_name
                and lab == LABEL_COLLECT
                and 1 <= i <= 40
            )
            # Montezuma ladder waypoints are COLLECT slots 4..11.
            is_montezuma_wp = (
                "montezuma" in self.env_name
                and lab == LABEL_COLLECT
                and 4 <= i <= 11
            )
            if not (is_small_nav or is_kangaroo_wp or is_montezuma_wp):
                continue
            sx, sy, sw, sh = self._object_screen_rect(
                xa[i], ya[i], wa[i], ha[i], min_game_px=NAVIGATE_MARKER_GAME_PX
            )
            is_target = i == int(self.current_target_idx)
            color = TARGET_COLOR if is_target else NAVIGATE_COLOR
            width = 3 if is_target else 2
            pygame.draw.rect(self.screen, color, (sx, sy, sw, sh), width)
            # Small filled centre so 1px waypoints stay readable when not selected.
            cx = sx + sw // 2
            cy = sy + sh // 2
            pygame.draw.circle(self.screen, color, (cx, cy), max(2, self.game_scale), 0)

    def _draw_avoid_target_marker(self):
        """Enlarge the selected AVOID object (coconuts are easy to miss at native size)."""
        if self.obs is None or self.current_target_idx == self.no_target_idx:
            return
        idx = int(self.current_target_idx)
        try:
            xa = np.array(self.obs.objects.x[-1])
            ya = np.array(self.obs.objects.y[-1])
            wa = np.array(self.obs.objects.width[-1])
            ha = np.array(self.obs.objects.height[-1])
            act = np.array(self.obs.objects.active[-1])
            labels = np.array(self.obs.objects.semantic_label[-1])
        except Exception:
            return
        if idx >= len(act) or act[idx] <= 0:
            return
        if int(labels[idx]) != LABEL_AVOID:
            return
        sx, sy, sw, sh = self._object_screen_rect(
            xa[idx], ya[idx], wa[idx], ha[idx], min_game_px=AVOID_MARKER_GAME_PX
        )
        pygame.draw.rect(self.screen, AVOID_COLOR, (sx, sy, sw, sh), 3)
        cx = sx + sw // 2
        cy = sy + sh // 2
        pygame.draw.circle(self.screen, TARGET_COLOR, (cx, cy), max(3, self.game_scale), 0)

    def _draw_target_outline(self):
        if self.obs is None or self.current_target_idx == self.no_target_idx:
            return
        idx = int(self.current_target_idx)
        try:
            xa = np.array(self.obs.objects.x[-1])
            ya = np.array(self.obs.objects.y[-1])
            wa = np.array(self.obs.objects.width[-1])
            ha = np.array(self.obs.objects.height[-1])
            act = np.array(self.obs.objects.active[-1])
            labels = np.array(self.obs.objects.semantic_label[-1])
        except Exception:
            return
        if idx >= len(act) or act[idx] <= 0:
            return
        lab = int(labels[idx])
        # Waypoint / coconut markers already highlight the selected checkpoint.
        if lab == LABEL_NAVIGATE and max(float(wa[idx]), float(ha[idx])) <= 12.0:
            return
        if "kangaroo" in self.env_name and lab == LABEL_COLLECT and 1 <= idx <= 40:
            return
        if "montezuma" in self.env_name and lab == LABEL_COLLECT and 4 <= idx <= 11:
            return
        if lab == LABEL_AVOID:
            return
        x, y, w, h = int(xa[idx]), int(ya[idx]), int(wa[idx]), int(ha[idx])
        sx, sy, sw, sh = self._object_screen_rect(x, y, w, h, min_game_px=2)
        pygame.draw.rect(self.screen, TARGET_COLOR, (sx, sy, sw, sh), 2)

    def _draw_panel(self):
        x0 = self.panel_x
        y  = 10
        lh = 16  # line height

        def text(s, px, py, color=WHITE, bold=False):
            f = self.bold_font if bold else self.font
            self.screen.blit(f.render(s, True, color), (px, py))
            return py + lh

        # ── header ────────────────────────────────────────────────────────────
        y = text(f"{self.env_name.upper()}  step={self.step_count}  "
                 f"env_total={self.total_reward:.0f}  fps={self.actual_fps:.0f}",
                 x0, y, YELLOW, bold=True)
        y += 4

        tgt_name = self._obj_name(int(self.current_target_idx))
        tgt_label = ""
        if self.obs is not None:
            try:
                lab = int(np.array(self.obs.objects.semantic_label[-1])[int(self.current_target_idx)])
                tgt_label = LABEL_NAMES.get(lab, "?")
            except Exception:
                pass
        y = text(f"Target: [{self.current_target_idx}] {tgt_name}  ({tgt_label})",
                 x0, y, TARGET_COLOR, bold=True)
        y += 6

        # ── skill reward legend ───────────────────────────────────────────────
        legend = "Skills: "
        for i, (name, col) in enumerate(zip(SKILL_NAMES, SKILL_COLORS)):
            lx = x0 + len(legend) * 8 if i == 0 else None  # approximate; drawn separately
        # Draw inline
        lx = x0
        self.screen.blit(self.font.render("Skills: ", True, GRAY), (lx, y))
        lx += self.font.size("Skills: ")[0]
        for name, col in zip(SKILL_NAMES, SKILL_COLORS):
            s = f"[{name}]  "
            self.screen.blit(self.font.render(s, True, col), (lx, y))
            lx += self.font.size(s)[0]
        y += lh + 4

        # ── active objects table ──────────────────────────────────────────────
        y = text("─── Active Objects ───────────────────────────────────────────",
                 x0, y, DARK_GRAY)

        if self.obs is not None:
            labels      = np.array(self.obs.objects.semantic_label[-1])
            active_flags= np.array(self.obs.objects.active[-1])
            was_killed_flags = np.array(self.obs.objects.was_killed[-1])
            num_objects = int(active_flags.shape[0])
            col_right   = x0 + self.panel_w - 5

            for obj_idx in range(num_objects):
                if active_flags[obj_idx] <= 0 and was_killed_flags[obj_idx] <= 0:
                    continue
                if y > self.game_h + 10:
                    text("  … more objects …", x0, y, GRAY)
                    y += lh
                    break

                is_tracked = (obj_idx == int(self.current_target_idx))
                label_id   = int(labels[obj_idx])
                label_str  = LABEL_NAMES.get(label_id, "?")
                obj_name   = self._obj_name(obj_idx)

                rvec = self.concept_rewards[:, obj_idx] \
                    if obj_idx < self.concept_rewards.shape[1] \
                    else np.zeros(3, dtype=np.float32)
                has_reward = np.any(np.abs(rvec) > 1e-6)

                # Row colour
                if is_tracked:
                    row_color = WHITE
                elif has_reward:
                    row_color = GREEN
                else:
                    row_color = GRAY

                # Object name + label
                row = f"{'▶ ' if is_tracked else '  '}[{obj_idx:2d}] {obj_name:<16s} {label_str:<7s}"
                text(row, x0, y, row_color, bold=is_tracked)

                # Skill reward values, right-aligned
                rx = col_right
                for skill_i in range(2, -1, -1):
                    rv = float(rvec[skill_i])
                    if abs(rv) > 1e-6:
                        s = f"{SKILL_NAMES[skill_i][0]}:{rv:+.1f} "
                        col = GREEN if rv > 0 else RED
                    else:
                        s = f"{SKILL_NAMES[skill_i][0]}:  0  "
                        col = DARK_GRAY
                    sw = self.font.size(s)[0]
                    rx -= sw
                    self.screen.blit(self.font.render(s, True, col), (rx, y))

                y += lh

        y += 8

        # ── recent events log ─────────────────────────────────────────────────
        y = text("─── Recent Reward Events ──────────────────────────────────────",
                 x0, y, DARK_GRAY)

        recent = [e for e in reversed(self.event_log)][:12]
        for e in recent:
            age   = self.step_count - e["step"]
            rvec  = e["rvec"]
            age_c = GREEN if age <= 5 else (YELLOW if age <= 15 else GRAY)
            tag   = "▶" if e["tracked"] else " "
            rv_str = "  ".join(
                f"{SKILL_NAMES[i][0]}:{rvec[i]:+.1f}"
                for i in range(3)
                if abs(rvec[i]) > 1e-6
            )
            line = f"{tag}[{e['obj_idx']:2d}] {e['obj_name']:<16s} {rv_str}  ({age}f ago)"
            y = text(line, x0, y, age_c)

        # ── controls ──────────────────────────────────────────────────────────
        cy = self.game_h + 20
        ctrl = "Arrows+Space: move/fire   R: reset   F: freeze   N: step   Q: quit"
        if self._is_montezuma():
            ctrl += "   W: reload WP*"
        text(ctrl, x0, cy, DARK_GRAY)
        if self.frozen:
            text("*** FROZEN — press N to step ***", x0, cy + lh, YELLOW, bold=True)

    # ── input ─────────────────────────────────────────────────────────────────

    def _ale_to_env_action(self, ale_action: int) -> int:
        """Convert a Seaquest-style ALE action id to an env Discrete index.

        Applies to **all** envs: compact ``ACTION_SET`` tables (Asteroids,
        NameThisGame, Phoenix, …) store ALE ids, so the Discrete index is not
        the ALE id.  Missing combos fall back to the nearest available action.
        """
        ale = int(ale_action)
        # Asteroids: DOWN/DOWNFIRE = hyperspace. Arrow-down must not teleport.
        if "asteroids" in self.env_name:
            if ale == ACTION_DOWN:
                ale = ACTION_NOOP
            elif ale == ACTION_DOWNFIRE:
                ale = ACTION_FIRE
            elif ale in (ACTION_DOWNRIGHT, ACTION_DOWNRIGHTFIRE):
                ale = ACTION_RIGHTFIRE if ale == ACTION_DOWNRIGHTFIRE else ACTION_RIGHT
            elif ale in (ACTION_DOWNLEFT, ACTION_DOWNLEFTFIRE):
                ale = ACTION_LEFTFIRE if ale == ACTION_DOWNLEFTFIRE else ACTION_LEFT

        if ale in self._ale_to_idx:
            return int(self._ale_to_idx[ale])
        for alt in _ALE_FALLBACKS.get(ale, ()):
            if alt in self._ale_to_idx:
                return int(self._ale_to_idx[alt])
        return int(self._ale_to_idx.get(ACTION_NOOP, 0))

    def _is_montezuma(self) -> bool:
        return "montezuma" in self.env_name

    def _rebind_montezuma_heuristic(self) -> None:
        """Pick up rebuilt callables after ``hot_reload`` / JSON reload."""
        from concept_based_learning.environments.montezumarevenge import (
            montezuma_policy as mp,
        )

        self.get_heuristic_with_state = mp.get_heuristic_target_with_state
        self.get_heuristic_target = mp.get_heuristic_target

    def _reload_montezuma_waypoints(self, *, from_py: bool) -> str:
        """Reload WP* without resetting the env. Returns a short status line."""
        from concept_based_learning.environments.montezumarevenge import (
            montezuma_policy as mp,
        )

        if from_py:
            mp = mp.hot_reload()
            self._rebind_montezuma_heuristic()
            self.policy_state = initial_policy_state(self.no_target_idx)
            return f"W: reloaded WP* from .py  ARRIVE_R={mp.ARRIVE_R} WP0={mp.WP0}"
        if mp.maybe_reload_waypoints_json():
            self._rebind_montezuma_heuristic()
            self.policy_state = initial_policy_state(self.no_target_idx)
            return f"JSON: reloaded waypoints  ARRIVE_R={mp.ARRIVE_R} WP0={mp.WP0}"
        return ""

    def handle_input(self) -> Tuple[int, bool, bool, bool, bool, bool]:
        keys = pygame.key.get_pressed()
        ale_action = ACTION_NOOP
        quit_, reset_, freeze_, step_, reload_wp_ = False, False, False, False, False

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

        u, d, l, r, sp = (keys[k] for k in (
            pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE
        ))
        if u and r:   ale_action = ACTION_UPRIGHTFIRE   if sp else ACTION_UPRIGHT
        elif u and l: ale_action = ACTION_UPLEFTFIRE    if sp else ACTION_UPLEFT
        elif d and r: ale_action = ACTION_DOWNRIGHTFIRE if sp else ACTION_DOWNRIGHT
        elif d and l: ale_action = ACTION_DOWNLEFTFIRE  if sp else ACTION_DOWNLEFT
        elif u:       ale_action = ACTION_UPFIRE        if sp else ACTION_UP
        elif d:       ale_action = ACTION_DOWNFIRE      if sp else ACTION_DOWN
        elif r:       ale_action = ACTION_RIGHTFIRE     if sp else ACTION_RIGHT
        elif l:       ale_action = ACTION_LEFTFIRE      if sp else ACTION_LEFT
        elif sp:      ale_action = ACTION_FIRE

        return (
            self._ale_to_env_action(ale_action),
            quit_,
            reset_,
            freeze_,
            step_,
            reload_wp_,
        )

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.reset()
        clock = pygame.time.Clock()
        running = True

        while running:
            action, quit_, reset_, freeze_, step_, reload_wp_ = self.handle_input()
            if quit_:
                break
            if reset_:
                self.reset()
                self.frozen = False
                continue
            if freeze_:
                self.frozen = not self.frozen
            if reload_wp_ and self._is_montezuma():
                msg = self._reload_montezuma_waypoints(from_py=True)
                if msg:
                    print(msg, flush=True)
            elif self._is_montezuma():
                msg = self._reload_montezuma_waypoints(from_py=False)
                if msg:
                    print(msg, flush=True)

            if not self.frozen or step_:
                if self.step(action):
                    self.reset()
                    self.frozen = False

            self.screen.fill(BLACK)
            self._render_game()
            self._draw_panel()
            pygame.display.flip()

            fps = 5 if self.frozen else self.target_fps
            ms = clock.tick(fps)
            if ms > 0:
                self.fps_history.append(1000.0 / ms)
                if len(self.fps_history) > 10:
                    self.fps_history.pop(0)
                self.actual_fps = sum(self.fps_history) / len(self.fps_history)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="seaquest_tandem",
                        help="Experiment config name (without .yaml)")
    parser.add_argument("--env-name", default=None,
                        help="Override ENV_NAME from the experiment config")
    args = parser.parse_args()

    repo_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_path   = os.path.join(repo_root, "configs", "config.yaml")
    exp_path    = os.path.join(repo_root, "configs", "experiment", f"{args.experiment}.yaml")
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    base_cfg = OmegaConf.to_container(OmegaConf.load(base_path), resolve=True)
    exp_cfg  = OmegaConf.to_container(OmegaConf.load(exp_path),  resolve=True)
    cfg      = {**base_cfg, **exp_cfg, **exp_cfg.get("alg", {})}
    env_name = str(args.env_name or cfg.get("ENV_NAME", "seaquest"))
    # Manual play: no action-repeat flicker; show every frame with objects+player.
    cfg["FRAME_SKIP"] = 1
    cfg["STICKY_ACTIONS"] = False

    manager    = EnvManager()
    env, renderer = manager.make_env(env_name, cfg, train=False)
    components = manager.get_components(env_name)

    dashboard = RewardDashboard(
        env, renderer,
        env_name=env_name,
        get_heuristic_target=resolve_heuristic_target_fn(cfg, env_name, components),
        get_heuristic_with_mission=resolve_heuristic_with_mission_fn(cfg, env_name),
        get_heuristic_with_state=resolve_heuristic_with_state_fn(
            cfg, env_name, components
        ),
        reset_policy_on_life_lost=bool(
            getattr(components, "reset_policy_on_life_lost", False)
        ),
        no_target_idx=components.no_target_idx,
        reward_extractor=components.reward_extractor,
        extract_life_lost=components.extract_life_lost,
        screen_width=1280,
        screen_height=630,
    )

    try:
        dashboard.run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
