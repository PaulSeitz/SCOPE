#!/usr/bin/env python3
"""
Interactive skill-network demo: click an on-screen object, then watch the trained
skill policy pursue it. Press F to freeze/unfreeze and pick a new target while frozen.

Usage (from project root):
    python scripts/skill_target_demo.py --model path/to/skills.safetensors
    python scripts/skill_target_demo.py --experiment seaquest_skills_pretrain \\
        --model ./outputs/models/seaquest_skills_pretrain/seaquest/hierarchical_pqn_seaquest_seed42_skills.safetensors

Controls:
    Click object (frozen) : select target
    Click empty (frozen)  : clear target (NOOP)
    0 / Backspace (frozen): clear target (NOOP)
    F                    : freeze / unfreeze
    N                    : single-step while frozen (optional)
    R                    : reset episode
    Q / ESC              : quit
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from concept_based_learning.environments.datatypes import (  # noqa: E402
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NOOP,
)
from concept_based_learning.environments.env_manager import EnvManager  # noqa: E402
from concept_based_learning.networks.skill_network import QNetwork  # noqa: E402
from concept_based_learning.observations.action_space import fire_ego_action_indices  # noqa: E402
from concept_based_learning.observations.egocentric import (  # noqa: E402
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    _center_y,
    _get_player,
    _uses_bottom_edge_y,
    distance_pixels_from_feature,
    ego_obs_channels_per_frame,
    get_ego_obs,
    set_ego_distance_clip_max_pixels,
    set_ego_obs_format,
    set_ego_size_features,
    set_ego_velocity_features,
    using_legacy_ego_obs,
)
from concept_based_learning.pqn_visualizer import (  # noqa: E402
    align_config_with_skill_checkpoint,
    create_eval_env,
    ensure_skill_params_stack,
    label_to_skill_idx,
    load_base_config,
    load_config_for_checkpoint,
    load_params_from_file,
    validate_skill_checkpoint_ego_dim,
)

# Avoid importing video_recorder (it forces SDL_VIDEODRIVER=dummy for headless clusters).
SKILL_NAMES = ["DESTROY", "COLLECT", "AVOID"]

# Object slot names aligned with SeaquestSemanticWrapper / reward_dashboard layout.
SEAQUEST_OBJECT_NAMES = (
    ["[NOOP]"]
    + [f"Diver {i}" for i in range(4)]
    + [f"Shark {i}" for i in range(12)]
    + [f"Sub {i}" for i in range(12)]
    + ["Surface Sub"]
    + [f"Missile {i}" for i in range(4)]
    + ["Surface WP"]
)

FREEWAY_OBJECT_NAMES = (
    ["[NOOP]"]
    + [f"Car {i}" for i in range(10)]
    + ["Goal"]
)

LABEL_NAMES = {
    LABEL_NOOP: "NOOP",
    LABEL_DESTROY: "DESTROY",
    LABEL_COLLECT: "COLLECT",
    LABEL_AVOID: "AVOID",
}

LABEL_COLORS = {
    LABEL_NOOP: (200, 200, 200),
    LABEL_DESTROY: (255, 80, 80),
    LABEL_COLLECT: (80, 255, 80),
    LABEL_AVOID: (255, 200, 60),
}

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (140, 140, 140)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 220, 0)
TARGET_COLOR = (255, 60, 60)
HOVER_COLOR = (100, 180, 255)
GREEN = (80, 220, 120)


def _restore_pygame_display() -> None:
    """
    pqn_visualizer imports video_recorder, which sets SDL_VIDEODRIVER=dummy.
    Undo that so an interactive window appears on the local machine.
    """
    for key in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER"):
        os.environ.pop(key, None)
    import pygame

    if pygame.get_init():
        pygame.display.quit()
        pygame.quit()
    pygame.init()
    return pygame


def object_names_for_env(env_name: str, num_slots: int) -> list[str]:
    name = env_name.lower()
    if "seaquest" in name:
        return list(SEAQUEST_OBJECT_NAMES)
    if "freeway" in name:
        return list(FREEWAY_OBJECT_NAMES)
    return [f"Obj {i}" for i in range(num_slots)]


def _object_field_at(obs, field: str, obj_idx: int, *, frame: int = -1) -> int:
    """
    Read a per-object int flag from the latest (or given) frame.

    Some envs leave ``was_collected`` as a scalar default (0); treat that as 0
    for every object instead of indexing.
    """
    arr = np.asarray(getattr(obs.objects, field))
    if arr.ndim == 0:
        return int(arr)
    row = arr[frame]
    if np.ndim(row) == 0:
        return int(row)
    return int(row[obj_idx])


def pick_object_at_point(
    obs,
    gx: float,
    gy: float,
    *,
    skip_indices: set[int],
) -> Optional[int]:
    """Return the smallest active bbox containing (gx, gy), or None."""
    xa = np.array(obs.objects.x[-1])
    ya = np.array(obs.objects.y[-1])
    wa = np.array(obs.objects.width[-1])
    ha = np.array(obs.objects.height[-1])
    active = np.array(obs.objects.active[-1])

    hits: list[tuple[int, float]] = []
    for i in range(int(active.shape[0])):
        if i in skip_indices or active[i] <= 0:
            continue
        w, h = float(wa[i]), float(ha[i])
        if w < 1 or h < 1:
            continue
        x0, y0 = float(xa[i]), float(ya[i])
        if x0 <= gx <= x0 + w and y0 <= gy <= y0 + h:
            hits.append((i, w * h))
    if not hits:
        return None
    return min(hits, key=lambda t: t[1])[0]


class SkillTargetDemo:
    def __init__(
        self,
        env,
        renderer,
        *,
        config: dict,
        skill_params,
        action_mapper,
        no_target_idx: int,
        env_name: str,
        screen_width: int = 1180,
        screen_height: int = 640,
    ):
        self.env = env
        self.renderer = renderer
        self.config = config
        self.skill_params = skill_params
        self.action_mapper = action_mapper
        self.no_target_idx = no_target_idx
        self.env_name = env_name.lower()

        self.num_skills = int(config["NUM_SKILLS"])
        self.ego_action_dim = int(config["EGO_ACTION_DIM"])
        self.restrict_fire = bool(config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False))

        self.skill_network = QNetwork(
            action_dim=self.ego_action_dim,
            hidden_size=int(config["HIDDEN_SIZE"]),
            num_layers=int(config["NUM_LAYERS"]),
        )

        pygame = _restore_pygame_display()
        self._pygame = pygame
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Skill Target Demo")
        self.font = pygame.font.SysFont("monospace", 13)
        self.bold_font = pygame.font.SysFont("monospace", 14, bold=True)

        self.game_scale = 2
        self.game_origin = (10, 10)
        self.game_w = 160 * self.game_scale
        self.game_h = 210 * self.game_scale
        self.panel_x = self.game_origin[0] + self.game_w + 20

        self.rng = jax.random.PRNGKey(int(config.get("SEED", 42)))
        self.obs = None
        self.state = None
        self.step_count = 0
        self.frozen = True
        self.target_idx: int = no_target_idx
        self.hover_idx: Optional[int] = None
        self.last_q_vals: Optional[np.ndarray] = None
        self.last_skill_idx: int = 0
        self.last_distance = float("nan")
        self.last_angle = float("nan")
        self.last_action_name = "—"
        self._obj_names: list[str] = []
        self._jax_env_reset = False
        self._jax_render = False
        self._jax_env_step = False
        self._jax_skill = False
        self._jax_warmup_phase = "not started"

    def _jax_ready(self) -> bool:
        return (
            self._jax_env_reset
            and self._jax_render
            and self._jax_env_step
            and self._jax_skill
        )

    def _jax_status_line(self) -> str:
        def tick(done: bool) -> str:
            return "ok" if done else ".."

        if self._jax_ready():
            return "JAX: ready (compilation done)"
        return (
            "JAX: "
            f"reset[{tick(self._jax_env_reset)}] "
            f"render[{tick(self._jax_render)}] "
            f"step[{tick(self._jax_env_step)}] "
            f"skill[{tick(self._jax_skill)}]"
        )

    def _show_status(self, message: str, *, allow_quit: bool = False) -> bool:
        """Draw a full-screen status message. Returns True if the user quit."""
        pygame = self._pygame
        self.screen.fill(BLACK)
        for i, line in enumerate(message.split("\n")):
            surf = self.bold_font.render(line, True, YELLOW)
            self.screen.blit(surf, (20, 20 + i * 22))
        pygame.display.flip()
        if allow_quit:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN) and (
                    event.type == pygame.QUIT
                    or event.key in (pygame.K_q, pygame.K_ESCAPE)
                ):
                    return True
        else:
            pygame.event.pump()
        return False

    def _finish_reset_state(self) -> None:
        self.step_count = 0
        self.target_idx = self.no_target_idx
        self.hover_idx = None
        self.last_q_vals = None
        self.last_skill_idx = 0
        self.last_distance = float("nan")
        self.last_angle = float("nan")
        self.last_action_name = "—"
        n_slots = int(np.array(self.obs.objects.active[-1]).shape[0])
        self._obj_names = object_names_for_env(self.env_name, n_slots)
        self.frozen = True

    def warmup(self) -> None:
        """
        Compile JAX env / renderer / step / skill network on a worker thread while
        the main thread shows a live progress screen.
        """
        state: dict = {"done": False, "error": None}
        spinner = "|/-\\"
        tick = 0

        def worker() -> None:
            try:
                self._jax_warmup_phase = "env.reset()"
                print(f"[warmup] {self._jax_warmup_phase}", flush=True)
                self.rng, sub = jax.random.split(self.rng)
                self.obs, self.state = self.env.reset(sub)
                jax.block_until_ready(self.state)
                self._jax_env_reset = True

                self._jax_warmup_phase = "renderer.render()"
                print(f"[warmup] {self._jax_warmup_phase}", flush=True)
                gs = self.state.atari_state.env_state
                frame = self.renderer.render(gs)
                jax.block_until_ready(frame)
                self._jax_render = True

                self._jax_warmup_phase = "env.step(NOOP)"
                print(f"[warmup] {self._jax_warmup_phase}", flush=True)
                self.obs, self.state, _reward, done, _info = self.env.step(self.state, 0)
                jax.block_until_ready(self.state)
                self._jax_env_step = True

                self._jax_warmup_phase = "skill network"
                print(f"[warmup] {self._jax_warmup_phase}", flush=True)
                active = np.array(self.obs.objects.active[-1])
                warmup_target = None
                for i in range(1, int(active.shape[0])):
                    if active[i] > 0:
                        warmup_target = i
                        break
                if warmup_target is not None:
                    saved_target = self.target_idx
                    self.target_idx = warmup_target
                    self._run_skill_step()
                    self.target_idx = saved_target
                else:
                    ego_obs = get_ego_obs(
                        self.obs.raw_observation,
                        self.obs.objects,
                        jnp.array(1, dtype=jnp.int32),
                    )
                    ego_obs_batch = ego_obs[jnp.newaxis, ...]

                    def apply_one(params_for_skill):
                        return self.skill_network.apply(
                            {"params": params_for_skill},
                            ego_obs_batch,
                            train=False,
                        )

                    q_vals = jax.vmap(apply_one)(self.skill_params)
                    jax.block_until_ready(q_vals)
                self._jax_skill = True
                self._jax_warmup_phase = "ready"
                print("[warmup] done", flush=True)
                self.rng, sub = jax.random.split(self.rng)
                self.obs, self.state = self.env.reset(sub)
                jax.block_until_ready(self.state)
                self._finish_reset_state()
            except Exception as exc:
                state["error"] = exc
            finally:
                state["done"] = True

        threading.Thread(target=worker, daemon=True).start()
        started = time.monotonic()
        while not state["done"]:
            elapsed = time.monotonic() - started
            spin = spinner[tick % len(spinner)]
            tick += 1
            if self._show_status(
                "JAX warmup in progress\n"
                f"{spin}  {elapsed:5.1f}s elapsed\n\n"
                f"Current: {self._jax_warmup_phase}\n\n"
                f"{self._jax_status_line()}\n\n"
                "First run may take 1-3 minutes.\n"
                "The UI updates while compilation runs.",
                allow_quit=True,
            ):
                raise SystemExit(0)
            time.sleep(0.1)

        if state["error"] is not None:
            raise state["error"]

        elapsed = time.monotonic() - started
        self._show_status(
            "JAX warmup complete\n\n"
            f"{self._jax_status_line()}\n\n"
            f"Finished in {elapsed:.1f}s\n\n"
            "Click an object (or empty for NOOP), then press F.",
        )
        time.sleep(1.2)

    def reset(self, *, show_status: bool = False) -> None:
        if show_status and not self._jax_env_reset:
            self._show_status(
                "Compiling JAX environment...\n"
                "First run can take 1-3 minutes. Please wait."
            )
        self.rng, sub = jax.random.split(self.rng)
        self.obs, self.state = self.env.reset(sub)
        jax.block_until_ready(self.state)
        self._jax_env_reset = True
        self._finish_reset_state()

    def _obj_name(self, idx: int) -> str:
        if 0 <= idx < len(self._obj_names):
            return self._obj_names[idx]
        return f"Obj {idx}"

    def _screen_to_game(self, mx: int, my: int) -> tuple[float, float]:
        ox, oy = self.game_origin
        gx = (mx - ox) / self.game_scale
        gy = (my - oy) / self.game_scale
        return gx, gy

    def _in_game_rect(self, mx: int, my: int) -> bool:
        ox, oy = self.game_origin
        return ox <= mx < ox + self.game_w and oy <= my < oy + self.game_h

    def _clear_target(self, *, reason: str | None = None) -> None:
        self.target_idx = self.no_target_idx
        self.hover_idx = None
        self.last_q_vals = None
        self.last_skill_idx = 0
        self.last_distance = float("nan")
        self.last_angle = float("nan")
        self.last_action_name = "NOOP"
        if reason:
            print(f"Target cleared ({reason}) — back to NOOP.")
        else:
            print("Target cleared — will run NOOP.")

    def _target_finished(
        self, prev_obs, next_obs, target_idx: int
    ) -> tuple[bool, str]:
        """True when the pursued object's skill objective completed this step."""
        if target_idx == self.no_target_idx:
            return False, ""

        prev_active = _object_field_at(prev_obs, "active", target_idx)
        next_active = _object_field_at(next_obs, "active", target_idx)
        was_killed = _object_field_at(next_obs, "was_killed", target_idx)
        was_collected = _object_field_at(next_obs, "was_collected", target_idx)
        label = _object_field_at(next_obs, "semantic_label", target_idx)

        if was_killed > 0:
            return True, "destroyed"
        if was_collected > 0:
            return True, "collected"
        if prev_active > 0 and next_active <= 0:
            if label == LABEL_DESTROY:
                return True, "target gone"
            if label == LABEL_COLLECT:
                return True, "collected"
            if label == LABEL_AVOID:
                return True, "threat gone"
        return False, ""

    def _release_target_after_objective(self, reason: str) -> None:
        self._clear_target(reason=reason)
        self.frozen = True

    def _run_skill_step(self) -> None:
        if self.target_idx == self.no_target_idx:
            self.last_action_name = "NOOP"
            self.last_q_vals = None
            self.last_distance = float("nan")
            self.last_angle = float("nan")
            self.obs, self.state, _reward, done, _info = self.env.step(self.state, 0)
            jax.block_until_ready(self.state)
            self._jax_env_step = True
            self.step_count += 1
            if bool(done):
                self.reset()
            return

        obs = self.obs
        target_obj_idx = int(self.target_idx)
        if int(np.array(obs.objects.active[-1])[target_obj_idx]) <= 0:
            self._release_target_after_objective("target no longer active")
            return

        raw_label = obs.objects.semantic_label[-1, target_obj_idx]
        active_skill_idx = int(label_to_skill_idx(raw_label, self.num_skills))
        self.last_skill_idx = active_skill_idx

        ego_obs = get_ego_obs(
            obs.raw_observation,
            obs.objects,
            jnp.array(target_obj_idx, dtype=jnp.int32),
        )
        ego_obs_batch = ego_obs[jnp.newaxis, ...]

        channels_per_frame = ego_obs_channels_per_frame()
        last_frame_ego = ego_obs[-channels_per_frame:]
        if using_legacy_ego_obs():
            dx_pix = float(last_frame_ego[0] * SCREEN_WIDTH)
            dy_pix = float(last_frame_ego[1] * SCREEN_HEIGHT)
            self.last_distance = float(jnp.sqrt(dx_pix ** 2 + dy_pix ** 2))
            self.last_angle = float(jnp.arctan2(dy_pix, dx_pix))
        else:
            self.last_distance = distance_pixels_from_feature(float(last_frame_ego[0]))
            player = _get_player(obs.raw_observation)
            y_is_bottom_edge = _uses_bottom_edge_y(obs.raw_observation)
            p_cx = float(player.x[-1] + player.width[-1] / 2.0)
            p_cy = float(
                _center_y(
                    player.y[-1],
                    player.height[-1],
                    y_is_bottom_edge=y_is_bottom_edge,
                )
            )
            t_cx = float(
                obs.objects.x[-1, target_obj_idx]
                + obs.objects.width[-1, target_obj_idx] / 2.0
            )
            t_cy = float(
                _center_y(
                    obs.objects.y[-1, target_obj_idx],
                    obs.objects.height[-1, target_obj_idx],
                    y_is_bottom_edge=y_is_bottom_edge,
                )
            )
            self.last_angle = float(jnp.arctan2(t_cy - p_cy, t_cx - p_cx))

        def apply_one(params_for_skill):
            return self.skill_network.apply(
                {"params": params_for_skill},
                ego_obs_batch,
                train=False,
            )

        q_vals = jax.vmap(apply_one)(self.skill_params)
        q_vals = jnp.squeeze(q_vals, axis=1)
        self.last_q_vals = np.array(jax.device_get(q_vals[active_skill_idx]), dtype=np.float32)

        q_active = q_vals[active_skill_idx]
        if self.restrict_fire and active_skill_idx != 0:
            fire_indices = fire_ego_action_indices(self.ego_action_dim)
            q_active = q_active.at[fire_indices].set(-jnp.inf)

        chosen_ego = int(jnp.argmax(q_active))
        global_action = int(
            self.action_mapper(
                chosen_ego,
                obs.raw_observation,
                jnp.array(target_obj_idx, dtype=jnp.int32),
                obs.objects,
            )
        )
        ego_names = [
            "NOOP", "FIRE", "FWD", "RIGHT", "LEFT", "BACK",
            "FWD_R", "FWD_L", "BACK_R", "BACK_L",
            "FWD+F", "RIGHT+F", "LEFT+F", "BACK+F",
            "FWD_R+F", "FWD_L+F", "BACK_R+F", "BACK_L+F",
        ]
        self.last_action_name = (
            ego_names[chosen_ego] if chosen_ego < len(ego_names) else f"ego_{chosen_ego}"
        )

        self.obs, self.state, _reward, done, _info = self.env.step(self.state, global_action)
        jax.block_until_ready(self.state)
        self._jax_env_step = True
        self._jax_skill = True
        self.step_count += 1

        finished, finish_reason = self._target_finished(obs, self.obs, target_obj_idx)
        if finished:
            name = self._obj_name(target_obj_idx)
            self._release_target_after_objective(f"{name}: {finish_reason}")
        elif bool(done):
            self.reset()

    def _render_frame(self) -> None:
        gs = self.state.atari_state.env_state
        frame = np.asarray(jax.device_get(self.renderer.render(gs)), dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        surf = self._pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = self._pygame.transform.scale(surf, (self.game_w, self.game_h))
        self.screen.blit(surf, self.game_origin)

    def _draw_object_overlays(self) -> None:
        if self.obs is None:
            return
        ox0, oy0 = self.game_origin
        xa = np.array(self.obs.objects.x[-1])
        ya = np.array(self.obs.objects.y[-1])
        wa = np.array(self.obs.objects.width[-1])
        ha = np.array(self.obs.objects.height[-1])
        active = np.array(self.obs.objects.active[-1])
        labels = np.array(self.obs.objects.semantic_label[-1])

        for i in range(int(active.shape[0])):
            if active[i] <= 0 or i == 0:
                continue
            x, y, w, h = int(xa[i]), int(ya[i]), int(wa[i]), int(ha[i])
            if w < 1 or h < 1:
                continue
            sx = ox0 + x * self.game_scale
            sy = oy0 + y * self.game_scale
            sw, sh = w * self.game_scale, h * self.game_scale

            label_id = int(labels[i])
            color = LABEL_COLORS.get(label_id, WHITE)
            width = 1
            if i == self.hover_idx:
                color = HOVER_COLOR
                width = 2
            if i == self.target_idx:
                color = TARGET_COLOR
                width = 3

            self._pygame.draw.rect(self.screen, color, (sx, sy, sw, sh), width)
            if self.frozen:
                name = self._obj_name(i)
                tag = LABEL_NAMES.get(label_id, "?")
                text = self.font.render(f"{name} [{tag}]", True, color)
                self.screen.blit(text, (sx, max(oy0, sy - 14)))

    def _draw_panel(self) -> None:
        x0 = self.panel_x
        y = 10
        lh = 16

        def line(text: str, color: str | tuple = WHITE, bold: bool = False) -> None:
            nonlocal y
            font = self.bold_font if bold else self.font
            self.screen.blit(font.render(text, True, color), (x0, y))
            y += lh

        mode = "FROZEN — click object or empty" if self.frozen else "RUNNING"
        line(f"Skill Target Demo  ({self.env_name})", YELLOW, bold=True)
        line(f"Step {self.step_count}   [{mode}]", bold=True)
        jax_color = GREEN if self._jax_ready() else YELLOW
        line(self._jax_status_line(), jax_color)
        y += 4

        if self.target_idx == self.no_target_idx:
            line("Target: NOOP (no object)", GRAY, bold=True)
            if not self.frozen:
                line(f"  action={self.last_action_name}")
        else:
            lab = int(np.array(self.obs.objects.semantic_label[-1])[self.target_idx])
            line(
                f"Target: [{self.target_idx}] {self._obj_name(self.target_idx)}",
                TARGET_COLOR,
                bold=True,
            )
            line(f"  label={LABEL_NAMES.get(lab, '?')}  skill={SKILL_NAMES[self.last_skill_idx]}")
            if not np.isnan(self.last_distance):
                line(f"  d={self.last_distance:.1f}px  θ={self.last_angle:.2f} rad")
            line(f"  action={self.last_action_name}")

        y += 6
        line("Controls:", bold=True)
        line("  Click object  select target (frozen)")
        line("  Click empty   NOOP / no target (frozen)")
        line("  0 / Backspace clear target (frozen)")
        line("  F             freeze / unfreeze")
        line("  N      step once while frozen")
        line("  R      reset episode")
        line("  Q/ESC  quit")

        if self.last_q_vals is not None:
            y += 6
            line(f"Q-values ({SKILL_NAMES[self.last_skill_idx]}):", bold=True)
            q = self.last_q_vals
            best = int(np.argmax(q))
            for j, val in enumerate(q):
                mark = " *" if j == best else ""
                line(f"  ego[{j:2d}] = {val:6.2f}{mark}")

        y += 8
        line("Active objects:", GRAY)
        if self.obs is not None:
            active = np.array(self.obs.objects.active[-1])
            labels = np.array(self.obs.objects.semantic_label[-1])
            for i in range(int(active.shape[0])):
                if active[i] <= 0 or i == 0:
                    continue
                tag = LABEL_NAMES.get(int(labels[i]), "?")
                prefix = ">" if i == self.target_idx else " "
                line(f"{prefix}[{i:2d}] {self._obj_name(i):<14s} {tag}")

    def handle_events(self) -> tuple[bool, bool, bool, bool]:
        quit_ = reset_ = freeze_ = step_ = False
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                quit_ = True
            elif event.type == self._pygame.KEYDOWN:
                if event.key in (self._pygame.K_q, self._pygame.K_ESCAPE):
                    quit_ = True
                elif event.key == self._pygame.K_r:
                    reset_ = True
                elif event.key == self._pygame.K_f:
                    freeze_ = True
                elif event.key == self._pygame.K_n:
                    step_ = True
                elif self.frozen and event.key in (
                    self._pygame.K_0,
                    self._pygame.K_BACKSPACE,
                ):
                    self._clear_target()
            elif event.type == self._pygame.MOUSEMOTION and self.frozen:
                mx, my = event.pos
                if self._in_game_rect(mx, my):
                    gx, gy = self._screen_to_game(mx, my)
                    self.hover_idx = pick_object_at_point(
                        self.obs, gx, gy, skip_indices={0}
                    )
                else:
                    self.hover_idx = None
            elif (
                event.type == self._pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.frozen
            ):
                mx, my = event.pos
                if self._in_game_rect(mx, my):
                    gx, gy = self._screen_to_game(mx, my)
                    picked = pick_object_at_point(self.obs, gx, gy, skip_indices={0})
                    if picked is not None:
                        self.target_idx = picked
                        print(
                            f"Selected target [{picked}] {self._obj_name(picked)} "
                            f"({LABEL_NAMES.get(int(np.array(self.obs.objects.semantic_label[-1])[picked]), '?')})"
                        )
                    else:
                        self._clear_target()
        return quit_, reset_, freeze_, step_

    def run(self) -> None:
        clock = self._pygame.time.Clock()
        running = True

        while running:
            quit_, reset_, freeze_, step_ = self.handle_events()
            if quit_:
                break
            if reset_:
                self.reset()
                continue
            if freeze_:
                self.frozen = not self.frozen

            if not self.frozen:
                self._run_skill_step()
            elif step_:
                self._run_skill_step()

            self.screen.fill(BLACK)
            self._render_frame()
            self._draw_object_overlays()
            self._draw_panel()
            self._pygame.display.flip()
            clock.tick(5 if self.frozen else 30)

        self._pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive skill-network demo with click-to-select targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/skill_target_demo.py --model path/to/skills.safetensors
  python scripts/skill_target_demo.py --experiment seaquest_skills_pretrain \\
      --model ./outputs/models/.../hierarchical_pqn_seaquest_seed42_skills.safetensors

Controls:
  Click object        select target (frozen)
  Click empty / 0     NOOP — no target (frozen)
  F                   freeze / unfreeze
  N                     single-step while frozen
  R                     reset episode
  Q / ESC               quit
""",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to skill Q-network checkpoint (*.safetensors).",
    )
    parser.add_argument(
        "--experiment",
        default="seaquest_skills_pretrain",
        help="Experiment config when no *_config.yaml sits beside the checkpoint.",
    )
    parser.add_argument("--env-name", default=None, help="Override ENV_NAME.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    config = load_base_config(experiment=args.experiment)
    config, _ = load_config_for_checkpoint(args.model, config)
    if args.env_name is not None:
        config["ENV_NAME"] = args.env_name
    if args.seed is not None:
        config["SEED"] = args.seed

    skill_params = load_params_from_file(args.model)
    config = align_config_with_skill_checkpoint(config, skill_params)

    set_ego_obs_format(bool(config.get("USE_LEGACY_EGO_OBS", False)))
    set_ego_size_features(bool(config.get("USE_EGO_SIZE_FEATURES", True)))
    set_ego_velocity_features(bool(config.get("USE_EGO_VELOCITY_FEATURES", True)))
    clip_px = config.get("EGO_DISTANCE_CLIP_MAX_PX", None)
    set_ego_distance_clip_max_pixels(None if clip_px is None else float(clip_px))

    env_name = str(config.get("ENV_NAME", "seaquest"))
    manager = EnvManager()
    components = manager.get_components(env_name)

    print("Building evaluation environment...", flush=True)
    env, renderer = create_eval_env(config)
    skill_params = ensure_skill_params_stack(skill_params, int(config["NUM_SKILLS"]))

    print("Opening window...", flush=True)
    demo = SkillTargetDemo(
        env,
        renderer,
        config=config,
        skill_params=skill_params,
        action_mapper=components.action_mapper,
        no_target_idx=components.no_target_idx,
        env_name=env_name,
    )

    print("Warming up JAX (live progress in the window)...", flush=True)
    demo.warmup()
    dummy_ego = get_ego_obs(
        demo.obs.raw_observation,
        demo.obs.objects,
        jnp.array(0, dtype=jnp.int32),
    )
    validate_skill_checkpoint_ego_dim(config, int(dummy_ego.shape[-1]))
    print(f"{demo._jax_status_line()} — ready to play.", flush=True)
    demo.run()


if __name__ == "__main__":
    main()
