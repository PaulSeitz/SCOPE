#!/usr/bin/env python3
"""Live pygame visualization for flat PQN (pixel) checkpoints.

Supports JAXAtari (train stack) and ALE Seaquest (domain-gap test) backends.

Usage::

    # JAXAtari (matches training env)
    python scripts/visualize_flat_pqn.py \\
      --model tmp/pqn_paper_t5/ckpts/seed45/flat_pqn_seaquest_pixel_seed45_vmap0.safetensors

    # ALE Stella Seaquest (same ckpt; standard 84×84 resize — not native downscale)
    python scripts/visualize_flat_pqn.py \\
      --model tmp/pqn_paper_t5/ckpts/seed45/flat_pqn_seaquest_pixel_seed45_vmap0.safetensors \\
      --backend ale

Controls: P pause, F frame-by-frame, N next frame, R reset, Q/Esc quit.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.pop("SDL_VIDEODRIVER", None)
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    print(
        "Warning: no DISPLAY/WAYLAND_DISPLAY — pygame window may fail.",
        file=sys.stderr,
    )

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pygame

from concept_based_learning.baselines.flat_pqn import (
    QNetwork,
    flat_pqn_network_kwargs,
    infer_flat_pqn_action_dim,
    infer_flat_pqn_object_centric,
    load_flat_pqn_params,
    load_flat_pqn_training_config,
    make_flat_pqn_eval_env,
    normalize_flat_pqn_config,
)
from concept_based_learning.pqn_visualizer import resolve_config_path
from concept_based_learning.training.episode_stats import unwrap_base_env_state

UPSCALE_FACTOR = 4


# ---------------------------------------------------------------------------
# Config / paper audit
# ---------------------------------------------------------------------------


def _load_config(model_path: str, config_path: str | None) -> dict:
    if config_path:
        from omegaconf import OmegaConf

        resolved = resolve_config_path(config_path)
        raw = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
        return normalize_flat_pqn_config(raw)
    return load_flat_pqn_training_config(model_path)


def _print_paper_audit(config: dict[str, Any]) -> None:
    """Compare saved YAML + train-time wrapper defaults vs paper PQN / ALE norms."""
    sticky_yaml = bool(config.get("STICKY_ACTIONS", False))
    # flat_pqn_train: sticky_actions=0.0 if not STICKY_ACTIONS else 0.25
    sticky_effective = 0.25 if sticky_yaml else 0.0
    # AtariWrapper default is 0.25 — we override via flat_pqn_train.
    noop = int(config.get("NOOP_MAX", 30))  # default in train if unset
    episodic = bool(config.get("EPISODIC_LIFE", True))
    first_fire = bool(config.get("FIRST_FIRE", True))
    native = bool(config.get("USE_NATIVE_DOWNSCALING", True))  # train default True
    frame_skip = int(config.get("FRAME_SKIP", 4))
    stack = int(config.get("FRAME_STACK_SIZE", 4))
    # AtariWrapper default — never set in our YAML / flat_pqn_train call:
    max_frames = 108_000

    print("\n=== Wrap-stack audit (pqn_paper_t5 vs paper / ALE norms) ===")
    rows = [
        ("NUM_ENVS / STEPS / EPOCHS / MINIBATCH", "128 / 32 / 2 / 32", f"{config.get('NUM_ENVS')} / {config.get('NUM_STEPS')} / {config.get('NUM_EPOCHS')} / {config.get('NUM_MINIBATCHES')}", "ok"),
        ("LR / decay / clip / γ / λ", "2.5e-4 / none / 10 / 0.99 / 0.65", f"{config.get('LR')} / decay={config.get('LR_LINEAR_DECAY')} / {config.get('MAX_GRAD_NORM')} / {config.get('GAMMA')} / {config.get('LAMBDA')}", "ok"),
        ("ε schedule", "1→0.001 / decay=0.1", f"{config.get('EPS_START')}→{config.get('EPS_FINISH')} / {config.get('EPS_DECAY')}", "ok"),
        ("NORM", "layer_norm, NORM_INPUT=false", f"{config.get('NORM_TYPE')}, NORM_INPUT={config.get('NORM_INPUT')}", "ok"),
        ("FRAME_SKIP / STACK", "4 / 4 (ALE standard)", f"{frame_skip} / {stack}", "ok"),
        ("EPISODIC_LIFE", "True (main paper / CleanRL EnvPool)", f"{episodic} (YAML)", "ok" if episodic else "MISMATCH"),
        ("STICKY_ACTIONS", "0.0 in main PQN; sticky is Dopamine ablation (Fig.12)", f"YAML={sticky_yaml} → effective sticky_prob={sticky_effective} (train overrides AtariWrapper default 0.25)", "ok"),
        ("NOOP_MAX", "typically 30", f"{noop} (default; not in YAML)", "ok"),
        ("FIRST_FIRE", "common Atari practice", f"{first_fire}", "ok"),
        ("max_frames_per_episode", "108000 (30min@60fps ALE)", f"{max_frames} (AtariWrapper default; NOT passed explicitly in flat_pqn_train)", "ok (implicit)"),
        (
            "USE_NATIVE_DOWNSCALING",
            "False for real ALE (cv2/area 84×84). Native is JAXAtari-only hot-swap.",
            f"{native}" + (" ← THIS RUN USED NATIVE" if native else " (ALE-style resize)"),
            "DEVIATION" if native else "ok",
        ),
        ("PIXEL_GRAYSCALE / 84×84", "True / [84,84]", f"{config.get('PIXEL_GRAYSCALE')} / {config.get('PIXEL_RESIZE_SHAPE')}", "ok"),
        ("CLIP_REWARD (train)", "True", f"{config.get('CLIP_REWARD')}", "ok"),
    ]
    for name, paper, ours, flag in rows:
        mark = "✓" if flag.startswith("ok") else "!"
        print(f"  {mark} {name}")
        print(f"      paper/norm: {paper}")
        print(f"      ours:       {ours}")
    if native:
        print(
            "\n  Note: native downscaling re-renders JAXAtari Seaquest at 84×84 inside the\n"
            "  game (apply_native_downscaling). Paper ALE PQN uses standard resize of\n"
            "  210×160 frames — a real preprocess mismatch for ALE transfer / paper parity.\n"
        )
    else:
        print()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _update_pygame(window, image, upscale_factor: int, native_w: int, native_h: int) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    scaled = pygame.transform.scale(
        surface, (native_w * upscale_factor, native_h * upscale_factor)
    )
    window.blit(scaled, (0, 0))
    pygame.display.flip()


def _base_state_for_render(env_state):
    cur = env_state
    for _ in range(8):
        if hasattr(cur, "atari_state"):
            cur = cur.atari_state
            continue
        if hasattr(cur, "env_state"):
            try:
                return unwrap_base_env_state(cur)
            except AttributeError:
                cur = cur.env_state
                continue
        break
    if hasattr(cur, "env_state"):
        return cur.env_state
    return cur


def _resize_gray_84(rgb: np.ndarray, shape: tuple[int, int] = (84, 84)) -> np.ndarray:
    """ALE-style preprocess: grayscale + area resize → (H, W, 1) uint8."""
    import cv2

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return resized[..., None].astype(np.uint8)


# ---------------------------------------------------------------------------
# ALE backend (numpy; policy still JAX)
# ---------------------------------------------------------------------------


class AleSeaquestPixelEnv:
    """Minimal ALE Seaquest matching flat-PQN pixel obs layout ``(4, 84, 84, 1)``.

    Matches train *intent* where possible (skip 4, max-pool, stack 4, sticky 0,
    18 actions). Uses standard 84×84 resize — ALE has no JAXAtari native downscale.
    """

    def __init__(
        self,
        *,
        seed: int,
        frame_skip: int = 4,
        frame_stack: int = 4,
        resize_shape: tuple[int, int] = (84, 84),
        max_frames: int = 108_000,
        noop_max: int = 30,
        sticky_prob: float = 0.0,
    ):
        import ale_py  # noqa: F401
        import gymnasium as gym

        gym.register_envs(ale_py)
        self.frame_skip = int(frame_skip)
        self.frame_stack = int(frame_stack)
        self.resize_shape = tuple(resize_shape)
        self.max_frames = int(max_frames)
        self.noop_max = int(noop_max)
        self.sticky_prob = float(sticky_prob)
        self._rng = np.random.default_rng(seed)
        self._env = gym.make(
            "ALE/Seaquest-v5",
            frameskip=1,
            repeat_action_probability=0.0,  # we apply sticky ourselves if needed
            full_action_space=True,
            render_mode="rgb_array",
        )
        assert self._env.action_space.n == 18, self._env.action_space.n
        self._stack: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._frame_count = 0
        self._last_action = 0
        self._last_rgb: np.ndarray | None = None

    @property
    def action_dim(self) -> int:
        return 18

    def close(self) -> None:
        self._env.close()

    def _obs_from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        return _resize_gray_84(rgb, self.resize_shape)

    def _stacked(self) -> np.ndarray:
        return np.stack(list(self._stack), axis=0)  # (4, H, W, 1)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rgb, _info = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        self._last_rgb = rgb
        self._frame_count = 0
        self._last_action = 0
        # Random no-ops (exclude FIRE-only if we want first-fire — approximate with noops).
        n_noop = int(self._rng.integers(0, self.noop_max + 1))
        for _ in range(n_noop):
            rgb, _, term, trunc, _ = self._env.step(0)
            self._last_rgb = rgb
            self._frame_count += 1
            if term or trunc:
                rgb, _info = self._env.reset()
                self._last_rgb = rgb
                self._frame_count = 0
                break
        # First-fire (common Atari): action 1 = FIRE in full ALE set.
        rgb, _, term, trunc, _ = self._env.step(1)
        self._last_rgb = rgb
        self._frame_count += 1
        if term or trunc:
            rgb, _info = self._env.reset()
            self._last_rgb = rgb
            self._frame_count = 0
        frame = self._obs_from_rgb(self._last_rgb)
        self._stack.clear()
        for _ in range(self.frame_stack):
            self._stack.append(frame)
        return self._stacked(), self._last_rgb

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        if self.sticky_prob > 0.0 and self._rng.random() < self.sticky_prob:
            action = self._last_action
        self._last_action = int(action)

        total_reward = 0.0
        terminated = False
        truncated = False
        # Max-pool over the last two frames of the skip window (DeepMind style).
        pooled = None
        rgb_frames: list[np.ndarray] = []
        for i in range(self.frame_skip):
            rgb, reward, term, trunc, info = self._env.step(action)
            self._last_rgb = rgb
            rgb_frames.append(np.asarray(rgb, dtype=np.uint8))
            self._frame_count += 1
            total_reward += float(reward)
            terminated = terminated or bool(term)
            truncated = truncated or bool(trunc)
            gray = self._obs_from_rgb(rgb)
            pooled = gray if pooled is None else np.maximum(pooled, gray)
            if terminated or truncated:
                break
            if self._frame_count >= self.max_frames:
                truncated = True
                break

        assert pooled is not None
        self._stack.append(pooled)
        done = terminated or truncated
        return (
            self._stacked(),
            self._last_rgb,
            total_reward,
            done,
            {
                "terminated": terminated,
                "truncated": truncated,
                "rgb_frames": rgb_frames,
            },
        )


def _write_rgb_mp4(frames_hwc: list[np.ndarray], path: str, fps: float) -> None:
    """Write RGB uint8 frames (H, W, 3) to mp4 via OpenCV."""
    import cv2

    if not frames_hwc:
        raise SystemExit("No frames to write.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_hwc[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {out}")
    for fr in frames_hwc:
        bgr = cv2.cvtColor(np.asarray(fr, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    print(f"Saved video ({len(frames_hwc)} frames @ {fps:g} fps) → {out.resolve()}")


def _upscale_rgb(rgb: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.asarray(rgb, dtype=np.uint8)
    import cv2

    h, w = rgb.shape[:2]
    return cv2.resize(
        np.asarray(rgb, dtype=np.uint8),
        (w * factor, h * factor),
        interpolation=cv2.INTER_NEAREST,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a flat PQN checkpoint (JAXAtari or ALE Seaquest)."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--game", type=str, default=None)
    parser.add_argument("--mods", nargs="+", type=str, default=None)
    parser.add_argument("--object-centric", action="store_true", default=None)
    parser.add_argument("--pixel", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("jaxatari", "ale"),
        default="jaxatari",
        help="Env backend. 'ale' = Stella Seaquest via gymnasium/ale_py (pixel only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after N agent steps (0 = unlimited). Useful with --record.",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Write an mp4 to this path (implies headless if no display needed).",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=60.0,
        help="mp4 FPS for --record (ALE raw frames during skip → 60 looks realtime).",
    )
    parser.add_argument(
        "--record-upscale",
        type=int,
        default=4,
        help="Nearest-neighbor upscale for recorded frames (default 4).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a pygame window (auto if --record and no DISPLAY).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip printing the paper/wrap-stack audit.",
    )
    args = parser.parse_args()

    if args.record and args.num_episodes == 0:
        args.num_episodes = 1
    headless = bool(args.headless) or (
        bool(args.record)
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    model_path = resolve_config_path(args.model)
    if not model_path or not Path(model_path).is_file():
        raise SystemExit(f"Model not found: {args.model}")

    config = _load_config(model_path, args.config)
    game_name = (args.game or config.get("ENV_NAME") or "seaquest").lower()

    if args.pixel or args.backend == "ale":
        object_centric = False
    elif args.object_centric is not None:
        object_centric = True
    else:
        object_centric = infer_flat_pqn_object_centric(model_path, config)

    if args.backend == "ale" and object_centric:
        raise SystemExit("--backend ale only supports pixel flat PQN (not OC).")
    if args.backend == "ale" and game_name != "seaquest":
        raise SystemExit("--backend ale currently only implements Seaquest.")

    print(f"Backend: {args.backend}")
    print(f"Game: {game_name}")
    print(f"Object-centric: {object_centric}")
    print(f"Loading model from: {model_path}")
    if not args.no_audit:
        _print_paper_audit(config)

    params = load_flat_pqn_params(model_path)
    bs_path = model_path.replace(".safetensors", "_bs.safetensors")
    if Path(bs_path).is_file():
        batch_stats = load_flat_pqn_params(bs_path)
        print(f"Loaded batch stats from: {bs_path}")
    else:
        print(f"Warning: no batch-stats file at {bs_path}")
        batch_stats = {}

    ckpt_action_dim = infer_flat_pqn_action_dim(params)
    resize = config.get("PIXEL_RESIZE_SHAPE", [84, 84])
    if isinstance(resize, list):
        resize = tuple(int(x) for x in resize)

    sticky_prob = 0.25 if bool(config.get("STICKY_ACTIONS", False)) else 0.0
    frame_skip = int(config.get("FRAME_SKIP", 4))
    frame_stack = int(config.get("FRAME_STACK_SIZE", 4))
    noop_max = int(config.get("NOOP_MAX", 30))

    # ---- build env + first obs / render frame ----
    jaxatari_renderer = None
    jitted_reset = jitted_step = None
    ale_env: AleSeaquestPixelEnv | None = None
    recorded: list[np.ndarray] = []

    if args.backend == "ale":
        ale_env = AleSeaquestPixelEnv(
            seed=args.seed,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            resize_shape=resize,
            max_frames=108_000,
            noop_max=noop_max,
            sticky_prob=sticky_prob,
        )
        action_dim = ale_env.action_dim
        if ckpt_action_dim is not None and int(ckpt_action_dim) != action_dim:
            raise SystemExit(
                f"Checkpoint action_dim={ckpt_action_dim} != ALE Seaquest n={action_dim}"
            )
        print("Resetting ALE env...")
        obs_np, rgb = ale_env.reset(seed=args.seed)
        obs = jnp.asarray(obs_np)
        render_frame = rgb
        if args.record:
            recorded.append(_upscale_rgb(rgb, args.record_upscale))
    else:
        import jaxtari

        mods = list(args.mods) if args.mods else []
        env = make_flat_pqn_eval_env(
            object_centric=object_centric,
            game_mods=mods,
            config=config,
            env_name=game_name,
            required_action_dim=ckpt_action_dim,
        )
        jaxatari_renderer = jaxtari.make(game_name, mods=mods or None).renderer
        action_dim = int(env.action_space().n)
        jitted_reset = jax.jit(env.reset)
        jitted_step = jax.jit(env.step)
        print("Resetting JAXAtari env (may compile)...")
        master_key = jrandom.PRNGKey(args.seed)
        obs, env_state = jitted_reset(jrandom.fold_in(master_key, 0))
        obs = jax.block_until_ready(obs)
        render_frame = jaxatari_renderer.render(_base_state_for_render(env_state))
        reset_counter = 1
        if args.record:
            recorded.append(_upscale_rgb(np.asarray(render_frame), args.record_upscale))

    network = QNetwork(
        **flat_pqn_network_kwargs(
            config,
            params,
            action_dim=action_dim,
            object_centric=object_centric,
        )
    )
    print(
        f"Network: action_dim={action_dim}, "
        f"hidden={config.get('HIDDEN_SIZE', 128)}, "
        f"layers={config.get('NUM_LAYERS', 2)}, "
        f"norm={config.get('NORM_TYPE', 'layer_norm')}"
    )

    @jax.jit
    def get_action(obs_):
        q_vals = network.apply(
            {"params": params, "batch_stats": batch_stats},
            obs_[None, ...],
            train=False,
        )
        return jnp.argmax(q_vals, axis=-1)[0]

    window = None
    clock = None
    if not headless:
        pygame.init()
        pygame.display.set_caption(f"Flat PQN [{args.backend}] — {game_name}")
        render_h, render_w = np.asarray(render_frame).shape[:2]
        window = pygame.display.set_mode(
            (render_w * UPSCALE_FACTOR, render_h * UPSCALE_FACTOR)
        )
        clock = pygame.time.Clock()
    else:
        print("Headless mode (no pygame window).")

    print("Warmup policy (may compile)...")
    _ = jax.block_until_ready(get_action(obs))

    running = True
    pause = False
    frame_by_frame = False
    next_frame = False
    episode_count = 0
    total_reward = 0.0
    step_count = 0
    if args.backend == "jaxatari":
        master_key = jrandom.PRNGKey(args.seed)
        reset_counter = 1

    if not headless:
        print(
            "\nControls:\n"
            "  P - Pause/Resume\n"
            "  F - Toggle frame-by-frame\n"
            "  N - Next frame (frame-by-frame)\n"
            "  R - Reset episode\n"
            "  ESC/Q - Quit\n"
        )

    def _do_reset():
        nonlocal obs, env_state, render_frame, total_reward, step_count, reset_counter, rgb
        total_reward = 0.0
        step_count = 0
        if args.backend == "ale":
            assert ale_env is not None
            obs_np, rgb = ale_env.reset()
            obs = jnp.asarray(obs_np)
            render_frame = rgb
        else:
            reset_key = jrandom.fold_in(master_key, reset_counter)
            obs, env_state = jitted_reset(reset_key)
            reset_counter += 1
            render_frame = jaxatari_renderer.render(_base_state_for_render(env_state))
        if args.record and episode_count < (args.num_episodes or 1):
            # Don't keep appending reset frames across episodes into one clip
            # unless still recording the first episode only — handled below.
            pass

    rgb = render_frame  # for ale branch typing
    env_state = locals().get("env_state", None)

    while running:
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_p:
                        pause = not pause
                        print("Paused" if pause else "Resumed")
                    elif event.key == pygame.K_f:
                        frame_by_frame = not frame_by_frame
                        print(f"Frame-by-frame: {'ON' if frame_by_frame else 'OFF'}")
                    elif event.key == pygame.K_n:
                        next_frame = True
                    elif event.key == pygame.K_r:
                        _do_reset()
                        print("Episode reset")

            if pause or (frame_by_frame and not next_frame):
                _update_pygame(window, render_frame, UPSCALE_FACTOR, render_w, render_h)
                clock.tick(args.fps)
                continue

        next_frame = False
        action = int(get_action(obs))
        if args.verbose:
            print(f"Step {step_count}: action={action}", end="")

        if args.backend == "ale":
            assert ale_env is not None
            obs_np, rgb, reward, done, info = ale_env.step(action)
            obs = jnp.asarray(obs_np)
            render_frame = rgb
            done_b = bool(done)
            if args.record:
                for fr in info.get("rgb_frames", [rgb]):
                    recorded.append(_upscale_rgb(fr, args.record_upscale))
        else:
            step_out = jitted_step(env_state, action)
            if len(step_out) == 6:
                obs, env_state, reward, terminated, truncated, _info = step_out
                done = jnp.logical_or(terminated, truncated)
            else:
                obs, env_state, reward, done, _info = step_out
            render_frame = jaxatari_renderer.render(_base_state_for_render(env_state))
            done_b = bool(done)
            reward = float(reward)
            if args.record:
                recorded.append(
                    _upscale_rgb(np.asarray(render_frame), args.record_upscale)
                )

        total_reward += float(reward)
        step_count += 1
        if args.verbose:
            print(f", reward={float(reward):.1f}, total={total_reward:.1f}")

        if not headless:
            _update_pygame(window, render_frame, UPSCALE_FACTOR, render_w, render_h)
            clock.tick(args.fps)

        if args.max_steps > 0 and step_count >= args.max_steps:
            print(
                f"Hit --max-steps={args.max_steps} "
                f"(return so far={total_reward:.1f})."
            )
            running = False
            break

        if done_b:
            episode_count += 1
            print(
                f"Episode {episode_count} finished: "
                f"steps={step_count}, return={total_reward:.1f}"
            )
            if args.num_episodes > 0 and episode_count >= args.num_episodes:
                print(f"\nCompleted {args.num_episodes} episodes.")
                running = False
            else:
                _do_reset()

    if args.record:
        _write_rgb_mp4(recorded, args.record, fps=args.record_fps)

    if ale_env is not None:
        ale_env.close()
    if not headless:
        pygame.quit()
    print("\nVisualization ended.")


if __name__ == "__main__":
    main()
