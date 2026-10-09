#!/usr/bin/env python3
"""Side-by-side flat PQN: JAXAtari Seaquest (left) vs ALE Seaquest (right).

Same checkpoint plays greedily in both envs (independent obs / actions).
Inspired by JAXAtari ``scripts/gameplay_comparison.py``, but the agent is PQN.

Example::

    PYTHONUNBUFFERED=1 PYENV_VERSION=jaxtari-fork JAX_PLATFORMS=cpu \\
      python -u scripts/compare_pqn_jaxatari_ale.py \\
      --model tmp/pqn_paper_t5_oct1/ckpts/seed45/flat_pqn_seaquest_pixel_seed45_vmap0.safetensors \\
      --record tmp/pqn_paper_t5_oct1/viz/pqn_seed45_side_by_side.mp4 \\
      --headless --num-episodes 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_SRC), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.pop("SDL_VIDEODRIVER", None)
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    print("Warning: no DISPLAY — use --headless --record.", file=sys.stderr)

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pygame

from concept_based_learning.baselines.flat_pqn import (
    QNetwork,
    flat_pqn_network_kwargs,
    infer_flat_pqn_action_dim,
    load_flat_pqn_params,
    load_flat_pqn_training_config,
    make_flat_pqn_eval_env,
    normalize_flat_pqn_config,
)
from concept_based_learning.pqn_visualizer import resolve_config_path

# Reuse ALE wrapper + video helpers from visualize_flat_pqn.
from visualize_flat_pqn import (  # type: ignore
    AleSeaquestPixelEnv,
    _base_state_for_render,
)

UPSCALE = 3
PANEL_GAP = 8
LABEL_H = 36
COLOR_BG = (20, 20, 20)
COLOR_WHITE = (255, 255, 255)
COLOR_JAX = (80, 200, 120)
COLOR_ALE = (80, 160, 255)


def _load_config(model_path: str, config_path: str | None) -> dict:
    if config_path:
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(
            OmegaConf.load(resolve_config_path(config_path)), resolve=True
        )
        return normalize_flat_pqn_config(raw)
    return load_flat_pqn_training_config(model_path)


def _fit_panel(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbor resize/pad to a fixed panel size (H, W, 3)."""
    import cv2

    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    h, w = rgb.shape[:2]
    scale = min(target_h / h, target_w / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _compose_frame(
    jax_rgb: np.ndarray,
    ale_rgb: np.ndarray,
    *,
    jax_ret: float,
    ale_ret: float,
    step: int,
    panel_h: int,
    panel_w: int,
    font: pygame.font.Font,
) -> np.ndarray:
    left = _fit_panel(jax_rgb, panel_h, panel_w)
    right = _fit_panel(ale_rgb, panel_h, panel_w)
    gap = np.zeros((panel_h, PANEL_GAP, 3), dtype=np.uint8)
    body = np.concatenate([left, gap, right], axis=1)
    out = np.empty((LABEL_H + panel_h, body.shape[1], 3), dtype=np.uint8)
    out[LABEL_H:] = body

    surf = pygame.Surface((out.shape[1], LABEL_H))
    surf.fill(COLOR_BG)
    t1 = font.render(f"JAXAtari  ret={jax_ret:.0f}  step={step}", True, COLOR_JAX)
    t2 = font.render(f"ALE  ret={ale_ret:.0f}  step={step}", True, COLOR_ALE)
    surf.blit(t1, (8, 8))
    surf.blit(t2, (panel_w + PANEL_GAP + 8, 8))
    out[:LABEL_H] = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side PQN on JAXAtari vs ALE Seaquest."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
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
    game_name = str(config.get("ENV_NAME") or "seaquest").lower()
    if game_name != "seaquest":
        raise SystemExit("Side-by-side compare currently only supports seaquest.")

    params = load_flat_pqn_params(model_path)
    bs_path = model_path.replace(".safetensors", "_bs.safetensors")
    batch_stats = (
        load_flat_pqn_params(bs_path) if Path(bs_path).is_file() else {}
    )
    ckpt_action_dim = infer_flat_pqn_action_dim(params)
    frame_skip = int(config.get("FRAME_SKIP", 4))
    frame_stack = int(config.get("FRAME_STACK_SIZE", 4))
    sticky_prob = 0.25 if bool(config.get("STICKY_ACTIONS", False)) else 0.0
    noop_max = int(config.get("NOOP_MAX", 30))
    resize = config.get("PIXEL_RESIZE_SHAPE", [84, 84])
    if isinstance(resize, list):
        resize = tuple(int(x) for x in resize)

    # ---- JAXAtari (train stack, pixel) ----
    import jaxtari

    jax_env = make_flat_pqn_eval_env(
        object_centric=False,
        game_mods=None,
        config=config,
        env_name=game_name,
        required_action_dim=ckpt_action_dim,
    )
    jax_renderer = jaxtari.make(game_name, mods=None).renderer
    jitted_reset = jax.jit(jax_env.reset)
    jitted_step = jax.jit(jax_env.step)
    master_key = jrandom.PRNGKey(args.seed)
    jax_obs, jax_state = jitted_reset(jrandom.fold_in(master_key, 0))
    jax_obs = jax.block_until_ready(jax_obs)
    jax_rgb = np.asarray(jax_renderer.render(_base_state_for_render(jax_state)))
    reset_counter = 1

    # ---- ALE ----
    ale_env = AleSeaquestPixelEnv(
        seed=args.seed,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        resize_shape=resize,
        max_frames=108_000,
        noop_max=noop_max,
        sticky_prob=sticky_prob,
    )
    ale_obs_np, ale_rgb = ale_env.reset(seed=args.seed)
    ale_obs = jnp.asarray(ale_obs_np)

    action_dim = int(jax_env.action_space().n)
    if ale_env.action_dim != action_dim:
        raise SystemExit(
            f"action_dim mismatch jax={action_dim} ale={ale_env.action_dim}"
        )
    network = QNetwork(
        **flat_pqn_network_kwargs(
            config,
            params,
            action_dim=action_dim,
            object_centric=False,
        )
    )

    @jax.jit
    def get_action(obs_):
        q_vals = network.apply(
            {"params": params, "batch_stats": batch_stats},
            obs_[None, ...],
            train=False,
        )
        return jnp.argmax(q_vals, axis=-1)[0]

    _ = jax.block_until_ready(get_action(jax_obs))
    _ = jax.block_until_ready(get_action(ale_obs))

    # Panel size from native Atari-ish render
    panel_h = max(jax_rgb.shape[0], ale_rgb.shape[0]) * UPSCALE
    panel_w = max(jax_rgb.shape[1], ale_rgb.shape[1]) * UPSCALE

    import cv2

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 18, bold=True)

    window = None
    clock = None
    if not headless:
        pygame.display.set_caption("PQN — JAXAtari (L) vs ALE (R)")
        window = pygame.display.set_mode(
            (panel_w * 2 + PANEL_GAP, panel_h + LABEL_H)
        )
        clock = pygame.time.Clock()
    else:
        print("Headless side-by-side mode.")

    writer = None
    n_written = 0
    if args.record:
        out_path = Path(args.record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame_w = panel_w * 2 + PANEL_GAP
        frame_h = panel_h + LABEL_H
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.record_fps),
            (frame_w, frame_h),
        )
        if not writer.isOpened():
            raise SystemExit(f"Could not open VideoWriter for {out_path}")
        print(f"Recording → {out_path} ({frame_w}x{frame_h} @ {args.record_fps}fps)")

    episode = 0
    step = 0
    jax_ret = 0.0
    ale_ret = 0.0
    jax_done = False
    ale_done = False
    running = True

    def _frame():
        return _compose_frame(
            jax_rgb,
            ale_rgb,
            jax_ret=jax_ret,
            ale_ret=ale_ret,
            step=step,
            panel_h=panel_h,
            panel_w=panel_w,
            font=font,
        )

    def _write(fr: np.ndarray) -> None:
        nonlocal n_written
        if writer is None:
            return
        bgr = cv2.cvtColor(np.asarray(fr, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        n_written += 1

    _write(_frame())

    print(
        f"Model: {model_path}\n"
        f"Seed={args.seed} episodes={args.num_episodes} "
        f"max_steps={args.max_steps or '∞'}"
    )

    while running:
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    running = False

        # Independent greedy actions from each env's own obs.
        if not jax_done:
            jax_action = int(get_action(jax_obs))
            step_out = jitted_step(jax_state, jax_action)
            if len(step_out) == 6:
                jax_obs, jax_state, reward, term, trunc, _ = step_out
                done = bool(jnp.logical_or(term, trunc))
            else:
                jax_obs, jax_state, reward, done, _ = step_out
                done = bool(done)
            jax_ret += float(reward)
            jax_rgb = np.asarray(
                jax_renderer.render(_base_state_for_render(jax_state))
            )
            jax_done = done

        if not ale_done:
            ale_action = int(get_action(ale_obs))
            ale_obs_np, ale_rgb, reward, done, _info = ale_env.step(ale_action)
            ale_obs = jnp.asarray(ale_obs_np)
            ale_ret += float(reward)
            ale_done = bool(done)

        step += 1
        fr = _frame()
        _write(fr)

        if not headless:
            surf = pygame.surfarray.make_surface(fr.transpose(1, 0, 2))
            window.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(args.fps)

        if step % 250 == 0:
            print(
                f"  step={step}  JAXAtari={jax_ret:.0f}"
                f"{' ✓' if jax_done else ''}  "
                f"ALE={ale_ret:.0f}{' ✓' if ale_done else ''}",
                flush=True,
            )

        if args.max_steps > 0 and step >= args.max_steps:
            print(
                f"Hit --max-steps={args.max_steps} "
                f"(jax={jax_ret:.0f}, ale={ale_ret:.0f})"
            )
            break

        if jax_done and ale_done:
            episode += 1
            print(
                f"Episode {episode} finished at step={step}: "
                f"JAXAtari={jax_ret:.0f}  ALE={ale_ret:.0f}"
            )
            if args.num_episodes > 0 and episode >= args.num_episodes:
                break
            # reset both
            jax_obs, jax_state = jitted_reset(
                jrandom.fold_in(master_key, reset_counter)
            )
            reset_counter += 1
            jax_rgb = np.asarray(
                jax_renderer.render(_base_state_for_render(jax_state))
            )
            ale_obs_np, ale_rgb = ale_env.reset()
            ale_obs = jnp.asarray(ale_obs_np)
            jax_ret = ale_ret = 0.0
            jax_done = ale_done = False
            step = 0

    if writer is not None:
        writer.release()
        print(f"Wrote {n_written} frames → {args.record}")

    ale_env.close()
    if not headless:
        pygame.quit()
    print("Comparison ended.")


if __name__ == "__main__":
    main()
