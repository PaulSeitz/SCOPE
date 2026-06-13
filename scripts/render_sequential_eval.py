"""
Render evaluation videos for all sequential_eval checkpoints.

Loads skill + meta checkpoints from the downloaded folder and renders one
MP4 per env per phase into a videos/ subfolder alongside the checkpoints.

Usage (from project root):
  python scripts/render_sequential_eval.py
  python scripts/render_sequential_eval.py --ckpt-root /home/paul/Downloads/sequential_eval
  python scripts/render_sequential_eval.py --phases phase2_meta --envs seaquest pong
  python scripts/render_sequential_eval.py --max-frames 1200
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from concept_based_learning.observations.egocentric import (  # noqa: E402
    set_ego_obs_format,
    set_ego_size_features,
    set_ego_velocity_features,
    set_ego_distance_clip_max_pixels,
)
from concept_based_learning.pqn_visualizer import (  # noqa: E402
    load_params_from_file,
    run_eval_rollout,
)
from concept_based_learning.video_recorder import _write_frames_mp4  # noqa: E402
import numpy as np  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "sequential_eval"
SEED = 44

ENVS = [
    "seaquest", "pong", "breakout", "spaceinvaders", "phoenix",
    "freeway", "choppercommand", "beamrider", "kangaroo", "skiing", "tennis",
]
PHASES = ["phase2_meta", "phase3_tandem"]

PHASE_SUFFIX = {
    "phase2_meta":   "phase2_meta",
    "phase3_tandem": "phase3_tandem",
}


def _load_config(env: str, phase_suffix: str) -> dict:
    path = CONFIG_DIR / f"{env}_{phase_suffix}.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    alg = raw.pop("alg", {})
    raw.update(alg)
    raw["SEED"] = SEED
    raw["USE_LEGACY_EGO_OBS"] = False
    raw["USE_EGO_SIZE_FEATURES"] = False
    raw["USE_EGO_VELOCITY_FEATURES"] = False
    raw.setdefault("EGO_DISTANCE_CLIP_MAX_PX", 20)
    return raw


def render_one(env: str, phase: str, ckpt_root: Path, max_frames: int) -> bool:
    phase_dir = ckpt_root / phase / env
    skills_path = phase_dir / f"hierarchical_pqn_{env}_seed{SEED}_skills.safetensors"
    meta_path   = phase_dir / f"hierarchical_pqn_{env}_seed{SEED}_meta.safetensors"

    if not skills_path.exists() or not meta_path.exists():
        print(f"  [skip] {phase}/{env}: checkpoint(s) missing")
        return False

    video_dir = ckpt_root / phase / "videos"
    video_dir.mkdir(exist_ok=True)
    out_path = str(video_dir / f"{env}.mp4")

    print(f"  Rendering {phase}/{env} → {out_path}")
    try:
        cfg = _load_config(env, PHASE_SUFFIX[phase])

        skill_params = load_params_from_file(str(skills_path))
        meta_params  = load_params_from_file(str(meta_path))

        (
            states, active_agents, combined_qs, dones,
            target_obj_idxs, objects_bbox, distances, angles, renderer,
        ) = run_eval_rollout(
            cfg,
            params_tree=skill_params,
            meta_params=meta_params,
            max_frames=max_frames,
        )

        # Render frames via JAX renderer
        import jax
        import jax.numpy as jnp

        raw_states = states.atari_state.env_state
        leaves = jax.tree_util.tree_leaves(raw_states)
        full_length = leaves[0].shape[0] if leaves else 0

        dones_col = dones[:, 0] if len(dones.shape) > 1 else dones
        has_done = jnp.any(dones_col > 0.5)
        first_done = jnp.argmax(dones_col)
        num_frames = int(jnp.where(has_done, first_done + 1, full_length))

        states_slice = jax.tree_util.tree_map(lambda x: x[:num_frames], raw_states)
        from jaxatari.renderers import JAXGameRenderer
        rasters = jax.vmap(renderer.render)(states_slice)
        frames = np.array(rasters, dtype=np.uint8)
        if frames.shape[-1] == 3:
            frames = np.transpose(frames, (0, 3, 1, 2))  # (T,H,W,3) → (T,3,H,W)

        # Draw target outlines
        from concept_based_learning.video_recorder import draw_target_outline, add_distance_angle_text, SidebarRenderer
        if target_obj_idxs is not None and objects_bbox is not None:
            ox, oy, ow, oh, oactive = objects_bbox
            for i in range(len(frames)):
                frames[i] = draw_target_outline(frames[i], int(target_obj_idxs[i]), ox[i], oy[i], ow[i], oh[i], oactive[i])

        if distances is not None and angles is not None:
            for i in range(len(frames)):
                frames[i] = add_distance_angle_text(frames[i], float(distances[i]), float(angles[i]))

        if active_agents is not None and combined_qs is not None:
            sb = SidebarRenderer(frames[0].shape)
            from concept_based_learning.video_recorder import SKILL_NAMES
            new_frames = np.zeros((len(frames), frames.shape[1], frames.shape[2], frames.shape[3] + sb.sidebar_width), dtype=np.uint8)
            for i in range(len(frames)):
                texts = [(f"{SKILL_NAMES[j] if j < len(SKILL_NAMES) else f'Opt{j}'}:", f"{q:.2f}") for j, q in enumerate(combined_qs[i])]
                new_frames[i] = sb.render(frames[i], texts, active_agents[i])
            frames = new_frames

        _write_frames_mp4(frames, out_path, fps=30)
        print(f"  Done: {out_path} ({num_frames} frames)")
        return True
    except Exception:
        print(f"  [FAILED] {phase}/{env}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", default="/home/paul/Downloads/sequential_eval",
                        dest="ckpt_root", help="Root folder with phase2_meta/ and phase3_tandem/")
    parser.add_argument("--phases", nargs="+", default=PHASES, choices=PHASES)
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--max-frames", type=int, default=1800, dest="max_frames",
                        help="Max frames per rollout (default: 1800 = 60s @ 30fps)")
    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root)
    results = []
    for phase in args.phases:
        for env in args.envs:
            ok = render_one(env, phase, ckpt_root, args.max_frames)
            results.append((f"{phase}/{env}", ok))

    print("\n" + "="*60)
    passed = sum(1 for _, ok in results if ok)
    print(f"  {passed}/{len(results)} videos rendered")
    for label, ok in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")


if __name__ == "__main__":
    main()
