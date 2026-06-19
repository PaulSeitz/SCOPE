"""
Render evaluation videos for sequential_eval or persistence_eval checkpoints.

Loads skill + meta tandem checkpoints and writes one MP4 per condition into a
``videos/`` subfolder next to the checkpoints.

Sequential layout (``--suite sequential``):
  {ckpt_root}/phase2_meta/{env}/*_skills.safetensors + *_meta.safetensors
  {ckpt_root}/phase3_tandem/{env}/...

Persistence layout (``--suite persistence``):
  {ckpt_root}/{variant_key}/{env}/*_skills.safetensors + *_meta.safetensors
  e.g. {ckpt_root}/until_done/seaquest/, {ckpt_root}/no_persist_srl0.5/phoenix/

Usage (from project root):
  python scripts/render_sequential_eval.py
  python scripts/render_sequential_eval.py --suite persistence \\
      --ckpt-root ./outputs/models/persistence_eval
  python scripts/render_sequential_eval.py --suite persistence \\
      --variants until_done no_persist_srl1.0 --envs seaquest
  python scripts/render_sequential_eval.py --suite auto --ckpt-root ./outputs/models/persistence_eval
  python scripts/render_sequential_eval.py --max-frames 1200
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from concept_based_learning.observations.egocentric import (  # noqa: E402
    set_ego_distance_clip_max_pixels,
    set_ego_obs_format,
    set_ego_size_features,
    set_ego_velocity_features,
)
from concept_based_learning.pqn_visualizer import (  # noqa: E402
    load_config_for_checkpoint,
    load_params_from_file,
    run_eval_rollout,
)
from concept_based_learning.video_recorder import (  # noqa: E402
    SKILL_NAMES,
    SidebarRenderer,
    _write_frames_mp4,
    add_distance_angle_text,
    draw_target_outline,
)

SEQUENTIAL_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "sequential_eval"
PERSISTENCE_CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "persistence_eval"
DEFAULT_PERSISTENCE_CKPT_ROOT = PROJECT_ROOT / "outputs" / "models" / "persistence_eval"
DEFAULT_SEQUENTIAL_CKPT_ROOT = Path("/home/paul/Downloads/sequential_eval")
SEED = 44

SEQUENTIAL_ENVS = [
    "seaquest", "pong", "breakout", "spaceinvaders", "phoenix",
    "freeway", "choppercommand", "beamrider", "kangaroo", "skiing", "tennis",
]
PERSISTENCE_ENVS = ["seaquest", "phoenix"]
PERSISTENCE_VARIANT_KEYS = [
    "no_persist_srl0.5",
    "no_persist_srl1.0",
    "until_done",
    "until_done_srl1.0",
    "mission_avoid",
    "mission_avoid_srl1.0",
    "fixed_steps",
    "fixed_steps_srl1.0",
]
SEQUENTIAL_PHASES = ["phase2_meta", "phase3_tandem"]
PHASE_SUFFIX = {
    "phase2_meta": "phase2_meta",
    "phase3_tandem": "phase3_tandem",
}

# (variant_key, persist_mode or None, skill_reward_lambda)
_PERSISTENCE_VARIANTS: list[tuple[str, str | None, float]] = [
    ("no_persist_srl0.5", None, 0.5),
    ("no_persist_srl1.0", None, 1.0),
    ("until_done", "until_done", 0.0),
    ("until_done_srl1.0", "until_done", 1.0),
    ("mission_avoid", "mission_avoid", 0.0),
    ("mission_avoid_srl1.0", "mission_avoid", 1.0),
    ("fixed_steps", "fixed_steps", 0.0),
    ("fixed_steps_srl1.0", "fixed_steps", 1.0),
]


def _apply_persistence_variant(
    cfg: dict[str, Any], variant_key: str
) -> dict[str, Any]:
    by_key = {v[0]: v for v in _PERSISTENCE_VARIANTS}
    if variant_key not in by_key:
        raise KeyError(
            f"Unknown persistence variant {variant_key!r}; "
            f"expected one of {PERSISTENCE_VARIANT_KEYS}"
        )
    _, persist_mode, skill_reward_lambda = by_key[variant_key]
    out = dict(cfg)
    out["SAVE_PATH"] = f"./outputs/models/persistence_eval/{variant_key}"
    out["SKILL_REWARD_LAMBDA"] = skill_reward_lambda
    for key in (
        "META_TARGET_PERSIST_MODE",
        "META_TARGET_PERSIST_STEPS",
        "META_AVOID_INTERRUPT_MARGIN",
    ):
        out.pop(key, None)
    if persist_mode is not None:
        out["META_TARGET_PERSIST_MODE"] = persist_mode
        if persist_mode == "fixed_steps":
            out["META_TARGET_PERSIST_STEPS"] = int(
                out.get("META_TARGET_PERSIST_STEPS", 300)
            )
        if persist_mode == "mission_avoid":
            out["META_AVOID_INTERRUPT_MARGIN"] = float(
                out.get("META_AVOID_INTERRUPT_MARGIN", 0.0)
            )
    return out


def _flatten_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    alg = raw.pop("alg", {})
    raw.update(alg)
    return raw


def _apply_ego_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["SEED"] = SEED
    cfg["USE_LEGACY_EGO_OBS"] = False
    cfg["USE_EGO_SIZE_FEATURES"] = False
    cfg["USE_EGO_VELOCITY_FEATURES"] = False
    cfg.setdefault("EGO_DISTANCE_CLIP_MAX_PX", 20)
    return cfg


def detect_suite(ckpt_root: Path) -> str:
    if (ckpt_root / "phase2_meta").is_dir() or (ckpt_root / "phase3_tandem").is_dir():
        return "sequential"
    for key in PERSISTENCE_VARIANT_KEYS:
        if (ckpt_root / key).is_dir():
            return "persistence"
    if "persistence_eval" in ckpt_root.name:
        return "persistence"
    return "sequential"


def _load_sequential_config(env: str, phase_suffix: str) -> dict[str, Any]:
    path = SEQUENTIAL_CONFIG_DIR / f"{env}_{phase_suffix}.yaml"
    return _apply_ego_defaults(_flatten_yaml(path))


def _load_persistence_config(env: str, variant_key: str) -> dict[str, Any]:
    base_name = (
        "seaquest_tandem_base.yaml"
        if env == "seaquest"
        else "phoenix_tandem_base.yaml"
    )
    cfg = _apply_ego_defaults(
        _apply_persistence_variant(
            _flatten_yaml(PERSISTENCE_CONFIG_DIR / base_name),
            variant_key,
        )
    )
    cfg["ENV_NAME"] = env
    return cfg


def _configure_ego_from_config(cfg: dict[str, Any]) -> None:
    set_ego_obs_format(bool(cfg.get("USE_LEGACY_EGO_OBS", False)))
    set_ego_size_features(bool(cfg.get("USE_EGO_SIZE_FEATURES", True)))
    set_ego_velocity_features(bool(cfg.get("USE_EGO_VELOCITY_FEATURES", True)))
    clip_px = cfg.get("EGO_DISTANCE_CLIP_MAX_PX", None)
    set_ego_distance_clip_max_pixels(
        None if clip_px is None else float(clip_px)
    )


def rollout_to_mp4(
    *,
    cfg: dict[str, Any],
    skills_path: Path,
    meta_path: Path,
    out_path: Path,
    max_frames: int,
) -> bool:
    if not skills_path.exists() or not meta_path.exists():
        print(f"  [skip] missing checkpoint(s): skills={skills_path.exists()}, meta={meta_path.exists()}")
        return False

    print(f"  Rendering → {out_path}")
    try:
        cfg, _ = load_config_for_checkpoint(str(skills_path), cfg)
        _configure_ego_from_config(cfg)

        skill_params = load_params_from_file(str(skills_path))
        meta_params = load_params_from_file(str(meta_path))

        (
            states,
            active_agents,
            combined_qs,
            dones,
            target_obj_idxs,
            objects_bbox,
            distances,
            angles,
            renderer,
        ) = run_eval_rollout(
            cfg,
            params_tree=skill_params,
            meta_params=meta_params,
            max_frames=max_frames,
        )

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
        rasters = jax.vmap(renderer.render)(states_slice)
        frames = np.array(rasters, dtype=np.uint8)
        if frames.shape[-1] == 3:
            frames = np.transpose(frames, (0, 3, 1, 2))

        if target_obj_idxs is not None and objects_bbox is not None:
            ox, oy, ow, oh, oactive = objects_bbox
            for i in range(len(frames)):
                frames[i] = draw_target_outline(
                    frames[i],
                    int(target_obj_idxs[i]),
                    ox[i],
                    oy[i],
                    ow[i],
                    oh[i],
                    oactive[i],
                )

        if distances is not None and angles is not None:
            for i in range(len(frames)):
                frames[i] = add_distance_angle_text(
                    frames[i], float(distances[i]), float(angles[i])
                )

        if active_agents is not None and combined_qs is not None:
            sb = SidebarRenderer(frames[0].shape)
            sidebar_w = sb.sidebar_width
            new_frames = np.zeros(
                (len(frames), frames.shape[1], frames.shape[2], frames.shape[3] + sidebar_w),
                dtype=np.uint8,
            )
            for i in range(len(frames)):
                texts = [
                    (
                        f"{SKILL_NAMES[j] if j < len(SKILL_NAMES) else f'Opt{j}'}:",
                        f"{q:.2f}",
                    )
                    for j, q in enumerate(combined_qs[i])
                ]
                new_frames[i] = sb.render(frames[i], texts, active_agents[i])
            frames = new_frames

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_frames_mp4(frames, str(out_path), fps=30)
        print(f"  Done: {out_path} ({num_frames} frames)")
        return True
    except Exception:
        print(f"  [FAILED] {out_path}")
        traceback.print_exc()
        return False


def render_sequential_one(
    env: str, phase: str, ckpt_root: Path, max_frames: int
) -> bool:
    phase_dir = ckpt_root / phase / env
    skills_path = phase_dir / f"hierarchical_pqn_{env}_seed{SEED}_skills.safetensors"
    meta_path = phase_dir / f"hierarchical_pqn_{env}_seed{SEED}_meta.safetensors"
    out_path = ckpt_root / phase / "videos" / f"{env}.mp4"
    cfg = _load_sequential_config(env, PHASE_SUFFIX[phase])
    return rollout_to_mp4(
        cfg=cfg,
        skills_path=skills_path,
        meta_path=meta_path,
        out_path=out_path,
        max_frames=max_frames,
    )


def render_persistence_one(
    env: str, variant: str, ckpt_root: Path, max_frames: int
) -> bool:
    variant_dir = ckpt_root / variant / env
    skills_path = variant_dir / f"hierarchical_pqn_{env}_seed{SEED}_skills.safetensors"
    meta_path = variant_dir / f"hierarchical_pqn_{env}_seed{SEED}_meta.safetensors"
    out_path = ckpt_root / variant / "videos" / f"{env}.mp4"
    cfg = _load_persistence_config(env, variant)
    return rollout_to_mp4(
        cfg=cfg,
        skills_path=skills_path,
        meta_path=meta_path,
        out_path=out_path,
        max_frames=max_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["auto", "sequential", "persistence"],
        default="auto",
        help="Checkpoint layout (default: auto-detect from --ckpt-root)",
    )
    parser.add_argument(
        "--ckpt-root",
        default=None,
        dest="ckpt_root",
        help="Checkpoint root (default: sequential download path or ./outputs/models/persistence_eval)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=SEQUENTIAL_PHASES,
        choices=SEQUENTIAL_PHASES,
        help="Sequential suite only",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=f"Persistence suite only (default: all). Keys: {', '.join(PERSISTENCE_VARIANT_KEYS)}",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environments to render (suite-specific defaults if omitted)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1800,
        dest="max_frames",
        help="Max frames per rollout (default: 1800 = 60s @ 30fps)",
    )
    args = parser.parse_args()

    if args.ckpt_root is not None:
        ckpt_root = Path(args.ckpt_root)
    elif args.suite == "persistence":
        ckpt_root = DEFAULT_PERSISTENCE_CKPT_ROOT
    else:
        ckpt_root = DEFAULT_SEQUENTIAL_CKPT_ROOT

    if args.suite == "auto":
        suite = detect_suite(ckpt_root)
    else:
        suite = args.suite

    results: list[tuple[str, bool]] = []
    if suite == "sequential":
        envs = args.envs or SEQUENTIAL_ENVS
        for phase in args.phases:
            for env in envs:
                ok = render_sequential_one(env, phase, ckpt_root, args.max_frames)
                results.append((f"{phase}/{env}", ok))
    else:
        variants = args.variants or PERSISTENCE_VARIANT_KEYS
        envs = args.envs or PERSISTENCE_ENVS
        unknown = [v for v in variants if v not in PERSISTENCE_VARIANT_KEYS]
        if unknown:
            parser.error(
                f"Unknown variant(s): {unknown}. "
                f"Choose from: {PERSISTENCE_VARIANT_KEYS}"
            )
        for variant in variants:
            for env in envs:
                ok = render_persistence_one(env, variant, ckpt_root, args.max_frames)
                results.append((f"{variant}/{env}", ok))

    print(f"\nSuite: {suite}  ckpt-root: {ckpt_root}")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    print(f"  {passed}/{len(results)} videos rendered")
    for label, ok in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")


if __name__ == "__main__":
    main()
