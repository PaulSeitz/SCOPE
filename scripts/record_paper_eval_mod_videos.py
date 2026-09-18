#!/usr/bin/env python3
"""
Record local MP4s of a trained paper_eval agent on each env mod (frozen weights).

Does **not** train or finetune — loads a checkpoint and rolls out once per mod.

Examples:
  # RQ1 ego skills on all Seaquest paper mods
  python scripts/record_paper_eval_mod_videos.py \\
    --skills ./outputs/models/paper_eval/rq1/seaquest_skills_ego/seaquest/hierarchical_pqn_seaquest_seed44.safetensors \\
    --env seaquest

  # Subset of mods, short clips
  python scripts/record_paper_eval_mod_videos.py \\
    --skills .../hierarchical_pqn_seaquest_seed44.safetensors \\
    --env seaquest --mods base gravity lethal_divers --max-frames 300

  # With learned meta (tandem)
  python scripts/record_paper_eval_mod_videos.py \\
    --skills .../hierarchical_pqn_seaquest_seed44_skills.safetensors \\
    --meta .../hierarchical_pqn_seaquest_seed44_meta.safetensors \\
    --env seaquest --paper-core

  # Trial subset (base + first non-control mod)
  python scripts/record_paper_eval_mod_videos.py --skills ... --env seaquest --trial
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_JAXATARI_FORK = Path("/home/paul/Documents/JAXAtari-seaquest-robustness")
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "videos" / "paper_eval"


def _prepend_jaxatari(root: Path | None) -> None:
    if root is None:
        return
    src = root / "src"
    if not src.is_dir():
        raise SystemExit(f"JAXAtari root missing src/: {root}")
    sys.path.insert(0, str(src))
    print(f"Using JAXAtari from {src}")


def _resolve_jaxatari(cli: str | None) -> Path | None:
    if cli:
        return Path(cli).expanduser().resolve()
    env = os.environ.get("PAPER_EVAL_JAXATARI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if DEFAULT_JAXATARI_FORK.is_dir():
        return DEFAULT_JAXATARI_FORK
    return None


def _load_config_for_skills(
    skills_path: Path,
    *,
    env_name: str,
    experiment: str | None,
) -> dict:
    from concept_based_learning.paper_eval.eval_harness import (
        _overlay_ego_flags_from_skill_config,
    )
    from concept_based_learning.environments.datatypes import (
        configure_label_routing_from_config,
    )
    from concept_based_learning.observations.egocentric import (
        configure_ego_obs_from_config,
    )
    from concept_based_learning.pqn_visualizer import (
        _print_ego_runtime_summary,
        align_config_with_skill_checkpoint,
        find_checkpoint_config_path,
        load_base_config,
        load_config_for_checkpoint,
        load_params_from_file,
    )

    # Prefer the training sidecar next to the checkpoint. If missing, try the
    # experiment yaml named after the run stem (parent of env/).
    exp = experiment
    sidecar = find_checkpoint_config_path(str(skills_path))
    if exp is None and sidecar is None:
        # .../tmp/eval18/<stem>/<env>/ckpt.safetensors → stem
        stem = skills_path.parent.parent.name
        for rq in ("rq1", "rq2", "rq3"):
            cand = (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "experiment"
                / "paper_eval"
                / rq
                / f"{stem}.yaml"
            )
            if cand.is_file():
                exp = f"paper_eval/{rq}/{stem}"
                print(f"No sidecar; falling back to experiment={exp}")
                break
        if exp is None:
            raise SystemExit(
                f"No training *_config.yaml next to {skills_path} and no matching "
                f"configs/experiment/paper_eval/*/{{stem}}.yaml for stem={stem!r}. "
                "Sync the sidecar or pass --experiment."
            )

    base = load_base_config(experiment=exp)
    base, loaded_sidecar = load_config_for_checkpoint(str(skills_path), base)
    if not loaded_sidecar and sidecar is None and exp is None:
        raise SystemExit(f"Failed to load training config for {skills_path}")
    base["ENV_NAME"] = env_name
    params = load_params_from_file(str(skills_path))
    base = _overlay_ego_flags_from_skill_config(base, skills_path)
    base = align_config_with_skill_checkpoint(base, params)
    base["WANDB_MODE"] = "disabled"
    configure_ego_obs_from_config(base)
    configure_label_routing_from_config(base)
    _print_ego_runtime_summary(base)
    return base, params


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skills",
        required=True,
        help="Path to skill checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Optional meta checkpoint (*_meta.safetensors); else heuristic targets",
    )
    parser.add_argument("--env", required=True, help="Environment name (mod allowlist key)")
    parser.add_argument(
        "--mods",
        nargs="*",
        default=None,
        help="Mod keys to record (default: full allowlist / paper-core / trial)",
    )
    parser.add_argument(
        "--paper-core",
        action="store_true",
        help="Only mods listed under paper_core in the allowlist",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Only base + first non-control mod",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Directory for MP4s (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument(
        "--stop-on-done",
        dest="stop_on_done",
        action="store_true",
        help="End the rollout at the first env done (one full game).",
    )
    parser.add_argument(
        "--no-stop-on-done",
        dest="stop_on_done",
        action="store_false",
        help="Keep rolling until --max-frames even after done (multi-episode demo).",
    )
    parser.set_defaults(stop_on_done=True)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Log/render every N env steps (VIDEO_FRAME_STRIDE). 4 is common in paper suites.",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--experiment",
        default=None,
        help="Hydra experiment stem if no sidecar *_config.yaml next to checkpoint",
    )
    parser.add_argument("--jaxatari-root", default=None)
    parser.add_argument(
        "--heuristic-mode",
        default=None,
        help="Override HEURISTIC_MODE (e.g. nearest / env_specific) to match transfer eval",
    )
    parser.add_argument(
        "--no-outline",
        action="store_true",
        default=False,
        help="Disable target-outline painting for pixel_embed CNN input",
    )
    parser.add_argument(
        "--black-pixel",
        action="store_true",
        default=False,
        help="Zero out pixel_obs before the skill CNN (vision ablation)",
    )
    parser.add_argument(
        "--list-mods",
        action="store_true",
        help="Print selected mods and exit",
    )
    args = parser.parse_args()

    _prepend_jaxatari(_resolve_jaxatari(args.jaxatari_root))

    from concept_based_learning.paper_eval.mods import mod_specs_for_env
    from concept_based_learning.pqn_visualizer import (
        apply_visualizer_config_overrides,
        load_params_from_file,
        run_eval_rollout,
    )
    from concept_based_learning.video_recorder import video_callback, wait_for_video_finish

    skills_path = Path(args.skills).expanduser().resolve()
    if not skills_path.is_file():
        raise SystemExit(f"Skills checkpoint not found: {skills_path}")

    mod_specs = mod_specs_for_env(
        args.env,
        paper_core_only=args.paper_core,
        trial=args.trial,
        keys=args.mods,
    )
    if not mod_specs:
        raise SystemExit(f"No mods selected for env={args.env}")

    if args.list_mods:
        for m in mod_specs:
            print(f"  {m.key:28s} {list(m.mods) or '(base)'}")
        return

    cfg, params = _load_config_for_skills(
        skills_path, env_name=args.env, experiment=args.experiment
    )
    cfg["SEED"] = int(args.seed)
    cfg["VIDEO_MAX_FRAMES"] = int(args.max_frames)
    cfg["VIDEO_STOP_ON_DONE"] = bool(args.stop_on_done)
    cfg["VIDEO_FRAME_STRIDE"] = max(1, int(args.frame_stride))
    apply_visualizer_config_overrides(
        cfg,
        no_outline=bool(args.no_outline),
        black_pixel=bool(args.black_pixel),
        heuristic_mode=args.heuristic_mode,
    )
    if args.heuristic_mode:
        print(f"HEURISTIC_MODE override → {cfg['HEURISTIC_MODE']}")
    if args.no_outline:
        print("SKILL_PIXEL_PAINT_TARGET_OUTLINE=False")
    if args.black_pixel:
        print("SKILL_PIXEL_BLACKOUT=True")

    meta_params = None
    meta_path = args.meta
    if meta_path is None:
        sibling = skills_path.with_name(
            skills_path.name.replace("_skills.safetensors", "_meta.safetensors")
        )
        if "_skills.safetensors" in skills_path.name and sibling.is_file():
            meta_path = str(sibling)
            print(f"Auto-detected meta: {sibling}")
    if meta_path is not None:
        meta_params = load_params_from_file(str(meta_path))

    out_root = Path(args.out_dir) / args.env / skills_path.stem
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing videos under {out_root}")

    for mod in mod_specs:
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg["GAME_MODS"] = list(mod.mods) if mod.mods else None
        eval_cfg.pop("EVAL_GAME_MODS", None)
        out_mp4 = out_root / f"{mod.key}.mp4"
        print(f"\n=== Recording {mod.key} → {out_mp4.name} (mods={list(mod.mods) or 'none'}) ===")
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
            pov_overlay,
        ) = run_eval_rollout(
            eval_cfg,
            params_tree=params,
            meta_params=meta_params,
            max_frames=args.max_frames,
        )
        video_callback(
            states,
            active_agents,
            combined_qs,
            dones,
            0,
            renderer,
            target_obj_idxs=target_obj_idxs,
            objects_bbox=objects_bbox,
            distances=distances,
            angles=angles,
            video_key=f"paper_eval/{args.env}/{mod.key}",
            video_output_path=str(out_mp4),
            draw_target_outline_overlay=bool(
                eval_cfg.get("VISUALIZER_DRAW_TARGET_OUTLINE", True)
            ),
            pov_overlay=pov_overlay,
        )
        wait_for_video_finish()

    print(f"\nDone. Videos in {out_root}")


if __name__ == "__main__":
    main()
