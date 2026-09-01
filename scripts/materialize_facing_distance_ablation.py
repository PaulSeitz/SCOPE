#!/usr/bin/env python3
"""Distance ablations on the facing φ8 COLLECT-surface spine.

Hypothesis: fine-grained normalized distance is an overfitting channel; direction
(ψ/φ bins) should carry transfer signal without it.

Arms (run sequentially: train → eval_full → pong → beamrider per arm):
  nodist      — direction only: ``EGO_DISTANCE_CLIP_MAX_PX: 0`` (channel zeroed).
  distflags5  — 5 nested distance tiers via
                ``EGO_DISTANCE_FLAG_THRESHOLDS_PX: [15, 25, 50, 100, 200]``.

Usage:
    python scripts/materialize_facing_distance_ablation.py
    python scripts/materialize_facing_distance_ablation.py --arm nodist
    python scripts/materialize_facing_distance_ablation.py --arm distflags5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SOURCE_STEM = "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface"
SUITE = "facing_distance_ablation"

# Nested flags: cumulative tiers out to 200px (no clip).
DISTFLAGS_THRESHOLDS_PX = (15, 25, 50, 100, 200)

ARMS: dict[str, dict[str, str]] = {
    "nodist": {
        "train_stem": "seaquest_skills_ego_pov_clip0_facing_psibins8_collectsurface",
        "pong_stem": "pong_transfer_ego_pov_clip0_facing_psibins8_collectsurface",
        "beam_stem": "beamrider_transfer_ego_pov_clip0_facing_psibins8_collectsurface",
        "probe": "pov_clip0_facing_psibins8_collectsurface",
        "ablation_tag": "ego:dist_zero",
    },
    "distflags5": {
        "train_stem": "seaquest_skills_ego_pov_facing_psibins8_collectsurface_distflags5",
        "pong_stem": "pong_transfer_ego_pov_facing_psibins8_collectsurface_distflags5",
        "beam_stem": "beamrider_transfer_ego_pov_facing_psibins8_collectsurface_distflags5",
        "probe": "pov_facing_psibins8_distflags5",
        "ablation_tag": "ego:distflags5",
    },
}


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _replace_suite_tags(text: str) -> str:
    return text.replace("suite:thesis_facing", f"suite:{SUITE}")


def _distflags_yaml_block() -> str:
    lines = ["EGO_DISTANCE_CLIP_MAX_PX: null", "EGO_DISTANCE_FLAG_THRESHOLDS_PX:"]
    lines.extend(f"- {t}" for t in DISTFLAGS_THRESHOLDS_PX)
    return "\n".join(lines) + "\n"


def _nodist_train(text: str, meta: dict[str, str]) -> str:
    out = text.replace(SOURCE_STEM, meta["train_stem"])
    out = out.replace("clip50", "clip0")
    out = _replace_suite_tags(out)
    out = re.sub(
        r"# Fresh RQ1: facing-locked FPS \+ clip\d+ \+ φ8.*",
        "# Direction-only ablation: φ8 COLLECT-surface, distance channel zeroed (clip0).",
        out,
        count=1,
    )
    out = re.sub(r"EGO_DISTANCE_CLIP_MAX_PX: 50\b", "EGO_DISTANCE_CLIP_MAX_PX: 0", out)
    out = re.sub(
        r"WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_clip\d+_facing_psibins8_v2",
        "WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_clip0_facing_psibins8_collectsurface",
        out,
    )
    if f"- {meta['ablation_tag']}" not in out:
        out = out.replace(
            "- destroy:living_penalty",
            f"- destroy:living_penalty\n- {meta['ablation_tag']}\n- probe:{meta['probe']}",
        )
    return out


def _nodist_eval(text: str, meta: dict[str, str]) -> str:
    out = text.replace(SOURCE_STEM, meta["train_stem"])
    out = out.replace("clip50", "clip0")
    out = _replace_suite_tags(out)
    out = re.sub(
        r"# Full Seaquest mod eval for fresh φ8/clip\d+ COLLECT-surface skills.*",
        "# Full Seaquest mod eval — direction-only (clip0) COLLECT-surface @ 27k.",
        out,
        count=1,
    )
    out = re.sub(r"EGO_DISTANCE_CLIP_MAX_PX: 50\b", "EGO_DISTANCE_CLIP_MAX_PX: 0", out)
    out = re.sub(
        r"probe:pov_clip\d+_facing_psibins8_full",
        f"probe:{meta['probe']}_full",
        out,
    )
    out = re.sub(
        r"WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_clip\d+_facing_psibins8_full_v2",
        "WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_clip0_facing_psibins8_full",
        out,
    )
    return out


def _distflags_train(text: str, meta: dict[str, str]) -> str:
    out = text.replace(SOURCE_STEM, meta["train_stem"])
    out = _replace_suite_tags(out)
    out = re.sub(
        r"# Fresh RQ1: facing-locked FPS \+ clip\d+ \+ φ8.*",
        "# 5-tier distance flags ablation: φ8 COLLECT-surface (nested tiers to 200px).",
        out,
        count=1,
    )
    out = re.sub(
        r"EGO_DISTANCE_CLIP_MAX_PX: 50\n",
        _distflags_yaml_block(),
        out,
        count=1,
    )
    out = re.sub(
        r"WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_clip\d+_facing_psibins8_v2",
        "WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_facing_psibins8_distflags5_collectsurface",
        out,
    )
    if f"- {meta['ablation_tag']}" not in out:
        out = out.replace(
            "- destroy:living_penalty",
            f"- destroy:living_penalty\n- {meta['ablation_tag']}\n- probe:{meta['probe']}",
        )
    return out


def _distflags_eval(text: str, meta: dict[str, str]) -> str:
    out = text.replace(SOURCE_STEM, meta["train_stem"])
    out = _replace_suite_tags(out)
    out = re.sub(
        r"# Full Seaquest mod eval for fresh φ8/clip\d+ COLLECT-surface skills.*",
        "# Full Seaquest mod eval — distflags5 COLLECT-surface @ 27k.",
        out,
        count=1,
    )
    out = re.sub(
        r"EGO_DISTANCE_CLIP_MAX_PX: 50\n",
        _distflags_yaml_block(),
        out,
        count=1,
    )
    out = re.sub(
        r"probe:pov_clip\d+_facing_psibins8_full",
        f"probe:{meta['probe']}_full",
        out,
    )
    out = re.sub(
        r"WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_clip\d+_facing_psibins8_full_v2",
        "WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_facing_psibins8_distflags5_full",
        out,
    )
    return out


def _transfer(
    *,
    meta: dict[str, str],
    env: str,
    stem_key: str,
    distflags: bool,
) -> str:
    stem = meta[stem_key]
    if distflags:
        dist_block = _distflags_yaml_block()
        ablation = "ablation:distflags5"
        wandb = f"{'po' if env == 'pong' else 'br'}_paper_rq2_transfer_ego_pov_facing_psibins8_distflags5"
    else:
        dist_block = "EGO_DISTANCE_CLIP_MAX_PX: 0\n"
        ablation = "ablation:dist_zero"
        wandb = f"{'po' if env == 'pong' else 'br'}_paper_rq2_transfer_ego_pov_clip0_facing_psibins8"

    heuristic = "nearest" if env == "pong" else "env_specific"
    max_objects = 2 if env == "pong" else 31

    return f"""# {env.title()} zero-shot ego — facing φ8 COLLECT-surface distance ablation.
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: transfer
PAPER_EVAL_MODS: {env}
PAPER_EVAL_REQUIRES_RQ1:
- {meta["train_stem"]}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{meta["train_stem"]}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {meta["train_stem"]}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: {env}
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
SAVE_PATH: ./outputs/models/paper_eval/rq2/{stem}
WANDB_RUN_NAME: {wandb}
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:{env}
- phase:zero_shot
- skill_input:ego
- mapper:pov_facing
- ego:phi_bins8
- {ablation}
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
{dist_block}EGO_PSI_ANGLE_BINS: 8
EGO_LEGAL_ACTION_MASK: false
GAME_MODS: null
EVAL_GAME_MODS: null
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: {heuristic}
SKILL_DEATH_PENALIZE_COLLECT: true
RECORD_VIDEO: false
VIDEO_AT_END: false
DESTROY_CONTACT_FAIL: true
DESTROY_LIVING_PENALTY: true
alg:
  NUM_SKILLS: 4
  SKILL_NAMES:
  - DESTROY
  - COLLECT
  - AVOID
  - NAVIGATE
  MAX_OBJECTS: {max_objects}
  EGO_ACTION_DIM: 18
  HIDDEN_SIZE: 128
  NUM_LAYERS: 3
  SKILL_EMBED_DIM: 256
  FRAME_SKIP: 4
  FRAME_STACK_SIZE: 4
  HEURISTIC_MODE: {heuristic}
  SKILL_INPUT_MODE: ego
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
"""


def materialize_arm(arm: str, *, dry: bool) -> None:
    meta = ARMS[arm]
    train_src = (_RQ1 / f"{SOURCE_STEM}.yaml").read_text(encoding="utf-8")
    eval_src = (_RQ1 / f"{SOURCE_STEM}_eval_full.yaml").read_text(encoding="utf-8")
    distflags = arm == "distflags5"

    if arm == "nodist":
        train = _nodist_train(train_src, meta)
        eval_full = _nodist_eval(eval_src, meta)
    elif arm == "distflags5":
        train = _distflags_train(train_src, meta)
        eval_full = _distflags_eval(eval_src, meta)
    else:
        raise ValueError(f"unknown arm: {arm}")

    _write(_RQ1 / f"{meta['train_stem']}.yaml", train, dry=dry)
    _write(_RQ1 / f"{meta['train_stem']}_eval_full.yaml", eval_full, dry=dry)
    _write(
        _RQ2 / f"{meta['pong_stem']}.yaml",
        _transfer(meta=meta, env="pong", stem_key="pong_stem", distflags=distflags),
        dry=dry,
    )
    _write(
        _RQ2 / f"{meta['beam_stem']}.yaml",
        _transfer(meta=meta, env="beamrider", stem_key="beam_stem", distflags=distflags),
        dry=dry,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        choices=sorted(ARMS),
        action="append",
        help="Arm to materialize (repeatable; default: both).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = bool(args.dry_run)

    arms = args.arm or sorted(ARMS)
    for arm in arms:
        materialize_arm(arm, dry=dry)
    print("done.")


if __name__ == "__main__":
    main()
