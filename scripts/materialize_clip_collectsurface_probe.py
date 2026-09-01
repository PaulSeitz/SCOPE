#!/usr/bin/env python3
"""Distance-clip probe on the COLLECT-surface φ8 facing spine (Seaquest + Pong).

Hypothesis: Pong zero-shot fails by ~1px because clip50 saturates distance tiers;
larger clips (100, 150, …) expose finer approach steps without changing angle encoding.

Creates per clip value:
  - seaquest_skills_ego_pov_clip{N}_facing_psibins8_collectsurface (+ eval_full)
  - pong_transfer_ego_pov_clip{N}_facing_psibins8_collectsurface

Usage:
    python scripts/materialize_clip_collectsurface_probe.py --clip 100
    python scripts/materialize_clip_collectsurface_probe.py --clip 150
    python scripts/materialize_clip_collectsurface_probe.py --all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SOURCE_CLIP = 50
SOURCE_STEM = "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface"
DEFAULT_CLIPS = (100, 150)


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _stems(clip: int) -> tuple[str, str, str]:
    train = f"seaquest_skills_ego_pov_clip{clip}_facing_psibins8_collectsurface"
    pong = f"pong_transfer_ego_pov_clip{clip}_facing_psibins8_collectsurface"
    suite = f"clip{clip}_collectsurface_probe"
    return train, pong, suite


def _bump_clip(text: str, *, clip: int, suite: str) -> str:
    train_stem, _, _ = _stems(clip)
    out = text.replace(SOURCE_STEM, train_stem)
    out = out.replace("clip50", f"clip{clip}")
    out = out.replace("suite:thesis_facing", f"suite:{suite}")
    out = re.sub(
        rf"EGO_DISTANCE_CLIP_MAX_PX: {SOURCE_CLIP}\b",
        f"EGO_DISTANCE_CLIP_MAX_PX: {clip}",
        out,
    )
    out = re.sub(
        rf"  EGO_DISTANCE_CLIP_MAX_PX: {SOURCE_CLIP}\b",
        f"  EGO_DISTANCE_CLIP_MAX_PX: {clip}",
        out,
    )
    return out


def _pong_transfer(*, clip: int, suite: str, train_stem: str, pong_stem: str) -> str:
    return f"""# Pong zero-shot ego — clip{clip} φ8 (Seaquest COLLECT-surface trained).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: transfer
PAPER_EVAL_MODS: pong
PAPER_EVAL_REQUIRES_RQ1:
- {train_stem}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{train_stem}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {train_stem}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: pong
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
SAVE_PATH: ./outputs/models/paper_eval/rq2/{pong_stem}
WANDB_RUN_NAME: po_paper_rq2_transfer_ego_pov_clip{clip}_facing_psibins8
WANDB_TAGS:
- suite:{suite}
- rq:2
- env:pong
- phase:zero_shot
- skill_input:ego
- mapper:pov_facing
- ego:phi_bins8
- ablation:clip{clip}
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: {clip}
EGO_PSI_ANGLE_BINS: 8
EGO_LEGAL_ACTION_MASK: false
GAME_MODS: null
EVAL_GAME_MODS: null
SKILL_INPUT_MODE: ego
HEURISTIC_MODE: nearest
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
  MAX_OBJECTS: 2
  EGO_ACTION_DIM: 18
  HIDDEN_SIZE: 128
  NUM_LAYERS: 3
  SKILL_EMBED_DIM: 256
  FRAME_SKIP: 4
  FRAME_STACK_SIZE: 4
  HEURISTIC_MODE: nearest
  SKILL_INPUT_MODE: ego
  SKILL_DEATH_PENALIZE_COLLECT: true
  DESTROY_CONTACT_FAIL: true
  DESTROY_LIVING_PENALTY: true
  USE_EGO_SHOT_FEATURES: false
"""


def materialize_clip(clip: int, *, dry: bool) -> None:
    train_stem, pong_stem, suite = _stems(clip)
    train_src = (_RQ1 / f"{SOURCE_STEM}.yaml").read_text(encoding="utf-8")
    eval_src = (_RQ1 / f"{SOURCE_STEM}_eval_full.yaml").read_text(encoding="utf-8")

    train = _bump_clip(train_src, clip=clip, suite=suite)
    train = re.sub(
        r"# Fresh RQ1: facing-locked FPS \+ clip\d+ \+ φ8.*",
        f"# Clip{clip} probe: φ8 COLLECT-surface (distance clip {clip}px vs {SOURCE_CLIP}).",
        train,
        count=1,
    )
    train = re.sub(
        r"WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_clip\d+_facing_psibins8_v2",
        f"WANDB_RUN_NAME: sq_paper_rq1_skills_ego_pov_clip{clip}_facing_psibins8_collectsurface",
        train,
    )

    eval_full = _bump_clip(eval_src, clip=clip, suite=suite)
    eval_full = re.sub(
        r"# Full Seaquest mod eval for fresh φ8/clip\d+ COLLECT-surface skills.*",
        f"# Full Seaquest mod eval clip{clip} COLLECT-surface @ 27k.",
        eval_full,
        count=1,
    )
    eval_full = re.sub(
        r"WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_clip\d+_facing_psibins8_full_v2",
        f"WANDB_RUN_NAME: sq_paper_rq1_eval_ego_pov_clip{clip}_facing_psibins8_full",
        eval_full,
    )
    eval_full = re.sub(
        r"probe:pov_clip\d+_facing_psibins8_full",
        f"probe:pov_clip{clip}_facing_psibins8_full",
        eval_full,
    )

    _write(_RQ1 / f"{train_stem}.yaml", train, dry=dry)
    _write(_RQ1 / f"{train_stem}_eval_full.yaml", eval_full, dry=dry)
    _write(_RQ2 / f"{pong_stem}.yaml", _pong_transfer(clip=clip, suite=suite, train_stem=train_stem, pong_stem=pong_stem), dry=dry)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", type=int, action="append", help=f"Clip px (repeatable). Default: {DEFAULT_CLIPS}")
    ap.add_argument("--all", action="store_true", help=f"Materialize {DEFAULT_CLIPS}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = bool(args.dry_run)

    clips = list(DEFAULT_CLIPS) if args.all else (args.clip or [100])
    for clip in clips:
        materialize_clip(int(clip), dry=dry)
    print("done.")


if __name__ == "__main__":
    main()
