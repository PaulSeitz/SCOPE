#!/usr/bin/env python3
"""Clip100 probe on the COLLECT-surface φ8 facing spine (Seaquest + Pong).

Hypothesis: Pong zero-shot fails by ~1px because clip50 saturates distance tiers;
clip100 should expose finer approach steps without changing angle encoding.

Creates:
  - seaquest_skills_ego_pov_clip100_facing_psibins8_collectsurface (+ eval_full)
  - pong_transfer_ego_pov_clip100_facing_psibins8_collectsurface

Usage:
    python scripts/materialize_clip100_collectsurface_probe.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "clip100_collectsurface_probe"
SOURCE_CLIP = 50
TARGET_CLIP = 100
SOURCE_STEM = "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface"
TARGET_STEM = "seaquest_skills_ego_pov_clip100_facing_psibins8_collectsurface"
PONG_STEM = "pong_transfer_ego_pov_clip100_facing_psibins8_collectsurface"


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _bump_clip(text: str) -> str:
    out = text.replace(SOURCE_STEM, TARGET_STEM)
    out = out.replace("clip50", "clip100")
    out = out.replace("suite:thesis_facing", f"suite:{SUITE}")
    out = re.sub(
        rf"EGO_DISTANCE_CLIP_MAX_PX: {SOURCE_CLIP}\b",
        f"EGO_DISTANCE_CLIP_MAX_PX: {TARGET_CLIP}",
        out,
    )
    out = re.sub(
        rf"  EGO_DISTANCE_CLIP_MAX_PX: {SOURCE_CLIP}\b",
        f"  EGO_DISTANCE_CLIP_MAX_PX: {TARGET_CLIP}",
        out,
    )
    return out


def _pong_transfer() -> str:
    return f"""# Pong zero-shot ego — clip100 φ8 (Seaquest COLLECT-surface trained).
PAPER_EVAL_RQ: 2
PAPER_EVAL_PHASE: eval
PAPER_EVAL_KIND: transfer
PAPER_EVAL_MODS: pong
PAPER_EVAL_REQUIRES_RQ1:
- {TARGET_STEM}
PAPER_EVAL_SKILLS_CKPT: ./outputs/models/paper_eval/rq1/{TARGET_STEM}/seaquest/hierarchical_pqn_seaquest_seed44.safetensors
PAPER_EVAL_ARCHITECTURE_OF: {TARGET_STEM}
NUM_SEEDS: 1
ALG_NAME: hierarchical_pqn
TRAIN_MODE: skills_only
ENV_NAME: pong
USE_POV_RELATIVE_MAPPER: true
USE_POV_FIXED_BILATERAL: false
USE_POV_FACING_FRAME: true
EGO_POSE_XY: false
SAVE_PATH: ./outputs/models/paper_eval/rq2/{PONG_STEM}
WANDB_RUN_NAME: po_paper_rq2_transfer_ego_pov_clip100_facing_psibins8
WANDB_TAGS:
- suite:{SUITE}
- rq:2
- env:pong
- phase:zero_shot
- skill_input:ego
- mapper:pov_facing
- ego:phi_bins8
- ablation:clip100
USE_LEGACY_EGO_OBS: false
USE_EGO_SIZE_FEATURES: false
USE_EGO_VELOCITY_FEATURES: false
USE_EGO_BLOCKING_FEATURES: true
USE_EGO_WEAPON_FEATURES: true
USE_EGO_SHOT_FEATURES: false
EGO_DISTANCE_CLIP_MAX_PX: {TARGET_CLIP}
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = bool(args.dry_run)

    train_src = (_RQ1 / f"{SOURCE_STEM}.yaml").read_text(encoding="utf-8")
    eval_src = (_RQ1 / f"{SOURCE_STEM}_eval_full.yaml").read_text(encoding="utf-8")

    train = _bump_clip(train_src)
    train = train.replace(
        "# Fresh RQ1: facing-locked FPS + clip50 + φ8; surface relabeled COLLECT",
        f"# Clip100 probe: φ8 COLLECT-surface (distance clip {TARGET_CLIP}px vs {SOURCE_CLIP}).",
    )
    train = train.replace(
        "sq_paper_rq1_skills_ego_pov_clip50_facing_psibins8_v2",
        "sq_paper_rq1_skills_ego_pov_clip100_facing_psibins8_collectsurface",
    )

    eval_full = _bump_clip(eval_src)
    eval_full = eval_full.replace(
        "# Full Seaquest mod eval for fresh φ8/clip50 COLLECT-surface skills",
        f"# Full Seaquest mod eval clip100 COLLECT-surface @ 27k.",
    )
    eval_full = eval_full.replace(
        "sq_paper_rq1_eval_ego_pov_clip50_facing_psibins8_full_v2",
        "sq_paper_rq1_eval_ego_pov_clip100_facing_psibins8_full",
    )
    eval_full = eval_full.replace(
        "probe:pov_clip50_facing_psibins8_full",
        "probe:pov_clip100_facing_psibins8_full",
    )

    _write(_RQ1 / f"{TARGET_STEM}.yaml", train, dry=dry)
    _write(_RQ1 / f"{TARGET_STEM}_eval_full.yaml", eval_full, dry=dry)
    _write(_RQ2 / f"{PONG_STEM}.yaml", _pong_transfer(), dry=dry)
    print("done.")


if __name__ == "__main__":
    main()
