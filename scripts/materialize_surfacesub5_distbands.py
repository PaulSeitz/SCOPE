#!/usr/bin/env python3
"""Exclusive near/mid/far distance + train sticky on the surfacesub5 φ8 spine.

Hypothesis: overlapping bins / nested flags reconstruct Seaquest-scale metres.
Exclusive one-hots ``d<12 / 12–40 / ≥40`` keep a contact bit (clip10) plus one
mid band for DESTROY without a 50px analog ruler. Sticky is Machado 0.25 at
train only (eval wrapper still forces 0.0).

    python scripts/materialize_surfacesub5_distbands.py
    python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_distbands --gpus 0 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "ego_min_noweapon_surfacesub5_distbands"
SPINE = "facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SRC_TRAIN = f"seaquest_skills_ego_pov_clip50_{SPINE}"
DST_TRAIN = f"seaquest_skills_ego_pov_distbands3_{SPINE}_sticky"

DIST_BLOCK = """\
EGO_DISTANCE_CLIP_MAX_PX: null
EGO_DISTANCE_HARD_EDGES_PX:
- 12
- 40
"""


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _family_paths(train_stem: str) -> dict[str, Path]:
    tok = train_stem.removeprefix("seaquest_skills_ego_pov_")
    return {
        "train": _RQ1 / f"{train_stem}.yaml",
        "eval": _RQ1 / f"{train_stem}_eval_full.yaml",
        "pong": _RQ2 / f"pong_transfer_ego_pov_{tok}.yaml",
        "beam": _RQ2 / f"beamrider_transfer_ego_pov_{tok}.yaml",
        "phoenix": _RQ2 / f"phoenix_transfer_ego_pov_{tok}.yaml",
    }


def _transform(text: str) -> str:
    src_tok = SRC_TRAIN.removeprefix("seaquest_skills_ego_pov_")
    dst_tok = DST_TRAIN.removeprefix("seaquest_skills_ego_pov_")
    out = text.replace(src_tok, dst_tok)
    out = out.replace("suite:ego_min_noweapon_surfacesub5_clipsweep", f"suite:{SUITE}")
    out = out.replace("- suite:ego_min_noweapon_heu_ablation\n", "")
    out = out.replace("EGO_DISTANCE_CLIP_MAX_PX: 50\n", DIST_BLOCK)
    out = out.replace("STICKY_ACTIONS: false", "STICKY_ACTIONS: true")
    out = out.replace("- ego:clip50", "- ego:distbands3\n- ego:sticky")
    return out


def materialize(*, dry: bool) -> None:
    src = _family_paths(SRC_TRAIN)
    dst = _family_paths(DST_TRAIN)
    for key, src_path in src.items():
        _write(dst[key], _transform(src_path.read_text(encoding="utf-8")), dry=dry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    materialize(dry=args.dry)


if __name__ == "__main__":
    main()
