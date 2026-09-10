#!/usr/bin/env python3
"""Materialize surfacesub5 train-time randomized distance-clip arm.

Train samples ``EGO_DISTANCE_CLIP_CHOICES_PX`` per env per episode; eval and
zero-shot transfer keep fixed ``EGO_DISTANCE_CLIP_MAX_PX: 50`` (competence lock)
so tables stay comparable to clip50.

Choices span the knife edge seen in the clipsweep: clip10 transferred, clip50
was competent in-domain. Excludes 0 (zeros the channel) and noclip.

    python scripts/materialize_surfacesub5_cliprand.py
    python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_cliprand \\
        --gpus 0 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SPINE = "facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SRC_TRAIN = f"seaquest_skills_ego_pov_clip50_{SPINE}"
DST_TRAIN = f"seaquest_skills_ego_pov_cliprand_{SPINE}"
SUITE = "ego_min_noweapon_surfacesub5_cliprand"

# Keep clip50 as the default/eval fallback; choices drive train-time sampling only.
TRAIN_DIST_BLOCK = """\
EGO_DISTANCE_CLIP_MAX_PX: 50
EGO_DISTANCE_CLIP_CHOICES_PX:
- 10
- 20
- 30
- 50
- 80
"""

EVAL_DIST_BLOCK = """\
EGO_DISTANCE_CLIP_MAX_PX: 50
EGO_DISTANCE_CLIP_CHOICES_PX: null
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


def _transform(text: str, *, train: bool) -> str:
    src_tok = SRC_TRAIN.removeprefix("seaquest_skills_ego_pov_")
    dst_tok = DST_TRAIN.removeprefix("seaquest_skills_ego_pov_")
    out = text.replace(src_tok, dst_tok)
    out = out.replace(
        "suite:ego_min_noweapon_surfacesub5_clipsweep", f"suite:{SUITE}"
    )
    out = out.replace("- suite:ego_min_noweapon_heu_ablation\n", "")
    dist = TRAIN_DIST_BLOCK if train else EVAL_DIST_BLOCK
    out = out.replace("EGO_DISTANCE_CLIP_MAX_PX: 50\n", dist)
    out = out.replace("- ego:clip50", "- ego:cliprand")
    return out


def materialize(*, dry: bool) -> None:
    src = _family_paths(SRC_TRAIN)
    dst = _family_paths(DST_TRAIN)
    for key, src_path in src.items():
        _write(
            dst[key],
            _transform(src_path.read_text(encoding="utf-8"), train=(key == "train")),
            dry=dry,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    materialize(dry=args.dry)


if __name__ == "__main__":
    main()
