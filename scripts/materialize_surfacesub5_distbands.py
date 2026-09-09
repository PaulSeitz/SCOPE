#!/usr/bin/env python3
"""Two independent surfacesub5 probes: exclusive distance bands vs clip50 sticky.

1. distbands3 — exclusive one-hots ``d<12 / 12–40 / ≥40``, no sticky.
2. clip50_sticky — locked clip50 scalar + Machado sticky 0.25 at train
   (eval wrapper still forces 0.0). Chosen over clip10 because clip50 is the
   noisy competence lock; clip10 sticky already exists in distabl.

    python scripts/materialize_surfacesub5_distbands.py
    python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_distbands --gpus 0 --wait-deps --lean
    python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_clip50_sticky --gpus 1 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SPINE = "facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SRC_TRAIN = f"seaquest_skills_ego_pov_clip50_{SPINE}"

DIST_BLOCK = """\
EGO_DISTANCE_CLIP_MAX_PX: null
EGO_DISTANCE_HARD_EDGES_PX:
- 12
- 40
"""

ARMS = (
    {
        "dst_train": f"seaquest_skills_ego_pov_distbands3_{SPINE}",
        "suite": "ego_min_noweapon_surfacesub5_distbands",
        "distbands": True,
        "sticky": False,
        "tag_from": "- ego:clip50",
        "tag_to": "- ego:distbands3",
    },
    {
        "dst_train": f"seaquest_skills_ego_pov_clip50_{SPINE}_sticky",
        "suite": "ego_min_noweapon_surfacesub5_clip50_sticky",
        "distbands": False,
        "sticky": True,
        "tag_from": "- ego:clip50",
        "tag_to": "- ego:clip50\n- ego:sticky",
    },
)


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


def _transform(text: str, arm: dict) -> str:
    src_tok = SRC_TRAIN.removeprefix("seaquest_skills_ego_pov_")
    dst_tok = arm["dst_train"].removeprefix("seaquest_skills_ego_pov_")
    out = text.replace(src_tok, dst_tok)
    out = out.replace("suite:ego_min_noweapon_surfacesub5_clipsweep", f"suite:{arm['suite']}")
    out = out.replace("- suite:ego_min_noweapon_heu_ablation\n", "")
    if arm["distbands"]:
        out = out.replace("EGO_DISTANCE_CLIP_MAX_PX: 50\n", DIST_BLOCK)
    if arm["sticky"]:
        out = out.replace("STICKY_ACTIONS: false", "STICKY_ACTIONS: true")
    out = out.replace(arm["tag_from"], arm["tag_to"])
    return out


def materialize(*, dry: bool) -> None:
    src = _family_paths(SRC_TRAIN)
    for arm in ARMS:
        dst = _family_paths(arm["dst_train"])
        for key, src_path in src.items():
            _write(
                dst[key],
                _transform(src_path.read_text(encoding="utf-8"), arm),
                dry=dry,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    materialize(dry=args.dry)


if __name__ == "__main__":
    main()
