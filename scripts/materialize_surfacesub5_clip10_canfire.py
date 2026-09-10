#!/usr/bin/env python3
"""clip10 surfacesub5 + can_fire only (facing_enemy off).

Probe whether restoring the DESTROY ``can_fire`` channel lifts clip10's weak
SQ/BR without killing its consistent Pong COLLECT. Same surfacesub5 heuristic
spine as the clipsweep; weapon extras beyond can_fire stay off.

    python scripts/materialize_surfacesub5_clip10_canfire.py
    python scripts/run_paper_eval.py --suite ego_min_surfacesub5_clip10_canfire \\
        --gpus 0 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SPINE_SRC = "facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SPINE_DST = "facing_psibins8_v2_noblocking_canfire_surfacesub5"
SRC_TRAIN = f"seaquest_skills_ego_pov_clip10_{SPINE_SRC}"
DST_TRAIN = f"seaquest_skills_ego_pov_clip10_{SPINE_DST}"
SUITE = "ego_min_surfacesub5_clip10_canfire"


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
    out = text.replace(SPINE_SRC, SPINE_DST)
    out = out.replace(
        "suite:ego_min_noweapon_surfacesub5_clipsweep", f"suite:{SUITE}"
    )
    out = out.replace("USE_EGO_WEAPON_FEATURES: false", "USE_EGO_WEAPON_FEATURES: true")
    # Insert facing_enemy: false after weapon line if missing.
    if "USE_EGO_FACING_ENEMY_FEATURES" not in out:
        out = out.replace(
            "USE_EGO_WEAPON_FEATURES: true\n",
            "USE_EGO_WEAPON_FEATURES: true\nUSE_EGO_FACING_ENEMY_FEATURES: false\n",
        )
    else:
        out = out.replace(
            "USE_EGO_FACING_ENEMY_FEATURES: true",
            "USE_EGO_FACING_ENEMY_FEATURES: false",
        )
    out = out.replace("- ego:noweapon", "- ego:can_fire_only")
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
