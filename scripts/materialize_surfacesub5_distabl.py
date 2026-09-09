#!/usr/bin/env python3
"""Distance / sticky probes on the noweapon+surfacesub5 φ8 spine.

Hypothesis: a short or non-metric distance channel forces COLLECT/DESTROY to
learn "go toward the object" instead of Seaquest-scale approach, which should
show up as more consistent Pong tracking (clip10 already ties 20–20).

Arms (each: SQ train → full mod eval → Pong/BR/PH ZS):
  clip0         — distance channel zeroed (direction only). Same obs dim.
  distbins8     — 8 triangular bins, no hard clip (coarse range, no 50px cap).
  clip10_sticky — clip10 recipe with ALE sticky actions *at train only*
                  (eval wrapper still forces sticky off).

Usage:
    python scripts/materialize_surfacesub5_distabl.py
    python scripts/materialize_surfacesub5_distabl.py --dry
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "ego_min_noweapon_surfacesub5_distabl"
SPINE = "facing_psibins8_v2_noblocking_noweapon_surfacesub5"

CLIP10_TRAIN = f"seaquest_skills_ego_pov_clip10_{SPINE}"
CLIP50_TRAIN = f"seaquest_skills_ego_pov_clip50_{SPINE}"

DISTBINS_BLOCK = """\
EGO_DISTANCE_CLIP_MAX_PX: null
EGO_DISTANCE_BINS: 8
EGO_DISTANCE_BIN_CENTERS_PX:
- 0
- 12
- 25
- 40
- 60
- 90
- 130
- 190
EGO_DISTANCE_BIN_MAX_PX: 220
"""


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _read(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text:
        raise FileNotFoundError(path)
    return text


def _retarget_suite(text: str) -> str:
    out = text.replace(
        "suite:ego_min_noweapon_surfacesub5_clipsweep", f"suite:{SUITE}"
    )
    out = out.replace("- suite:ego_min_noweapon_heu_ablation\n", "")
    return out


def _rename_stem(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def _family_paths(train_stem: str) -> dict[str, Path]:
    return {
        "train": _RQ1 / f"{train_stem}.yaml",
        "eval": _RQ1 / f"{train_stem}_eval_full.yaml",
        "pong": _RQ2 / f"pong_transfer_ego_pov_{train_stem.removeprefix('seaquest_skills_ego_pov_')}.yaml",
        "beam": _RQ2 / f"beamrider_transfer_ego_pov_{train_stem.removeprefix('seaquest_skills_ego_pov_')}.yaml",
        "phoenix": _RQ2 / f"phoenix_transfer_ego_pov_{train_stem.removeprefix('seaquest_skills_ego_pov_')}.yaml",
    }


def _write_family(
    src_train: str,
    dst_train: str,
    *,
    transform,
    dry: bool,
) -> None:
    src = _family_paths(src_train)
    dst = _family_paths(dst_train)
    src_tok = src_train.removeprefix("seaquest_skills_ego_pov_")
    dst_tok = dst_train.removeprefix("seaquest_skills_ego_pov_")
    for key, src_path in src.items():
        text = transform(_read(src_path), key)
        # One token swap covers train stems, eval_full, and transfer names.
        text = _rename_stem(text, src_tok, dst_tok)
        text = _retarget_suite(text)
        _write(dst[key], text, dry=dry)


def _clip0_transform(text: str, _key: str) -> str:
    text = text.replace("EGO_DISTANCE_CLIP_MAX_PX: 10", "EGO_DISTANCE_CLIP_MAX_PX: 0")
    text = text.replace("- ego:clip10", "- ego:clip0\n- ego:dist_zero")
    return text


def _sticky_transform(text: str, _key: str) -> str:
    text = text.replace("STICKY_ACTIONS: false", "STICKY_ACTIONS: true")
    if "- ego:sticky" not in text:
        text = text.replace("- ego:clip10", "- ego:clip10\n- ego:sticky")
    return text


def _distbins_transform(text: str, _key: str) -> str:
    text = text.replace(
        "EGO_DISTANCE_CLIP_MAX_PX: 50\n",
        DISTBINS_BLOCK,
    )
    text = text.replace("- ego:clip50", "- ego:distbins8")
    return text


def materialize(*, dry: bool) -> None:
    _write_family(
        CLIP10_TRAIN,
        f"seaquest_skills_ego_pov_clip0_{SPINE}",
        transform=_clip0_transform,
        dry=dry,
    )
    _write_family(
        CLIP50_TRAIN,
        f"seaquest_skills_ego_pov_distbins8_{SPINE}",
        transform=_distbins_transform,
        dry=dry,
    )
    _write_family(
        CLIP10_TRAIN,
        f"seaquest_skills_ego_pov_clip10_{SPINE}_sticky",
        transform=_sticky_transform,
        dry=dry,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    materialize(dry=args.dry)


if __name__ == "__main__":
    main()
